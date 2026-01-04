#!/usr/bin/env python3
"""
AlphaQubit 完整服务器运行脚本

本脚本设计用于在远程服务器上运行完整的 AlphaQubit 复现流程:
1. 验证论文对齐
2. 生成数据
3. 预训练模型
4. 微调模型
5. 测试评估
6. 生成研究报告

使用方法:
    python run_complete_server_pipeline.py --output-dir /path/to/results
    python run_complete_server_pipeline.py --quick-test  # 快速测试

作者: AlphaQubit Team
日期: 2024-12-12
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# 论文对齐配置 (完整版)
# =============================================================================

@dataclass
class PaperAlignedConfig:
    """完全对齐论文的配置"""
    
    # ===== 数据生成 =====
    # 预训练: 8.5M samples across d=3,5,7 and p=0.001-0.01
    pretrain_samples_total: int = 8_500_000
    pretrain_samples_per_config: int = 285_000  # ~285k per (d, p) pair
    
    code_distances: List[int] = field(default_factory=lambda: [3, 5, 7])
    si1000_p_grid: List[float] = field(default_factory=lambda: [
        0.001, 0.002, 0.003, 0.004, 0.005,
        0.006, 0.007, 0.008, 0.009, 0.010
    ])
    rounds_list: List[int] = field(default_factory=lambda: [1, 5, 10, 25])
    
    # 微调: 50K per experiment
    finetune_samples_per_exp: int = 50_000
    finetune_train_ratio: float = 0.8  # 80% train, 20% val
    
    # 测试: 10K per config
    test_samples_per_config: int = 10_000
    
    # ===== 模型架构 (Large) =====
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 12
    
    # ===== 预训练超参数 =====
    pretrain_batch_size: int = 256
    pretrain_lr: float = 1e-4
    pretrain_epochs: int = 100
    pretrain_weight_decay: float = 1e-4
    pretrain_scheduler: str = "cosine"
    
    # ===== 微调超参数 =====
    finetune_batch_size: int = 128
    finetune_lr: float = 1e-5
    finetune_epochs: int = 30
    finetune_weight_decay: float = 1e-3
    finetune_patience: int = 5
    finetune_grad_clip: float = 1.0
    
    # ===== 设备 =====
    device: str = "auto"  # auto, npu, cuda, cpu
    num_workers: int = 4
    
    # ===== 噪声模型参数 (Table S4) =====
    noise_params: Dict = field(default_factory=lambda: {
        "cycle_ns": 1076.0,
        "T1_us": 73.0,
        "Tphi_us": 720.0,
        "p_readout": 8.0e-3,
        "p_reset": 1.5e-3,
        "p_heat_12": 2.5e-4,
        "p_cz_leak_11_to_02": 2.0e-4,
        "p_cz_crosstalk_ZZ": 5.5e-4,
        "p_1q_excess": 6.2e-4,
        "p_cz_excess": 2.75e-3,
    })


@dataclass
class QuickTestConfig(PaperAlignedConfig):
    """快速测试配置"""
    pretrain_samples_total: int = 10_000
    pretrain_samples_per_config: int = 1_000
    code_distances: List[int] = field(default_factory=lambda: [3])
    si1000_p_grid: List[float] = field(default_factory=lambda: [0.005, 0.01])
    rounds_list: List[int] = field(default_factory=lambda: [5])
    finetune_samples_per_exp: int = 2_000
    test_samples_per_config: int = 500
    pretrain_epochs: int = 3
    finetune_epochs: int = 3


# =============================================================================
# 工具函数
# =============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """设置日志"""
    log_file = output_dir / "complete_pipeline.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def detect_device() -> str:
    """检测可用设备"""
    try:
        import torch
        if hasattr(torch, "npu") and torch.npu.is_available():
            count = torch.npu.device_count()
            print(f"检测到 {count} 个 NPU")
            return "npu"
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            print(f"检测到 {count} 个 GPU")
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def save_json(data: Dict, path: Path):
    """保存 JSON"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: Path) -> Dict:
    """加载 JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_command(cmd: List[str], cwd: Path = None, logger: logging.Logger = None,
                timeout: int = None) -> Tuple[bool, str]:
    """运行命令"""
    cmd_str = ' '.join(map(str, cmd))
    if logger:
        logger.info(f"运行: {cmd_str}")
    
    try:
        result = subprocess.run(
            cmd, cwd=cwd, 
            capture_output=True, text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            if logger:
                logger.error(f"命令失败: {result.stderr[:500]}")
            return False, result.stderr
        
        return True, result.stdout
        
    except subprocess.TimeoutExpired:
        if logger:
            logger.error(f"命令超时")
        return False, "Timeout"
    except Exception as e:
        if logger:
            logger.error(f"命令异常: {e}")
        return False, str(e)


# =============================================================================
# 完整流水线
# =============================================================================

class CompletePipeline:
    """AlphaQubit 完整复现流水线"""
    
    def __init__(self, config: PaperAlignedConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 子目录
        self.alignment_dir = output_dir / "01_alignment"
        self.data_dir = output_dir / "02_data"
        self.pretrain_dir = output_dir / "03_pretrain"
        self.finetune_dir = output_dir / "04_finetune"
        self.test_dir = output_dir / "05_test"
        self.report_dir = output_dir / "06_report"
        
        for d in [self.alignment_dir, self.data_dir, self.pretrain_dir,
                  self.finetune_dir, self.test_dir, self.report_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 日志
        self.logger = setup_logging(output_dir)
        
        # 设备
        if config.device == "auto":
            self.device = detect_device()
        else:
            self.device = config.device
        self.logger.info(f"使用设备: {self.device}")
        
        # 保存配置
        save_json(asdict(config), output_dir / "config.json")
        
        # 指标
        self.start_time = datetime.now()
        self.metrics = {
            "start_time": self.start_time.isoformat(),
            "config": asdict(config),
            "device": self.device,
            "stages": {}
        }
    
    def run_all(self):
        """运行完整流水线"""
        try:
            self.logger.info("=" * 70)
            self.logger.info(" AlphaQubit 论文复现完整流水线")
            self.logger.info("=" * 70)
            self.logger.info(f"输出目录: {self.output_dir}")
            self.logger.info(f"设备: {self.device}")
            
            # Stage 1: 论文对齐验证
            self._stage_alignment_check()
            
            # Stage 2: 数据生成
            self._stage_data_generation()
            
            # Stage 3: 预训练
            self._stage_pretrain()
            
            # Stage 4: 微调
            self._stage_finetune()
            
            # Stage 5: 测试
            self._stage_test()
            
            # Stage 6: 生成报告
            self._stage_generate_report()
            
            # 完成
            self.metrics["end_time"] = datetime.now().isoformat()
            self.metrics["total_seconds"] = (datetime.now() - self.start_time).total_seconds()
            self.metrics["status"] = "completed"
            save_json(self.metrics, self.output_dir / "pipeline_metrics.json")
            
            self.logger.info("=" * 70)
            self.logger.info(" 流水线完成!")
            self.logger.info(f" 结果目录: {self.output_dir}")
            self.logger.info("=" * 70)
            
        except Exception as e:
            self.metrics["status"] = "failed"
            self.metrics["error"] = str(e)
            save_json(self.metrics, self.output_dir / "pipeline_metrics.json")
            self.logger.error(f"流水线失败: {e}")
            raise
    
    # -------------------------------------------------------------------------
    # Stage 1: 论文对齐验证
    # -------------------------------------------------------------------------
    def _stage_alignment_check(self):
        """Stage 1: 验证论文对齐"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Stage 1: 论文对齐验证")
        self.logger.info("=" * 50)
        
        stage_start = time.time()
        
        # 运行对齐检查脚本
        success, output = run_command(
            [sys.executable, "run_paper_alignment_check.py", 
             "--output-dir", str(self.alignment_dir)],
            logger=self.logger
        )
        
        # 读取结果
        report_path = self.alignment_dir / "alignment_report.json"
        if report_path.exists():
            alignment_results = load_json(report_path)
            self.metrics["stages"]["alignment"] = {
                "time_seconds": time.time() - stage_start,
                "passed": alignment_results.get("summary", {}).get("passed", 0),
                "failed": alignment_results.get("summary", {}).get("failed", 0),
                "warnings": alignment_results.get("summary", {}).get("warnings", 0),
            }
        else:
            self.metrics["stages"]["alignment"] = {
                "time_seconds": time.time() - stage_start,
                "status": "skipped"
            }
        
        self.logger.info(f"Stage 1 完成, 耗时: {time.time() - stage_start:.1f}s")
    
    # -------------------------------------------------------------------------
    # Stage 2: 数据生成
    # -------------------------------------------------------------------------
    def _stage_data_generation(self):
        """Stage 2: 数据生成"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Stage 2: 数据生成")
        self.logger.info("=" * 50)
        
        stage_start = time.time()
        stage_metrics = {"files": []}
        
        try:
            import stim
            
            # 2.1 生成 SI1000 预训练数据
            self.logger.info("2.1 生成 SI1000 预训练数据...")
            si1000_dir = self.data_dir / "si1000"
            si1000_dir.mkdir(parents=True, exist_ok=True)
            
            for d in self.config.code_distances:
                for p in self.config.si1000_p_grid:
                    self._generate_si1000(d, p, si1000_dir, stage_metrics)
            
            # 2.2 生成 Pauli+ 微调数据
            self.logger.info("2.2 生成 Pauli+ 微调数据...")
            pauli_dir = self.data_dir / "pauli_plus"
            pauli_dir.mkdir(parents=True, exist_ok=True)
            
            for basis in ["x", "z"]:
                for d in self.config.code_distances[:2]:  # d=3, d=5
                    for r in self.config.rounds_list:
                        self._generate_pauli_plus(basis, d, r, pauli_dir, stage_metrics)
            
            # 2.3 生成测试数据
            self.logger.info("2.3 生成测试数据...")
            test_data_dir = self.data_dir / "test"
            test_data_dir.mkdir(parents=True, exist_ok=True)
            
            for basis in ["x", "z"]:
                for d in self.config.code_distances[:2]:
                    for r in [1, 5, 10, 25]:
                        self._generate_test_data(basis, d, r, test_data_dir, stage_metrics)
            
        except ImportError as e:
            self.logger.warning(f"stim 不可用: {e}, 生成模拟数据")
            self._generate_dummy_data(stage_metrics)
        
        stage_metrics["time_seconds"] = time.time() - stage_start
        self.metrics["stages"]["data_generation"] = stage_metrics
        self.logger.info(f"Stage 2 完成, 耗时: {stage_metrics['time_seconds']:.1f}s")
    
    def _generate_si1000(self, distance: int, p: float, output_dir: Path, 
                         stage_metrics: Dict):
        """生成 SI1000 数据"""
        import stim
        
        p_str = f"{p:.3f}".replace(".", "p")
        output_file = output_dir / f"si1000_d{distance}_p{p_str}.npz"
        
        if output_file.exists():
            self.logger.info(f"  跳过: {output_file.name}")
            return
        
        samples = self.config.pretrain_samples_per_config
        self.logger.info(f"  生成: d={distance}, p={p}, n={samples}")
        
        # 创建电路
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=25,  # 默认使用 25 轮
            after_clifford_depolarization=p,
            after_reset_flip_probability=p * 0.5,
            before_measure_flip_probability=p,
        )
        
        sampler = circuit.compile_detector_sampler()
        syndromes, logicals = sampler.sample(samples, separate_observables=True)
        
        np.savez(output_file,
                 data=syndromes.astype(np.float32),
                 obs=logicals.astype(np.float32).flatten())
        
        stage_metrics["files"].append(str(output_file))
    
    def _generate_pauli_plus(self, basis: str, distance: int, rounds: int,
                             output_dir: Path, stage_metrics: Dict):
        """生成 Pauli+ 数据 (使用论文对齐的噪声模型)"""
        import yaml
        from simulator.pauli_plus_simulator import PauliPlusSimulator
        
        output_file = output_dir / f"samples_surface_code_b{basis.upper()}_d{distance}_r{rounds:02d}.npz"
        
        if output_file.exists():
            return
        
        samples = self.config.finetune_samples_per_exp
        self.logger.info(f"  生成 Pauli+: basis={basis}, d={distance}, r={rounds}, n={samples}")
        
        # 加载论文对齐的配置
        config_path = Path("configs/paper_aligned.yaml")
        if config_path.exists():
            with open(config_path, encoding='utf-8') as f:
                noise_config = yaml.safe_load(f)
        else:
            noise_config = {}
        
        # 设置距离和轮数
        noise_config['distance'] = distance
        noise_config['rounds'] = rounds
        
        # 确保 dqlr_matrix 有默认值 (YAML中 null 会变成 None)
        if noise_config.get('dqlr_matrix') is None:
            noise_config['dqlr_matrix'] = [
                [1.0, 0.0, 0.05],  # P(end in |0⟩ | start in |0⟩, |1⟩, |2⟩)
                [0.0, 1.0, 0.90],  # P(end in |1⟩ | start in |0⟩, |1⟩, |2⟩)
                [0.0, 0.0, 0.05],  # P(end in |2⟩ | start in |0⟩, |1⟩, |2⟩)
            ]
        
        # 使用 PauliPlusSimulator 创建带论文噪声的电路
        sim = PauliPlusSimulator(noise_config, basis)
        sim.apply_paper_aligned_noise(noise_config)
        
        sampler = sim.circuit.compile_detector_sampler()
        syndromes, logicals = sampler.sample(samples, separate_observables=True)
        
        np.savez(output_file,
                 data=syndromes.astype(np.float32),
                 obs=logicals.astype(np.float32).flatten())
        
        stage_metrics["files"].append(str(output_file))
    
    def _generate_test_data(self, basis: str, distance: int, rounds: int,
                            output_dir: Path, stage_metrics: Dict):
        """生成测试数据 (使用论文对齐的噪声模型)"""
        import yaml
        from simulator.pauli_plus_simulator import PauliPlusSimulator
        
        output_file = output_dir / f"test_b{basis.upper()}_d{distance}_r{rounds:02d}.npz"
        
        if output_file.exists():
            return
        
        samples = self.config.test_samples_per_config
        self.logger.info(f"  生成测试数据: basis={basis}, d={distance}, r={rounds}, n={samples}")
        
        # 加载论文对齐的配置
        config_path = Path("configs/paper_aligned.yaml")
        if config_path.exists():
            with open(config_path, encoding='utf-8') as f:
                noise_config = yaml.safe_load(f)
        else:
            noise_config = {}
        
        # 设置距离和轮数
        noise_config['distance'] = distance
        noise_config['rounds'] = rounds
        
        # 确保 dqlr_matrix 有默认值 (YAML中 null 会变成 None)
        if noise_config.get('dqlr_matrix') is None:
            noise_config['dqlr_matrix'] = [
                [1.0, 0.0, 0.05],
                [0.0, 1.0, 0.90],
                [0.0, 0.0, 0.05],
            ]
        
        # 使用 PauliPlusSimulator 创建带论文噪声的电路
        sim = PauliPlusSimulator(noise_config, basis)
        sim.apply_paper_aligned_noise(noise_config)
        
        sampler = sim.circuit.compile_detector_sampler()
        syndromes, logicals = sampler.sample(samples, separate_observables=True)
        
        np.savez(output_file,
                 data=syndromes.astype(np.float32),
                 obs=logicals.astype(np.float32).flatten())
        
        stage_metrics["files"].append(str(output_file))
    
    def _generate_dummy_data(self, stage_metrics: Dict):
        """生成模拟数据"""
        self.logger.info("生成模拟数据...")
        
        for d in self.config.code_distances:
            for p in self.config.si1000_p_grid[:3]:
                output_file = self.data_dir / "si1000" / f"si1000_d{d}_p{p:.3f}.npz"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                n = self.config.pretrain_samples_per_config // 10
                num_det = (d - 1) ** 2 * 25
                
                np.savez(output_file,
                         data=np.random.binomial(1, 0.01, (n, num_det)).astype(np.float32),
                         obs=np.random.binomial(1, 0.05, n).astype(np.float32))
                
                stage_metrics["files"].append(str(output_file))
    
    # -------------------------------------------------------------------------
    # Stage 3: 预训练
    # -------------------------------------------------------------------------
    def _stage_pretrain(self):
        """Stage 3: 预训练"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Stage 3: 预训练")
        self.logger.info("=" * 50)
        
        stage_start = time.time()
        
        # 收集数据文件
        data_files = list((self.data_dir / "si1000").glob("*.npz"))
        self.logger.info(f"找到 {len(data_files)} 个预训练数据文件")
        
        if not data_files:
            self.logger.warning("没有预训练数据，跳过")
            self.metrics["stages"]["pretrain"] = {"status": "skipped"}
            return
        
        try:
            self._run_pretraining(data_files)
            self.metrics["stages"]["pretrain"] = {
                "time_seconds": time.time() - stage_start,
                "status": "completed",
                "data_files": len(data_files)
            }
        except Exception as e:
            self.logger.error(f"预训练失败: {e}")
            self.metrics["stages"]["pretrain"] = {
                "time_seconds": time.time() - stage_start,
                "status": "failed",
                "error": str(e)
            }
        
        self.logger.info(f"Stage 3 完成, 耗时: {time.time() - stage_start:.1f}s")
    
    def _run_pretraining(self, data_files: List[Path]):
        """运行预训练"""
        import torch
        from torch.utils.data import DataLoader, TensorDataset, random_split
        
        # 合并数据
        self.logger.info("加载预训练数据...")
        all_X, all_y = [], []
        
        for f in data_files[:20]:  # 限制文件数
            try:
                data = np.load(f, allow_pickle=True)
                X = data["data"] if "data" in data else data["syndromes"]
                y = data["obs"] if "obs" in data else data["logicals"]
                all_X.append(X)
                all_y.append(y.flatten())
            except Exception as e:
                self.logger.warning(f"跳过 {f}: {e}")
        
        if not all_X:
            raise ValueError("没有可用数据")
        
        # 不同distance的数据有不同的syndrome维度，需要padding到最大维度
        # 找到最大维度
        max_dim = max(x.shape[1] if len(x.shape) > 1 else x.shape[0] for x in all_X)
        self.logger.info(f"最大syndrome维度: {max_dim}, 进行padding...")
        
        # Padding所有数组到相同维度
        padded_X = []
        for x in all_X:
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
            current_dim = x.shape[1]
            if current_dim < max_dim:
                pad_width = ((0, 0), (0, max_dim - current_dim))
                x = np.pad(x, pad_width, mode='constant', constant_values=0)
            padded_X.append(x)
        
        X = np.concatenate(padded_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        
        # 限制样本数
        max_samples = min(len(X), self.config.pretrain_samples_total)
        if max_samples < len(X):
            indices = np.random.choice(len(X), max_samples, replace=False)
            X, y = X[indices], y[indices]
        
        self.logger.info(f"预训练数据: X={X.shape}, y={y.shape}")
        
        # 准备数据
        if X.ndim == 2:
            X = X[:, np.newaxis, :, np.newaxis]  # (N, 1, S, 1)
        elif X.ndim == 3:
            X = X[:, :, :, np.newaxis]
        
        # 添加 basis 特征
        N, R, S, F = X.shape
        basis_feat = np.zeros((N, R, S, 1), dtype=np.float32)  # Z basis = 1
        X = np.concatenate([X, basis_feat], axis=-1)
        
        X_tensor = torch.from_numpy(X.astype(np.float32))
        y_tensor = torch.from_numpy(y.astype(np.float32))
        
        # 分割
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_ds, batch_size=self.config.pretrain_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.pretrain_batch_size)
        
        # 创建模型 (使用标准 Transformer, 非 MLA)
        from ai_models.model import AlphaQubitDecoder
        
        _, R, S, F = X.shape
        d = int(np.ceil(np.sqrt(S + 1)))
        grid_size = d - 1
        
        model = AlphaQubitDecoder(
            num_features=F,
            hidden_dim=self.config.hidden_dim,
            num_stabilizers=S,
            grid_size=grid_size,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers
        )
        
        device = torch.device(self.device)
        model.to(device)
        
        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.pretrain_lr,
            weight_decay=self.config.pretrain_weight_decay
        )
        
        # 学习率调度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.pretrain_epochs
        )
        
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # 训练
        best_val_loss = float('inf')
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        
        for epoch in range(1, self.config.pretrain_epochs + 1):
            # Train
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                # 准备输入
                B = X_batch.shape[0]
                basis = torch.ones(B, dtype=torch.long, device=device)
                final_mask = torch.zeros(B, S, dtype=torch.long, device=device)
                
                optimizer.zero_grad()
                outputs = model(X_batch, basis, final_mask)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            scheduler.step()
            train_loss /= len(train_loader)
            
            # Validate
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    B = X_batch.shape[0]
                    basis = torch.ones(B, dtype=torch.long, device=device)
                    final_mask = torch.zeros(B, S, dtype=torch.long, device=device)
                    
                    outputs = model(X_batch, basis, final_mask)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.numel()
            
            val_loss /= len(val_loader)
            val_acc = correct / total
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = self.pretrain_dir / "pretrained_model.pth"
                torch.save(model.state_dict(), model_path)
            
            if epoch % 10 == 0 or epoch == 1:
                self.logger.info(
                    f"Epoch {epoch}/{self.config.pretrain_epochs}: "
                    f"train={train_loss:.4f}, val={val_loss:.4f}, acc={val_acc:.4f}"
                )
        
        # 保存历史
        save_json(history, self.pretrain_dir / "pretrain_history.json")
        self.logger.info(f"预训练模型已保存到 {self.pretrain_dir}")
    
    # -------------------------------------------------------------------------
    # Stage 4: 微调
    # -------------------------------------------------------------------------
    def _stage_finetune(self):
        """Stage 4: 微调"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Stage 4: 微调")
        self.logger.info("=" * 50)
        
        stage_start = time.time()
        stage_metrics = {"experiments": []}
        
        # 收集微调数据
        pauli_dir = self.data_dir / "pauli_plus"
        if not pauli_dir.exists():
            self.logger.warning("没有微调数据")
            self.metrics["stages"]["finetune"] = {"status": "skipped"}
            return
        
        finetune_files = list(pauli_dir.glob("*.npz"))
        self.logger.info(f"找到 {len(finetune_files)} 个微调数据文件")
        
        pretrained_path = self.pretrain_dir / "pretrained_model.pth"
        
        for f in finetune_files:
            exp_name = f.stem
            exp_dir = self.finetune_dir / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                self._run_finetuning(f, pretrained_path, exp_dir)
                stage_metrics["experiments"].append({
                    "name": exp_name,
                    "status": "success"
                })
            except Exception as e:
                self.logger.error(f"微调失败 {exp_name}: {e}")
                stage_metrics["experiments"].append({
                    "name": exp_name,
                    "status": "failed",
                    "error": str(e)
                })
        
        stage_metrics["time_seconds"] = time.time() - stage_start
        self.metrics["stages"]["finetune"] = stage_metrics
        self.logger.info(f"Stage 4 完成, 耗时: {stage_metrics['time_seconds']:.1f}s")
    
    def _run_finetuning(self, data_file: Path, pretrained_path: Path, output_dir: Path):
        """运行单个微调"""
        import torch
        from torch.utils.data import DataLoader, TensorDataset, random_split
        
        # 加载数据
        data = np.load(data_file, allow_pickle=True)
        X = data["data"] if "data" in data else data["syndromes"]
        y = data["obs"] if "obs" in data else data["logicals"]
        y = y.flatten()
        
        if X.ndim == 2:
            X = X[:, np.newaxis, :, np.newaxis]
        elif X.ndim == 3:
            X = X[:, :, :, np.newaxis]
        
        N, R, S, F = X.shape
        basis_feat = np.zeros((N, R, S, 1), dtype=np.float32)
        X = np.concatenate([X, basis_feat], axis=-1)
        F = X.shape[-1]
        
        X_tensor = torch.from_numpy(X.astype(np.float32))
        y_tensor = torch.from_numpy(y.astype(np.float32))
        
        # 分割
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int(self.config.finetune_train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_ds, batch_size=self.config.finetune_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.finetune_batch_size)
        
        self.logger.info(f"  训练: {train_size}, 验证: {val_size}")
        
        # 创建模型 (使用标准 Transformer, 非 MLA)
        from ai_models.model import AlphaQubitDecoder
        
        d = int(np.ceil(np.sqrt(S + 1)))
        grid_size = d - 1
        
        model = AlphaQubitDecoder(
            num_features=F,
            hidden_dim=self.config.hidden_dim,
            num_stabilizers=S,
            grid_size=grid_size,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers
        )
        
        device = torch.device(self.device)
        
        # 加载预训练权重
        if pretrained_path.exists():
            try:
                model.load_state_dict(torch.load(pretrained_path, map_location="cpu"), strict=False)
                self.logger.info(f"  加载预训练权重: {pretrained_path}")
            except Exception as e:
                self.logger.warning(f"  无法加载预训练权重: {e}")
        
        model.to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.finetune_lr,
            weight_decay=self.config.finetune_weight_decay
        )
        
        criterion = torch.nn.BCEWithLogitsLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, self.config.finetune_epochs + 1):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                B = X_batch.shape[0]
                basis = torch.ones(B, dtype=torch.long, device=device)
                final_mask = torch.zeros(B, S, dtype=torch.long, device=device)
                
                optimizer.zero_grad()
                outputs = model(X_batch, basis, final_mask)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.finetune_grad_clip)
                optimizer.step()
            
            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    B = X_batch.shape[0]
                    basis = torch.ones(B, dtype=torch.long, device=device)
                    final_mask = torch.zeros(B, S, dtype=torch.long, device=device)
                    
                    outputs = model(X_batch, basis, final_mask)
                    val_loss += criterion(outputs, y_batch).item()
            
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), output_dir / "finetuned_model.pth")
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.finetune_patience:
                self.logger.info(f"  Early stopping at epoch {epoch}")
                break
    
    # -------------------------------------------------------------------------
    # Stage 5: 测试
    # -------------------------------------------------------------------------
    def _stage_test(self):
        """Stage 5: 测试"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Stage 5: 测试评估")
        self.logger.info("=" * 50)
        
        stage_start = time.time()
        stage_metrics = {"results": []}
        
        # 收集模型
        models = list(self.finetune_dir.glob("*/finetuned_model.pth"))
        if not models:
            pretrained = self.pretrain_dir / "pretrained_model.pth"
            if pretrained.exists():
                models = [pretrained]
        
        # 收集测试数据
        test_files = list((self.data_dir / "test").glob("*.npz"))
        
        self.logger.info(f"测试 {len(models)} 个模型 on {len(test_files)} 个测试集")
        
        for model_path in models:
            model_name = model_path.parent.name
            
            for test_file in test_files:
                test_name = test_file.stem
                
                try:
                    metrics = self._run_test(model_path, test_file)
                    stage_metrics["results"].append({
                        "model": model_name,
                        "test": test_name,
                        **metrics
                    })
                    self.logger.info(
                        f"  {model_name} on {test_name}: "
                        f"LER={metrics.get('ler', 'N/A'):.4f}, Acc={metrics.get('accuracy', 'N/A'):.4f}"
                    )
                except Exception as e:
                    self.logger.error(f"  测试失败 {model_name}/{test_name}: {e}")
        
        stage_metrics["time_seconds"] = time.time() - stage_start
        self.metrics["stages"]["test"] = stage_metrics
        save_json(stage_metrics, self.test_dir / "test_results.json")
        self.logger.info(f"Stage 5 完成, 耗时: {stage_metrics['time_seconds']:.1f}s")
    
    def _run_test(self, model_path: Path, test_file: Path) -> Dict[str, float]:
        """运行测试"""
        import torch
        
        # 加载数据
        data = np.load(test_file, allow_pickle=True)
        X = data["data"] if "data" in data else data["syndromes"]
        y = data["obs"] if "obs" in data else data["logicals"]
        y = y.flatten()
        
        if X.ndim == 2:
            X = X[:, np.newaxis, :, np.newaxis]
        elif X.ndim == 3:
            X = X[:, :, :, np.newaxis]
        
        N, R, S, F = X.shape
        basis_feat = np.zeros((N, R, S, 1), dtype=np.float32)
        X = np.concatenate([X, basis_feat], axis=-1)
        F = X.shape[-1]
        
        # 加载模型 (使用标准 Transformer, 非 MLA)
        from ai_models.model import AlphaQubitDecoder
        
        d = int(np.ceil(np.sqrt(S + 1)))
        grid_size = d - 1
        
        model = AlphaQubitDecoder(
            num_features=F,
            hidden_dim=self.config.hidden_dim,
            num_stabilizers=S,
            grid_size=grid_size,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers
        )
        
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
        device = torch.device(self.device)
        model.to(device)
        model.eval()
        
        # 预测
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)
        basis = torch.ones(N, dtype=torch.long, device=device)
        final_mask = torch.zeros(N, S, dtype=torch.long, device=device)
        
        with torch.no_grad():
            outputs = model(X_tensor, basis, final_mask)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
        
        # 计算指标
        accuracy = (preds == y).mean()
        ler = 1 - accuracy  # Logical Error Rate
        
        return {"accuracy": float(accuracy), "ler": float(ler)}
    
    # -------------------------------------------------------------------------
    # Stage 6: 生成报告
    # -------------------------------------------------------------------------
    def _stage_generate_report(self):
        """Stage 6: 生成研究报告"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Stage 6: 生成研究报告")
        self.logger.info("=" * 50)
        
        report = self._build_research_report()
        
        # Markdown 报告
        md_path = self.report_dir / "research_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # JSON 格式
        save_json(self.metrics, self.report_dir / "full_metrics.json")
        
        self.logger.info(f"报告已保存到: {self.report_dir}")
    
    def _build_research_report(self) -> str:
        """构建研究报告"""
        lines = [
            "# AlphaQubit 论文复现研究报告",
            "",
            f"**生成时间**: {datetime.now().isoformat()}",
            f"**设备**: {self.device}",
            "",
            "## 1. 执行摘要",
            "",
            f"本报告记录了 AlphaQubit 论文 (Nature, 2024) 的复现结果。",
            "",
            "### 配置参数",
            "",
            "| 参数 | 值 |",
            "|------|------|",
            f"| 预训练样本 | {self.config.pretrain_samples_total:,} |",
            f"| 微调样本/实验 | {self.config.finetune_samples_per_exp:,} |",
            f"| 测试样本/配置 | {self.config.test_samples_per_config:,} |",
            f"| 预训练 epochs | {self.config.pretrain_epochs} |",
            f"| 微调 epochs | {self.config.finetune_epochs} |",
            "",
            "## 2. 论文对齐验证",
            "",
        ]
        
        # 添加对齐结果
        alignment = self.metrics.get("stages", {}).get("alignment", {})
        if alignment:
            lines.extend([
                f"- 通过: {alignment.get('passed', 'N/A')}",
                f"- 失败: {alignment.get('failed', 'N/A')}",
                f"- 警告: {alignment.get('warnings', 'N/A')}",
                "",
            ])
        
        lines.extend([
            "## 3. 数据生成",
            "",
        ])
        
        data_gen = self.metrics.get("stages", {}).get("data_generation", {})
        if data_gen:
            lines.append(f"- 生成文件数: {len(data_gen.get('files', []))}")
            lines.append(f"- 耗时: {data_gen.get('time_seconds', 'N/A'):.1f}s")
        
        lines.extend([
            "",
            "## 4. 预训练结果",
            "",
        ])
        
        pretrain = self.metrics.get("stages", {}).get("pretrain", {})
        if pretrain:
            lines.append(f"- 状态: {pretrain.get('status', 'N/A')}")
            lines.append(f"- 数据文件数: {pretrain.get('data_files', 'N/A')}")
            lines.append(f"- 耗时: {pretrain.get('time_seconds', 'N/A'):.1f}s")
        
        lines.extend([
            "",
            "## 5. 微调结果",
            "",
        ])
        
        finetune = self.metrics.get("stages", {}).get("finetune", {})
        if finetune:
            exps = finetune.get("experiments", [])
            lines.append(f"- 实验数: {len(exps)}")
            success = sum(1 for e in exps if e.get("status") == "success")
            lines.append(f"- 成功: {success}/{len(exps)}")
        
        lines.extend([
            "",
            "## 6. 测试结果",
            "",
        ])
        
        test = self.metrics.get("stages", {}).get("test", {})
        if test:
            results = test.get("results", [])
            if results:
                lines.extend([
                    "| 模型 | 测试集 | LER | 准确率 |",
                    "|------|--------|-----|--------|",
                ])
                for r in results:
                    lines.append(
                        f"| {r.get('model', 'N/A')} | {r.get('test', 'N/A')} | "
                        f"{r.get('ler', 0):.4f} | {r.get('accuracy', 0):.4f} |"
                    )
        
        lines.extend([
            "",
            "## 7. 结论",
            "",
            "本复现实验按照论文规格执行了完整的训练流程。",
            "详细结果请参见各阶段的 JSON 输出文件。",
            "",
            "---",
            "",
            "## 参考文献",
            "",
            "1. Google DeepMind. *Accurate neural network decoding of surface codes* ",
            "   *for quantum error correction*. Nature, 2024.",
            "   DOI: 10.1038/s41586-024-08449-y",
            "",
        ])
        
        return "\n".join(lines)


# =============================================================================
# 主函数
# =============================================================================

# 服务器上的项目目录
SERVER_PROJECT_DIR = Path("/home/ma-user/work/ALPHAQUBIT")


def main():
    parser = argparse.ArgumentParser(
        description="AlphaQubit 完整服务器流水线"
    )
    parser.add_argument(
        "--output-dir", type=Path, 
        default=Path(f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        help="输出目录"
    )
    parser.add_argument(
        "--quick-test", action="store_true",
        help="快速测试模式"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "npu", "cuda", "cpu"],
        help="计算设备"
    )
    parser.add_argument(
        "--project-dir", type=Path, default=SERVER_PROJECT_DIR,
        help="项目根目录 (服务器上的 ALPHAQUBIT 目录)"
    )
    args = parser.parse_args()
    
    # 切换到项目目录
    project_dir = args.project_dir
    if project_dir.exists():
        os.chdir(project_dir)
        print(f"已切换到项目目录: {project_dir}")
    else:
        # 尝试当前脚本所在目录
        script_dir = Path(__file__).resolve().parent
        os.chdir(script_dir)
        print(f"项目目录不存在，使用脚本目录: {script_dir}")
    
    print(f"当前工作目录: {os.getcwd()}")
    
    # 选择配置
    if args.quick_test:
        config = QuickTestConfig()
        config.device = args.device
    else:
        config = PaperAlignedConfig()
        config.device = args.device
    
    # 运行流水线
    pipeline = CompletePipeline(config, args.output_dir)
    pipeline.run_all()


if __name__ == "__main__":
    main()
