#!/usr/bin/env python3
"""
AlphaQubit 代码正确性深度诊断
============================

检查代码实现与论文的差异，找出可能导致性能差距的原因。

使用方法:
    python diagnose_code_issues.py
"""

import os
import sys
import re
import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class CodeIssue:
    """代码问题"""
    file: str
    line: int
    severity: str  # "critical", "warning", "info"
    category: str
    description: str
    suggestion: str
    paper_reference: str = ""


class CodeDiagnostic:
    """代码诊断器"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(".")
        self.issues: List[CodeIssue] = []
        
    def check_model_mla(self) -> List[CodeIssue]:
        """检查 model_mla.py 的问题"""
        issues = []
        file_path = self.project_root / "ai_models" / "model_mla.py"
        
        if not file_path.exists():
            return issues
            
        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        
        # 1. 检查是否使用 DeepSeek MLA
        for i, line in enumerate(lines):
            if "DeepSeekMLA" in line and "import" not in line:
                issues.append(CodeIssue(
                    file=str(file_path),
                    line=i + 1,
                    severity="critical",
                    category="模型架构",
                    description="使用 DeepSeek MLA 注意力机制而非标准 Multi-Head Attention",
                    suggestion="考虑切换到 model.py 中的标准 Transformer 实现",
                    paper_reference="论文使用 Recurrent Transformer with standard MHA"
                ))
                break
        
        # 2. 检查 SyndromeTransformerLayer 是否使用了 events 参数
        in_layer_class = False
        for i, line in enumerate(lines):
            if "class SyndromeTransformerLayer" in line:
                in_layer_class = True
            if in_layer_class and "def forward" in line:
                # 检查后面的实现
                method_lines = []
                for j in range(i, min(i + 20, len(lines))):
                    method_lines.append(lines[j])
                method_code = "\n".join(method_lines)
                
                if "events" not in method_code.replace("def forward", ""):
                    issues.append(CodeIssue(
                        file=str(file_path),
                        line=i + 1,
                        severity="warning",
                        category="模型架构",
                        description="SyndromeTransformerLayer.forward() 接收 events 和 prev_events 参数但未使用",
                        suggestion="论文可能使用 events 来调制注意力或提供额外特征",
                        paper_reference="检查论文中 Transformer 层是否利用 detection events"
                    ))
                break
        
        # 3. 检查 ReadoutNetwork 的网格计算
        for i, line in enumerate(lines):
            if "actual_grid = int(math.ceil(math.sqrt(current_size)))" in line:
                issues.append(CodeIssue(
                    file=str(file_path),
                    line=i + 1,
                    severity="warning",
                    category="模型架构",
                    description="ReadoutNetwork 使用动态网格大小，可能与不同 code distance 不兼容",
                    suggestion="验证 d=3, d=5, d=7 下网格计算是否正确",
                    paper_reference="论文中 ReadoutNetwork 的具体实现"
                ))
                break
        
        # 4. 检查 weight_decay
        for i, line in enumerate(lines):
            if "weight_decay" in line and "0.01" in line:
                issues.append(CodeIssue(
                    file=str(file_path),
                    line=i + 1,
                    severity="critical",
                    category="训练配置",
                    description="weight_decay=0.01，但论文预训练用 1e-4，微调用 1e-3",
                    suggestion="修改为: 预训练 weight_decay=1e-4, 微调 weight_decay=1e-3",
                    paper_reference="Paper: pretraining weight_decay=1e-4, finetuning=1e-3"
                ))
                break
        
        # 5. 检查学习率调度器
        for i, line in enumerate(lines):
            if "OneCycleLR" in line:
                issues.append(CodeIssue(
                    file=str(file_path),
                    line=i + 1,
                    severity="warning",
                    category="训练配置",
                    description="使用 OneCycleLR 调度器，论文使用 cosine_annealing",
                    suggestion="改用 torch.optim.lr_scheduler.CosineAnnealingLR",
                    paper_reference="Paper: scheduler=cosine_annealing"
                ))
                break
        
        return issues
    
    def check_train_py(self) -> List[CodeIssue]:
        """检查 train.py 的问题"""
        issues = []
        file_path = self.project_root / "ai_models" / "train.py"
        
        if not file_path.exists():
            return issues
            
        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        
        # 检查 NPU 检测
        has_npu_check = "torch.npu.is_available()" in content or "HAS_NPU" in content
        if not has_npu_check:
            issues.append(CodeIssue(
                file=str(file_path),
                line=1,
                severity="warning",
                category="硬件支持",
                description="可能缺少 NPU 设备检测",
                suggestion="添加 torch_npu 导入和 NPU 可用性检查",
                paper_reference="训练应该利用可用的加速器"
            ))
        
        return issues
    
    def check_fine_tune_npz(self) -> List[CodeIssue]:
        """检查 fine_tune_npz.py 的问题"""
        issues = []
        file_path = self.project_root / "ai_models" / "fine_tune_npz.py"
        
        if not file_path.exists():
            return issues
            
        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        
        # 检查 observable 转换逻辑
        for i, line in enumerate(lines):
            if "48.0" in line and "49.0" in line:
                # 找到了转换逻辑，检查是否正确
                issues.append(CodeIssue(
                    file=str(file_path),
                    line=i + 1,
                    severity="info",
                    category="数据处理",
                    description="Observable 转换: 48.0='0', 49.0='1' (ASCII)",
                    suggestion="验证这与 Google 数据格式一致",
                    paper_reference="Google QEC v3.5 数据格式"
                ))
                break
        
        # 检查 final_mask 计算
        for i, line in enumerate(lines):
            if "checkerboard" in line.lower() or ("1 if (r + c) % 2" in line):
                issues.append(CodeIssue(
                    file=str(file_path),
                    line=i + 1,
                    severity="warning",
                    category="数据处理",
                    description="Final mask 使用 checkerboard pattern",
                    suggestion="验证 pattern 与论文中 X/Z stabilizer 布局一致",
                    paper_reference="Surface code stabilizer 布局"
                ))
                break
        
        return issues
    
    def check_noise_model(self) -> List[CodeIssue]:
        """检查噪声模型的问题"""
        issues = []
        file_path = self.project_root / "my_noise_model" / "paper_aligned.py"
        
        if not file_path.exists():
            return issues
            
        content = file_path.read_text(encoding="utf-8")
        
        # 检查关键参数是否与论文一致
        expected_params = {
            "cycle_ns": "1076",
            "T1_us": "73",
            "Tphi_us": "720",
            "p_readout": "8.0e-3",
            "p_reset": "1.5e-3",
            "p_heat_12": "2.5e-4",
            "p_cz_leak_11_to_02": "2.0e-4",
            "p_cz_crosstalk_ZZ": "5.5e-4",
            "p_1q_excess": "6.2e-4",
            "p_cz_excess": "2.75e-3",
        }
        
        missing_params = []
        for param, expected in expected_params.items():
            if param not in content:
                missing_params.append(param)
        
        if missing_params:
            issues.append(CodeIssue(
                file=str(file_path),
                line=1,
                severity="warning",
                category="噪声模型",
                description=f"可能缺少参数: {', '.join(missing_params)}",
                suggestion="对照论文 Table S4 检查所有噪声参数",
                paper_reference="Paper Table S4: Pauli+ noise parameters"
            ))
        
        return issues
    
    def check_data_generation(self) -> List[CodeIssue]:
        """检查数据生成的问题"""
        issues = []
        file_path = self.project_root / "make_all_pretraining_noise.py"
        
        if not file_path.exists():
            return issues
            
        content = file_path.read_text(encoding="utf-8")
        
        # 检查预训练数据量
        if "8_500_000" not in content and "8500000" not in content:
            # 查找实际配置的样本数
            match = re.search(r"samples.*=.*(\d+)", content)
            if match:
                actual_samples = match.group(1)
                issues.append(CodeIssue(
                    file=str(file_path),
                    line=1,
                    severity="critical",
                    category="训练数据",
                    description=f"预训练数据量可能不足: 发现 {actual_samples}, 论文要求 8.5M",
                    suggestion="确保生成 8,500,000 个预训练样本",
                    paper_reference="Paper: 8.5M synthetic pretraining samples"
                ))
        
        return issues
    
    def check_soft_readout(self) -> List[CodeIssue]:
        """检查 soft readout 实现"""
        issues = []
        
        # 检查 data_manager.py
        file_path = self.project_root / "google_qec_simulator" / "data_manager.py"
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            
            # 检查是否有 3 通道处理
            if "3" not in content or "channel" not in content.lower():
                issues.append(CodeIssue(
                    file=str(file_path),
                    line=1,
                    severity="critical",
                    category="数据处理",
                    description="可能没有正确处理 3 通道 soft readout 特征",
                    suggestion="确保输入包含: [detection_bit, p_computed, p_leaked]",
                    paper_reference="Paper: 3-channel input with soft readout posteriors"
                ))
        
        return issues
    
    def check_ler_calculation(self) -> List[CodeIssue]:
        """检查 LER 计算方法"""
        issues = []
        
        # 检查 decode.py
        file_path = self.project_root / "ai_models" / "decode.py"
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            
            # 检查 LER 计算
            if "LER" in content or "logical_error" in content.lower():
                # 检查是否是 per-round LER
                if "per_round" not in content.lower() and "/ rounds" not in content:
                    issues.append(CodeIssue(
                        file=str(file_path),
                        line=1,
                        severity="warning",
                        category="评估方法",
                        description="LER 计算可能不是 per-round 的",
                        suggestion="论文使用 LER per round: 1 - (1-p_logical)^(1/r)",
                        paper_reference="Paper: Logical error rate per round"
                    ))
        
        return issues
    
    def run_all_checks(self) -> List[CodeIssue]:
        """运行所有检查"""
        print("=" * 70)
        print("AlphaQubit 代码诊断")
        print("=" * 70)
        
        all_issues = []
        
        print("\n检查 model_mla.py...")
        all_issues.extend(self.check_model_mla())
        
        print("检查 train.py...")
        all_issues.extend(self.check_train_py())
        
        print("检查 fine_tune_npz.py...")
        all_issues.extend(self.check_fine_tune_npz())
        
        print("检查噪声模型...")
        all_issues.extend(self.check_noise_model())
        
        print("检查数据生成...")
        all_issues.extend(self.check_data_generation())
        
        print("检查 soft readout...")
        all_issues.extend(self.check_soft_readout())
        
        print("检查 LER 计算...")
        all_issues.extend(self.check_ler_calculation())
        
        self.issues = all_issues
        return all_issues
    
    def print_report(self):
        """打印诊断报告"""
        print("\n" + "=" * 70)
        print("诊断结果")
        print("=" * 70)
        
        # 按严重性分类
        critical = [i for i in self.issues if i.severity == "critical"]
        warnings = [i for i in self.issues if i.severity == "warning"]
        infos = [i for i in self.issues if i.severity == "info"]
        
        print(f"\n总计发现 {len(self.issues)} 个问题:")
        print(f"  - 严重 (Critical): {len(critical)}")
        print(f"  - 警告 (Warning): {len(warnings)}")
        print(f"  - 信息 (Info): {len(infos)}")
        
        if critical:
            print("\n" + "=" * 70)
            print("🔴 严重问题 (Critical)")
            print("=" * 70)
            for i, issue in enumerate(critical, 1):
                print(f"\n[C{i}] {issue.category}: {issue.description}")
                print(f"    文件: {issue.file}:{issue.line}")
                print(f"    建议: {issue.suggestion}")
                if issue.paper_reference:
                    print(f"    论文参考: {issue.paper_reference}")
        
        if warnings:
            print("\n" + "=" * 70)
            print("🟡 警告 (Warning)")
            print("=" * 70)
            for i, issue in enumerate(warnings, 1):
                print(f"\n[W{i}] {issue.category}: {issue.description}")
                print(f"    文件: {issue.file}:{issue.line}")
                print(f"    建议: {issue.suggestion}")
        
        if infos:
            print("\n" + "=" * 70)
            print("🔵 信息 (Info)")
            print("=" * 70)
            for i, issue in enumerate(infos, 1):
                print(f"\n[I{i}] {issue.category}: {issue.description}")
                print(f"    文件: {issue.file}:{issue.line}")
        
        # 生成修复优先级
        print("\n" + "=" * 70)
        print("修复优先级建议")
        print("=" * 70)
        print("""
1. [最高] 切换到标准 Transformer (model.py)
   - 修改 run_full_pipeline.py 和 fine_tune_npz.py
   - 使用 AlphaQubitDecoderTransformer 而非 AlphaQubitDecoderMLA

2. [高] 修复 weight_decay 参数
   - 预训练: 1e-4
   - 微调: 1e-3

3. [高] 修复学习率调度器
   - 改用 CosineAnnealingLR

4. [中] 验证预训练数据量
   - 确保达到 8.5M 样本

5. [中] 验证 soft readout 特征
   - 确保 3 通道输入正确

6. [低] 检查 LER 计算方法
   - 确认使用 per-round LER
""")
    
    def save_report(self, output_path: Path):
        """保存诊断报告"""
        report = {
            "timestamp": str(Path),
            "total_issues": len(self.issues),
            "critical": len([i for i in self.issues if i.severity == "critical"]),
            "warnings": len([i for i in self.issues if i.severity == "warning"]),
            "infos": len([i for i in self.issues if i.severity == "info"]),
            "issues": [
                {
                    "file": i.file,
                    "line": i.line,
                    "severity": i.severity,
                    "category": i.category,
                    "description": i.description,
                    "suggestion": i.suggestion,
                    "paper_reference": i.paper_reference,
                }
                for i in self.issues
            ]
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 诊断报告已保存: {output_path}")


def main():
    print("=" * 70)
    print("AlphaQubit 代码正确性深度诊断")
    print("=" * 70)
    
    diagnostic = CodeDiagnostic(Path("."))
    diagnostic.run_all_checks()
    diagnostic.print_report()
    
    # 保存报告
    output_dir = Path("research_report")
    output_dir.mkdir(exist_ok=True)
    diagnostic.save_report(output_dir / "code_diagnostic.json")


if __name__ == "__main__":
    main()
