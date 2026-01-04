"""paper_aligned.py —— 论文对齐的 Pauli+ 噪声高层封装。

该模块将 `my_noise_model` 中的底层 Kraus/GPT 工具整合，构建与论文参数一致的
噪声配置对象，并驱动 :class:`~simulator.pauli_plus_simulator.PauliPlusSimulator`
完成电路“穿衣”。"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from simulator.pauli_plus_simulator import PauliPlusSimulator
from .gpt import amp_phase_kraus
from .gpta import twirl_to_pauli_channel
from .channels import lift_qubit_to_qutrit, dqlr_kraus
from . import kraus_utils


@dataclass
class PaperAlignedNoiseConfig:
    """存储与论文一致的噪声参数。
    
    This config includes ALL noise parameters from the AlphaQubit Nature 2024 paper:
    - Basic decoherence (T1, Tphi, heating)
    - Readout and reset errors
    - DQLR (Dissipative Qubit-Level Reset) imperfections
    - CZ gate mechanisms (leakage, crosstalk, swap-like)
    - Residual Pauli noise
    - NEW: Readout crosstalk, burst errors, temporal drift, etc.
    """

    # Timing (ns)
    cycle_ns: float = 1076.0
    
    # === Decoherence (Table S3) ===
    # Note on T2 vs Tphi:
    #   - T1 = 73 µs (energy relaxation time)
    #   - Tphi = 720 µs (pure dephasing time, aka T_φ)
    #   - T2_CPMG ≈ 80 µs (measured via CPMG echo sequence)
    #   - Relation: 1/T2 = 1/(2*T1) + 1/T_φ
    #   - Verification: 1/T2 = 1/146 + 1/720 ≈ 0.00822 → T2 ≈ 122 µs (Ramsey)
    #   - T2_CPMG < T2_Ramsey due to low-frequency noise not refocused
    # The paper uses Tphi for simulation but calibrates against T2_CPMG measurements.
    T1_us: float = 73.0
    Tphi_us: float = 720.0  # Pure dephasing time (T_φ), NOT T2
    T2_CPMG_us: float = 80.0  # For reference: CPMG-measured coherence
    p_heat_01: float = 0.0  # |0> -> |1> heating
    p_heat_12: float = 2.5e-4  # |1> -> |2> heating
    
    # === Readout / reset (Table S4) ===
    p_readout: float = 8.0e-3
    p_reset: float = 1.5e-3
    
    # === DQLR imperfection matrix P_{j->i} on |0>,|1>,|2> ===
    # Column j = starting state, Row i = ending state
    # |0⟩ stays |0⟩, |1⟩ stays |1⟩, |2⟩ → 5% to |0⟩, 90% to |1⟩, 5% remains |2⟩
    dqlr_matrix: Tuple[Tuple[float, ...], ...] = (
        (1.0, 0.0, 0.05),   # P(end in |0⟩ | start in |0⟩, |1⟩, |2⟩)
        (0.0, 1.0, 0.90),   # P(end in |1⟩ | start in |0⟩, |1⟩, |2⟩)
        (0.0, 0.0, 0.05),   # P(end in |2⟩ | start in |0⟩, |1⟩, |2⟩)
    )
    
    # === CZ gate mechanisms (Table S5) ===
    p_cz_depolarizing: float = 7.0e-3  # Paper: 2Q gate depolarizing error
    p_cz_leak_11_to_02: float = 2.0e-4
    p_cz_leak_11_to_20: float = 2.0e-4  # Symmetric leakage to |20⟩
    p_cz_crosstalk_ZZ: float = 5.5e-4
    p_cz_swap_like: float = 0.0
    p_leak_transport_12_to_30: float = 0.0005
    
    # === Residual Pauli noise (Table S6) ===
    p_1q_excess: float = 6.2e-4
    p_cz_excess: float = 2.75e-3
    p_idle_excess: float = 0.0
    
    # === NEW: Readout Crosstalk (Section 3.2.1 of Supplementary) ===
    # Probability that measuring one qubit affects neighbor's readout
    p_readout_crosstalk: float = 1e-3
    
    # === NEW: Burst/Cosmic Ray Errors (Section 3.3.2) ===
    # Probability of a burst error event (cosmic ray, TLS)
    p_burst: float = 1e-5
    # Depolarization probability given a burst occurred
    p_depol_given_burst: float = 0.5
    
    # === NEW: Measurement-Induced Reset (Section 3.2.3) ===
    # State-dependent reset probabilities after measurement
    p_reset_from_0: float = 0.001  # Reset error starting from |0⟩
    p_reset_from_1: float = 0.002  # Reset error starting from |1⟩
    p_reset_from_L: float = 0.05   # Reset error starting from |2⟩ (leakage)
    
    # === NEW: Leakage Seepage (Section 3.4.1) ===
    # Probability that neighbor's leakage seeps into this qubit
    p_leakage_seepage: float = 5e-5
    
    # === NEW: State-Dependent T1 (Section 3.1.2) ===
    # Different decay rates for different transitions
    gamma_10: float = 0.0  # |1⟩→|0⟩ rate (computed from T1 if 0)
    gamma_21: float = 0.0  # |2⟩→|1⟩ rate (typically ~2x gamma_10)
    
    # === NEW: Temporal Drift (Section 4.2) ===
    # Enable time-varying noise to simulate drift
    enable_temporal_drift: bool = False
    drift_amplitude: float = 0.1  # Fractional amplitude of drift
    
    # === NEW: Frequency Collision (Section 3.4.3) ===
    # Probability of coherent exchange when qubits are near resonance
    p_frequency_collision: float = 1e-4


class PaperAlignedNoiseModel:
    """高层封装：将论文噪声注入 PauliPlusSimulator。"""

    def __init__(self, config: Dict, basis: str = "z"):
        self.cfg = self._load_cfg(config)
        self.basis = basis.lower()
        if self.basis not in ("x", "z"):
            raise ValueError("basis must be 'x' or 'z'")

        # Build simulator and instrument with paper noise
        self.sim = PauliPlusSimulator(config, basis)
        self.sim.apply_paper_aligned_noise(config=self.cfg.__dict__)

        # Precompute GPT-twirled single-qubit channels
        self._ptm_1q_idle, self._p_idle_leak = self._build_idle_ptm()
        self._ptm_dqlr, self._p_dqlr_leak = self._build_dqlr_ptm()
        self._ptm_1q_excess = {
            "I": 1 - self.cfg.p_1q_excess,
            "X": self.cfg.p_1q_excess / 3,
            "Y": self.cfg.p_1q_excess / 3,
            "Z": self.cfg.p_1q_excess / 3,
        }

    def _load_cfg(self, raw: Dict) -> PaperAlignedNoiseConfig:
        """将字典配置映射到 :class:`PaperAlignedNoiseConfig` 对象。"""
        cfg = PaperAlignedNoiseConfig()
        for k, v in (raw or {}).items():
            if k == "p_heat":
                cfg.p_heat_12 = float(v)
                continue
            # Allow configs to explicitly set ``dqlr_matrix: null`` to use the
            # dataclass default instead of passing ``None`` into downstream
            # Kraus builders that expect a 3x3 matrix.
            if k == "dqlr_matrix" and v is None:
                continue
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def _build_idle_ptm(self) -> Tuple[Dict[str, float], float]:
        """构造空闲段的 GPT 后 Pauli 传输矩阵与泄漏概率。"""
        dt_us = self.cfg.cycle_ns / 1000.0
        K_amp_phase = amp_phase_kraus(dt_us, self.cfg.T1_us, self.cfg.Tphi_us)
        K = lift_qubit_to_qutrit(K_amp_phase)
        if self.cfg.p_heat_01 > 0 or self.cfg.p_heat_12 > 0:
            heat = kraus_utils.kraus_leakage_heating(self.cfg.p_heat_01, self.cfg.p_heat_12)
            K = kraus_utils.combine_kraus_channels(K, heat)
        probs, leak = twirl_to_pauli_channel(K, 1)
        ptm = {
            "I": float(probs[0]),
            "X": float(probs[1]),
            "Y": float(probs[2]),
            "Z": float(probs[3]),
        }
        return ptm, float(leak)

    def _build_dqlr_ptm(self) -> Tuple[Dict[str, float], float]:
        """基于 DQLR Kraus 集计算等效 Pauli 概率与泄漏。"""
        Ks = dqlr_kraus(self.cfg.dqlr_matrix)
        probs, leak = twirl_to_pauli_channel(Ks, 1)
        ptm = {
            "I": float(probs[0]),
            "X": float(probs[1]),
            "Y": float(probs[2]),
            "Z": float(probs[3]),
        }
        return ptm, float(leak)

    def sample(self, num_samples: int):
        """调用底层模拟器采样 ``num_samples`` 次检测事件。"""
        sampler = self.sim.circuit.compile_detector_sampler()
        return sampler.sample(num_samples, separate_observables=True)
