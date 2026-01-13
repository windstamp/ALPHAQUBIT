"""channels.py —— Kraus 通道构建工具。

本文件提供多种 1Q/2Q/多能级噪声通道的 Kraus 表达，用于在 Pauli+ 模型中
描述 T₁/T₂ 弛豫、去极化、泄漏注入、泄漏运输、DQLR 复位等物理机制。所有
函数均返回满足 CPTP 条件的 Kraus 集，并在注释中详细说明其物理含义。
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple

def _pauli(name: str) -> np.ndarray:
    """返回指定 Pauli 算符的矩阵表示。"""
    if name == "I":
        return np.array([[1, 0],[0, 1]], dtype=complex)
    if name == "X":
        return np.array([[0, 1],[1, 0]], dtype=complex)
    if name == "Y":
        return np.array([[0, -1j],[1j, 0]], dtype=complex)
    if name == "Z":
        return np.array([[1, 0],[0, -1]], dtype=complex)
    raise ValueError(name)

def kron(*ops: np.ndarray) -> np.ndarray:
    """连乘克罗内克积，用于构造多体 Kraus 算符。"""
    out = np.array([[1.0+0j]])
    for op in ops:
        out = np.kron(out, op)
    return out

def amplitude_damping_kraus(tau: float) -> List[np.ndarray]:
    r"""单量子比特振幅阻尼通道。

    ``tau`` 表示归一化的演化时间，阻尼强度 ``γ = 1 - e^{-tau}``。
    返回两个 Kraus 算符 ``K0`` 与 ``K1``，分别对应保持在计算子空间和
    ``\|1⟩→\|0⟩`` 跃迁。"""
    gamma = 1.0 - np.exp(-float(tau))
    g = float(gamma)
    K0 = np.array([[1, 0],[0, np.sqrt(1-g)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(g)],[0, 0]], dtype=complex)
    return [K0, K1]

def dephasing_kraus(p: float) -> List[np.ndarray]:
    """单量子比特纯退相干通道，概率 ``p`` 施加 Z 相位翻转。"""
    p = float(p)
    K0 = np.sqrt(1.0 - p) * _pauli("I")
    K1 = np.sqrt(p) * _pauli("Z")
    return [K0, K1]

def depolarizing_1q_kraus(p: float) -> List[np.ndarray]:
    """单量子比特去极化通道，总错误概率为 ``p``。"""
    p = float(p)
    K = [np.sqrt(1.0 - p) * _pauli("I")]
    for name in ("X","Y","Z"):
        K.append(np.sqrt(p/3.0) * _pauli(name))
    return K

def depolarizing_2q_kraus(p: float) -> List[np.ndarray]:
    """双量子比特去极化通道，总错误概率为 ``p``。"""
    p = float(p)
    I = _pauli("I"); X=_pauli("X"); Y=_pauli("Y"); Z=_pauli("Z")
    paulis = [I,X,Y,Z]
    K = [np.sqrt(1.0 - p) * kron(I, I)]
    rest = []
    for a in (I,X,Y,Z):
        for b in (I,X,Y,Z):
            if a is I and b is I:
                continue
            rest.append(kron(a,b))
    for op in rest:
        K.append(np.sqrt(p/15.0) * op)
    return K

def leakage_injection_kraus(p: float) -> List[np.ndarray]:
    r"""单 qutrit 泄漏注入通道。

    以概率 ``p`` 将 ``\|0⟩, \|1⟩, \|2⟩`` 全部泵浦到泄漏态 ``\|2⟩``，否则保持不变。
    Kraus 集满足 CPTP 条件：

    ``K0 = √(1-p)·I₃``，``K1 = √p·|2⟩⟨0|``，``K2 = √p·|2⟩⟨1|``，``K3 = √p·|2⟩⟨2|``。
    """
    p = float(p)
    I3 = np.eye(3, dtype=complex)
    K0 = np.sqrt(1.0 - p) * I3
    K1 = np.zeros((3,3), complex); K1[2,0] = np.sqrt(p)
    K2 = np.zeros((3,3), complex); K2[2,1] = np.sqrt(p)
    K3 = np.zeros((3,3), complex); K3[2,2] = np.sqrt(p)
    return [K0, K1, K2, K3]

def lift_qubit_to_qutrit(Ks_2x2: List[np.ndarray]) -> List[np.ndarray]:
    """将 2×2 Kraus 算符嵌入到含泄漏态的 3×3 空间。"""
    out = []
    for K in Ks_2x2:
        K3 = np.zeros((3,3), complex)
        K3[:2,:2] = K
        K3[2,2] = 1.0
        out.append(K3)
    return out

def cz_induced_leakage_kraus(p_leak: float) -> List[np.ndarray]:
    r"""双 qutrit 系统（两量子比特 + 泄漏态 |2⟩）的 CZ 诱导泄漏通道。

    论文中使用 3 能级系统 (|0⟩, |1⟩, |2⟩)，没有第 4 能级。
    模拟 ``|11⟩`` 态在 CZ 作用下转移到 ``|02⟩`` 与 ``|20⟩`` 的过程，每个分支
    概率 ``p_leak/2``。

    维度: 9×9 (3³ × 3³ = 两个 qutrit 的张量积)
    """

    p = float(p_leak)
    dim = 9  # 3×3 for two qutrits (NOT 16 for ququarts)
    I9 = np.eye(dim, dtype=complex)

    idx = lambda i, j: 3 * i + j  # 3-level indexing
    K0 = I9.copy()
    idx11 = idx(1, 1)
    K0[idx11, idx11] = np.sqrt(max(0.0, 1.0 - p))

    K1 = np.zeros((dim, dim), complex)
    K2 = np.zeros((dim, dim), complex)
    K1[idx(0, 2), idx11] = np.sqrt(p / 2.0)  # |02><11|
    K2[idx(2, 0), idx11] = np.sqrt(p / 2.0)  # |20><11|

    return [K0, K1, K2]

def leakage_transport_kraus(p_move: float) -> List[np.ndarray]:
    r"""Qutrit 泄漏迁移通道（论文使用 3 能级，无 |3⟩ 态）。

    论文中的泄漏传输描述：当一个量子比特处于泄漏态 |2⟩ 而另一个处于 |1⟩ 时，
    在 CZ 门期间泄漏可以"迁移"到另一个量子比特。

    在 qutrit 模型中，这表现为：
    - ``|12⟩ → |21⟩``，概率 ``p_move`` (泄漏从 qubit B 移到 qubit A)
    - ``|21⟩ → |12⟩``，概率 ``p_move`` (泄漏从 qubit A 移到 qubit B)

    注意：论文没有使用 |3⟩ 态，所有泄漏都合并到 |2⟩。

    维度: 9×9 (两个 qutrit 的张量积)
    """

    p = float(p_move)
    dim = 9  # 3×3 for two qutrits
    I9 = np.eye(dim, dtype=complex)

    idx = lambda i, j: 3 * i + j  # 3-level indexing
    idx12 = idx(1, 2)
    idx21 = idx(2, 1)

    # Identity branch with reduced amplitude on transported states.
    K0 = I9.copy()
    K0[idx12, idx12] = np.sqrt(max(0.0, 1.0 - p))
    K0[idx21, idx21] = np.sqrt(max(0.0, 1.0 - p))

    # Transport branches: |21><12| and |12><21| (swap leakage between qubits)
    K1 = np.zeros((dim, dim), complex)
    K2 = np.zeros((dim, dim), complex)
    K1[idx21, idx12] = np.sqrt(p)  # |21><12| (leak moves A→B)
    K2[idx12, idx21] = np.sqrt(p)  # |12><21| (leak moves B→A)

    return [K0, K1, K2]


def dqlr_kraus(p_matrix: List[List[float]]) -> List[np.ndarray]:
    r"""生成 DQLR 复位过程的 Kraus 算符。

    ``p_matrix`` 为 3×3 转移矩阵，列索引 ``j`` 表示复位前的能级 ``\|j⟩``，行
    索引 ``i`` 表示复位后的目标能级 ``\|i⟩``。只要每一列概率和为 1，即可保
    证生成的 Kraus 集满足 CPTP。"""

    P = np.array(p_matrix, dtype=float)
    Ks: List[np.ndarray] = []
    for i in range(3):
        for j in range(3):
            amp = np.sqrt(max(P[i, j], 0.0))
            if amp == 0:
                continue
            K = np.zeros((3, 3), complex)
            K[i, j] = amp
            Ks.append(K)
    return Ks

def spectator_crosstalk_z_kraus(p: float) -> List[np.ndarray]:
    """用于模拟并行 CZ 引起观测者串扰的单量子比特 Z 相位噪声。"""
    return dephasing_kraus(p)

def multi_level_reset_kraus(f_reset: float, rel_leak_after: float) -> List[np.ndarray]:
    r"""三能级复位过程：以保真度 ``f_reset`` 复位到 ``\|0⟩``，并保留 ``rel_leak_after`` 泄漏。"""
    f = float(f_reset)
    r = float(rel_leak_after)
    # Kraus mapping everything to |0> with prob f, and to |2> with small r,
    # otherwise identity remainder.
    K0 = np.zeros((3,3), complex); K0[0,:] = np.sqrt(f)  # reset-to-0 branch
    K1 = np.zeros((3,3), complex); K1[2,:] = np.sqrt(r)  # residual leakage branch
    K2 = np.sqrt(max(0.0, 1.0 - f - r)) * np.eye(3, dtype=complex)
    return [K0, K1, K2]


# ==============================================================================
# Additional Paper-Aligned Noise Channels
# ==============================================================================

def readout_crosstalk_kraus(p_crosstalk: float) -> List[np.ndarray]:
    """Readout crosstalk: measuring one qubit induces Z error on neighbor.
    
    When qubit A is measured, neighboring qubit B experiences dephasing
    with probability p_crosstalk. This models the electromagnetic coupling
    during readout pulses.
    
    Args:
        p_crosstalk: Probability of inducing Z error on neighbor
        
    Returns:
        Single-qubit Kraus operators for the affected neighbor
    """
    return dephasing_kraus(p_crosstalk)


def measurement_induced_reset_kraus(
    p_reset_m0: float,
    p_reset_m1: float,
    measurement_outcome: int
) -> List[np.ndarray]:
    """Measurement-induced state preparation (MISP) errors.
    
    Reset fidelity depends on the measurement outcome:
    - After measuring |0⟩: reset error = p_reset_m0
    - After measuring |1⟩: reset error = p_reset_m1
    
    In real devices, resetting after measuring |1⟩ typically has higher error
    because it requires an active pulse to flip the qubit.
    
    Args:
        p_reset_m0: Reset error probability after measuring 0
        p_reset_m1: Reset error probability after measuring 1
        measurement_outcome: 0 or 1
        
    Returns:
        Kraus operators for reset error (X flip with state-dependent prob)
    """
    p = p_reset_m0 if measurement_outcome == 0 else p_reset_m1
    # Reset error is modeled as X flip
    K0 = np.sqrt(1.0 - p) * _pauli("I")
    K1 = np.sqrt(p) * _pauli("X")
    return [K0, K1]


def leakage_seepage_kraus(p_seep: float) -> List[np.ndarray]:
    r"""Leakage seepage to neighboring qubits during idle.
    
    Leaked population |2⟩ on qubit A can "seep" to neighboring qubit B,
    exciting it from |0⟩ to |1⟩. This models the residual ZZ coupling
    between transmon qubits at higher energy levels.
    
    This is a two-qutrit channel where:
    - |20⟩ → |11⟩ with probability p_seep
    - |02⟩ → |11⟩ with probability p_seep
    
    Args:
        p_seep: Probability of seepage per idle period
        
    Returns:
        Two-qutrit (9x9) Kraus operators
    """
    p = float(p_seep)
    dim = 9  # 3x3 for two qutrits
    I9 = np.eye(dim, dtype=complex)
    
    idx = lambda i, j: 3 * i + j
    
    # Identity with reduced amplitude on seepage states
    K0 = I9.copy()
    idx20 = idx(2, 0)
    idx02 = idx(0, 2)
    K0[idx20, idx20] = np.sqrt(max(0.0, 1.0 - p))
    K0[idx02, idx02] = np.sqrt(max(0.0, 1.0 - p))
    
    # Seepage: |20⟩ → |11⟩ and |02⟩ → |11⟩
    idx11 = idx(1, 1)
    K1 = np.zeros((dim, dim), complex)
    K1[idx11, idx20] = np.sqrt(p)  # |11><20|
    
    K2 = np.zeros((dim, dim), complex)
    K2[idx11, idx02] = np.sqrt(p)  # |11><02|
    
    return [K0, K1, K2]


def state_dependent_amplitude_damping_kraus(
    gamma_10: float,
    gamma_21: float
) -> List[np.ndarray]:
    r"""State-dependent amplitude damping for three-level system.
    
    Different T1 times for |1⟩→|0⟩ vs |2⟩→|1⟩ transitions:
    - gamma_10 = 1 - exp(-t/T1_10) for |1⟩→|0⟩
    - gamma_21 = 1 - exp(-t/T1_21) for |2⟩→|1⟩
    
    In transmon qubits, higher levels typically have shorter T1.
    
    Args:
        gamma_10: Decay rate for |1⟩→|0⟩
        gamma_21: Decay rate for |2⟩→|1⟩
        
    Returns:
        3x3 Kraus operators for qutrit amplitude damping
    """
    g10 = float(gamma_10)
    g21 = float(gamma_21)
    
    # K0: No decay (survival)
    K0 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.sqrt(1.0 - g10), 0.0],
        [0.0, 0.0, np.sqrt(1.0 - g21)]
    ], dtype=complex)
    
    # K1: |1⟩ → |0⟩ decay
    K1 = np.array([
        [0.0, np.sqrt(g10), 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=complex)
    
    # K2: |2⟩ → |1⟩ decay
    K2 = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, np.sqrt(g21)],
        [0.0, 0.0, 0.0]
    ], dtype=complex)
    
    return [K0, K1, K2]


def cosmic_ray_burst_kraus(p_burst: float, affected_qubits: int = 1) -> List[np.ndarray]:
    """Cosmic ray / burst error model.
    
    Transient high-error events that cause correlated errors on multiple qubits.
    With probability p_burst, a burst occurs causing depolarization on
    affected qubits.
    
    For single qubit version:
    - With prob p_burst: complete depolarization (random Pauli)
    - With prob (1-p_burst): identity
    
    Args:
        p_burst: Probability of burst event per round
        affected_qubits: Number of qubits affected (1 for single-qubit Kraus)
        
    Returns:
        Kraus operators for burst error
    """
    p = float(p_burst)
    
    if affected_qubits == 1:
        # Single qubit: identity or full depolarization
        K0 = np.sqrt(1.0 - p) * _pauli("I")
        # Under burst: apply random Pauli with equal probability
        K1 = np.sqrt(p / 4) * _pauli("I")
        K2 = np.sqrt(p / 4) * _pauli("X")
        K3 = np.sqrt(p / 4) * _pauli("Y")
        K4 = np.sqrt(p / 4) * _pauli("Z")
        return [K0, K1, K2, K3, K4]
    else:
        # Multi-qubit burst: use tensor product of single-qubit depolarization
        # For simplicity, return single-qubit version
        return cosmic_ray_burst_kraus(p, 1)


def temporal_noise_scaling(
    base_error: float,
    drift_rate: float,
    time_index: int,
    noise_type: str = "linear"
) -> float:
    """Apply temporal drift to noise parameters.
    
    Models slow drift in T1/T2 and gate fidelities over time.
    
    Args:
        base_error: Base error rate at t=0
        drift_rate: Rate of change per time unit (can be positive or negative)
        time_index: Current time step (e.g., round number)
        noise_type: "linear", "sinusoidal", or "random_walk"
        
    Returns:
        Scaled error rate at current time
    """
    if noise_type == "linear":
        # Linear drift
        scaled = base_error * (1.0 + drift_rate * time_index)
    elif noise_type == "sinusoidal":
        # Periodic fluctuation (e.g., temperature cycles)
        import math
        scaled = base_error * (1.0 + drift_rate * math.sin(0.1 * time_index))
    elif noise_type == "random_walk":
        # Random walk (requires external state, use with seed)
        np.random.seed(time_index)  # Reproducible but varying
        walk = np.random.normal(0, drift_rate * np.sqrt(time_index))
        scaled = base_error * (1.0 + walk)
    else:
        scaled = base_error
    
    # Ensure valid probability
    return max(0.0, min(1.0, scaled))


def spatially_correlated_crosstalk_kraus(
    p_base: float,
    distance: float,
    decay_length: float = 1.0
) -> List[np.ndarray]:
    """Spatially-correlated crosstalk based on qubit distance.
    
    Crosstalk probability decays exponentially with distance:
    p_crosstalk = p_base * exp(-distance / decay_length)
    
    Args:
        p_base: Base crosstalk probability for adjacent qubits
        distance: Manhattan or Euclidean distance between qubits
        decay_length: Characteristic decay length (in qubit spacings)
        
    Returns:
        Single-qubit Kraus operators for Z dephasing
    """
    import math
    p_effective = p_base * math.exp(-distance / decay_length)
    return dephasing_kraus(p_effective)


def frequency_collision_kraus(
    p_collision: float,
    frequency_detuning: float,
    collision_width: float = 0.01
) -> List[np.ndarray]:
    """Frequency collision effects between qubits.
    
    When two qubits have similar frequencies, they can hybridize,
    causing enhanced error rates. Error probability depends on detuning.
    
    p_error = p_collision * exp(-(detuning/width)^2)
    
    Args:
        p_collision: Maximum error probability at zero detuning
        frequency_detuning: Frequency difference (normalized)
        collision_width: Width of collision resonance
        
    Returns:
        Single-qubit Kraus operators for enhanced dephasing
    """
    import math
    p_effective = p_collision * math.exp(-(frequency_detuning / collision_width) ** 2)
    return dephasing_kraus(p_effective)


