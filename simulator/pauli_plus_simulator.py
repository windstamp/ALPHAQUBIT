"""Stim-based surface code simulator with paper-aligned noise injection.

This module constructs a rotated memory surface-code circuit using Stim and
provides a method ``apply_paper_aligned_noise`` that instruments the circuit
with the detailed error mechanisms described in Google's "quantum error
correction below the surface code threshold" paper.  Each mechanism is modelled
via Kraus operators and converted to a generalized Pauli channel using the
generalized Pauli twirling approximation (GPTA) before being appended to the
Stim circuit.

The implementation supports the parameters defined in
``configs/paper_aligned.yaml``.  Key mappings are:

* ``T1_us`` / ``Tphi_us`` – single qubit amplitude/phase damping combined with
  passive heating and twirled to a Pauli+ channel applied after 1Q gates.
* ``p_cz_crosstalk_ZZ`` – correlated ZZ after parallel CZ windows.
* ``p_cz_swap_like`` – swap‑like correlated errors modelled as (XX+YY)/2.
* ``p_cz_leak_11_to_02`` – Kraus model of dephasing‑induced leakage twirled to
  a Pauli channel.
* ``p_leak_transport_12_to_30`` – leakage transport channel applied during CZs.
* ``p_readout`` / ``p_reset`` – classical flips preceding measurement/reset.
* ``dqlr_matrix`` – imperfect DQLR reset modelled with Kraus operators.
* ``p_1q_excess``, ``p_cz_excess`` and ``p_idle_excess`` – residual Pauli noise
  around gates and idles.

The resulting circuit can be sampled using Stim's detector sampler to generate
synthetic syndromes and logical observables.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import stim

from my_noise_model.gpta import twirl_to_pauli_channel
from my_noise_model.gpt import amp_phase_kraus
from my_noise_model.channels import (
    lift_qubit_to_qutrit,
    dqlr_kraus,
    leakage_injection_kraus,     # kept for completeness
    cz_induced_leakage_kraus,    # new: two‑qutrit CZ‑leakage model
    leakage_transport_kraus,     # new: two‑qutrit leakage transport
)
from my_noise_model.kraus_utils import (
    combine_kraus_channels,
    kraus_leakage_heating,
)


ONE_Q_GATES = {
    "H",
    "X",
    "Y",
    "Z",
    "S",
    "SQRT_X",
    "SQRT_Y",
    "RX",
    "RY",
    "RZ",
    "H_XZ",
    "H_YZ",
    "T",
    "SQRT_Z",
}
TWO_Q_GATES = {"CZ", "CNOT", "CX"}


class PauliPlusSimulator:
    """Constructs a base surface-code circuit and attaches noise channels."""

    def __init__(self, config: Dict, basis: str) -> None:
        self.config = config
        self.distance = config.get("distance")
        self.rounds = config.get("rounds")
        if self.distance is None or self.rounds is None:
            raise ValueError("config must include 'distance' and 'rounds'")

        self.depolarization = config.get("depolarization", 0.001)
        self.leakage_rate = config.get("leakage_rate", 0.01)
        self.cross_talk = config.get("cross_talk", 0.002)

        self.basis = basis.upper()
        if self.basis not in ("X", "Z"):
            raise ValueError("basis must be 'X' or 'Z'")

        # Build the base circuit and attach simple two-qubit noise for backwards
        # compatibility.  Paper-aligned noise is injected later via
        # ``apply_paper_aligned_noise``.
        self.circuit = self._build_base_circuit(self.basis)
        self.circuit = self._attach_noise_to_two_qubit_gates(self.circuit)

    # ------------------------------------------------------------------
    # Circuit construction helpers
    # ------------------------------------------------------------------
    def _build_base_circuit(self, basis: str) -> stim.Circuit:
        return stim.Circuit.generated(
            f"surface_code:rotated_memory_{basis.lower()}",
            rounds=self.rounds,
            distance=self.distance,
        )

    def _attach_noise_to_two_qubit_gates(self, circuit: stim.Circuit) -> stim.Circuit:
        """Insert simple leakage and depolarization after each two-qubit gate."""
        noisy = stim.Circuit()
        for inst in circuit:
            noisy.append(inst)
            if inst.name in ("CX", "CZ"):
                targets = [t.value for t in inst.targets_copy()]
                noisy.append_operation(
                    "PAULI_CHANNEL_2",
                    targets,
                    [
                        self.leakage_rate / 3,
                        self.leakage_rate / 3,
                        self.leakage_rate / 3,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                )
                noisy.append_operation("DEPOLARIZE2", targets, self.cross_talk)
        return noisy

    # ------------------------------------------------------------------
    # Paper aligned noise instrumentation
    # ------------------------------------------------------------------
    def apply_paper_aligned_noise(self, config: Dict) -> None:
        """Instrument the circuit with paper-aligned noise channels.

        Parameters are read from ``config`` and correspond directly to the
        fields in ``configs/paper_aligned.yaml``.  Each physical mechanism is
        converted to an equivalent Pauli channel before being appended to the
        Stim circuit.
        
        NEW: Supports spatial variation via use_spatial_variation=True.
        When enabled, each qubit/edge gets unique noise parameters drawn
        from distributions matching real Sycamore device heterogeneity.
        """
        from simulator.device_calibration import SpatialErrorMap

        cfg = config or {}

        # Spatial variation settings (for replicating Google's final results)
        use_spatial = bool(cfg.get("use_spatial_variation", False))
        spatial_seed = int(cfg.get("spatial_seed", 42))
        spatial_map = None
        if use_spatial:
            spatial_map = SpatialErrorMap(self.distance, seed=spatial_seed)

        cycle_ns = float(cfg.get("cycle_ns", 1076.0))
        dt_us = cycle_ns / 1000.0
        
        # Base parameters (used when spatial variation is off)
        base_T1_us = float(cfg.get("T1_us", 73.0))
        base_Tphi_us = float(cfg.get("Tphi_us", 720.0))
        p_heat_01 = float(cfg.get("p_heat_01", 0.0))
        p_heat_12 = float(cfg.get("p_heat_12", cfg.get("p_heat", 2.5e-4)))

        base_p_readout = float(cfg.get("p_readout", 8.0e-3))
        base_p_reset = float(cfg.get("p_reset", 1.5e-3))
        dqlr_matrix: List[List[float]] = cfg.get(
            "dqlr_matrix", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.05, 0.90, 0.05))
        )

        base_p_1q_excess = float(cfg.get("p_1q_excess", 6.2e-4))
        p_idle_excess = float(cfg.get("p_idle_excess", 0.0))
        base_p_cz_excess = float(cfg.get("p_cz_excess", 2.75e-3))
        base_p_cz_zz = float(cfg.get("p_cz_crosstalk_ZZ", 5.5e-4))
        p_cz_swap = float(cfg.get("p_cz_swap_like", 0.0))
        base_p_cz_leak = float(cfg.get("p_cz_leak_11_to_02", 2.0e-4))
        p_leak_transport = float(cfg.get("p_leak_transport_12_to_30", 0.0005))

        twirl_idles_each_tick = bool(cfg.get("twirl_idles_each_tick", False))
        twirl_after_1q_gates = bool(cfg.get("twirl_after_1q_gates", True))

        # Caches for per-qubit and per-edge channel parameters
        _cache_1q = {}  # q -> (px, py, pz)
        _cache_2q = {}  # (a,b) -> (args15, p_zz)
        _cache_readout = {}  # q -> p_readout
        _cache_reset = {}    # q -> p_reset

        def get_1q_params(q: int):
            """Get per-qubit 1Q Pauli channel parameters."""
            if q in _cache_1q:
                return _cache_1q[q]
            
            if use_spatial and spatial_map:
                t1 = spatial_map.get_qubit_error(q, 't1_us')
                t2 = spatial_map.get_qubit_error(q, 't2_us')
                # Convert T2 to Tphi: 1/Tphi = 1/T2 - 1/(2*T1)
                rate_phi = 1.0/t2 - 1.0/(2.0*t1)
                tphi = 1.0/rate_phi if rate_phi > 1e-9 else 1e9
                p_excess = spatial_map.get_qubit_error(q, '1q_error')
            else:
                t1 = base_T1_us
                tphi = base_Tphi_us
                p_excess = base_p_1q_excess

            # Build single-qubit channel via Kraus + GPTA
            K_amp = amp_phase_kraus(dt_us=dt_us, T1_us=t1, Tphi_us=tphi)
            K = lift_qubit_to_qutrit(K_amp)
            if p_heat_01 > 0 or p_heat_12 > 0:
                K = combine_kraus_channels(K, kraus_leakage_heating(p_heat_01, p_heat_12))
            idle_probs, _ = twirl_to_pauli_channel(K, 1)
            px = float(idle_probs[1]) + p_excess / 3.0
            py = float(idle_probs[2]) + p_excess / 3.0
            pz = float(idle_probs[3]) + p_excess / 3.0
            
            _cache_1q[q] = (px, py, pz)
            return px, py, pz

        def get_readout_error(q: int):
            """Get per-qubit readout error probability."""
            if q in _cache_readout:
                return _cache_readout[q]
            if use_spatial and spatial_map:
                p = spatial_map.get_qubit_error(q, 'readout_error')
            else:
                p = base_p_readout
            _cache_readout[q] = p
            return p

        def get_reset_error(q: int):
            """Get per-qubit reset error probability."""
            if q in _cache_reset:
                return _cache_reset[q]
            if use_spatial and spatial_map:
                p = spatial_map.get_qubit_error(q, 'reset_error')
            else:
                p = base_p_reset
            _cache_reset[q] = p
            return p

        def get_2q_params(a: int, b: int):
            """Get per-edge 2Q channel parameters."""
            pair = tuple(sorted((a, b)))
            if pair in _cache_2q:
                return _cache_2q[pair]
            
            if use_spatial and spatial_map:
                p_cz_ex = spatial_map.get_edge_error(a, b, 'cz_error')
                p_cz_lk = spatial_map.get_edge_error(a, b, 'cz_leakage')
                p_zz = spatial_map.get_edge_error(a, b, 'zz_crosstalk')
            else:
                p_cz_ex = base_p_cz_excess
                p_cz_lk = base_p_cz_leak
                p_zz = base_p_cz_zz

            # Build 2-qutrit channel
            cz_Ks: List = []
            if p_cz_lk > 0:
                cz_Ks = cz_induced_leakage_kraus(p_cz_lk)
            if p_leak_transport > 0:
                Ks_move = leakage_transport_kraus(p_leak_transport)
                cz_Ks = Ks_move if not cz_Ks else combine_kraus_channels(cz_Ks, Ks_move)
            
            # Twirl to 2-qubit Pauli channel
            cz_probs16 = None
            if cz_Ks:
                cz_probs16, _ = twirl_to_pauli_channel(cz_Ks, 2)
            
            # Convert to Stim args (15 probs excluding II)
            if cz_probs16 is not None:
                order = [1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15]
                args15 = [float(cz_probs16[i]) for i in order]
            else:
                args15 = [0.0] * 15

            # Add excess depolarizing
            if p_cz_ex > 0:
                u = p_cz_ex / 15.0
                scale = 1.0 - p_cz_ex
                args15 = [val * scale + u for val in args15]
            
            _cache_2q[pair] = (args15, p_zz)
            return args15, p_zz

        # DQLR reset imperfections (constant across qubits for now)
        dqlr_probs, _ = twirl_to_pauli_channel(dqlr_kraus(dqlr_matrix), 1)
        dqlr_px, dqlr_py, dqlr_pz = (
            float(dqlr_probs[1]),
            float(dqlr_probs[2]),
            float(dqlr_probs[3]),
        )

        new_circuit = stim.Circuit()
        current_cz_pairs: List[Tuple[int, int]] = []
        seen_qubits: set[int] = set()

        def add_pauli_ch1(q: int, out_circuit: stim.Circuit) -> None:
            px, py, pz = get_1q_params(q)
            if px <= 0 and py <= 0 and pz <= 0:
                return
            out_circuit.append_operation("PAULI_CHANNEL_1", [q], [px, py, pz])

        def inject_cz_pair(a: int, b: int, out_circuit: stim.Circuit) -> None:
            args15, p_zz = get_2q_params(a, b)
            
            if p_zz > 0:
                out_circuit.append_operation(
                    "CORRELATED_ERROR",
                    [stim.target_z(a), stim.target_z(b)],
                    p_zz,
                )
            if p_cz_swap > 0:
                out_circuit.append_operation(
                    "CORRELATED_ERROR",
                    [stim.target_x(a), stim.target_x(b)],
                    0.5 * p_cz_swap,
                )
                out_circuit.append_operation(
                    "CORRELATED_ERROR",
                    [stim.target_y(a), stim.target_y(b)],
                    0.5 * p_cz_swap,
                )
            
            if any(v > 0 for v in args15):
                out_circuit.append_operation("PAULI_CHANNEL_2", [a, b], args15)

        def process_instruction(inst, out_circuit: stim.Circuit):
            """Process a single instruction, adding noise as needed."""
            nonlocal current_cz_pairs, seen_qubits
            
            # Handle REPEAT blocks recursively
            if isinstance(inst, stim.CircuitRepeatBlock):
                inner = stim.Circuit()
                for sub_inst in inst.body_copy():
                    process_instruction(sub_inst, inner)
                out_circuit += inner * inst.repeat_count
                return
            
            name = inst.name
            targs = inst.targets_copy()
            gargs = inst.gate_args_copy()

            if name == "M":
                for t in targs:
                    if t.is_qubit_target:
                        p = get_readout_error(t.value)
                        if p > 0:
                            out_circuit.append_operation("X_ERROR", [t], [p])
                out_circuit.append_operation(name, targs, gargs)
                return

            if name in ("R", "RX", "RY", "RZ"):
                out_circuit.append_operation(name, targs, gargs)
                for t in targs:
                    if not t.is_qubit_target:
                        continue
                    p_rst = get_reset_error(t.value)
                    if p_rst > 0:
                        out_circuit.append_operation("X_ERROR", [t], [p_rst])
                    out_circuit.append_operation("PAULI_CHANNEL_1", [t], [dqlr_px, dqlr_py, dqlr_pz])
                return

            if name in ONE_Q_GATES:
                out_circuit.append_operation(name, targs, gargs)
                if twirl_after_1q_gates:
                    for t in targs:
                        if t.is_qubit_target:
                            q = t.value
                            seen_qubits.add(q)
                            add_pauli_ch1(q, out_circuit)
                return

            if name in TWO_Q_GATES:
                qs = [t.value for t in targs if t.is_qubit_target]
                for i in range(0, len(qs), 2):
                    if i + 1 < len(qs):
                        a, b = qs[i], qs[i + 1]
                        current_cz_pairs.append((a, b))
                        seen_qubits.update((a, b))
                out_circuit.append_operation(name, targs, gargs)
                return

            if name == "TICK":
                for a, b in current_cz_pairs:
                    inject_cz_pair(a, b, out_circuit)
                current_cz_pairs.clear()
                if twirl_idles_each_tick and seen_qubits:
                    s = p_idle_excess / 3.0
                    for q in sorted(seen_qubits):
                        px, py, pz = get_1q_params(q)
                        out_circuit.append_operation("PAULI_CHANNEL_1", [q], [px + s, py + s, pz + s])
                out_circuit.append_operation(name, targs, gargs)
                return

            out_circuit.append_operation(name, targs, gargs)

        for inst in self.circuit:
            process_instruction(inst, new_circuit)

        for a, b in current_cz_pairs:
            inject_cz_pair(a, b, new_circuit)
        self.circuit = new_circuit
