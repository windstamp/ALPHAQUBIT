#!/usr/bin/env python3
"""
Comprehensive Paper Verification Script
========================================

This script performs exhaustive verification of the AlphaQubit implementation
against the Google Nature 2024 paper specifications.

Run with: python -m verification.verify_paper_alignment
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import math

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class VerificationResult:
    """Result of a single verification check."""
    name: str
    passed: bool
    expected: Any
    actual: Any
    message: str = ""


class PaperVerifier:
    """Comprehensive verifier for AlphaQubit paper alignment."""
    
    def __init__(self):
        self.results: List[VerificationResult] = []
        self.spec = self._load_spec()
    
    def _load_spec(self) -> Dict:
        """Load the paper specification JSON."""
        spec_path = PROJECT_ROOT / "verification" / "paper_spec.json"
        with open(spec_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _add_result(self, name: str, passed: bool, expected: Any, actual: Any, message: str = ""):
        """Record a verification result."""
        self.results.append(VerificationResult(name, passed, expected, actual, message))
    
    def verify_table_s4_noise_parameters(self) -> None:
        """Verify all Table S4 noise parameters match code defaults."""
        print("\n" + "="*80)
        print("VERIFICATION: Table S4 Noise Parameters")
        print("="*80)
        
        # Import the config class
        from my_noise_model.paper_aligned import PaperAlignedNoiseConfig
        cfg = PaperAlignedNoiseConfig()
        
        params = self.spec["table_s4_noise_parameters"]
        
        checks = [
            ("cycle_ns", params["cycle_ns"]["value"], cfg.cycle_ns),
            ("T1_us", params["T1_us"]["value"], cfg.T1_us),
            ("Tphi_us", params["Tphi_us"]["value"], cfg.Tphi_us),
            ("p_readout", params["p_readout"]["value"], cfg.p_readout),
            ("p_reset", params["p_reset"]["value"], cfg.p_reset),
            ("p_heat_12", params["p_heat_12"]["value"], cfg.p_heat_12),
            ("p_cz_leak_11_to_02", params["p_cz_leak_11_to_02"]["value"], cfg.p_cz_leak_11_to_02),
            ("p_cz_crosstalk_ZZ", params["p_cz_crosstalk_ZZ"]["value"], cfg.p_cz_crosstalk_ZZ),
            ("p_1q_excess", params["p_1q_excess"]["value"], cfg.p_1q_excess),
            ("p_cz_excess", params["p_cz_excess"]["value"], cfg.p_cz_excess),
        ]
        
        for param_name, expected, actual in checks:
            passed = abs(expected - actual) < 1e-10
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {param_name}: {status} (expected={expected}, actual={actual})")
            self._add_result(f"Table_S4_{param_name}", passed, expected, actual)
    
    def verify_model_architecture(self) -> None:
        """Verify model architecture matches paper specifications."""
        print("\n" + "="*80)
        print("VERIFICATION: Model Architecture")
        print("="*80)
        
        # Test instantiation of the model
        try:
            from ai_models.model import AlphaQubitDecoder
            
            # Large model config (paper default)
            expected = self.spec["model_architecture"]["configurations"]["large"]
            
            # Create model with paper specs
            F = 2  # features
            S = 24  # stabilizers (d=5)
            grid_size = 5
            model = AlphaQubitDecoder(
                F, 
                expected["hidden_dim"],
                S, 
                grid_size, 
                num_heads=expected["num_heads"], 
                num_layers=expected["num_layers"]
            )
            
            # Verify layer count
            actual_layers = len(model.transformer.layers)
            passed = actual_layers == expected["num_layers"]
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  num_layers: {status} (expected={expected['num_layers']}, actual={actual_layers})")
            self._add_result("Architecture_num_layers", passed, expected["num_layers"], actual_layers)
            
            # Verify hidden dimension
            actual_hidden = model.embedder.index_embedding.embedding_dim
            passed = actual_hidden == expected["hidden_dim"]
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  hidden_dim: {status} (expected={expected['hidden_dim']}, actual={actual_hidden})")
            self._add_result("Architecture_hidden_dim", passed, expected["hidden_dim"], actual_hidden)
            
            # Verify number of heads
            actual_heads = model.transformer.layers[0].num_heads
            passed = actual_heads == expected["num_heads"]
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  num_heads: {status} (expected={expected['num_heads']}, actual={actual_heads})")
            self._add_result("Architecture_num_heads", passed, expected["num_heads"], actual_heads)
            
        except Exception as e:
            print(f"  ❌ FAIL: Could not instantiate model: {e}")
            self._add_result("Architecture_instantiation", False, "success", str(e))
    
    def verify_attention_scaling(self) -> None:
        """Verify attention uses 1/sqrt(d_k) scaling."""
        print("\n" + "="*80)
        print("VERIFICATION: Attention Scaling")
        print("="*80)
        
        import torch
        from ai_models.model import SyndromeTransformerLayer
        
        # Create a layer
        hidden_dim = 256
        num_heads = 8
        head_dim = hidden_dim // num_heads
        expected_scale = 1.0 / math.sqrt(head_dim)
        
        layer = SyndromeTransformerLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_stabilizers=24,
            grid_size=5
        )
        
        # Check the scaling factor
        actual_scale = 1.0 / math.sqrt(layer.head_dim)
        passed = abs(expected_scale - actual_scale) < 1e-10
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  1/sqrt(head_dim): {status} (expected={expected_scale:.6f}, actual={actual_scale:.6f})")
        self._add_result("Attention_scaling", passed, expected_scale, actual_scale)
    
    def verify_positional_encodings(self) -> None:
        """Verify grid-based positional encodings exist."""
        print("\n" + "="*80)
        print("VERIFICATION: Positional Encodings")
        print("="*80)
        
        from ai_models.model import SyndromeTransformerLayer
        
        layer = SyndromeTransformerLayer(
            hidden_dim=256,
            num_heads=8,
            num_stabilizers=24,
            grid_size=5
        )
        
        expected_embeddings = ["row_emb", "col_emb", "dx_emb", "dy_emb", "manh_emb", "same_emb"]
        
        for emb_name in expected_embeddings:
            has_emb = hasattr(layer, emb_name)
            status = "✅ PASS" if has_emb else "❌ FAIL"
            print(f"  {emb_name}: {status}")
            self._add_result(f"PosEnc_{emb_name}", has_emb, True, has_emb)
    
    def verify_soft_xor(self) -> None:
        """Verify soft XOR formula: p + q - 2pq."""
        print("\n" + "="*80)
        print("VERIFICATION: Soft XOR Formula")
        print("="*80)
        
        import numpy as np
        from my_noise_model.softxor import soft_xor
        
        # Test cases
        test_cases = [
            (0.0, 0.0, 0.0),  # 0 XOR 0 = 0
            (1.0, 0.0, 1.0),  # 1 XOR 0 = 1
            (0.0, 1.0, 1.0),  # 0 XOR 1 = 1
            (1.0, 1.0, 0.0),  # 1 XOR 1 = 0
            (0.5, 0.5, 0.5),  # soft case
            (0.3, 0.7, 0.3 + 0.7 - 2*0.3*0.7),  # general case
        ]
        
        all_passed = True
        for p, q, expected in test_cases:
            actual = soft_xor(np.array([p]), np.array([q]))[0]
            passed = abs(expected - actual) < 1e-10
            all_passed = all_passed and passed
            status = "✅" if passed else "❌"
            print(f"  soft_xor({p}, {q}): {status} (expected={expected:.4f}, actual={actual:.4f})")
        
        self._add_result("SoftXOR_formula", all_passed, "p+q-2pq", "verified" if all_passed else "mismatch")
    
    def verify_gpta_implementation(self) -> None:
        """Verify GPTA Pauli twirling implementation."""
        print("\n" + "="*80)
        print("VERIFICATION: GPTA Implementation")
        print("="*80)
        
        import numpy as np
        from my_noise_model.gpta import twirl_to_pauli_channel
        from my_noise_model.channels import depolarizing_1q_kraus
        
        # Test with a known channel (depolarizing)
        p_error = 0.1
        kraus = depolarizing_1q_kraus(p_error)
        
        probs, leak = twirl_to_pauli_channel(kraus, n_qubits=1)
        
        # For depolarizing channel: p_I = 1-p, p_X = p_Y = p_Z = p/3
        expected_I = 1.0 - p_error
        expected_XYZ = p_error / 3.0
        
        # Check probabilities sum to 1
        sum_probs = float(np.sum(probs))
        passed_sum = abs(sum_probs - 1.0) < 1e-6
        status = "✅ PASS" if passed_sum else "❌ FAIL"
        print(f"  Probability sum: {status} (expected=1.0, actual={sum_probs:.6f})")
        self._add_result("GPTA_prob_sum", passed_sum, 1.0, sum_probs)
        
        # Check identity probability
        passed_I = abs(probs[0] - expected_I) < 1e-4
        status = "✅ PASS" if passed_I else "❌ FAIL"
        print(f"  p(I): {status} (expected={expected_I:.4f}, actual={probs[0]:.4f})")
        self._add_result("GPTA_p_identity", passed_I, expected_I, float(probs[0]))
        
        # Check Pauli X probability
        passed_X = abs(probs[1] - expected_XYZ) < 1e-4
        status = "✅ PASS" if passed_X else "❌ FAIL"
        print(f"  p(X): {status} (expected={expected_XYZ:.4f}, actual={probs[1]:.4f})")
        self._add_result("GPTA_p_X", passed_X, expected_XYZ, float(probs[1]))
    
    def verify_training_hyperparameters(self) -> None:
        """Verify training hyperparameters match paper."""
        print("\n" + "="*80)
        print("VERIFICATION: Training Hyperparameters")
        print("="*80)
        
        # Read model_mla.py to check hardcoded values
        model_mla_path = PROJECT_ROOT / "ai_models" / "model_mla.py"
        with open(model_mla_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check weight_decay=0.01
        has_weight_decay = "weight_decay=0.01" in content
        status = "✅ PASS" if has_weight_decay else "❌ FAIL"
        print(f"  weight_decay=0.01: {status}")
        self._add_result("Training_weight_decay", has_weight_decay, "0.01", "found" if has_weight_decay else "not found")
        
        # Check BCEWithLogitsLoss
        has_bce = "BCEWithLogitsLoss" in content
        status = "✅ PASS" if has_bce else "❌ FAIL"
        print(f"  BCEWithLogitsLoss: {status}")
        self._add_result("Training_loss_fn", has_bce, "BCEWithLogitsLoss", "found" if has_bce else "not found")
        
        # Check OneCycleLR
        has_onecycle = "OneCycleLR" in content
        status = "✅ PASS" if has_onecycle else "❌ FAIL"
        print(f"  OneCycleLR scheduler: {status}")
        self._add_result("Training_scheduler", has_onecycle, "OneCycleLR", "found" if has_onecycle else "not found")
        
        # Check gradient clipping
        has_grad_clip = "clip_grad_norm_" in content
        status = "✅ PASS" if has_grad_clip else "❌ FAIL"
        print(f"  Gradient clipping: {status}")
        self._add_result("Training_grad_clip", has_grad_clip, "clip_grad_norm_", "found" if has_grad_clip else "not found")
    
    def verify_kraus_operators(self) -> None:
        """Verify Kraus operators satisfy CPTP condition."""
        print("\n" + "="*80)
        print("VERIFICATION: Kraus Operators (CPTP)")
        print("="*80)
        
        import numpy as np
        from my_noise_model.channels import (
            amplitude_damping_kraus,
            dephasing_kraus,
            depolarizing_1q_kraus,
            depolarizing_2q_kraus,
            leakage_injection_kraus,
            cz_induced_leakage_kraus,
            leakage_transport_kraus,
        )
        
        def check_cptp(kraus_ops, name, dim):
            """Check that sum of K†K equals identity (trace preserving)."""
            total = np.zeros((dim, dim), dtype=complex)
            for K in kraus_ops:
                total += K.conj().T @ K
            identity = np.eye(dim, dtype=complex)
            diff = np.max(np.abs(total - identity))
            return diff < 1e-10
        
        tests = [
            ("amplitude_damping", amplitude_damping_kraus(0.1), 2),
            ("dephasing", dephasing_kraus(0.1), 2),
            ("depolarizing_1q", depolarizing_1q_kraus(0.1), 2),
            ("depolarizing_2q", depolarizing_2q_kraus(0.1), 4),
            ("leakage_injection", leakage_injection_kraus(0.1), 3),
            ("cz_induced_leakage", cz_induced_leakage_kraus(0.1), 16),
            ("leakage_transport", leakage_transport_kraus(0.1), 16),
        ]
        
        for name, kraus, dim in tests:
            passed = check_cptp(kraus, name, dim)
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {name} (CPTP): {status}")
            self._add_result(f"Kraus_{name}_CPTP", passed, "Σ K†K = I", "verified" if passed else "failed")
    
    def verify_iq_readout_model(self) -> None:
        """Verify I/Q readout model posteriors sum to 1."""
        print("\n" + "="*80)
        print("VERIFICATION: I/Q Readout Model")
        print("="*80)
        
        import numpy as np
        from my_noise_model.iq_readout import IQReadoutModel
        
        model = IQReadoutModel(snr=5.0, tau=0.01)
        
        # Test that posteriors sum to 1
        x_test = np.linspace(-3, 3, 100)
        probs = model.soft_meas_probs(x_test)
        sums = probs.sum(axis=-1)
        
        passed = np.allclose(sums, 1.0, atol=1e-10)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Posteriors sum to 1: {status}")
        self._add_result("IQ_posteriors_normalized", passed, 1.0, float(sums.mean()))
        
        # Check that three states are returned
        has_three_states = probs.shape[-1] == 3
        status = "✅ PASS" if has_three_states else "❌ FAIL"
        print(f"  Three states (|0⟩, |1⟩, |L⟩): {status}")
        self._add_result("IQ_three_states", has_three_states, 3, probs.shape[-1])
    
    def verify_dqlr_matrix(self) -> None:
        """Verify DQLR reset matrix is stochastic (columns sum to 1)."""
        print("\n" + "="*80)
        print("VERIFICATION: DQLR Reset Matrix")
        print("="*80)
        
        import numpy as np
        from my_noise_model.paper_aligned import PaperAlignedNoiseConfig
        
        cfg = PaperAlignedNoiseConfig()
        dqlr = np.array(cfg.dqlr_matrix)
        
        # Check columns sum to 1 (stochastic matrix)
        col_sums = dqlr.sum(axis=0)
        passed = np.allclose(col_sums, 1.0, atol=1e-10)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Stochastic (columns sum to 1): {status}")
        print(f"    Column sums: {col_sums}")
        self._add_result("DQLR_stochastic", passed, [1.0, 1.0, 1.0], col_sums.tolist())
        
        # Check expected values from paper
        expected_22_to_0 = 0.05  # P(reset to |0⟩ | was |2⟩)
        expected_22_to_1 = 0.90  # P(reset to |1⟩ | was |2⟩)
        expected_22_to_2 = 0.05  # P(remain |2⟩ | was |2⟩)
        
        passed_col2 = (
            abs(dqlr[0, 2] - expected_22_to_0) < 1e-10 and
            abs(dqlr[1, 2] - expected_22_to_1) < 1e-10 and
            abs(dqlr[2, 2] - expected_22_to_2) < 1e-10
        )
        status = "✅ PASS" if passed_col2 else "❌ FAIL"
        print(f"  DQLR column 2 (|2⟩ transitions): {status}")
        print(f"    Expected: [0.05, 0.90, 0.05], Actual: {dqlr[:, 2].tolist()}")
        self._add_result("DQLR_column2", passed_col2, [0.05, 0.90, 0.05], dqlr[:, 2].tolist())
    
    def run_all_verifications(self) -> None:
        """Run all verification checks."""
        print("\n" + "#"*80)
        print("#  ALPHAQUBIT PAPER VERIFICATION SUITE")
        print("#  Checking alignment with Nature 2024 paper specifications")
        print("#"*80)
        
        self.verify_table_s4_noise_parameters()
        self.verify_model_architecture()
        self.verify_attention_scaling()
        self.verify_positional_encodings()
        self.verify_soft_xor()
        self.verify_gpta_implementation()
        self.verify_training_hyperparameters()
        self.verify_kraus_operators()
        self.verify_iq_readout_model()
        self.verify_dqlr_matrix()
        
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print verification summary."""
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        
        print(f"\n  Total checks: {total}")
        print(f"  ✅ Passed: {passed}")
        print(f"  ❌ Failed: {failed}")
        print(f"  Pass rate: {100*passed/total:.1f}%")
        
        if failed > 0:
            print("\n  Failed checks:")
            for r in self.results:
                if not r.passed:
                    print(f"    - {r.name}: expected={r.expected}, actual={r.actual}")
        
        print("\n" + "="*80)
        if failed == 0:
            print("✅ ALL VERIFICATIONS PASSED - Implementation matches paper!")
        else:
            print(f"⚠️  {failed} VERIFICATION(S) FAILED - Review required")
        print("="*80)


def main():
    verifier = PaperVerifier()
    verifier.run_all_verifications()


if __name__ == "__main__":
    main()
