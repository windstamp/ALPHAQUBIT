"""test_realistic_noise_model.py - Tests for realistic noise model components.

Run with: pytest tests/test_realistic_noise_model.py -v
"""

import json
import numpy as np
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from my_noise_model.calibration_loader import (
    QubitCalibration,
    EdgeCalibration,
    DeviceCalibration,
    generate_random_calibration,
)
from my_noise_model.realistic_noise_model import (
    RealisticNoiseModel,
    RealisticNoiseConfig,
)
from my_noise_model.realistic_data_generator import (
    RealisticDataGenerator,
    GeneratedDataset,
    perturb_calibration,
    interpolate_calibrations,
)


class TestQubitCalibration:
    """Tests for QubitCalibration dataclass."""
    
    def test_from_dict(self):
        """Test creating QubitCalibration from dictionary."""
        data = {
            "qubit_id": 5,
            "row": 1,
            "col": 0,
            "t1_us": 75.0,
            "t2_us": 80.0,
            "tphi_us": 720.0,
            "readout_error": 0.008,
            "reset_error": 0.0015,
            "oneq_error": 0.0006,
            "is_bad_qubit": False,
        }
        
        qcal = QubitCalibration.from_dict(data)
        
        assert qcal.qubit_id == 5
        assert qcal.t1_us == 75.0
        assert qcal.readout_error == 0.008
        assert not qcal.is_bad_qubit
    
    def test_compute_tphi(self):
        """Test computing Tphi from T1 and T2."""
        qcal = QubitCalibration(
            qubit_id=0,
            t1_us=73.0,
            t2_us=80.0,
            tphi_us=0,  # Will compute
        )
        
        # 1/T2 = 1/(2*T1) + 1/Tphi
        # 1/80 = 1/146 + 1/Tphi
        # Tphi should be around 177 µs
        tphi = qcal.compute_tphi_from_t1_t2()
        assert 100 < tphi < 300


class TestEdgeCalibration:
    """Tests for EdgeCalibration dataclass."""
    
    def test_from_dict(self):
        """Test creating EdgeCalibration from dictionary."""
        data = {
            "edge": [0, 1],
            "qubit_a": 0,
            "qubit_b": 1,
            "cz_error": 0.003,
            "cz_leakage": 0.0002,
            "zz_crosstalk": 0.00055,
            "xeb_fidelity": 0.997,
        }
        
        ecal = EdgeCalibration.from_dict(data)
        
        assert ecal.edge == (0, 1)
        assert ecal.cz_error == 0.003
        assert ecal.xeb_fidelity == 0.997
    
    def test_edge_sorted(self):
        """Test that edge property returns sorted tuple."""
        ecal = EdgeCalibration(qubit_a=5, qubit_b=2)
        assert ecal.edge == (2, 5)


class TestDeviceCalibration:
    """Tests for DeviceCalibration dataclass."""
    
    def test_from_dict(self):
        """Test creating DeviceCalibration from dictionary."""
        data = {
            "metadata": {"distance": 3, "seed": 42},
            "qubit_calibration": {
                "0": {"qubit_id": 0, "t1_us": 70.0},
                "1": {"qubit_id": 1, "t1_us": 75.0},
            },
            "edge_calibration": {
                "(0, 1)": {"edge": [0, 1], "cz_error": 0.003},
            },
            "bad_qubits": [1],
        }
        
        cal = DeviceCalibration.from_dict(data)
        
        assert cal.distance == 3
        assert len(cal.qubit_calibrations) == 2
        assert len(cal.edge_calibrations) == 1
        assert cal.bad_qubits == [1]
    
    def test_get_qubit_fallback(self):
        """Test that get_qubit returns default for missing qubit."""
        cal = DeviceCalibration()
        qcal = cal.get_qubit(999)
        
        assert qcal.qubit_id == 999
        assert qcal.t1_us == 73.0  # Default
    
    def test_get_edge_fallback(self):
        """Test that get_edge returns default for missing edge."""
        cal = DeviceCalibration()
        ecal = cal.get_edge(10, 20)
        
        assert ecal.edge == (10, 20)
        assert ecal.cz_error == 0.0035  # Default
    
    def test_json_roundtrip(self):
        """Test saving and loading calibration from JSON."""
        cal = generate_random_calibration(distance=3, seed=42)
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            # Save
            data = {
                "metadata": {"distance": cal.distance},
                "qubit_calibration": {
                    str(k): {
                        "qubit_id": v.qubit_id,
                        "t1_us": v.t1_us,
                        "t2_us": v.t2_us,
                        "readout_error": v.readout_error,
                    }
                    for k, v in cal.qubit_calibrations.items()
                },
                "edge_calibration": {
                    str(k): {
                        "edge": list(k),
                        "cz_error": v.cz_error,
                    }
                    for k, v in cal.edge_calibrations.items()
                },
                "bad_qubits": cal.bad_qubits,
            }
            f.write(json.dumps(data).encode())
            path = f.name
        
        try:
            # Load
            loaded = DeviceCalibration.from_json_file(path)
            assert loaded.distance == cal.distance
            assert len(loaded.qubit_calibrations) == len(cal.qubit_calibrations)
        finally:
            Path(path).unlink()


class TestGenerateRandomCalibration:
    """Tests for generate_random_calibration function."""
    
    def test_basic_generation(self):
        """Test basic calibration generation."""
        cal = generate_random_calibration(distance=5, seed=42)
        
        assert cal.distance == 5
        assert cal.num_qubits == 81  # (2*5-1)^2
        assert len(cal.qubit_calibrations) == 81
        assert len(cal.edge_calibrations) > 0
    
    def test_reproducibility(self):
        """Test that same seed produces same calibration."""
        cal1 = generate_random_calibration(distance=3, seed=123)
        cal2 = generate_random_calibration(distance=3, seed=123)
        
        q0_1 = cal1.get_qubit(0)
        q0_2 = cal2.get_qubit(0)
        
        assert q0_1.t1_us == q0_2.t1_us
        assert q0_1.readout_error == q0_2.readout_error
    
    def test_statistics_match(self):
        """Test that generated statistics roughly match inputs."""
        cal = generate_random_calibration(
            distance=5,
            seed=42,
            t1_median=73.0,
            cz_error_median=0.0035,
        )
        
        stats = cal.get_statistics()
        
        # Should be within 30% of target
        assert 50 < stats["t1_median"] < 100
        assert 0.002 < stats["cz_error_median"] < 0.006
    
    def test_bad_qubits(self):
        """Test that bad qubits are generated."""
        cal = generate_random_calibration(
            distance=5,
            seed=42,
            bad_qubit_fraction=0.1,  # 10%
        )
        
        # Should have some bad qubits
        assert len(cal.bad_qubits) > 0
        assert len(cal.bad_qubits) < cal.num_qubits


class TestRealisticNoiseModel:
    """Tests for RealisticNoiseModel class."""
    
    def test_build_circuit(self):
        """Test building a noisy circuit."""
        cal = generate_random_calibration(distance=3, seed=42)
        model = RealisticNoiseModel(
            calibration=cal,
            distance=3,
            rounds=5,
        )
        
        circuit = model.build_noisy_circuit(basis="z")
        
        assert circuit is not None
        # Circuit should have instructions
        assert len(list(circuit)) > 0
    
    def test_sampling(self):
        """Test sampling from noisy circuit."""
        cal = generate_random_calibration(distance=3, seed=42)
        model = RealisticNoiseModel(
            calibration=cal,
            distance=3,
            rounds=5,
        )
        model.build_noisy_circuit(basis="z")
        
        det_events, obs = model.sample(num_samples=100)
        
        assert det_events.shape[0] == 100
        assert obs.shape[0] == 100
    
    def test_without_calibration(self):
        """Test model works without calibration (uses defaults)."""
        config = RealisticNoiseConfig(distance=3, rounds=5)
        config.use_calibration = False
        
        model = RealisticNoiseModel(config=config, distance=3, rounds=5)
        circuit = model.build_noisy_circuit(basis="z")
        
        assert circuit is not None
    
    def test_both_bases(self):
        """Test both X and Z basis work."""
        cal = generate_random_calibration(distance=3, seed=42)
        model = RealisticNoiseModel(calibration=cal, distance=3, rounds=5)
        
        circuit_z = model.build_noisy_circuit(basis="z")
        circuit_x = model.build_noisy_circuit(basis="x")
        
        assert circuit_z is not None
        assert circuit_x is not None


class TestRealisticDataGenerator:
    """Tests for RealisticDataGenerator class."""
    
    def test_generate_samples(self):
        """Test generating training samples."""
        generator = RealisticDataGenerator.with_random_calibration(
            distance=3,
            rounds=5,
            seed=42,
        )
        
        dataset = generator.generate_samples(num_samples=100, basis="z")
        
        assert dataset.num_samples == 100
        assert dataset.distance == 3
        assert dataset.rounds == 5
        assert dataset.detection_events.shape[0] == 100
        assert dataset.observables.shape[0] == 100
    
    def test_save_load_npz(self):
        """Test saving and loading dataset."""
        generator = RealisticDataGenerator.with_random_calibration(
            distance=3, rounds=5, seed=42
        )
        dataset = generator.generate_samples(num_samples=100)
        
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        
        try:
            generator.save_to_npz(path, dataset)
            loaded = GeneratedDataset.from_npz(path)
            
            assert loaded.num_samples == dataset.num_samples
            assert loaded.distance == dataset.distance
            np.testing.assert_array_equal(
                loaded.detection_events, dataset.detection_events
            )
        finally:
            Path(path).unlink()


class TestCalibrationPerturbation:
    """Tests for calibration perturbation utilities."""
    
    def test_perturb_calibration(self):
        """Test perturbing calibration data."""
        cal = generate_random_calibration(distance=3, seed=42)
        perturbed = perturb_calibration(cal, perturbation_scale=0.1, seed=123)
        
        # Should have same structure
        assert len(perturbed.qubit_calibrations) == len(cal.qubit_calibrations)
        
        # Values should be different
        q0_orig = cal.get_qubit(0)
        q0_pert = perturbed.get_qubit(0)
        assert q0_orig.t1_us != q0_pert.t1_us
    
    def test_interpolate_calibrations(self):
        """Test interpolating between calibrations."""
        cal1 = generate_random_calibration(distance=3, seed=42)
        cal2 = generate_random_calibration(distance=3, seed=123)
        
        interpolated = interpolate_calibrations(cal1, cal2, alpha=0.5)
        
        q0_1 = cal1.get_qubit(0)
        q0_2 = cal2.get_qubit(0)
        q0_i = interpolated.get_qubit(0)
        
        # Should be between the two
        expected_t1 = 0.5 * q0_1.t1_us + 0.5 * q0_2.t1_us
        assert abs(q0_i.t1_us - expected_t1) < 0.001


class TestIntegrationWithRealCalibration:
    """Integration tests using real calibration file if available."""
    
    @pytest.fixture
    def calibration_path(self):
        """Path to real calibration file."""
        path = Path("configs/realistic_calibration_d5.json")
        if not path.exists():
            pytest.skip("Calibration file not found")
        return str(path)
    
    def test_load_real_calibration(self, calibration_path):
        """Test loading real calibration file."""
        cal = DeviceCalibration.from_json_file(calibration_path)
        
        assert cal.distance == 5
        assert len(cal.qubit_calibrations) == 49
        assert len(cal.edge_calibrations) == 83
    
    def test_generate_from_real_calibration(self, calibration_path):
        """Test generating data from real calibration."""
        generator = RealisticDataGenerator.from_calibration_file(
            calibration_path,
            distance=5,
            rounds=10,
        )
        
        dataset = generator.generate_samples(num_samples=100)
        
        assert dataset.num_samples == 100
        assert dataset.distance == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
