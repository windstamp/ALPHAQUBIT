# google_qec_simulator/data_helpers.py
import numpy as np
import torch
from pathlib import Path

# Lazy import to avoid stim dependency for soft_channels
def _get_stim_helpers():
    from .stim_helpers import extract_rounds_and_dets
    return extract_rounds_and_dets


def reshape_detectors(det_flat: np.ndarray, stim_path: Path, shots: int) -> np.ndarray:
    """
    Reshapes the flat detector array into a (shots, rounds, dets_per_round, 1) shape.
    """
    # Lazy import to avoid stim dependency when only using soft_channels
    extract_rounds_and_dets = _get_stim_helpers()
    
    # Get rounds and detectors per round
    R, S = extract_rounds_and_dets(stim_path)
    
    # Debugging: Print out rounds, detectors, and shot information
    print(f"Shots (N): {shots}")
    print(f"Extracted rounds (R): {R}, Detectors per round (S): {S}")
    
    # Calculate the expected reshape size
    expected_size = shots * R * S
    print(f"Expected reshape size (shots * rounds * detectors_per_round): {expected_size}")
    
    # Ensure the reshaping size matches
    print(f"Actual detected data size: {det_flat.size}")
    if det_flat.size != expected_size:
        raise ValueError(f"Cannot reshape {det_flat.size} elements into "
                         f"shape ({shots}, {R}, {S}, 1). "
                         f"Expected size: {expected_size}. Please check the dimensions of your data.")

    return det_flat.reshape(shots, R, S, 1).astype(np.float32)


def soft_channels(n, snr=10.0, tau=0.01, leak_p=0.001, device: str = "cpu"):
    """Vectorized soft-channel sampling aligned with paper's I/Q readout model.

    Paper-aligned implementation:
    - Uses symmetric means: mu0 = +SNR/2, mu1 = -alpha*SNR/2 where alpha = exp(-tau)
    - Leakage prior: 0.1% (p_leak_prior = 0.001) per paper spec
    - Three-state posterior probabilities for {|0>, |1>, |L>}

    Parameters
    ----------
    n : int
        Number of samples to draw.
    snr : float
        Signal-to-noise ratio of the readout channel.
    tau : float
        Amplitude damping time constant. alpha = exp(-tau) controls |1> mean shift.
    leak_p : float
        Probability of preparing in the leakage state. Default 0.001 (0.1%) per paper.
    device : str
        Torch device string (e.g. ``"cpu"``, ``"cuda"``, ``"npu"``).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Posterior probabilities for states 1 (p1) and leakage (pL).
    """
    import math

    normalized_device = device.lower()

    # ``torch.device`` does not yet expose an "npu" backend in stock
    # PyTorch builds.  We treat it (and any other unsupported device
    # aliases) as a request to fall back to CPU execution so the data
    # pipeline keeps running on machines without specialised hardware.
    device_aliases = {
        "npu": "cpu",
    }
    normalized_device = device_aliases.get(normalized_device, normalized_device)

    try:
        dev = torch.device(normalized_device)
    except RuntimeError:
        print(f"[soft_channels] Unsupported device '{device}', falling back to CPU.")
        dev = torch.device("cpu")

    # Paper-aligned I/Q model parameters
    # mu0 = +SNR/2, mu1 = -alpha*SNR/2 where alpha = exp(-tau)
    alpha = math.exp(-tau)
    mu0 = 0.5 * snr      # Mean for |0> state (positive)
    mu1 = -alpha * 0.5 * snr  # Mean for |1> state (negative, damped)
    mu_leak = 0.0        # Leakage state centered at 0 with wider sigma
    sigma = 1.0          # Standard deviation for computational states
    sigma_leak = 1.6 * sigma  # Wider sigma for leakage (paper: leak_sigma_scale=1.6)

    # Sample physical states {0, 1, 2=leakage}
    # Paper-aligned priors: p0 = p1 = (1 - leak_p)/2, pL = leak_p
    p0_prior = (1.0 - leak_p) / 2.0
    p1_prior = (1.0 - leak_p) / 2.0
    pL_prior = leak_p
    probs = torch.tensor([p0_prior, p1_prior, pL_prior], device=dev)
    states = torch.multinomial(probs, n, replacement=True)

    # Generate I/Q samples with paper-aligned means
    z = torch.zeros(n, device=dev)
    
    # State 0: Gaussian centered at mu0
    mask0 = (states == 0)
    if mask0.any():
        z[mask0] = torch.normal(mu0, sigma, size=(mask0.sum().item(),), device=dev)
    
    # State 1: Gaussian centered at mu1 (with amplitude damping shift)
    mask1 = (states == 1)
    if mask1.any():
        z[mask1] = torch.normal(mu1, sigma, size=(mask1.sum().item(),), device=dev)
    
    # State 2 (leakage): Gaussian centered at 0 with wider sigma
    mask2 = (states == 2)
    if mask2.any():
        z[mask2] = torch.normal(mu_leak, sigma_leak, size=(mask2.sum().item(),), device=dev)

    # Compute likelihoods (Gaussian PDFs)
    def gauss_pdf(x, mu, sig):
        return torch.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))
    
    L0 = gauss_pdf(z, mu0, sigma)
    L1 = gauss_pdf(z, mu1, sigma)
    LL = gauss_pdf(z, mu_leak, sigma_leak)

    # Compute posteriors with paper-aligned priors (0.1% leakage)
    # w0, w1, wL = 0.4995, 0.4995, 0.001  # Paper: p_leak_prior = 1e-3
    w0, w1, wL = p0_prior, p1_prior, pL_prior
    
    num0 = w0 * L0
    num1 = w1 * L1
    numL = wL * LL
    norm = num0 + num1 + numL + 1e-18  # Avoid division by zero
    
    post1 = num1 / norm  # P(state=|1>)
    postL = numL / norm  # P(state=|L>)

    return post1.cpu().numpy().astype(np.float32), postL.cpu().numpy().astype(np.float32)






# Test each function in this file
if __name__ == "__main__":
    stim_path = Path("path_to_a_stim_file/stim_example.stim")

    # Test reshape_detectors
    det_flat = np.random.randint(0, 2, (1000, 100))  # Fake data
    reshaped_det = reshape_detectors(det_flat, stim_path)
    print(f"Reshaped detector shape: {reshaped_det.shape}")

    # Test soft channels generation
    post1, post2 = soft_channels(1000)
    print(f"Generated post1 and post2 channels shapes: {post1.shape}, {post2.shape}")
