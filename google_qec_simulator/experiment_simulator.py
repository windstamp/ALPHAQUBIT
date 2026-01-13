# google_qec_simulator/experiment_simulator.py
# --------------------------------------------------------------------
"""
Monte-Carlo simulator for Google-style DEM data with soft I/Q channels.

Usage
-----
    python -m google_qec_simulator.experiment_simulator  <exp_root>  \
           [--shots 1000] [--out ./sim_soft_data.npz]

`<exp_root>` is the directory that contains many sub-folders; every
sub-folder that contains   *.stim  and  *.dem  files with the same stem
is sampled.  A single compressed .npz bundle is written.
"""

from pathlib import Path
import argparse
import numpy as np

from circuit_utils import sample_detectors_obs
from data_manager  import DataManager
from main          import reshape_detectors, rounds_dets_per_round

# --------------------------------------------------------------------
# Soft-I/Q helpers (Paper-aligned implementation)
# --------------------------------------------------------------------
import math

def iq_sample_paper_aligned(state: int, snr: float, tau: float) -> float:
    """Sample I/Q value with paper-aligned means.
    
    Paper spec: mu0 = +SNR/2, mu1 = -alpha*SNR/2 where alpha = exp(-tau)
    """
    alpha = math.exp(-tau)
    sigma = 1.0
    sigma_leak = 1.6 * sigma
    
    if state == 0:
        mu = 0.5 * snr
        z = np.random.normal(mu, sigma)
    elif state == 1:
        mu = -alpha * 0.5 * snr
        z = np.random.normal(mu, sigma)
    else:  # leakage state
        z = np.random.normal(0.0, sigma_leak)
    return z


def gauss_pdf(z: float, mu: float, sigma: float) -> float:
    """Gaussian PDF."""
    return np.exp(-0.5 * ((z - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def posteriors_paper_aligned(z: float, snr: float, tau: float, leak_prior: float = 0.001):
    """Compute posteriors with paper-aligned model.
    
    Paper spec:
    - Leakage prior: 0.1% (not 1%)
    - Symmetric means: mu0 = +SNR/2, mu1 = -alpha*SNR/2
    """
    alpha = math.exp(-tau)
    mu0 = 0.5 * snr
    mu1 = -alpha * 0.5 * snr
    sigma = 1.0
    sigma_leak = 1.6 * sigma
    
    # Priors (paper-aligned: 0.1% leakage)
    w0 = (1.0 - leak_prior) / 2.0
    w1 = (1.0 - leak_prior) / 2.0
    wL = leak_prior
    
    # Likelihoods
    L0 = gauss_pdf(z, mu0, sigma)
    L1 = gauss_pdf(z, mu1, sigma)
    LL = gauss_pdf(z, 0.0, sigma_leak)
    
    # Posteriors
    norm = w0 * L0 + w1 * L1 + wL * LL + 1e-18
    post1 = (w1 * L1) / norm
    postL = (wL * LL) / norm
    
    return post1, postL


def soft_channels(num, snr=10.0, tau=0.01, leak_p=0.001):
    """Paper-aligned soft channel generation.
    
    Fixed issues:
    - Leakage prior: 0.1% (was 1%)
    - I/Q means: symmetric +/- SNR/2 model (was 0/1/2)
    """
    # Sample states with paper-aligned priors
    p0 = (1.0 - leak_p) / 2.0
    p1 = (1.0 - leak_p) / 2.0
    pL = leak_p
    states = np.random.choice([0, 1, 2], size=num, p=[p0, p1, pL])
    
    # Generate I/Q samples with paper-aligned model
    z_vals = np.array([iq_sample_paper_aligned(s, snr, tau) for s in states])
    
    # Compute posteriors
    post_results = [posteriors_paper_aligned(z, snr, tau, leak_p) for z in z_vals]
    post1 = np.array([p[0] for p in post_results], dtype=np.float32)
    postL = np.array([p[1] for p in post_results], dtype=np.float32)
    
    return post1, postL

# --------------------------------------------------------------------
def simulate_experiment_root(
    exp_root: Path,
    out_npz: Path,
    shots: int,
    snr: float = 10.0,
    tau: float = 0.01,
    leak_p: float = 0.001,  # Paper-aligned: 0.1% leakage prior
) -> None:
    dm = DataManager()

    for stim_path in exp_root.rglob("*.stim"):
        dem_path = stim_path.with_suffix(".dem")
        if not dem_path.exists():
            continue  # skip unpaired files

        dets_flat, obs = sample_detectors_obs(stim_path, dem_path, shots=shots)
        dets = reshape_detectors(dets_flat, dem_path)        # (N,R,S,1)

        R, S = rounds_dets_per_round(dem_path)
        post1, post2 = soft_channels(shots * R * S, snr=snr, tau=tau, leak_p=leak_p)
        post1 = post1.reshape(shots, R, S, 1)
        post2 = post2.reshape(shots, R, S, 1)

        data  = np.concatenate([dets, post1, post2], axis=-1)  # (N,R,S,3)
        dm.store(data, obs)

        print(f"[OK] {stim_path.relative_to(exp_root)}  shots={shots}")

    if not dm._data_blocks:
        raise RuntimeError("No (*.stim, *.dem) pairs were found!")

    dm.save(out_npz)

# --------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_root", type=Path,
                    help="Root folder that contains many sub-folders each "
                         "with *.stim & *.dem files.")
    ap.add_argument("--shots", type=int, default=1000,
                    help="Monte-Carlo shots per circuit (default 1000).")
    ap.add_argument("--out", type=Path, default=Path("./sim_soft_data.npz"),
                    help="Output .npz filename (default ./sim_soft_data.npz)")
    args = ap.parse_args()

    simulate_experiment_root(
        args.exp_root,
        args.out,
        shots=args.shots,
    )
    print("DONE.  →", args.out.resolve())
