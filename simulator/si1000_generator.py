import stim


def si1000_noise_model(config: dict) -> stim.Circuit:
    """Construct a surface-code circuit using Stim's SI1000 noise model.
    
    SI1000 noise model parameters (per Google paper):
    - meas_bitflip: 5p (before measurement)
    - reset_bitflip: 2p (after reset)
    - twoq_depol: p (after 2Q Clifford gates)
    - oneq_depol: p/10 (after 1Q Clifford gates)
    - idle: p/10 (before round data depolarization)
    
    Note: stim.Circuit.generated() uses the same after_clifford_depolarization
    for both 1Q and 2Q gates. We set it to p for correct 2Q gate noise.
    """
    p = config.get("p", 0.001)
    distance = config.get("distance", 3)
    rounds = config.get("rounds", 25)
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=p,          # p for 2Q gates (DEPOLARIZE2)
        before_round_data_depolarization=p / 10,  # p/10 for idle
        before_measure_flip_probability=5 * p,    # 5p for measurement
        after_reset_flip_probability=2 * p,       # 2p for reset (SI1000 spec)
    )
    return circuit
