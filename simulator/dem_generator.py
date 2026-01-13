import stim

def generate_dem_data(num_samples: int, dem_config: dict):
    # generate dem model from .dem file
    dem = stim.DetectorErrorModel.from_file(dem_config["dem_path"])
    # compile sampler
    sampler = dem.compile_sampler()
    # sample syndromes and logical errors
    syndromes, logical_errors, _ = sampler.sample(shots=num_samples, return_errors=True)
    return syndromes, logical_errors