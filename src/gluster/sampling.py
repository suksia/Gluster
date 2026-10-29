import random
import numpy as np

class Sampler:
    def __init__(self, seed: int = None):
        self.py_rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def sample_exp_distr(self, k: int, a: float):
        """Sample discrete integer values following the exponential distribution."""
        i = np.arange(0, k)
        w = a*np.exp(-a*i/k)
        return self.py_rng.choices(range(0, k), weights=w)[0]

    def sample_uni_distr(self, k: int):
        return self.py_rng.choices(range(0, k))[0]