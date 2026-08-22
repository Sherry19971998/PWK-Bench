"""Determinism helpers — one global seed drives the whole pipeline."""
import random
import numpy as np
from .spec import GLOBAL_SEED

def set_seed(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

def rng(seed: int = GLOBAL_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)
