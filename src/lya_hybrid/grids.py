from __future__ import annotations

import numpy as np


def log_k_grid(kmin: float, kmax: float, nk: int) -> np.ndarray:
    return np.logspace(np.log10(kmin), np.log10(kmax), nk)


def mu_grid(nmu: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, nmu)
