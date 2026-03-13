from __future__ import annotations

import numpy as np


def linear_flux_prefactor(mu: np.ndarray, *, b1: float, b_eta: float, f_growth: float) -> np.ndarray:
    # eta enters with an LOS weighting; this toy form is the minimal anisotropic structure.
    return b1 + b_eta * f_growth * mu**2


def counterterm_component(
    k: np.ndarray,
    mu: np.ndarray,
    p_lin: np.ndarray,
    *,
    c0: float,
    c2: float,
    c4: float,
) -> np.ndarray:
    mu2 = mu**2
    mu4 = mu2**2
    return -2.0 * (c0 + c2 * mu2 + c4 * mu4) * (k**2) * p_lin


def loop_component_toy(
    k: np.ndarray,
    mu: np.ndarray,
    p_lin: np.ndarray,
    *,
    loop_amp: float,
    loop_mu2: float,
    loop_mu4: float,
    loop_k_nl: float,
) -> np.ndarray:
    mu2 = mu**2
    mu4 = mu2**2
    shape = (k / loop_k_nl) ** 2 / (1.0 + (k / loop_k_nl) ** 2)
    anis = 1.0 + loop_mu2 * mu2 + loop_mu4 * mu4
    return loop_amp * shape * anis * p_lin
