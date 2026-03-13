from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d

from .operators_ivanov import counterterm_component, loop_component_toy


@dataclass
class HybridToyParams:
    b_delta: float
    b_eta: float
    b_t: float
    c0: float
    c2: float
    c4: float
    loop_amp: float
    loop_mu2: float
    loop_mu4: float
    loop_k_nl: float
    sigma_th: float
    stochastic: float = 0.0


class HybridToyModel:
    """Minimal source+LOS hybrid model.

    Layer A (source): anisotropic source bias + one-loop toy corrections.
    Layer B (LOS): explicit thermal broadening kernel exp[-(k*mu*sigma_th)^2].
    """

    def __init__(self, k_lin: np.ndarray, p_lin: np.ndarray, f_growth: float, *, k_t: float = 0.5) -> None:
        self.f_growth = f_growth
        self.k_t = max(float(k_t), 1.0e-4)
        self._interp = interp1d(k_lin, p_lin, kind="linear", fill_value="extrapolate")

    def evaluate_components(
        self,
        k: np.ndarray,
        mu: np.ndarray,
        params: HybridToyParams,
    ) -> dict[str, np.ndarray]:
        k = np.asarray(k)
        mu = np.asarray(mu)
        p_lin = self._interp(k)

        # Coarse-grained temperature-like source response: large on long scales, suppressed in UV.
        temp_shape = 1.0 / (1.0 + (k / self.k_t) ** 2)
        source_pref = params.b_delta + params.b_eta * self.f_growth * mu**2 + params.b_t * temp_shape

        tree = source_pref**2 * p_lin
        loop = loop_component_toy(
            k,
            mu,
            p_lin,
            loop_amp=params.loop_amp,
            loop_mu2=params.loop_mu2,
            loop_mu4=params.loop_mu4,
            loop_k_nl=params.loop_k_nl,
        )
        counter = counterterm_component(
            k,
            mu,
            p_lin,
            c0=params.c0,
            c2=params.c2,
            c4=params.c4,
        )
        stochastic = np.full_like(tree, params.stochastic)

        source_total = tree + loop + counter + stochastic
        los_kernel = np.exp(-((k * mu * params.sigma_th) ** 2))
        total = los_kernel * source_total

        return {
            "tree": tree,
            "loop": loop,
            "counterterm": counter,
            "stochastic": stochastic,
            "source_total": source_total,
            "los_kernel": los_kernel,
            "total": total,
        }

    def evaluate_grid(
        self,
        k_values: np.ndarray,
        mu_values: np.ndarray,
        params: HybridToyParams,
    ) -> dict[str, np.ndarray]:
        kk = np.broadcast_to(k_values[None, :], (mu_values.size, k_values.size))
        mm = np.broadcast_to(mu_values[:, None], (mu_values.size, k_values.size))
        return self.evaluate_components(kk, mm, params)
