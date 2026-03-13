from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d

from .operators_ivanov import counterterm_component, linear_flux_prefactor, loop_component_toy


@dataclass
class IvanovToyParams:
    b1: float
    b_eta: float
    c0: float
    c2: float
    c4: float
    loop_amp: float
    loop_mu2: float
    loop_mu4: float
    loop_k_nl: float
    stochastic: float = 0.0


class IvanovToyModel:
    def __init__(self, k_lin: np.ndarray, p_lin: np.ndarray, f_growth: float) -> None:
        self.f_growth = f_growth
        self._interp = interp1d(k_lin, p_lin, kind="linear", fill_value="extrapolate")

    def evaluate_components(
        self,
        k: np.ndarray,
        mu: np.ndarray,
        params: IvanovToyParams,
    ) -> dict[str, np.ndarray]:
        k = np.asarray(k)
        mu = np.asarray(mu)
        p_lin = self._interp(k)

        pref = linear_flux_prefactor(mu, b1=params.b1, b_eta=params.b_eta, f_growth=self.f_growth)
        tree = pref**2 * p_lin
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

        total = tree + loop + counter + stochastic
        return {
            "tree": tree,
            "loop": loop,
            "counterterm": counter,
            "stochastic": stochastic,
            "total": total,
        }

    def evaluate_grid(
        self,
        k_values: np.ndarray,
        mu_values: np.ndarray,
        params: IvanovToyParams,
    ) -> dict[str, np.ndarray]:
        kk = np.broadcast_to(k_values[None, :], (mu_values.size, k_values.size))
        mm = np.broadcast_to(mu_values[:, None], (mu_values.size, k_values.size))
        return self.evaluate_components(kk, mm, params)
