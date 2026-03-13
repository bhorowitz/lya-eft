from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from velocileptors.EPT.moment_expansion_fftw import MomentExpansion


@dataclass
class EPTReducedParams:
    b1: float = 1.0
    b2: float = 0.0
    bs: float = 0.0
    b3: float = 0.0
    alpha0: float = 0.5
    alpha2: float = 0.2
    alpha4: float = 0.2
    sn: float = 0.0
    sn2: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([
            self.b1,
            self.b2,
            self.bs,
            self.b3,
            self.alpha0,
            self.alpha2,
            self.alpha4,
            self.sn,
            self.sn2,
        ])


class VelocileptorsEPTBackend:
    def __init__(
        self,
        k_lin: np.ndarray,
        p_lin: np.ndarray,
        p_nw: np.ndarray,
        *,
        kmin: float,
        kmax: float,
        nk: int,
        threads: int,
        beyond_gauss: bool,
    ) -> None:
        self.model = MomentExpansion(
            k_lin,
            p_lin,
            pnw=p_nw,
            kmin=kmin,
            kmax=kmax,
            nk=nk,
            threads=threads,
            beyond_gauss=beyond_gauss,
        )

    def power_at_mu(
        self,
        *,
        mu: float,
        f_growth: float,
        params: EPTReducedParams,
        counterterm_c3: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.model.compute_redshift_space_power_at_mu(
            pars=params.as_array(),
            f=f_growth,
            mu_obs=mu,
            reduced=True,
            counterterm_c3=counterterm_c3,
            beyond_gauss=False,
        )

    def multipoles(
        self,
        *,
        f_growth: float,
        params: EPTReducedParams,
        counterterm_c3: float = 0.0,
        ngauss: int = 4,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.model.compute_redshift_space_power_multipoles(
            pars=params.as_array(),
            f=f_growth,
            reduced=True,
            counterterm_c3=counterterm_c3,
            ngauss=ngauss,
            beyond_gauss=False,
        )

    def evaluate_grid(
        self,
        *,
        k_eval: np.ndarray,
        mu_eval: np.ndarray,
        f_growth: float,
        params: EPTReducedParams,
        counterterm_c3: float = 0.0,
    ) -> np.ndarray:
        p_grid = np.zeros((mu_eval.size, k_eval.size))
        for i, mu in enumerate(mu_eval):
            k_native, p_native = self.power_at_mu(
                mu=float(mu), f_growth=f_growth, params=params, counterterm_c3=counterterm_c3
            )
            p_grid[i, :] = interp1d(k_native, p_native, kind="linear", fill_value="extrapolate")(k_eval)
        return p_grid
