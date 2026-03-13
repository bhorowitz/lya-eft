from __future__ import annotations

from dataclasses import dataclass

import camb
import numpy as np
from scipy.signal import savgol_filter


@dataclass
class LinearPowerResult:
    k_hmpc: np.ndarray
    p_lin_h3mpc3: np.ndarray
    p_nw_h3mpc3: np.ndarray
    f_growth: float
    sigma8_0: float | None = None


def omega_m_z(omega_m0: float, z: float) -> float:
    ez2 = omega_m0 * (1.0 + z) ** 3 + (1.0 - omega_m0)
    return omega_m0 * (1.0 + z) ** 3 / ez2


def growth_rate_linder(omega_m0: float, z: float) -> float:
    return omega_m_z(omega_m0, z) ** 0.55


def smooth_no_wiggle(pk: np.ndarray, window: int = 61, polyorder: int = 3) -> np.ndarray:
    if window >= pk.size:
        window = pk.size - 1
    if window % 2 == 0:
        window -= 1
    window = max(window, 5)
    polyorder = min(polyorder, window - 2)
    return savgol_filter(pk, window_length=window, polyorder=polyorder)


def compute_linear_power_camb(
    *,
    h: float,
    omega_b: float,
    omega_cdm: float,
    ns: float,
    As: float,
    z: float,
    kmin: float,
    kmax: float,
    nk: int,
) -> LinearPowerResult:
    omega_m0 = omega_b + omega_cdm

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100.0 * h,
        ombh2=omega_b * h**2,
        omch2=omega_cdm * h**2,
        mnu=0.06,
        omk=0.0,
        tau=0.054,
    )
    pars.InitPower.set_params(As=As, ns=ns)
    redshift_list = [float(z)]
    if abs(float(z)) > 1.0e-10:
        redshift_list.append(0.0)
    pars.set_matter_power(redshifts=redshift_list, kmax=max(2.0 * kmax, 10.0))
    pars.NonLinear = camb.model.NonLinear_none

    results = camb.get_results(pars)

    kh = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    pk = results.get_matter_power_interpolator(nonlinear=False, hubble_units=True, k_hunit=True)
    p_lin = pk.P(z, kh)
    p_nw = smooth_no_wiggle(p_lin)

    f = growth_rate_linder(omega_m0=omega_m0, z=z)

    sigma8_0 = None
    try:
        sigma8_0 = float(results.get_sigma8_0())
    except Exception:
        sigma8_0 = None

    return LinearPowerResult(
        k_hmpc=kh,
        p_lin_h3mpc3=p_lin,
        p_nw_h3mpc3=p_nw,
        f_growth=f,
        sigma8_0=sigma8_0,
    )
