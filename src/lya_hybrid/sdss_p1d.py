from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class P1DBlock:
    z: float
    k_kms: np.ndarray
    p_kms: np.ndarray
    cov_kms: np.ndarray
    k_hmpc: np.ndarray
    p_hmpc: np.ndarray
    cov_hmpc: np.ndarray


def _hubble_kms_per_mpc(*, z: float, h: float, omega_b: float, omega_cdm: float) -> float:
    omega_m = omega_b + omega_cdm
    ez = np.sqrt(omega_m * (1.0 + z) ** 3 + (1.0 - omega_m))
    return 100.0 * h * ez


def _kms_to_hmpc_factor(*, z: float, h: float, omega_b: float, omega_cdm: float) -> float:
    # k[h/Mpc] = k[s/km] * H(z)/(1+z)/h
    hub = _hubble_kms_per_mpc(z=z, h=h, omega_b=omega_b, omega_cdm=omega_cdm)
    return hub / (1.0 + z) / h


def _convert_block_to_h_units(
    *,
    z: float,
    k_kms: np.ndarray,
    p_kms: np.ndarray,
    cov_kms: np.ndarray,
    h: float,
    omega_b: float,
    omega_cdm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fac = _kms_to_hmpc_factor(z=z, h=h, omega_b=omega_b, omega_cdm=omega_cdm)
    k_h = fac * k_kms
    p_h = p_kms / fac
    cov_h = cov_kms / (fac**2)
    return k_h, p_h, cov_h


def load_chabanier2019_blocks(
    *,
    data_dir: str | Path,
    z_min: float,
    z_max: float,
    h: float,
    omega_b: float,
    omega_cdm: float,
    include_syst: bool = True,
) -> list[P1DBlock]:
    data_dir = Path(data_dir)
    p1d_file = data_dir / "Pk1D_data.dat"
    corr_file = data_dir / "Pk1D_cor.dat"
    syst_file = data_dir / "Pk1D_syst.dat"

    if not p1d_file.exists():
        raise FileNotFoundError(f"Missing file: {p1d_file}")
    if not corr_file.exists():
        raise FileNotFoundError(f"Missing file: {corr_file}")

    zs_raw, k_raw, p_raw, stat_raw, _, _ = np.loadtxt(p1d_file, unpack=True)
    var_raw = stat_raw**2

    if include_syst:
        if not syst_file.exists():
            raise FileNotFoundError(f"Missing file: {syst_file}")
        syst_cols = np.loadtxt(syst_file, unpack=True)
        var_raw = var_raw + np.sum(syst_cols**2, axis=0)

    incorr = np.loadtxt(corr_file, unpack=True)
    z_unique = np.unique(zs_raw)

    blocks: list[P1DBlock] = []
    for z in z_unique:
        if z < z_min or z > z_max:
            continue

        mask = np.argwhere((zs_raw == z) & np.isfinite(p_raw) & (var_raw > 0))[:, 0]
        kk = np.asarray(k_raw[mask], dtype=float)
        pp = np.asarray(p_raw[mask], dtype=float)
        sigma = np.sqrt(np.asarray(var_raw[mask], dtype=float))
        cov_kms = np.multiply(incorr[:, mask], np.outer(sigma, sigma))

        k_h, p_h, cov_h = _convert_block_to_h_units(
            z=float(z),
            k_kms=kk,
            p_kms=pp,
            cov_kms=np.asarray(cov_kms, dtype=float),
            h=h,
            omega_b=omega_b,
            omega_cdm=omega_cdm,
        )
        blocks.append(
            P1DBlock(
                z=float(z),
                k_kms=kk,
                p_kms=pp,
                cov_kms=np.asarray(cov_kms, dtype=float),
                k_hmpc=k_h,
                p_hmpc=p_h,
                cov_hmpc=cov_h,
            )
        )

    return blocks


def load_eboss_mock_blocks(
    *,
    data_dir: str | Path,
    z_min: float,
    z_max: float,
    h: float,
    omega_b: float,
    omega_cdm: float,
) -> list[P1DBlock]:
    data_dir = Path(data_dir)
    mock_file = data_dir / "pk_1d_Nyx_emu_fiducial_mock.out"
    invcov_file = data_dir / "pk_1d_DR12_13bins_invCov.out"

    if not mock_file.exists():
        raise FileNotFoundError(f"Missing file: {mock_file}")
    if not invcov_file.exists():
        raise FileNotFoundError(f"Missing file: {invcov_file}")

    z_all, k_all, p_all, *_ = np.loadtxt(mock_file, unpack=True)
    inv_cov_full = np.loadtxt(invcov_file)

    z_unique = np.unique(z_all)
    blocks: list[P1DBlock] = []
    for z in z_unique:
        if z < z_min or z > z_max:
            continue
        mask = np.argwhere(z_all == z)[:, 0]
        kk = np.asarray(k_all[mask], dtype=float)
        pp = np.asarray(p_all[mask], dtype=float)
        cov_kms = np.linalg.inv(inv_cov_full[mask][:, mask])

        k_h, p_h, cov_h = _convert_block_to_h_units(
            z=float(z),
            k_kms=kk,
            p_kms=pp,
            cov_kms=np.asarray(cov_kms, dtype=float),
            h=h,
            omega_b=omega_b,
            omega_cdm=omega_cdm,
        )
        blocks.append(
            P1DBlock(
                z=float(z),
                k_kms=kk,
                p_kms=pp,
                cov_kms=np.asarray(cov_kms, dtype=float),
                k_hmpc=k_h,
                p_hmpc=p_h,
                cov_hmpc=cov_h,
            )
        )

    return blocks
