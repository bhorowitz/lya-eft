from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import fitsio
except Exception as exc:  # pragma: no cover
    fitsio = None
    _fitsio_error = exc
else:
    _fitsio_error = None


@dataclass
class SherwoodP3DData:
    p3d_hmpc3: np.ndarray
    k_hmpc: np.ndarray
    mu: np.ndarray
    counts: np.ndarray
    mean_flux: float
    n_k_bins: int
    n_mu_bins: int
    k_hmpc_max: float

    def valid_mask(self) -> np.ndarray:
        return (
            np.isfinite(self.p3d_hmpc3)
            & np.isfinite(self.k_hmpc)
            & np.isfinite(self.mu)
            & (self.counts > 0)
        )

    def flatten_valid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mask = self.valid_mask()
        return (
            self.k_hmpc[mask],
            self.mu[mask],
            self.p3d_hmpc3[mask],
            self.counts[mask],
        )


@dataclass
class SherwoodP1DData:
    kp_hmpc: np.ndarray
    p1d_hmpc: np.ndarray
    mean_flux: float

    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.kp_hmpc) & np.isfinite(self.p1d_hmpc)


def _require_fitsio() -> None:
    if fitsio is None:  # pragma: no cover
        raise ImportError(
            "fitsio is required to load Sherwood FITS files. "
            "Install with `pip install fitsio`."
        ) from _fitsio_error


def load_sherwood_flux_p3d(path: str | Path) -> SherwoodP3DData:
    _require_fitsio()
    fp = Path(path)
    arr = fitsio.read(fp, ext="FLUX_P3D")
    header = fitsio.read_header(fp, ext="FLUX_P3D")

    data = SherwoodP3DData(
        p3d_hmpc3=np.asarray(arr["P3D_HMPC"], dtype=float),
        k_hmpc=np.asarray(arr["K_HMPC"], dtype=float),
        mu=np.asarray(arr["MU"], dtype=float),
        counts=np.asarray(arr["COUNTS"], dtype=float),
        mean_flux=float(header["MEAN_FLUX"]),
        n_k_bins=int(header["N_K_BINS"]),
        n_mu_bins=int(header["N_MU_BINS"]),
        k_hmpc_max=float(header["K_HMPC_MAX"]),
    )

    if data.p3d_hmpc3.shape != data.k_hmpc.shape:
        raise ValueError("P3D and k arrays must have identical shapes.")
    if data.mu.shape != data.p3d_hmpc3.shape:
        raise ValueError("mu array shape mismatch.")
    if data.counts.shape != data.p3d_hmpc3.shape:
        raise ValueError("counts array shape mismatch.")

    return data


def load_sherwood_flux_p1d(path: str | Path) -> SherwoodP1DData:
    _require_fitsio()
    fp = Path(path)
    arr = fitsio.read(fp, ext="FLUX_P1D")
    header = fitsio.read_header(fp, ext="FLUX_P1D")

    data = SherwoodP1DData(
        kp_hmpc=np.asarray(arr["KP_HMPC"], dtype=float),
        p1d_hmpc=np.asarray(arr["P1D_HMPC"], dtype=float),
        mean_flux=float(header["MEAN_FLUX"]),
    )

    if data.kp_hmpc.shape != data.p1d_hmpc.shape:
        raise ValueError("P1D and k_parallel arrays must have identical shapes.")

    return data
