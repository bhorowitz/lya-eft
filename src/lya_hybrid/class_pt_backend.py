from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from classy import Class


@dataclass
class ClassPTTracerParams:
    b1: float = 1.0
    b2: float = 0.0
    bG2: float = 0.0
    bGamma3: float = 0.0
    cs0: float = 0.0
    cs2: float = 0.0
    cs4: float = 0.0
    Pshot: float = 0.0
    b4: float = 0.0


class ClassPTBackend:
    """Thin wrapper around the local CLASS-PT Python interface."""

    def __init__(
        self,
        *,
        h: float,
        omega_b: float,
        omega_cdm: float,
        ns: float,
        As: float,
        z: float,
        tau_reio: float = 0.054,
        P_k_max_hmpc: float = 6.0,
        # toggles for CLASS-PT features (kept as strings where CLASS expects Yes/No)
        ir_resummation: str = "Yes",
        rsd: str = "Yes",
        cb: str = "Yes",
        non_linear: str = "PT",
    ) -> None:
        self.z = float(z)
        self._cosmo = Class()
        self._cosmo.set(
            {
                "A_s": float(As),
                "n_s": float(ns),
                "omega_b": float(omega_b) * float(h) ** 2,
                "omega_cdm": float(omega_cdm) * float(h) ** 2,
                "h": float(h),
                "tau_reio": float(tau_reio),
                "output": "mPk",
                "P_k_max_h/Mpc": float(P_k_max_hmpc),
                "z_pk": float(z),
                "non linear": str(non_linear),
                "IR resummation": str(ir_resummation),
                "Bias tracers": "Yes",
                "cb": str(cb),
                "RSD": str(rsd),
                # optional settings (AP/PNG) intentionally omitted to avoid
                # input parameter parsing issues in some CLASS builds
            }
        )
        self._cosmo.compute()
        self._k_init: np.ndarray | None = None

    @property
    def cosmo(self) -> Class:
        return self._cosmo

    def initialize_output(self, k_hmpc: np.ndarray) -> None:
        k_hmpc = np.asarray(k_hmpc, dtype=float)
        self._cosmo.initialize_output(k_hmpc, self.z, int(k_hmpc.size))
        self._k_init = k_hmpc

    def matter_multipoles(self, *, cs0: float = 0.0, cs2: float = 0.0, cs4: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._k_init is None:
            raise RuntimeError("initialize_output() must be called before requesting spectra")
        return (
            self._k_init.copy(),
            np.asarray(self._cosmo.pk_mm_l0(float(cs0)), dtype=float),
            np.asarray(self._cosmo.pk_mm_l2(float(cs2)), dtype=float),
            np.asarray(self._cosmo.pk_mm_l4(float(cs4)), dtype=float),
        )

    def tracer_multipoles(self, params: ClassPTTracerParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._k_init is None:
            raise RuntimeError("initialize_output() must be called before requesting spectra")
        return (
            self._k_init.copy(),
            np.asarray(
                self._cosmo.pk_gg_l0(
                    float(params.b1),
                    float(params.b2),
                    float(params.bG2),
                    float(params.bGamma3),
                    float(params.cs0),
                    float(params.Pshot),
                    float(params.b4),
                ),
                dtype=float,
            ),
            np.asarray(
                self._cosmo.pk_gg_l2(
                    float(params.b1),
                    float(params.b2),
                    float(params.bG2),
                    float(params.bGamma3),
                    float(params.cs2),
                    float(params.b4),
                ),
                dtype=float,
            ),
            np.asarray(
                self._cosmo.pk_gg_l4(
                    float(params.b1),
                    float(params.b2),
                    float(params.bG2),
                    float(params.bGamma3),
                    float(params.cs4),
                    float(params.b4),
                ),
                dtype=float,
            ),
        )

    def close(self) -> None:
        # CLASS exposes explicit cleanup hooks.
        self._cosmo.struct_cleanup()
        self._cosmo.empty()

    def __enter__(self) -> "ClassPTBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
