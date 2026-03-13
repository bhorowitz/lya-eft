#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from lya_hybrid.config import load_config
from lya_hybrid.linear_power import compute_linear_power_camb
from lya_hybrid.logging_utils import build_repro_metadata, init_run_dir, write_json
from lya_hybrid.model_hybrid import HybridToyModel, HybridToyParams
from lya_hybrid.model_ivanov import IvanovToyModel, IvanovToyParams
from lya_hybrid.projection_1d import Polynomial1DCounterterms, project_to_1d
from lya_hybrid.sdss_p1d import P1DBlock, load_chabanier2019_blocks, load_eboss_mock_blocks

DEFAULT_SDSS_DIR = Path("data/external/cup1d/data/p1d_measurements/Chabanier2019")
DEFAULT_MOCK_DIR = Path("data/external/cup1d/data/p1d_measurements/eBOSS_mock")
THEORY_CHOICES = ("one_loop", "hybrid")
COUNTERTERM_MODES = ("proxy", "paper_baseline", "paper_rescaled")
PAPER_COUNTERTERM_RELATIONS = {
    "paper_baseline": {
        "C0_1d": (-0.17, -0.2053),
        "C2_1d": (-9.7e-4, 2.16e-2),
        "label": "Published LaCE calibration from Eq. (S12)",
    },
    "paper_rescaled": {
        "C0_1d": (-0.26, -1.1175),
        "C2_1d": (1.68e-3, 0.1107),
        "label": "Published LaCE calibration from Eq. (S13)",
    },
}
PAPER_C0_SIGMA = 5.0 * 5.0e-3
PAPER_C2_SIGMA = 0.5 * 5.0e-4
PAPER_FSIIII = 8.7e-3
PAPER_VSIIII_KMS = 2.0 * np.pi / 0.0028
PAPER_KS_THERMAL_KMS_INV = 0.11


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage-1 2405-style SDSS DR14 baseline with simulation-based prior calibration: "
            "recalibrate C0/C2 on eBOSS mock proxy, then fit SDSS with informative and conservative priors."
        )
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--theory", choices=THEORY_CHOICES, default="one_loop")
    p.add_argument("--prior-csv", type=Path, default=None)
    p.add_argument("--sdss-dir", type=Path, default=DEFAULT_SDSS_DIR)
    p.add_argument("--mock-dir", type=Path, default=DEFAULT_MOCK_DIR)
    p.add_argument("--z-min", type=float, default=3.2)
    p.add_argument("--z-max", type=float, default=4.2)
    p.add_argument("--kmin-fit-hmpc", type=float, default=0.03)
    p.add_argument("--kmax-fit-hmpc", type=float, default=3.0)
    p.add_argument("--kmax-proj", type=float, default=3.0)
    p.add_argument("--nint-proj", type=int, default=320)
    p.add_argument("--mode", choices=["informative", "conservative", "both"], default="both")
    p.add_argument("--conservative-inflate", type=float, default=5.0)
    p.add_argument("--counterterm-mode", choices=COUNTERTERM_MODES, default="proxy")
    p.add_argument("--apply-paper-systematics", action="store_true")
    p.add_argument("--b-eta-min", type=float, default=-2.0)
    p.add_argument("--b-eta-max", type=float, default=2.0)
    p.add_argument("--n-starts", type=int, default=12)
    p.add_argument("--max-nfev", type=int, default=2600)
    p.add_argument("--seed", type=int, default=20260313)
    return p.parse_args()


@dataclass
class FitBlock:
    z: float
    k_hmpc: np.ndarray
    p_hmpc: np.ndarray
    cov_hmpc: np.ndarray
    chol: np.ndarray


def latest_stage1_prior_csv(theory: Literal["one_loop", "hybrid"]) -> Path:
    if theory == "hybrid":
        patterns = ["*_repro_2405_stage1_sherwood_hybrid/logs/sherwood_prior_linear_fits.csv"]
    else:
        patterns = [
            "*_repro_2405_stage1_sherwood/logs/sherwood_prior_linear_fits.csv",
            "*_repro_2405_stage1_sherwood_one_loop/logs/sherwood_prior_linear_fits.csv",
        ]

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(Path("results/runs").glob(pattern)))
    if not candidates:
        raise FileNotFoundError(
            f"No stage-1 Sherwood prior CSV found for theory={theory!r} under results/runs."
        )
    return candidates[-1]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def load_prior_relations(path: Path, theory: Literal["one_loop", "hybrid"]) -> dict[str, dict[str, float]]:
    rows = read_csv_rows(path)
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        out[row["operator_name"]] = {
            "A": float(row["A_O"]),
            "B": float(row["B_O"]),
            "rmse": float(row["fit_rmse"]),
        }
    needed = {"c0_3d", "c2_3d", "c4_3d", "loop_amp"}
    if theory == "hybrid":
        needed |= {"b_t", "sigma_th"}
    missing = needed.difference(out.keys())
    if missing:
        raise ValueError(f"Missing required operators in prior CSV {path}: {sorted(missing)}")
    return out


def build_fit_blocks(
    blocks: list[P1DBlock], *, kmin_fit_hmpc: float, kmax_fit_hmpc: float
) -> list[FitBlock]:
    out: list[FitBlock] = []
    for b in blocks:
        mask = (b.k_hmpc >= kmin_fit_hmpc) & (b.k_hmpc <= kmax_fit_hmpc)
        kk = b.k_hmpc[mask]
        pp = b.p_hmpc[mask]
        cov = b.cov_hmpc[np.ix_(mask, mask)]
        # small diagonal jitter for robust Cholesky
        jitter = max(1.0e-12, 1.0e-9 * float(np.median(np.diag(cov))))
        cov_reg = cov + jitter * np.eye(cov.shape[0], dtype=float)
        chol = np.linalg.cholesky(cov_reg)
        out.append(FitBlock(z=b.z, k_hmpc=kk, p_hmpc=pp, cov_hmpc=cov_reg, chol=chol))
    return out


def relation_eval(rel: dict[str, dict[str, float]], name: str, b1: float) -> float:
    return rel[name]["A"] * b1 + rel[name]["B"]


def _hubble_kms_per_mpc(*, z: float, h: float, omega_b: float, omega_cdm: float) -> float:
    omega_m = omega_b + omega_cdm
    ez = np.sqrt(omega_m * (1.0 + z) ** 3 + (1.0 - omega_m))
    return 100.0 * h * ez


def _hmpc_to_kms_factor(*, z: float, h: float, omega_b: float, omega_cdm: float) -> float:
    hub = _hubble_kms_per_mpc(z=z, h=h, omega_b=omega_b, omega_cdm=omega_cdm)
    return h * (1.0 + z) / hub


def paper_systematics_factor(*, z: float, kpar_hmpc: np.ndarray, cfg) -> np.ndarray:
    kpar_hmpc = np.asarray(kpar_hmpc, dtype=float)
    fac = _hmpc_to_kms_factor(
        z=float(z),
        h=float(cfg.cosmology.h),
        omega_b=float(cfg.cosmology.omega_b),
        omega_cdm=float(cfg.cosmology.omega_cdm),
    )
    kpar_kms = fac * kpar_hmpc
    fbar_fid = np.exp(-0.0025 * (1.0 + float(z)) ** 3.7)
    siii_ratio = PAPER_FSIIII / max(1.0 - fbar_fid, 1.0e-8)
    kappa_siiii = 1.0 + 2.0 * siii_ratio * np.cos(PAPER_VSIIII_KMS * kpar_kms) + siii_ratio**2
    thermal = np.exp(-((kpar_kms / PAPER_KS_THERMAL_KMS_INV) ** 2))
    return kappa_siiii * thermal


def published_counterterm_relations(mode: str) -> tuple[tuple[float, float], tuple[float, float], list[dict[str, float]], list[dict[str, float]]]:
    rel = PAPER_COUNTERTERM_RELATIONS[mode]
    c0 = tuple(float(x) for x in rel["C0_1d"])
    c2 = tuple(float(x) for x in rel["C2_1d"])
    fit_rows = [
        {"operator_name": "C0_1d", "A_O": c0[0], "B_O": c0[1], "fit_rmse": 0.0, "n_points": 0},
        {"operator_name": "C2_1d", "A_O": c2[0], "B_O": c2[1], "fit_rmse": 0.0, "n_points": 0},
    ]
    return c0, c2, [], fit_rows


def p1d_prediction(
    *,
    theory: Literal["one_loop", "hybrid"],
    model: IvanovToyModel | HybridToyModel,
    z: float,
    kpar_hmpc: np.ndarray,
    b1: float,
    b_eta: float,
    sigma8: float,
    sigma8_ref: float,
    prior3d: dict[str, dict[str, float]],
    c0_1d_rel: tuple[float, float],
    c2_1d_rel: tuple[float, float],
    apply_paper_systematics: bool,
    kmax_proj: float,
    nint_proj: int,
    cfg,
    offsets: dict[str, float] | None = None,
) -> np.ndarray:
    if offsets is None:
        offsets = {}
    d = lambda key: float(offsets.get(key, 0.0))

    if theory == "one_loop":
        params = IvanovToyParams(
            b1=float(b1),
            b_eta=float(b_eta),
            c0=float(relation_eval(prior3d, "c0_3d", b1) + d("c0_3d")),
            c2=float(relation_eval(prior3d, "c2_3d", b1) + d("c2_3d")),
            c4=float(relation_eval(prior3d, "c4_3d", b1) + d("c4_3d")),
            loop_amp=float(relation_eval(prior3d, "loop_amp", b1) + d("loop_amp")),
            loop_mu2=float(cfg.ivanov_toy.loop_mu2),
            loop_mu4=float(cfg.ivanov_toy.loop_mu4),
            loop_k_nl=float(cfg.ivanov_toy.loop_k_nl),
            stochastic=float(cfg.ivanov_toy.stochastic),
        )
    else:
        params = HybridToyParams(
            b_delta=float(b1),
            b_eta=float(b_eta),
            b_t=float(relation_eval(prior3d, "b_t", b1) + d("b_t")),
            c0=float(relation_eval(prior3d, "c0_3d", b1) + d("c0_3d")),
            c2=float(relation_eval(prior3d, "c2_3d", b1) + d("c2_3d")),
            c4=float(relation_eval(prior3d, "c4_3d", b1) + d("c4_3d")),
            loop_amp=float(relation_eval(prior3d, "loop_amp", b1) + d("loop_amp")),
            loop_mu2=float(cfg.hybrid_toy.loop_mu2),
            loop_mu4=float(cfg.hybrid_toy.loop_mu4),
            loop_k_nl=float(cfg.hybrid_toy.loop_k_nl),
            sigma_th=float(max(0.0, relation_eval(prior3d, "sigma_th", b1) + d("sigma_th"))),
            stochastic=float(cfg.hybrid_toy.stochastic),
        )
    amp = (float(sigma8) / float(sigma8_ref)) ** 2

    def p3d_callable(kk: np.ndarray, mm: np.ndarray) -> np.ndarray:
        return amp * model.evaluate_components(kk, mm, params)["total"]

    raw = project_to_1d(
        kpar_values=kpar_hmpc,
        p3d_callable=p3d_callable,
        kmax_proj=float(kmax_proj),
        nint=int(nint_proj),
        method="trapz",
        counterterms=Polynomial1DCounterterms(),
    )["raw"]

    c0_1d = c0_1d_rel[0] * b1 + c0_1d_rel[1] + d("C0_1d")
    c2_1d = c2_1d_rel[0] * b1 + c2_1d_rel[1] + d("C2_1d")
    pred = raw + c0_1d + c2_1d * kpar_hmpc**2
    if apply_paper_systematics:
        pred = pred * paper_systematics_factor(z=float(z), kpar_hmpc=kpar_hmpc, cfg=cfg)
    return pred


def fit_mock_counterterm_relations(
    *,
    theory: Literal["one_loop", "hybrid"],
    mock_blocks: list[FitBlock],
    models_by_z: dict[float, IvanovToyModel | HybridToyModel],
    sigma8_ref: float,
    prior3d: dict[str, dict[str, float]],
    cfg,
    args,
) -> tuple[tuple[float, float], tuple[float, float], list[dict[str, float]], list[dict[str, float]]]:
    # Per-z fits with free (b1, b_eta, C0, C2).
    rows = []
    x0 = np.array([-0.5, 0.0, 0.0, 0.0], dtype=float)

    for block in mock_blocks:
        model = models_by_z[block.z]

        def residual(theta: np.ndarray) -> np.ndarray:
            b1, b_eta, c0, c2 = [float(x) for x in theta]
            pred = p1d_prediction(
                theory=theory,
                model=model,
                z=float(block.z),
                kpar_hmpc=block.k_hmpc,
                b1=b1,
                b_eta=b_eta,
                sigma8=sigma8_ref,
                sigma8_ref=sigma8_ref,
                prior3d=prior3d,
                c0_1d_rel=(0.0, c0),
                c2_1d_rel=(0.0, c2),
                apply_paper_systematics=bool(args.apply_paper_systematics),
                kmax_proj=float(args.kmax_proj),
                nint_proj=int(args.nint_proj),
                cfg=cfg,
            )
            diff = pred - block.p_hmpc
            return np.linalg.solve(block.chol, diff)

        lo = np.array([-1.5, -2.0, -1.5, -1.5], dtype=float)
        hi = np.array([0.2, 2.0, 1.5, 1.5], dtype=float)
        x0 = np.clip(x0, lo + 1.0e-8, hi - 1.0e-8)
        fit = least_squares(
            residual,
            x0=x0,
            bounds=(lo, hi),
            method="trf",
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=int(args.max_nfev),
        )
        x0 = fit.x.copy()
        b1, b_eta, c0, c2 = [float(x) for x in fit.x]
        chi2 = float(np.sum(residual(fit.x) ** 2))
        dof = max(int(block.k_hmpc.size - 4), 1)
        rows.append(
            {
                "z": float(block.z),
                "b1": b1,
                "b_eta": b_eta,
                "C0_1d_fit": c0,
                "C2_1d_fit": c2,
                "chi2_dof": float(chi2 / dof),
            }
        )

    b1_all = np.asarray([r["b1"] for r in rows], dtype=float)
    c0_all = np.asarray([r["C0_1d_fit"] for r in rows], dtype=float)
    c2_all = np.asarray([r["C2_1d_fit"] for r in rows], dtype=float)

    def _linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        A = np.column_stack([x, np.ones_like(x)])
        coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coeff
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        return float(coeff[0]), float(coeff[1]), rmse

    c0_A, c0_B, c0_rmse = _linfit(b1_all, c0_all)
    c2_A, c2_B, c2_rmse = _linfit(b1_all, c2_all)

    fit_rows = [
        {"operator_name": "C0_1d", "A_O": c0_A, "B_O": c0_B, "fit_rmse": c0_rmse, "n_points": int(b1_all.size)},
        {"operator_name": "C2_1d", "A_O": c2_A, "B_O": c2_B, "fit_rmse": c2_rmse, "n_points": int(b1_all.size)},
    ]
    return (c0_A, c0_B), (c2_A, c2_B), rows, fit_rows


def run_sdss_fit(
    *,
    theory: Literal["one_loop", "hybrid"],
    mode: str,
    sdss_blocks: list[FitBlock],
    models_by_z: dict[float, IvanovToyModel | HybridToyModel],
    sigma8_ref: float,
    prior3d: dict[str, dict[str, float]],
    c0_rel: tuple[float, float],
    c2_rel: tuple[float, float],
    conservative_sigmas: dict[str, float],
    linear_start_hint: np.ndarray | None,
    cfg,
    args,
) -> dict[str, object]:
    zvals = [b.z for b in sdss_blocks]
    nz = len(sdss_blocks)
    use_offsets = mode == "conservative"

    bias_label = "b1" if theory == "one_loop" else "b_delta"
    names = ["sigma8"] + [f"{bias_label}_z{z:.1f}" for z in zvals] + [f"b_eta_z{z:.1f}" for z in zvals]
    if use_offsets:
        names += ["d_c0_3d", "d_c2_3d", "d_c4_3d", "d_loop_amp", "d_C0_1d", "d_C2_1d"]
        if theory == "hybrid":
            names += ["d_b_t", "d_sigma_th"]

    n_dim = len(names)
    lo = np.full(n_dim, -np.inf)
    hi = np.full(n_dim, np.inf)

    lo[0], hi[0] = 0.6, 1.1
    lo[1 : 1 + nz], hi[1 : 1 + nz] = -1.5, 0.2
    lo[1 + nz : 1 + 2 * nz], hi[1 + nz : 1 + 2 * nz] = float(args.b_eta_min), float(args.b_eta_max)
    if use_offsets:
        if theory == "one_loop":
            lo[-6:], hi[-6:] = -1.5, 1.5
        else:
            lo[-8:-2], hi[-8:-2] = -1.5, 1.5
            lo[-2:], hi[-2:] = [-0.5, -0.2], [0.5, 0.2]

    x0 = np.zeros(n_dim, dtype=float)
    x0[0] = sigma8_ref
    for i, z in enumerate(zvals):
        x0[1 + i] = -0.45 - 0.18 * (z - min(zvals)) / max(max(zvals) - min(zvals), 1.0e-6)
        x0[1 + nz + i] = 0.0
    if linear_start_hint is not None and linear_start_hint.size >= (1 + 2 * nz):
        x0[: 1 + 2 * nz] = linear_start_hint[: 1 + 2 * nz]

    def unpack(theta: np.ndarray):
        sigma8 = float(theta[0])
        b1 = [float(x) for x in theta[1 : 1 + nz]]
        b_eta = [float(x) for x in theta[1 + nz : 1 + 2 * nz]]
        offsets = {}
        if use_offsets:
            if theory == "one_loop":
                offsets = {
                    "c0_3d": float(theta[-6]),
                    "c2_3d": float(theta[-5]),
                    "c4_3d": float(theta[-4]),
                    "loop_amp": float(theta[-3]),
                    "C0_1d": float(theta[-2]),
                    "C2_1d": float(theta[-1]),
                }
            else:
                offsets = {
                    "c0_3d": float(theta[-8]),
                    "c2_3d": float(theta[-7]),
                    "c4_3d": float(theta[-6]),
                    "loop_amp": float(theta[-5]),
                    "C0_1d": float(theta[-4]),
                    "C2_1d": float(theta[-3]),
                    "b_t": float(theta[-2]),
                    "sigma_th": float(theta[-1]),
                }
        return sigma8, b1, b_eta, offsets

    def residual(theta: np.ndarray) -> np.ndarray:
        sigma8, b1_list, beta_list, offsets = unpack(theta)
        chunks = []
        for i, block in enumerate(sdss_blocks):
            pred = p1d_prediction(
                theory=theory,
                model=models_by_z[block.z],
                z=float(block.z),
                kpar_hmpc=block.k_hmpc,
                b1=b1_list[i],
                b_eta=beta_list[i],
                sigma8=sigma8,
                sigma8_ref=sigma8_ref,
                prior3d=prior3d,
                c0_1d_rel=c0_rel,
                c2_1d_rel=c2_rel,
                apply_paper_systematics=bool(args.apply_paper_systematics),
                kmax_proj=float(args.kmax_proj),
                nint_proj=int(args.nint_proj),
                cfg=cfg,
                offsets=offsets,
            )
            diff = pred - block.p_hmpc
            chunks.append(np.linalg.solve(block.chol, diff))

        if use_offsets:
            chunks.append(np.array([offsets["c0_3d"] / conservative_sigmas["c0_3d"]], dtype=float))
            chunks.append(np.array([offsets["c2_3d"] / conservative_sigmas["c2_3d"]], dtype=float))
            chunks.append(np.array([offsets["c4_3d"] / conservative_sigmas["c4_3d"]], dtype=float))
            chunks.append(np.array([offsets["loop_amp"] / conservative_sigmas["loop_amp"]], dtype=float))
            chunks.append(np.array([offsets["C0_1d"] / conservative_sigmas["C0_1d"]], dtype=float))
            chunks.append(np.array([offsets["C2_1d"] / conservative_sigmas["C2_1d"]], dtype=float))
            if theory == "hybrid":
                chunks.append(np.array([offsets["b_t"] / conservative_sigmas["b_t"]], dtype=float))
                chunks.append(np.array([offsets["sigma_th"] / conservative_sigmas["sigma_th"]], dtype=float))
        return np.concatenate(chunks)

    x0 = np.clip(x0, lo + 1.0e-8, hi - 1.0e-8)
    n_starts = max(1, int(args.n_starts))
    seed_offset = (0 if theory == "one_loop" else 10000) + (0 if mode == "informative" else 1000)
    rng = np.random.default_rng(int(args.seed) + seed_offset)

    starts = [x0]
    if n_starts > 1:
        trend = x0.copy()
        trend[1 : 1 + nz] = np.linspace(-0.30, -0.60, nz, dtype=float)
        trend[1 + nz : 1 + 2 * nz] = np.linspace(-0.20, -0.55, nz, dtype=float)
        starts.append(np.clip(trend, lo + 1.0e-8, hi - 1.0e-8))
    if n_starts > 2:
        middle = 0.5 * (lo + hi)
        middle[0] = sigma8_ref
        starts.append(np.clip(middle, lo + 1.0e-8, hi - 1.0e-8))
    while len(starts) < n_starts:
        rand = rng.uniform(lo, hi)
        rand[0] = np.clip(rng.normal(loc=sigma8_ref, scale=0.08), lo[0] + 1.0e-8, hi[0] - 1.0e-8)
        starts.append(np.clip(rand, lo + 1.0e-8, hi - 1.0e-8))

    fit = None
    best_obj = float("inf")
    best_idx = -1
    start_objectives: list[float] = []
    start_nfev: list[int] = []
    for i, start in enumerate(starts):
        fit_i = least_squares(
            residual,
            x0=start,
            bounds=(lo, hi),
            method="trf",
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=int(args.max_nfev),
        )
        obj_i = float(np.sum(fit_i.fun**2))
        start_objectives.append(obj_i)
        start_nfev.append(int(fit_i.nfev))
        if obj_i < best_obj:
            best_obj = obj_i
            best_idx = i
            fit = fit_i

    if fit is None:
        raise RuntimeError("Least-squares multi-start did not produce a valid fit.")
    theta = fit.x.copy()
    sigma8, b1_list, beta_list, offsets = unpack(theta)

    # Data-only chi2 for reporting.
    data_res = []
    pred_by_z: dict[float, np.ndarray] = {}
    for i, block in enumerate(sdss_blocks):
        pred = p1d_prediction(
            theory=theory,
            model=models_by_z[block.z],
            z=float(block.z),
            kpar_hmpc=block.k_hmpc,
            b1=b1_list[i],
            b_eta=beta_list[i],
            sigma8=sigma8,
            sigma8_ref=sigma8_ref,
            prior3d=prior3d,
            c0_1d_rel=c0_rel,
            c2_1d_rel=c2_rel,
            apply_paper_systematics=bool(args.apply_paper_systematics),
            kmax_proj=float(args.kmax_proj),
            nint_proj=int(args.nint_proj),
            cfg=cfg,
            offsets=offsets,
        )
        pred_by_z[block.z] = pred
        data_res.append(np.linalg.solve(block.chol, pred - block.p_hmpc))

    data_res_vec = np.concatenate(data_res)
    data_chi2 = float(np.sum(data_res_vec**2))
    n_data = int(data_res_vec.size)
    dof = max(n_data - n_dim, 1)

    # Approximate covariance from Jacobian.
    sigma8_std = float("nan")
    try:
        jtj = fit.jac.T @ fit.jac
        cov = np.linalg.pinv(jtj) * max(data_chi2 / dof, 1.0)
        if cov[0, 0] > 0:
            sigma8_std = float(np.sqrt(cov[0, 0]))
    except Exception:
        sigma8_std = float("nan")

    return {
        "mode": mode,
        "param_names": names,
        "best_theta": theta.tolist(),
        "sigma8": float(sigma8),
        "sigma8_std_approx": sigma8_std,
        "chi2": data_chi2,
        "chi2_dof": float(data_chi2 / dof),
        "n_data": n_data,
        "n_dim": n_dim,
        "bias_label": bias_label,
        "bias_linear_by_z": {f"{z:.1f}": float(b1_list[i]) for i, z in enumerate(zvals)},
        "b1_by_z": {f"{z:.1f}": float(b1_list[i]) for i, z in enumerate(zvals)},
        "b_delta_by_z": (
            {f"{z:.1f}": float(b1_list[i]) for i, z in enumerate(zvals)} if theory == "hybrid" else {}
        ),
        "b_eta_by_z": {f"{z:.1f}": float(beta_list[i]) for i, z in enumerate(zvals)},
        "offsets": {k: float(v) for k, v in offsets.items()},
        "pred_by_z": {f"{z:.1f}": pred_by_z[z].tolist() for z in zvals},
        "multistart": {
            "n_starts": int(n_starts),
            "seed": int(args.seed) + seed_offset,
            "best_start_index": int(best_idx),
            "best_objective": float(best_obj),
            "start_objectives": [float(x) for x in start_objectives],
            "start_nfev": [int(x) for x in start_nfev],
        },
    }


def make_fit_plot(
    *,
    sdss_blocks: list[FitBlock],
    pred_by_mode: dict[str, dict[float, np.ndarray]],
    theory: Literal["one_loop", "hybrid"],
    out_path: Path,
) -> None:
    nz = len(sdss_blocks)
    fig, axes = plt.subplots(nz, 1, figsize=(9, 3.0 * nz), sharex=False)
    if nz == 1:
        axes = [axes]

    for i, block in enumerate(sdss_blocks):
        ax = axes[i]
        err = np.sqrt(np.diag(block.cov_hmpc))
        ax.errorbar(block.k_hmpc, block.p_hmpc, yerr=err, fmt="o", ms=3, color="k", alpha=0.8, label="SDSS DR14")
        for mode, preds in pred_by_mode.items():
            color = "#1f77b4" if mode == "informative" else "#d62728"
            ax.plot(block.k_hmpc, preds[block.z], lw=2.0, color=color, label=mode if i == 0 else None)
        ax.set_xscale("log")
        ax.set_ylabel(r"$P_{1D}\ [h^{-1}\mathrm{Mpc}]$")
        ax.set_title(f"z = {block.z:.1f}")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel(r"$k_\parallel\ [h\,\mathrm{Mpc}^{-1}]$")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    title_model = "One-loop toy" if theory == "one_loop" else "Hybrid source+LOS toy"
    fig.suptitle(f"SDSS DR14 1D Fits: 2405 Stage-1 Baseline ({title_model})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_sigma8_plot(results: list[dict[str, object]], out_path: Path) -> None:
    labels = [r["mode"] for r in results]
    means = [float(r["sigma8"]) for r in results]
    errs = [float(r["sigma8_std_approx"]) if np.isfinite(float(r["sigma8_std_approx"])) else 0.0 for r in results]

    x = np.arange(len(labels))
    plt.figure(figsize=(7, 4.8))
    plt.errorbar(x, means, yerr=errs, fmt="o", ms=8, lw=1.8, capsize=4)
    plt.xticks(x, labels)
    plt.ylabel(r"$\sigma_8$")
    plt.title(r"$\sigma_8$ Comparison: Informative vs Conservative Priors")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    _ = np.random.default_rng(args.seed).integers(0, 10)

    prior_csv = latest_stage1_prior_csv(args.theory) if args.prior_csv is None else args.prior_csv
    prior3d = load_prior_relations(prior_csv, args.theory)

    run_paths = init_run_dir(cfg.run.output_root, tag=f"repro_2405_stage1_sdss_{args.theory}")
    meta = build_repro_metadata(args.config)
    meta.update(
        {
            "theory": str(args.theory),
            "prior_csv": str(prior_csv),
            "sdss_dir": str(args.sdss_dir),
            "mock_dir": str(args.mock_dir),
            "z_min": float(args.z_min),
            "z_max": float(args.z_max),
            "kmin_fit_hmpc": float(args.kmin_fit_hmpc),
            "kmax_fit_hmpc": float(args.kmax_fit_hmpc),
            "kmax_proj": float(args.kmax_proj),
            "nint_proj": int(args.nint_proj),
            "mode": args.mode,
            "conservative_inflate": float(args.conservative_inflate),
            "counterterm_mode": str(args.counterterm_mode),
            "apply_paper_systematics": bool(args.apply_paper_systematics),
            "b_eta_min": float(args.b_eta_min),
            "b_eta_max": float(args.b_eta_max),
            "n_starts": int(args.n_starts),
            "max_nfev": int(args.max_nfev),
            "seed": int(args.seed),
        }
    )
    write_json(run_paths.logs_dir / "repro_metadata.json", meta)

    # Load SDSS DR14 blocks and optionally eBOSS mock blocks for the proxy C0/C2 calibration.
    sdss_raw = load_chabanier2019_blocks(
        data_dir=args.sdss_dir,
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        h=cfg.cosmology.h,
        omega_b=cfg.cosmology.omega_b,
        omega_cdm=cfg.cosmology.omega_cdm,
        include_syst=True,
    )
    mock_raw: list[P1DBlock] = []
    if args.counterterm_mode == "proxy":
        mock_raw = load_eboss_mock_blocks(
            data_dir=args.mock_dir,
            z_min=float(args.z_min),
            z_max=float(args.z_max),
            h=cfg.cosmology.h,
            omega_b=cfg.cosmology.omega_b,
            omega_cdm=cfg.cosmology.omega_cdm,
        )

    if not sdss_raw:
        raise ValueError(f"No SDSS blocks found in z range [{args.z_min}, {args.z_max}]")
    if args.counterterm_mode == "proxy" and not mock_raw:
        raise ValueError(f"No mock blocks found in z range [{args.z_min}, {args.z_max}]")

    sdss_blocks = build_fit_blocks(
        sdss_raw, kmin_fit_hmpc=float(args.kmin_fit_hmpc), kmax_fit_hmpc=float(args.kmax_fit_hmpc)
    )
    mock_blocks = (
        build_fit_blocks(mock_raw, kmin_fit_hmpc=float(args.kmin_fit_hmpc), kmax_fit_hmpc=float(args.kmax_fit_hmpc))
        if args.counterterm_mode == "proxy"
        else []
    )

    zvals = sorted({b.z for b in sdss_blocks})
    models_by_z: dict[float, IvanovToyModel | HybridToyModel] = {}
    sigma8_vals = []
    for z in zvals:
        lp = compute_linear_power_camb(
            h=cfg.cosmology.h,
            omega_b=cfg.cosmology.omega_b,
            omega_cdm=cfg.cosmology.omega_cdm,
            ns=cfg.cosmology.ns,
            As=cfg.cosmology.As,
            z=float(z),
            kmin=cfg.k_grid.kmin,
            kmax=max(cfg.k_grid.kmax, float(args.kmax_fit_hmpc)),
            nk=cfg.k_grid.nk,
        )
        if args.theory == "one_loop":
            models_by_z[z] = IvanovToyModel(lp.k_hmpc, lp.p_lin_h3mpc3, lp.f_growth)
        else:
            models_by_z[z] = HybridToyModel(lp.k_hmpc, lp.p_lin_h3mpc3, lp.f_growth, k_t=cfg.hybrid_toy.k_t)
        if lp.sigma8_0 is not None:
            sigma8_vals.append(float(lp.sigma8_0))
    sigma8_ref = float(np.median(sigma8_vals)) if sigma8_vals else 0.83

    if args.counterterm_mode == "proxy":
        c0_rel, c2_rel, mock_rows, mock_fit_rows = fit_mock_counterterm_relations(
            theory=args.theory,
            mock_blocks=mock_blocks,
            models_by_z=models_by_z,
            sigma8_ref=sigma8_ref,
            prior3d=prior3d,
            cfg=cfg,
            args=args,
        )
        counterterm_label = "Mock-based eBOSS proxy for C0/C2"
    else:
        c0_rel, c2_rel, mock_rows, mock_fit_rows = published_counterterm_relations(args.counterterm_mode)
        counterterm_label = PAPER_COUNTERTERM_RELATIONS[args.counterterm_mode]["label"]

    # Conservative prior widths.
    conservative_sigmas = {
        "c0_3d": float(args.conservative_inflate) * max(prior3d["c0_3d"]["rmse"], 1.0e-4),
        "c2_3d": float(args.conservative_inflate) * max(prior3d["c2_3d"]["rmse"], 1.0e-4),
        "c4_3d": float(args.conservative_inflate) * max(prior3d["c4_3d"]["rmse"], 1.0e-4),
        "loop_amp": float(args.conservative_inflate) * max(prior3d["loop_amp"]["rmse"], 1.0e-4),
        "C0_1d": (
            PAPER_C0_SIGMA
            if args.counterterm_mode != "proxy"
            else float(args.conservative_inflate) * max(mock_fit_rows[0]["fit_rmse"], 1.0e-4)
        ),
        "C2_1d": (
            PAPER_C2_SIGMA
            if args.counterterm_mode != "proxy"
            else float(args.conservative_inflate) * max(mock_fit_rows[1]["fit_rmse"], 1.0e-4)
        ),
    }
    if args.theory == "hybrid":
        conservative_sigmas["b_t"] = float(args.conservative_inflate) * max(prior3d["b_t"]["rmse"], 1.0e-4)
        conservative_sigmas["sigma_th"] = float(args.conservative_inflate) * max(prior3d["sigma_th"]["rmse"], 1.0e-4)

    modes = [args.mode] if args.mode in {"informative", "conservative"} else ["conservative", "informative"]
    fit_results = []
    pred_by_mode: dict[str, dict[float, np.ndarray]] = {}
    linear_start_hint: np.ndarray | None = None
    for mode in modes:
        res = run_sdss_fit(
            theory=args.theory,
            mode=mode,
            sdss_blocks=sdss_blocks,
            models_by_z=models_by_z,
            sigma8_ref=sigma8_ref,
            prior3d=prior3d,
            c0_rel=c0_rel,
            c2_rel=c2_rel,
            conservative_sigmas=conservative_sigmas,
            linear_start_hint=linear_start_hint,
            cfg=cfg,
            args=args,
        )
        fit_results.append(res)
        pred_by_mode[mode] = {float(k): np.asarray(v, dtype=float) for k, v in res["pred_by_z"].items()}
        linear_start_hint = np.asarray(res["best_theta"][: 1 + 2 * len(sdss_blocks)], dtype=float)

    # Save arrays.
    np.savez(
        run_paths.arrays_dir / "sdss_stage1_fit_arrays.npz",
        z=np.array([b.z for b in sdss_blocks], dtype=float),
        k_concat=np.concatenate([b.k_hmpc for b in sdss_blocks]),
        p_data_concat=np.concatenate([b.p_hmpc for b in sdss_blocks]),
        pred_informative=(
            np.concatenate([pred_by_mode["informative"][b.z] for b in sdss_blocks]) if "informative" in pred_by_mode else np.array([])
        ),
        pred_conservative=(
            np.concatenate([pred_by_mode["conservative"][b.z] for b in sdss_blocks]) if "conservative" in pred_by_mode else np.array([])
        ),
    )

    # Plots.
    f_fit = run_paths.figures_dir / "71_sdss_p1d_fit_multiz.png"
    make_fit_plot(sdss_blocks=sdss_blocks, pred_by_mode=pred_by_mode, theory=args.theory, out_path=f_fit)

    f_s8 = run_paths.figures_dir / "72_sdss_sigma8_comparison.png"
    make_sigma8_plot(fit_results, f_s8)

    # Tables.
    summary_rows = []
    for r in fit_results:
        summary_rows.append(
            {
                "model_name": f"{args.theory}_2405_stage1_{args.counterterm_mode}",
                "k_max": float(args.kmax_fit_hmpc),
                "chi2": float(r["chi2"]),
                "sigma8_mean": float(r["sigma8"]),
                "sigma8_std": float(r["sigma8_std_approx"]),
                "prior_variant": r["mode"],
                "notes": (
                    f"SDSS DR14 + {counterterm_label} ({args.theory}); "
                    f"paper_systematics={bool(args.apply_paper_systematics)}"
                ),
            }
        )
    write_csv(
        run_paths.logs_dir / "sdss_comparison_summary.csv",
        summary_rows,
        fieldnames=["model_name", "k_max", "chi2", "sigma8_mean", "sigma8_std", "prior_variant", "notes"],
    )
    write_csv(
        run_paths.logs_dir / "lace_counterterm_fits_proxy.csv",
        mock_fit_rows,
        fieldnames=["operator_name", "A_O", "B_O", "fit_rmse", "n_points"],
    )
    write_csv(
        run_paths.logs_dir / "lace_counterterm_points_proxy.csv",
        mock_rows,
        fieldnames=["z", "b1", "b_eta", "C0_1d_fit", "C2_1d_fit", "chi2_dof"],
    )

    # Copy key outputs to shared dirs.
    fig_target = Path("results/figures")
    fig_target.mkdir(parents=True, exist_ok=True)
    for fp in [f_fit, f_s8]:
        shutil.copy2(fp, fig_target / f"{fp.stem}_{args.theory}{fp.suffix}")

    table_target = Path("results/tables")
    table_target.mkdir(parents=True, exist_ok=True)
    for fp in [
        run_paths.logs_dir / "sdss_comparison_summary.csv",
        run_paths.logs_dir / "lace_counterterm_fits_proxy.csv",
        run_paths.logs_dir / "lace_counterterm_points_proxy.csv",
    ]:
        shutil.copy2(fp, table_target / f"{fp.stem}_{args.theory}{fp.suffix}")

    summary = {
        "run_dir": str(run_paths.run_dir),
        "theory": str(args.theory),
        "prior_csv": str(prior_csv),
        "sdss_z_bins": [float(b.z) for b in sdss_blocks],
        "sigma8_ref": sigma8_ref,
        "counterterm_mode": str(args.counterterm_mode),
        "counterterm_label": counterterm_label,
        "apply_paper_systematics": bool(args.apply_paper_systematics),
        "c0_rel_proxy": {"A": c0_rel[0], "B": c0_rel[1]},
        "c2_rel_proxy": {"A": c2_rel[0], "B": c2_rel[1]},
        "conservative_sigmas": conservative_sigmas,
        "fit_results": fit_results,
        "figures": [str(f_fit), str(f_s8)],
        "tables": [
            str(run_paths.logs_dir / "sdss_comparison_summary.csv"),
            str(run_paths.logs_dir / "lace_counterterm_fits_proxy.csv"),
            str(run_paths.logs_dir / "lace_counterterm_points_proxy.csv"),
        ],
        "notes": [
            "SDSS DR14 P1D data/covariance loaded from cup1d Chabanier2019 public files.",
            (
                "LaCE calibration step implemented as mock-based proxy using cup1d eBOSS_mock files."
                if args.counterterm_mode == "proxy"
                else f"Using published paper counterterm calibration: {counterterm_label}."
            ),
            (
                "Paper-like multiplicative SiIII + thermal-broadening systematics are enabled."
                if args.apply_paper_systematics
                else "Paper-like multiplicative SiIII + thermal-broadening systematics are disabled."
            ),
            "This remains a reduced one-loop toy basis rather than the full Ivanov 11-parameter correlator.",
            f"Theory choice for this run: {args.theory}.",
        ],
    }
    write_json(run_paths.logs_dir / "summary.json", summary)

    print(f"2405 Stage-1 SDSS baseline ({args.theory}) complete: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
