#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import binned_statistic

from lya_hybrid.config import load_config
from lya_hybrid.io import load_sherwood_flux_p1d, load_sherwood_flux_p3d
from lya_hybrid.linear_power import compute_linear_power_camb
from lya_hybrid.logging_utils import build_repro_metadata, init_run_dir, write_json
from lya_hybrid.model_ivanov_full import IvanovFullModel, IvanovFullParams
from lya_hybrid.projection_1d import Polynomial1DCounterterms, project_to_1d

SNAP_TO_Z = {8: 3.2, 9: 2.8, 10: 2.4, 11: 2.0}
DEFAULT_P3D_DIR = Path("data/external/sherwood_p3d/data/flux_p3d")
DEFAULT_P1D_DIR = Path("data/external/sherwood_p3d/data/flux_p1d")

PUBLISHED_RELATIONS = {
    "b_G2": (0.154, -0.252),
    "b_delta2": (0.061, -0.480),
    "b_eta2": (-2.84, 0.11),
    "b_delta_eta": (4.31, -0.0745),
    "b_KK_par": (1.55, 0.205),
    "b_Pi2_par": (2.48, 0.011),
    "b_Pi3_par": (-3.08, 1.86),
    "b_delta_Pi2_par": (20.7, 1.34),
    "b_KPi2_par": (5.83, -1.99),
    "b_eta_Pi2_par": (1.60, 1.07),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Single-redshift Sherwood reproduction run with the fuller Ivanov one-loop basis. "
            "Fits P3D directly, then projects to P1D and refits C0/C2."
        )
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--snapshot", type=int, default=9)
    p.add_argument("--p3d-dir", type=Path, default=DEFAULT_P3D_DIR)
    p.add_argument("--p1d-dir", type=Path, default=DEFAULT_P1D_DIR)
    p.add_argument("--kmax-fit", type=float, default=3.0)
    p.add_argument("--kpar-max-fit", type=float, default=2.5)
    p.add_argument("--kmax-proj", type=float, default=3.0)
    p.add_argument("--nint-proj", type=int, default=500)
    p.add_argument("--qmax-loop", type=float, default=8.0)
    p.add_argument("--nq-loop", type=int, default=24)
    p.add_argument("--nmuq-loop", type=int, default=10)
    p.add_argument("--nphi-loop", type=int, default=10)
    p.add_argument("--stage-a-max-nfev", type=int, default=60)
    p.add_argument("--stage-b-max-nfev", type=int, default=120)
    p.add_argument("--seed", type=int, default=20260313)
    return p.parse_args()


def choose_p3d_file(p3d_dir: Path, snapshot: int) -> Path:
    preferred = p3d_dir / f"p3d_80_1024_{snapshot}_0_512_1024_20_16_20.fits"
    if preferred.exists():
        return preferred
    candidates = sorted(p3d_dir.glob(f"p3d_*_{snapshot}_0_*_20_16_20.fits"))
    if not candidates:
        raise FileNotFoundError(f"No P3D file found for snapshot {snapshot} in {p3d_dir}")
    return candidates[0]


def choose_p1d_file(p1d_dir: Path, snapshot: int) -> Path:
    preferred = p1d_dir / f"p1d_80_1024_{snapshot}_0_512_1024.fits"
    if preferred.exists():
        return preferred
    candidates = sorted(p1d_dir.glob(f"p1d_*_{snapshot}_0_*_*.fits"))
    if not candidates:
        raise FileNotFoundError(f"No P1D file found for snapshot {snapshot} in {p1d_dir}")
    return candidates[0]


def pseudo_sigma(p: np.ndarray, counts: np.ndarray, sigma_frac: float, sigma_floor: float) -> np.ndarray:
    return sigma_frac * np.maximum(np.abs(p), 1.0e-8) + np.maximum(1.0 / np.sqrt(np.maximum(counts, 1.0)), sigma_floor)


def binned_curve(k: np.ndarray, y: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yb, edges, _ = binned_statistic(k, y, statistic="median", bins=bins)
    xc = np.sqrt(edges[:-1] * edges[1:])
    keep = np.isfinite(yb)
    return xc[keep], yb[keep]


def published_guess(b1: float, b_eta: float) -> IvanovFullParams:
    vals = {name: slope * b1 + intercept for name, (slope, intercept) in PUBLISHED_RELATIONS.items()}
    return IvanovFullParams(b1=b1, b_eta=b_eta, **vals)


def params_to_theta(p: IvanovFullParams) -> np.ndarray:
    return np.array(
        [
            p.b1,
            p.b_eta,
            p.b_delta2,
            p.b_G2,
            p.b_KK_par,
            p.b_delta_eta,
            p.b_eta2,
            p.b_Pi2_par,
            p.b_Pi3_par,
            p.b_delta_Pi2_par,
            p.b_KPi2_par,
            p.b_eta_Pi2_par,
        ],
        dtype=float,
    )


def theta_to_params(theta: np.ndarray) -> IvanovFullParams:
    return IvanovFullParams(
        b1=float(theta[0]),
        b_eta=float(theta[1]),
        b_delta2=float(theta[2]),
        b_G2=float(theta[3]),
        b_KK_par=float(theta[4]),
        b_delta_eta=float(theta[5]),
        b_eta2=float(theta[6]),
        b_Pi2_par=float(theta[7]),
        b_Pi3_par=float(theta[8]),
        b_delta_Pi2_par=float(theta[9]),
        b_KPi2_par=float(theta[10]),
        b_eta_Pi2_par=float(theta[11]),
        b_gamma3=0.0,
    )


def make_p3d_residual_plot(
    *,
    k_all: np.ndarray,
    mu_all: np.ndarray,
    p_data: np.ndarray,
    p_model: np.ndarray,
    z: float,
    out_path: Path,
) -> None:
    mu_bins = np.array([0.0, 0.25, 0.5, 0.75, 1.01])
    k_bins = np.logspace(np.log10(0.04), np.log10(3.0), 14)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for i in range(4):
        lo, hi = mu_bins[i], mu_bins[i + 1]
        mask = (mu_all >= lo) & (mu_all < hi) & (k_all >= 0.03) & (k_all <= 3.0)
        ax = axes[i]
        if np.count_nonzero(mask) < 8:
            ax.set_visible(False)
            continue
        kx, rr = binned_curve(k_all[mask], p_model[mask] / p_data[mask] - 1.0, bins=k_bins)
        ax.semilogx(kx, rr, marker="o", lw=1.6)
        ax.axhline(0.0, color="k", lw=1)
        ax.grid(alpha=0.25)
        ax.set_title(fr"$\mu \in [{lo:.2f},{hi:.2f})$")
        ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
        ax.set_ylim(-0.35, 0.35)
    axes[0].set_ylabel(r"$P_{\rm model}/P_{\rm data}-1$")
    fig.suptitle(fr"Full Ivanov Sherwood Residuals ($z={z:.1f}$)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_p1d_projection_plot(
    *,
    kp: np.ndarray,
    p_data: np.ndarray,
    p_raw: np.ndarray,
    p_fit: np.ndarray,
    z: float,
    out_path: Path,
) -> None:
    plt.figure(figsize=(7.5, 5))
    plt.loglog(kp, np.abs(p_data), label="Sherwood P1D")
    plt.loglog(kp, np.abs(p_raw), ls="--", label="Projected raw")
    plt.loglog(kp, np.abs(p_fit), ls="-.", label="Projected + C0 + C2 k^2")
    plt.xlabel(r"$k_\parallel\ [h\,{\rm Mpc}^{-1}]$")
    plt.ylabel(r"$|P_{1D}|$")
    plt.title(fr"Full-basis 1D Projection Check ($z={z:.1f}$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.snapshot not in SNAP_TO_Z:
        raise ValueError(f"Unsupported snapshot {args.snapshot}; supported: {sorted(SNAP_TO_Z)}")

    z = SNAP_TO_Z[args.snapshot]
    p3d_path = choose_p3d_file(args.p3d_dir, args.snapshot)
    p1d_path = choose_p1d_file(args.p1d_dir, args.snapshot)
    p3d = load_sherwood_flux_p3d(p3d_path)
    p1d = load_sherwood_flux_p1d(p1d_path)

    k_all, mu_all, p_all, counts_all = p3d.flatten_valid()
    mask_3d = (k_all >= cfg.fit.kmin_fit) & (k_all <= float(args.kmax_fit))
    k_fit = k_all[mask_3d]
    mu_fit = mu_all[mask_3d]
    p_fit = p_all[mask_3d]
    counts_fit = counts_all[mask_3d]
    sigma_3d = pseudo_sigma(p_fit, counts_fit, sigma_frac=cfg.fit.sigma_frac, sigma_floor=cfg.fit.sigma_floor)

    lp = compute_linear_power_camb(
        h=cfg.cosmology.h,
        omega_b=cfg.cosmology.omega_b,
        omega_cdm=cfg.cosmology.omega_cdm,
        ns=cfg.cosmology.ns,
        As=cfg.cosmology.As,
        z=float(z),
        kmin=cfg.k_grid.kmin,
        kmax=max(cfg.k_grid.kmax, float(args.qmax_loop) + float(args.kmax_fit)),
        nk=max(cfg.k_grid.nk, 1200),
    )
    model = IvanovFullModel(
        lp.k_hmpc,
        lp.p_lin_h3mpc3,
        lp.f_growth,
        qmin=float(cfg.k_grid.kmin),
        qmax=float(args.qmax_loop),
        nq=int(args.nq_loop),
        nmuq=int(args.nmuq_loop),
        nphi=int(args.nphi_loop),
    )

    run_paths = init_run_dir(cfg.run.output_root, tag="ivanov_full_sherwood_singlez")
    meta = build_repro_metadata(args.config)
    meta.update(
        {
            "snapshot": int(args.snapshot),
            "redshift": float(z),
            "p3d_file": str(p3d_path),
            "p1d_file": str(p1d_path),
            "kmax_fit": float(args.kmax_fit),
            "kpar_max_fit": float(args.kpar_max_fit),
            "kmax_proj": float(args.kmax_proj),
            "nint_proj": int(args.nint_proj),
            "qmax_loop": float(args.qmax_loop),
            "nq_loop": int(args.nq_loop),
            "nmuq_loop": int(args.nmuq_loop),
            "nphi_loop": int(args.nphi_loop),
            "stage_a_max_nfev": int(args.stage_a_max_nfev),
            "stage_b_max_nfev": int(args.stage_b_max_nfev),
            "seed": int(args.seed),
        }
    )
    write_json(run_paths.logs_dir / "repro_metadata.json", meta)

    def stage_a_residual(theta: np.ndarray) -> np.ndarray:
        b1, b_eta = [float(x) for x in theta]
        params = published_guess(b1=b1, b_eta=b_eta)
        pred = model.evaluate_components(k_fit, mu_fit, params)["total"]
        return (pred - p_fit) / sigma_3d

    x0_a = np.array([cfg.ivanov_toy.b1, cfg.ivanov_toy.b_eta], dtype=float)
    lo_a = np.array([-1.2, -2.5], dtype=float)
    hi_a = np.array([0.05, 1.0], dtype=float)
    t0 = time.perf_counter()
    fit_a = least_squares(
        stage_a_residual,
        x0=np.clip(x0_a, lo_a + 1.0e-8, hi_a - 1.0e-8),
        bounds=(lo_a, hi_a),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(args.stage_a_max_nfev),
    )
    params_a = published_guess(b1=float(fit_a.x[0]), b_eta=float(fit_a.x[1]))

    x0_b = params_to_theta(params_a)
    lo_b = x0_b.copy()
    hi_b = x0_b.copy()
    lo_b[0], hi_b[0] = -1.2, 0.05
    lo_b[1], hi_b[1] = -2.5, 1.0
    spreads = np.array([0.0, 0.0, 2.5, 2.5, 2.5, 6.0, 4.0, 6.0, 8.0, 20.0, 8.0, 8.0], dtype=float)
    lo_b[2:] = x0_b[2:] - spreads[2:]
    hi_b[2:] = x0_b[2:] + spreads[2:]

    def stage_b_residual(theta: np.ndarray) -> np.ndarray:
        params = theta_to_params(theta)
        pred = model.evaluate_components(k_fit, mu_fit, params)["total"]
        return (pred - p_fit) / sigma_3d

    fit_b = least_squares(
        stage_b_residual,
        x0=np.clip(x0_b, lo_b + 1.0e-8, hi_b - 1.0e-8),
        bounds=(lo_b, hi_b),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(args.stage_b_max_nfev),
    )
    runtime = time.perf_counter() - t0
    params_b = theta_to_params(fit_b.x)

    pred_fit_full = model.evaluate_components(k_fit, mu_fit, params_b)["total"]
    chi2_3d = float(np.sum(((pred_fit_full - p_fit) / sigma_3d) ** 2))
    dof_3d = max(int(k_fit.size - fit_b.x.size), 1)

    kp_all = p1d.kp_hmpc[p1d.valid_mask()]
    p1d_all = p1d.p1d_hmpc[p1d.valid_mask()]
    mask_1d = (kp_all >= 0.03) & (kp_all <= float(args.kpar_max_fit))
    kp_fit = kp_all[mask_1d]
    p1d_fit = p1d_all[mask_1d]

    raw1d = project_to_1d(
        kpar_values=kp_fit,
        p3d_callable=lambda kk, mm: model.evaluate_components(kk, mm, params_b)["total"],
        kmax_proj=float(args.kmax_proj),
        nint=int(args.nint_proj),
        method="trapz",
        counterterms=Polynomial1DCounterterms(),
    )["raw"]

    sigma_1d = 0.05 * np.maximum(np.abs(p1d_fit), 1.0e-8) + 0.02

    def residual_1d(theta: np.ndarray) -> np.ndarray:
        c0, c2 = [float(x) for x in theta]
        pred = raw1d + c0 + c2 * kp_fit**2
        return (pred - p1d_fit) / sigma_1d

    fit_1d = least_squares(
        residual_1d,
        x0=np.array([0.0, 0.0], dtype=float),
        bounds=(np.array([-4.0, -4.0]), np.array([4.0, 4.0])),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=8000,
    )
    c0_1d, c2_1d = [float(x) for x in fit_1d.x]
    pred1d = raw1d + c0_1d + c2_1d * kp_fit**2
    chi2_1d = float(np.sum(((pred1d - p1d_fit) / sigma_1d) ** 2))
    dof_1d = max(int(kp_fit.size - 2), 1)

    pred_all = model.evaluate_components(k_all, mu_all, params_b)["total"]
    f_p3d = run_paths.figures_dir / "51_full_ivanov_p3d_residuals.png"
    f_p1d = run_paths.figures_dir / "52_full_ivanov_p1d_projection.png"
    make_p3d_residual_plot(
        k_all=k_all,
        mu_all=mu_all,
        p_data=p_all,
        p_model=pred_all,
        z=z,
        out_path=f_p3d,
    )
    make_p1d_projection_plot(
        kp=kp_fit,
        p_data=p1d_fit,
        p_raw=raw1d,
        p_fit=pred1d,
        z=z,
        out_path=f_p1d,
    )

    np.savez(
        run_paths.arrays_dir / "full_ivanov_singlez_arrays.npz",
        k_all=k_all,
        mu_all=mu_all,
        p3d_data=p_all,
        p3d_pred=pred_all,
        kp_fit=kp_fit,
        p1d_data=p1d_fit,
        p1d_raw=raw1d,
        p1d_fit=pred1d,
        theta_stage_a=fit_a.x,
        theta_stage_b=fit_b.x,
    )

    fig_target = Path("results/figures")
    fig_target.mkdir(parents=True, exist_ok=True)
    for fp in [f_p3d, f_p1d]:
        shutil.copy2(fp, fig_target / fp.name)

    summary = {
        "run_dir": str(run_paths.run_dir),
        "snapshot": int(args.snapshot),
        "redshift": float(z),
        "runtime_sec": float(runtime),
        "kmax_fit": float(args.kmax_fit),
        "qmax_loop": float(args.qmax_loop),
        "integration_grid": {
            "nq": int(args.nq_loop),
            "nmuq": int(args.nmuq_loop),
            "nphi": int(args.nphi_loop),
        },
        "stage_a": {
            "best_theta": fit_a.x.tolist(),
            "params": asdict(params_a),
            "chi2_dof": float(np.sum(stage_a_residual(fit_a.x) ** 2) / max(k_fit.size - 2, 1)),
            "nfev": int(fit_a.nfev),
        },
        "stage_b": {
            "best_theta": fit_b.x.tolist(),
            "params": asdict(params_b),
            "chi2": chi2_3d,
            "chi2_dof": float(chi2_3d / dof_3d),
            "n_data": int(k_fit.size),
            "n_dim": int(fit_b.x.size),
            "nfev": int(fit_b.nfev),
        },
        "projection_1d": {
            "C0_1d": c0_1d,
            "C2_1d": c2_1d,
            "chi2": chi2_1d,
            "chi2_dof": float(chi2_1d / dof_1d),
        },
        "figures": [str(f_p3d), str(f_p1d)],
        "notes": [
            "This is the first dedicated fuller-basis Ivanov reproduction path in the repo.",
            "b_gamma3 is fixed to zero following the supplement discussion of its degeneracy with b_G2.",
            "The stage-A warm start uses the published Table III linear relations only as an initialization manifold.",
        ],
    }
    write_json(run_paths.logs_dir / "summary.json", summary)
    print(f"Full Ivanov Sherwood single-z run complete: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
