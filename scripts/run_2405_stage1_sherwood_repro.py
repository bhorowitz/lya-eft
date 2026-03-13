#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
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
from lya_hybrid.model_ivanov import IvanovToyModel, IvanovToyParams
from lya_hybrid.projection_1d import Polynomial1DCounterterms, project_to_1d

SNAP_TO_Z = {8: 3.2, 9: 2.8, 10: 2.4, 11: 2.0}
DEFAULT_P3D_DIR = Path("data/external/sherwood_p3d/data/flux_p3d")
DEFAULT_P1D_DIR = Path("data/external/sherwood_p3d/data/flux_p1d")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage-1 reproduction step for 2405.13208 using public Sherwood data: "
            "fit one-loop P3D in z bins, project to P1D, and extract linear b_O(b1) prior relations."
        )
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--p3d-dir", type=Path, default=DEFAULT_P3D_DIR)
    p.add_argument("--p1d-dir", type=Path, default=DEFAULT_P1D_DIR)
    p.add_argument("--snapshots", type=str, default="11,10,9,8")
    p.add_argument("--kmax-fit", type=float, default=3.0)
    p.add_argument("--kmax-proj", type=float, default=3.0)
    p.add_argument("--kpar-max-fit", type=float, default=2.5)
    p.add_argument("--nint-proj", type=int, default=1200)
    p.add_argument("--seed", type=int, default=20260313)
    return p.parse_args()


def pseudo_sigma(p: np.ndarray, counts: np.ndarray, sigma_frac: float, sigma_floor: float) -> np.ndarray:
    return sigma_frac * np.maximum(np.abs(p), 1.0e-8) + np.maximum(1.0 / np.sqrt(np.maximum(counts, 1.0)), sigma_floor)


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


def binned_curve(k: np.ndarray, y: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yb, edges, _ = binned_statistic(k, y, statistic="median", bins=bins)
    xc = np.sqrt(edges[:-1] * edges[1:])
    keep = np.isfinite(yb)
    return xc[keep], yb[keep]


def fit_p3d_one_loop(
    *,
    model: IvanovToyModel,
    k: np.ndarray,
    mu: np.ndarray,
    p_data: np.ndarray,
    counts: np.ndarray,
    cfg,
    kmax_fit: float,
    x0: np.ndarray,
) -> tuple[IvanovToyParams, np.ndarray, dict[str, float]]:
    mask = (k >= cfg.fit.kmin_fit) & (k <= kmax_fit)
    kf, muf, pf, cf = k[mask], mu[mask], p_data[mask], counts[mask]
    sigma = pseudo_sigma(pf, cf, sigma_frac=cfg.fit.sigma_frac, sigma_floor=cfg.fit.sigma_floor)

    lo = np.array([-1.0, -2.0, -0.2, -0.2, -0.2, -0.3], dtype=float)
    hi = np.array([0.2, 2.0, 0.2, 0.2, 0.2, 0.3], dtype=float)

    def unpack(theta: np.ndarray) -> IvanovToyParams:
        return IvanovToyParams(
            b1=float(theta[0]),
            b_eta=float(theta[1]),
            c0=float(theta[2]),
            c2=float(theta[3]),
            c4=float(theta[4]),
            loop_amp=float(theta[5]),
            loop_mu2=float(cfg.ivanov_toy.loop_mu2),
            loop_mu4=float(cfg.ivanov_toy.loop_mu4),
            loop_k_nl=float(cfg.ivanov_toy.loop_k_nl),
            stochastic=float(cfg.ivanov_toy.stochastic),
        )

    def residual(theta: np.ndarray) -> np.ndarray:
        pred = model.evaluate_components(kf, muf, unpack(theta))["total"]
        return (pred - pf) / sigma

    x0 = np.clip(x0, lo + 1.0e-8, hi - 1.0e-8)
    fit = least_squares(
        residual,
        x0=x0,
        bounds=(lo, hi),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=15000,
    )

    params = unpack(fit.x)
    pred = model.evaluate_components(kf, muf, params)["total"]
    chi2 = float(np.sum(((pred - pf) / sigma) ** 2))
    dof = max(int(kf.size - fit.x.size), 1)

    out = {
        "chi2": chi2,
        "chi2_dof": float(chi2 / dof),
        "n_data": int(kf.size),
    }
    return params, fit.x.copy(), out


def fit_p1d_counterterms(
    *,
    model: IvanovToyModel,
    params: IvanovToyParams,
    p1d_k: np.ndarray,
    p1d_data: np.ndarray,
    kmax_proj: float,
    kpar_max_fit: float,
    nint_proj: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    mask = (p1d_k >= 0.03) & (p1d_k <= kpar_max_fit)
    kp = p1d_k[mask]
    pd = p1d_data[mask]

    def p3d_callable(kvals: np.ndarray, muvals: np.ndarray) -> np.ndarray:
        return model.evaluate_components(kvals, muvals, params)["total"]

    proj = project_to_1d(
        kpar_values=kp,
        p3d_callable=p3d_callable,
        kmax_proj=kmax_proj,
        nint=nint_proj,
        method="trapz",
        counterterms=Polynomial1DCounterterms(),
    )
    raw = proj["raw"]

    sigma = 0.05 * np.maximum(np.abs(pd), 1.0e-8) + 0.02

    def residual(theta: np.ndarray) -> np.ndarray:
        c0, c2 = float(theta[0]), float(theta[1])
        pred = raw + c0 + c2 * kp**2
        return (pred - pd) / sigma

    fit = least_squares(
        residual,
        x0=np.array([0.0, 0.0], dtype=float),
        bounds=(np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=8000,
    )
    c0_fit, c2_fit = [float(x) for x in fit.x]
    pred_fit = raw + c0_fit + c2_fit * kp**2
    chi2 = float(np.sum(((pred_fit - pd) / sigma) ** 2))
    dof = max(int(kp.size - 2), 1)
    return kp, pred_fit, c0_fit, c2_fit, float(chi2 / dof)


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
    fig.suptitle(fr"2405 Stage-1 Sherwood Fit Residuals ($z={z:.1f}$)")
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
    plt.title(fr"1D Projection Check ($z={z:.1f}$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def fit_linear_relation(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    A = np.column_stack([x, np.ones_like(x)])
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(coeff[0])
    intercept = float(coeff[1])
    pred = slope * x + intercept
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return slope, intercept, rmse


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_linear_prior_plot(
    *,
    b1: np.ndarray,
    y_by_name: dict[str, np.ndarray],
    fit_rows: list[dict[str, object]],
    out_path: Path,
) -> None:
    names = list(y_by_name.keys())
    n = len(names)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    xx = np.linspace(np.min(b1) - 0.03, np.max(b1) + 0.03, 120)

    for i, name in enumerate(names):
        ax = axes[i]
        yy = y_by_name[name]
        row = next(r for r in fit_rows if r["operator_name"] == name)
        slope = float(row["A_O"])
        intercept = float(row["B_O"])
        ax.scatter(b1, yy, s=45, color="#1f77b4")
        ax.plot(xx, slope * xx + intercept, color="#d62728", lw=2)
        ax.set_xlabel(r"$b_1$")
        ax.set_ylabel(name)
        ax.grid(alpha=0.25)
        ax.set_title(f"{name} (RMSE={row['fit_rmse']:.3g})")

    for j in range(n, axes.size):
        axes[j].set_visible(False)

    fig.suptitle("Stage-1 Linear Prior Manifold Fits: b_O(b1) = A_O b1 + B_O")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    snapshots = [int(x) for x in args.snapshots.split(",") if x.strip()]
    unknown = [s for s in snapshots if s not in SNAP_TO_Z]
    if unknown:
        raise ValueError(f"Unsupported snapshots {unknown}; supported: {sorted(SNAP_TO_Z)}")

    run_paths = init_run_dir(cfg.run.output_root, tag="repro_2405_stage1_sherwood")
    meta = build_repro_metadata(args.config)
    meta.update(
        {
            "p3d_dir": str(args.p3d_dir),
            "p1d_dir": str(args.p1d_dir),
            "snapshots": snapshots,
            "kmax_fit": float(args.kmax_fit),
            "kmax_proj": float(args.kmax_proj),
            "kpar_max_fit": float(args.kpar_max_fit),
            "nint_proj": int(args.nint_proj),
            "seed": int(args.seed),
        }
    )
    write_json(run_paths.logs_dir / "repro_metadata.json", meta)

    rng = np.random.default_rng(args.seed)
    _ = rng.integers(0, 10)

    rows_z: list[dict[str, object]] = []
    b1_values = []
    op_data: dict[str, list[float]] = {
        "b_eta": [],
        "c0_3d": [],
        "c2_3d": [],
        "c4_3d": [],
        "loop_amp": [],
        "C0_1d": [],
        "C2_1d": [],
    }

    warm = np.array(
        [
            cfg.ivanov_toy.b1,
            cfg.ivanov_toy.b_eta,
            cfg.ivanov_toy.c0,
            cfg.ivanov_toy.c2,
            cfg.ivanov_toy.c4,
            cfg.ivanov_toy.loop_amp,
        ],
        dtype=float,
    )

    for snapshot in snapshots:
        z = SNAP_TO_Z[snapshot]
        p3d_path = choose_p3d_file(args.p3d_dir, snapshot)
        p1d_path = choose_p1d_file(args.p1d_dir, snapshot)

        p3d = load_sherwood_flux_p3d(p3d_path)
        p1d = load_sherwood_flux_p1d(p1d_path)
        k_all, mu_all, p_all, counts_all = p3d.flatten_valid()

        lp = compute_linear_power_camb(
            h=cfg.cosmology.h,
            omega_b=cfg.cosmology.omega_b,
            omega_cdm=cfg.cosmology.omega_cdm,
            ns=cfg.cosmology.ns,
            As=cfg.cosmology.As,
            z=float(z),
            kmin=cfg.k_grid.kmin,
            kmax=max(cfg.k_grid.kmax, args.kmax_fit),
            nk=cfg.k_grid.nk,
        )
        model = IvanovToyModel(lp.k_hmpc, lp.p_lin_h3mpc3, lp.f_growth)

        params, warm, p3d_diag = fit_p3d_one_loop(
            model=model,
            k=k_all,
            mu=mu_all,
            p_data=p_all,
            counts=counts_all,
            cfg=cfg,
            kmax_fit=float(args.kmax_fit),
            x0=warm,
        )

        pred_all = model.evaluate_components(k_all, mu_all, params)["total"]

        kp_valid = p1d.kp_hmpc[p1d.valid_mask()]
        p1d_valid = p1d.p1d_hmpc[p1d.valid_mask()]
        kp_fit, p1d_fit, c0_1d, c2_1d, p1d_chi2_dof = fit_p1d_counterterms(
            model=model,
            params=params,
            p1d_k=kp_valid,
            p1d_data=p1d_valid,
            kmax_proj=float(args.kmax_proj),
            kpar_max_fit=float(args.kpar_max_fit),
            nint_proj=int(args.nint_proj),
        )

        # Recompute raw curve for plotting.
        proj_raw = project_to_1d(
            kpar_values=kp_fit,
            p3d_callable=lambda kk, mm: model.evaluate_components(kk, mm, params)["total"],
            kmax_proj=float(args.kmax_proj),
            nint=int(args.nint_proj),
            method="trapz",
            counterterms=Polynomial1DCounterterms(),
        )["raw"]

        z_tag = f"z{z:.1f}".replace(".", "p")
        f_p3d = run_paths.figures_dir / f"51_{z_tag}_p3d_residuals.png"
        f_p1d = run_paths.figures_dir / f"52_{z_tag}_p1d_projection.png"
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
            p_data=p1d_valid[(kp_valid >= 0.03) & (kp_valid <= float(args.kpar_max_fit))],
            p_raw=proj_raw,
            p_fit=p1d_fit,
            z=z,
            out_path=f_p1d,
        )

        rows_z.append(
            {
                "snapshot": int(snapshot),
                "redshift": float(z),
                "p3d_file": str(p3d_path),
                "p1d_file": str(p1d_path),
                "kmax_fit": float(args.kmax_fit),
                "kmax_proj": float(args.kmax_proj),
                "b1": float(params.b1),
                "b_eta": float(params.b_eta),
                "c0_3d": float(params.c0),
                "c2_3d": float(params.c2),
                "c4_3d": float(params.c4),
                "loop_amp": float(params.loop_amp),
                "C0_1d": float(c0_1d),
                "C2_1d": float(c2_1d),
                "p3d_chi2_dof": float(p3d_diag["chi2_dof"]),
                "p3d_n_data": int(p3d_diag["n_data"]),
                "p1d_chi2_dof": float(p1d_chi2_dof),
                "params_dict": asdict(params),
                "figure_p3d": str(f_p3d),
                "figure_p1d": str(f_p1d),
            }
        )

        b1_values.append(float(params.b1))
        op_data["b_eta"].append(float(params.b_eta))
        op_data["c0_3d"].append(float(params.c0))
        op_data["c2_3d"].append(float(params.c2))
        op_data["c4_3d"].append(float(params.c4))
        op_data["loop_amp"].append(float(params.loop_amp))
        op_data["C0_1d"].append(float(c0_1d))
        op_data["C2_1d"].append(float(c2_1d))

        np.savez(
            run_paths.arrays_dir / f"{z_tag}_fit_arrays.npz",
            k_all=k_all,
            mu_all=mu_all,
            p3d_data=p_all,
            p3d_pred=pred_all,
            kp_fit=kp_fit,
            p1d_data=p1d_valid[(kp_valid >= 0.03) & (kp_valid <= float(args.kpar_max_fit))],
            p1d_raw=proj_raw,
            p1d_fit=p1d_fit,
        )

    b1 = np.asarray(b1_values, dtype=float)
    y_by_name = {k: np.asarray(v, dtype=float) for k, v in op_data.items()}

    prior_rows: list[dict[str, object]] = []
    for name, yy in y_by_name.items():
        slope, intercept, rmse = fit_linear_relation(b1, yy)
        prior_rows.append(
            {
                "operator_name": name,
                "A_O": slope,
                "B_O": intercept,
                "fit_rmse": rmse,
                "n_points": int(b1.size),
                "fit_variant": "sherwood_public_stage1_repro",
            }
        )

    f_prior = run_paths.figures_dir / "61_linear_prior_relations.png"
    make_linear_prior_plot(
        b1=b1,
        y_by_name=y_by_name,
        fit_rows=prior_rows,
        out_path=f_prior,
    )

    write_csv(
        run_paths.logs_dir / "z_bin_fit_summary.csv",
        rows_z,
        fieldnames=[
            "snapshot",
            "redshift",
            "p3d_file",
            "p1d_file",
            "kmax_fit",
            "kmax_proj",
            "b1",
            "b_eta",
            "c0_3d",
            "c2_3d",
            "c4_3d",
            "loop_amp",
            "C0_1d",
            "C2_1d",
            "p3d_chi2_dof",
            "p3d_n_data",
            "p1d_chi2_dof",
            "params_dict",
            "figure_p3d",
            "figure_p1d",
        ],
    )
    write_csv(
        run_paths.logs_dir / "sherwood_prior_linear_fits.csv",
        prior_rows,
        fieldnames=["operator_name", "A_O", "B_O", "fit_rmse", "n_points", "fit_variant"],
    )

    fig_target = Path("results/figures")
    fig_target.mkdir(parents=True, exist_ok=True)
    table_target = Path("results/tables")
    table_target.mkdir(parents=True, exist_ok=True)
    for row in rows_z:
        shutil.copy2(Path(row["figure_p3d"]), fig_target / Path(row["figure_p3d"]).name)
        shutil.copy2(Path(row["figure_p1d"]), fig_target / Path(row["figure_p1d"]).name)
    shutil.copy2(f_prior, fig_target / f_prior.name)
    shutil.copy2(run_paths.logs_dir / "z_bin_fit_summary.csv", table_target / "z_bin_fit_summary.csv")
    shutil.copy2(run_paths.logs_dir / "sherwood_prior_linear_fits.csv", table_target / "sherwood_prior_linear_fits.csv")

    summary = {
        "run_dir": str(run_paths.run_dir),
        "snapshots": snapshots,
        "redshifts": [float(SNAP_TO_Z[s]) for s in snapshots],
        "kmax_fit": float(args.kmax_fit),
        "kmax_proj": float(args.kmax_proj),
        "rows_z": rows_z,
        "linear_prior_fits": prior_rows,
        "figures": [str(f_prior)] + [str(r["figure_p3d"]) for r in rows_z] + [str(r["figure_p1d"]) for r in rows_z],
        "tables": [
            str(run_paths.logs_dir / "z_bin_fit_summary.csv"),
            str(run_paths.logs_dir / "sherwood_prior_linear_fits.csv"),
        ],
        "notes": [
            "This reproduces the public Sherwood prior-extraction stage with the current reduced one-loop toy basis.",
            "Full 2405 SDSS+LaCE likelihood reproduction requires additional SDSS DR14 data vectors/covariances and LaCE mock products.",
        ],
    }
    write_json(run_paths.logs_dir / "summary.json", summary)

    print(f"2405 Stage-1 Sherwood reproduction complete: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
