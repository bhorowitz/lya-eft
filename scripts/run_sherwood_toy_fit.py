#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
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


DEFAULT_P3D = Path("data/external/sherwood_p3d/data/flux_p3d/p3d_160_2048_9_0_1024_2048_20_16_20.fits")
DEFAULT_P1D = Path("data/external/sherwood_p3d/data/flux_p1d/p1d_160_2048_9_0_1024_2048.fits")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a first Sherwood z=2.8 toy EFT fit and diagnostics.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--p3d", type=Path, default=DEFAULT_P3D)
    p.add_argument("--p1d", type=Path, default=DEFAULT_P1D)
    p.add_argument("--kmax-list", type=str, default="1.0,1.5,2.0,2.5,3.0,4.0,5.0")
    p.add_argument("--kmax-proj", type=float, default=6.0)
    p.add_argument("--kpar-max", type=float, default=4.0)
    return p.parse_args()


def fit_vector_to_params(vec: np.ndarray, cfg_params) -> IvanovToyParams:
    return IvanovToyParams(
        b1=float(vec[0]),
        b_eta=float(vec[1]),
        c0=float(vec[2]),
        c2=float(vec[3]),
        c4=float(vec[4]),
        loop_amp=float(vec[5]),
        loop_mu2=float(cfg_params.loop_mu2),
        loop_mu4=float(cfg_params.loop_mu4),
        loop_k_nl=float(cfg_params.loop_k_nl),
        stochastic=float(cfg_params.stochastic),
    )


def binned_residual_curve(k: np.ndarray, residual: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med, edges, _ = binned_statistic(k, residual, statistic="median", bins=bins)
    xc = np.sqrt(edges[:-1] * edges[1:])
    return xc, med


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    kmax_values = [float(x) for x in args.kmax_list.split(",")]

    run_paths = init_run_dir(cfg.run.output_root, tag="sherwood_toy_fit")
    meta = build_repro_metadata(args.config)
    meta.update({"p3d_path": str(args.p3d), "p1d_path": str(args.p1d)})
    write_json(run_paths.logs_dir / "repro_metadata.json", meta)

    p3d_data = load_sherwood_flux_p3d(args.p3d)
    p1d_data = load_sherwood_flux_p1d(args.p1d)

    k_all, mu_all, p_all, counts_all = p3d_data.flatten_valid()

    lp = compute_linear_power_camb(
        h=cfg.cosmology.h,
        omega_b=cfg.cosmology.omega_b,
        omega_cdm=cfg.cosmology.omega_cdm,
        ns=cfg.cosmology.ns,
        As=cfg.cosmology.As,
        z=cfg.cosmology.z,
        kmin=cfg.k_grid.kmin,
        kmax=max(cfg.k_grid.kmax, args.kmax_proj),
        nk=cfg.k_grid.nk,
    )

    model = IvanovToyModel(lp.k_hmpc, lp.p_lin_h3mpc3, lp.f_growth)

    x0 = np.array(
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

    bounds = (
        np.array([-1.0, -4.0, -8.0, -8.0, -8.0, -4.0]),
        np.array([0.2, 4.0, 8.0, 8.0, 8.0, 4.0]),
    )

    fit_rows: list[dict[str, float]] = []
    best_params_by_kmax: dict[float, np.ndarray] = {}

    for kmax in kmax_values:
        m = (k_all <= kmax) & (k_all > 0.03)
        kf = k_all[m]
        muf = mu_all[m]
        pf = p_all[m]
        cf = counts_all[m]

        # Pseudo-errors for first-pass diagnostics only.
        sigma = 0.05 * np.maximum(np.abs(pf), 1e-8) + np.maximum(1.0 / np.sqrt(cf), 0.02)

        def residuals(x: np.ndarray) -> np.ndarray:
            params = fit_vector_to_params(x, cfg.ivanov_toy)
            pred = model.evaluate_components(kf, muf, params)["total"]
            return (pred - pf) / sigma

        fit = least_squares(
            residuals,
            x0=x0,
            bounds=bounds,
            method="trf",
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=10000,
        )

        x0 = fit.x.copy()
        best_params_by_kmax[kmax] = fit.x.copy()
        chi2 = float(np.sum(fit.fun**2))
        dof = max(int(fit.fun.size - fit.x.size), 1)
        fit_rows.append(
            {
                "kmax": kmax,
                "npts": int(fit.fun.size),
                "chi2": chi2,
                "chi2_dof": chi2 / dof,
                "b1": float(fit.x[0]),
                "b_eta": float(fit.x[1]),
                "c0": float(fit.x[2]),
                "c2": float(fit.x[3]),
                "c4": float(fit.x[4]),
                "loop_amp": float(fit.x[5]),
                "success": bool(fit.success),
            }
        )

    kmax_ref = min(kmax_values, key=lambda x: abs(x - 3.0))
    params_ref = fit_vector_to_params(best_params_by_kmax[kmax_ref], cfg.ivanov_toy)

    # Residual map for the reference fit.
    pred_all = model.evaluate_components(k_all, mu_all, params_ref)["total"]
    resid_all = pred_all / p_all - 1.0

    # 1D comparison
    m1d = p1d_data.valid_mask() & (p1d_data.kp_hmpc > 0.03) & (p1d_data.kp_hmpc <= args.kpar_max)
    kp = p1d_data.kp_hmpc[m1d]
    p1d_target = p1d_data.p1d_hmpc[m1d]

    def p3d_callable(kvals: np.ndarray, muvals: np.ndarray) -> np.ndarray:
        return model.evaluate_components(kvals, muvals, params_ref)["total"]

    p1d_raw = project_to_1d(
        kpar_values=kp,
        p3d_callable=p3d_callable,
        kmax_proj=args.kmax_proj,
        nint=1400,
        method="trapz",
        counterterms=Polynomial1DCounterterms(),
    )["raw"]

    A = np.column_stack([np.ones_like(kp), kp**2, kp**4])
    coeff, *_ = np.linalg.lstsq(A, p1d_target - p1d_raw, rcond=None)
    p1d_poly = p1d_raw + A @ coeff

    # Save arrays.
    np.savez(
        run_paths.arrays_dir / "sherwood_toy_fit_arrays.npz",
        k_all=k_all,
        mu_all=mu_all,
        p_all=p_all,
        counts_all=counts_all,
        pred_all=pred_all,
        residual_all=resid_all,
        kmax_values=np.array(kmax_values),
        kp=kp,
        p1d_data=p1d_target,
        p1d_raw=p1d_raw,
        p1d_poly=p1d_poly,
        poly_coeff=coeff,
    )

    # Plot 1: data coverage and k-mu points.
    plt.figure(figsize=(8, 5))
    sc = plt.scatter(k_all, mu_all, c=np.log10(np.maximum(counts_all, 1.0)), s=12, cmap="viridis")
    plt.xscale("log")
    cbar = plt.colorbar(sc)
    cbar.set_label(r"$\log_{10}(\mathrm{counts})$")
    plt.xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    plt.ylabel(r"$\mu$")
    plt.title("Sherwood z=2.8 P3D sampling points")
    plt.tight_layout()
    f1 = run_paths.figures_dir / "21_sherwood_sampling.png"
    plt.savefig(f1, dpi=160)
    plt.close()

    # Plot 2: chi2/dof vs kmax.
    plt.figure(figsize=(8, 5))
    plt.plot([r["kmax"] for r in fit_rows], [r["chi2_dof"] for r in fit_rows], marker="o")
    plt.xlabel(r"$k_{\max}\ [h\,{\rm Mpc}^{-1}]$")
    plt.ylabel(r"$\chi^2/\mathrm{dof}$ (pseudo)")
    plt.title("Toy-model scale-cut trend on Sherwood P3D")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    f2 = run_paths.figures_dir / "22_toy_chi2_vs_kmax.png"
    plt.savefig(f2, dpi=160)
    plt.close()

    # Plot 3: fitted parameters vs kmax.
    keys = ["b1", "b_eta", "c0", "c2", "c4", "loop_amp"]
    labels = [r"$b_1$", r"$b_\eta$", r"$c_0$", r"$c_2$", r"$c_4$", r"$A_{loop}$"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True)
    x = [r["kmax"] for r in fit_rows]
    for ax, key, lab in zip(axes.ravel(), keys, labels):
        ax.plot(x, [r[key] for r in fit_rows], marker="o")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25)
    for ax in axes[1, :]:
        ax.set_xlabel(r"$k_{\max}\ [h\,{\rm Mpc}^{-1}]$")
    fig.suptitle("Toy parameter drift under sliding 3D scale cuts")
    fig.tight_layout()
    f3 = run_paths.figures_dir / "23_toy_params_vs_kmax.png"
    fig.savefig(f3, dpi=160)
    plt.close(fig)

    # Plot 4: residuals by mu bins for reference kmax.
    mu_bins = np.array([0.0, 0.25, 0.5, 0.75, 1.01])
    k_bins = np.logspace(np.log10(0.04), np.log10(5.5), 16)
    plt.figure(figsize=(9, 6))
    for i in range(mu_bins.size - 1):
        mm = (mu_all >= mu_bins[i]) & (mu_all < mu_bins[i + 1])
        if np.count_nonzero(mm) < 8:
            continue
        xc, med = binned_residual_curve(k_all[mm], resid_all[mm], bins=k_bins)
        plt.semilogx(xc, med, marker="o", label=fr"$\mu \in [{mu_bins[i]:.2f},{mu_bins[i+1]:.2f})$")
    plt.axhline(0.0, color="k", lw=1)
    plt.xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    plt.ylabel(r"$P_{\rm toy}/P_{\rm data}-1$")
    plt.title(fr"Sherwood residual medians at reference fit $k_{{\max}}={kmax_ref:.1f}$")
    plt.legend(fontsize=9)
    plt.tight_layout()
    f4 = run_paths.figures_dir / "24_toy_residuals_by_mu.png"
    plt.savefig(f4, dpi=160)
    plt.close()

    # Plot 5: 1D projection comparison.
    plt.figure(figsize=(8, 5))
    plt.loglog(kp, p1d_target, label="Sherwood P1D")
    plt.loglog(kp, np.abs(p1d_raw), label="Projected from 3D toy fit", ls="--")
    plt.loglog(kp, np.abs(p1d_poly), label="Projected + C0+C2k^2+C4k^4", ls="-.")
    plt.xlabel(r"$k_\parallel\ [h\,{\rm Mpc}^{-1}]$")
    plt.ylabel(r"$P_{1D}$")
    plt.title("3D-fit projection to 1D and polynomial correction")
    plt.legend()
    plt.tight_layout()
    f5 = run_paths.figures_dir / "25_toy_1d_projection_vs_data.png"
    plt.savefig(f5, dpi=160)
    plt.close()

    fig_target = Path("results/figures")
    fig_target.mkdir(parents=True, exist_ok=True)
    for fp in [f1, f2, f3, f4, f5]:
        shutil.copy2(fp, fig_target / fp.name)

    summary = {
        "run_dir": str(run_paths.run_dir),
        "kmax_values": kmax_values,
        "fit_rows": fit_rows,
        "reference_kmax": float(kmax_ref),
        "poly_coeff_c0_c2_c4": [float(x) for x in coeff],
        "p1d_raw_median_frac_abs_error": float(np.median(np.abs(p1d_raw / p1d_target - 1.0))),
        "p1d_poly_median_frac_abs_error": float(np.median(np.abs(p1d_poly / p1d_target - 1.0))),
    }
    write_json(run_paths.logs_dir / "summary.json", summary)

    print(f"Sherwood toy fit run complete: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
