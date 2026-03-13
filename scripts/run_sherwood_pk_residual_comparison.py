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
from lya_hybrid.model_hybrid import HybridToyModel, HybridToyParams
from lya_hybrid.model_ivanov import IvanovToyModel, IvanovToyParams
from lya_hybrid.projection_1d import project_to_1d


DEFAULT_P3D = Path("data/external/sherwood_p3d/data/flux_p3d/p3d_160_2048_9_0_1024_2048_20_16_20.fits")
DEFAULT_P1D = Path("data/external/sherwood_p3d/data/flux_p1d/p1d_160_2048_9_0_1024_2048.fits")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare original/tree, one-loop, and hybrid toy models on Sherwood P(k,mu)."
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--p3d", type=Path, default=DEFAULT_P3D)
    p.add_argument("--p1d", type=Path, default=DEFAULT_P1D)
    p.add_argument("--kmax-fit", type=float, default=None)
    p.add_argument("--kmax-proj", type=float, default=6.0)
    return p.parse_args()


def _pseudo_sigma(p: np.ndarray, counts: np.ndarray, sigma_frac: float, sigma_floor: float) -> np.ndarray:
    return sigma_frac * np.maximum(np.abs(p), 1.0e-8) + np.maximum(1.0 / np.sqrt(np.maximum(counts, 1.0)), sigma_floor)


def _chi2_dof(residual_vec: np.ndarray, n_params: int) -> tuple[float, float]:
    chi2 = float(np.sum(residual_vec**2))
    dof = max(int(residual_vec.size - n_params), 1)
    return chi2, chi2 / dof


def _binned_curve(k: np.ndarray, y: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yb, edges, _ = binned_statistic(k, y, statistic="median", bins=bins)
    xc = np.sqrt(edges[:-1] * edges[1:])
    keep = np.isfinite(yb)
    return xc[keep], yb[keep]


def _make_pk_residual_figure(
    *,
    k: np.ndarray,
    mu: np.ndarray,
    p_data: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    label_a: str,
    label_b: str,
    title: str,
    out_path: Path,
    kmin_plot: float = 0.04,
    kmax_plot: float = 5.5,
) -> None:
    mu_bins = np.array([0.0, 0.25, 0.5, 0.75, 1.01])
    k_bins = np.logspace(np.log10(kmin_plot), np.log10(kmax_plot), 16)

    fig, axes = plt.subplots(
        2,
        mu_bins.size - 1,
        figsize=(17, 7),
        sharex="col",
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.05, "wspace": 0.18},
    )

    for i in range(mu_bins.size - 1):
        lo, hi = mu_bins[i], mu_bins[i + 1]
        m = (mu >= lo) & (mu < hi) & (k >= kmin_plot) & (k <= kmax_plot)

        ax_top = axes[0, i]
        ax_bot = axes[1, i]

        if np.count_nonzero(m) < 5:
            ax_top.set_visible(False)
            ax_bot.set_visible(False)
            continue

        kx, pd = _binned_curve(k[m], p_data[m], bins=k_bins)
        _, pa = _binned_curve(k[m], p_a[m], bins=k_bins)
        _, pb = _binned_curve(k[m], p_b[m], bins=k_bins)

        _, ra = _binned_curve(k[m], p_a[m] / p_data[m] - 1.0, bins=k_bins)
        _, rb = _binned_curve(k[m], p_b[m] / p_data[m] - 1.0, bins=k_bins)

        if kx.size > 0:
            ax_top.loglog(kx, np.abs(pd), marker="o", color="black", lw=1.2, ms=4, label="Sherwood")
            ax_top.loglog(kx, np.abs(pa), color="#1f77b4", lw=2.0, label=label_a)
            ax_top.loglog(kx, np.abs(pb), color="#d62728", lw=2.0, label=label_b)

            ax_bot.semilogx(kx, ra, color="#1f77b4", lw=1.8)
            ax_bot.semilogx(kx, rb, color="#d62728", lw=1.8)

        ax_top.set_title(fr"$\mu \in [{lo:.2f},{hi:.2f})$")
        ax_top.grid(alpha=0.25)
        ax_bot.grid(alpha=0.25)
        ax_bot.axhline(0.0, color="k", lw=1)

        ax_bot.set_ylim(-0.7, 0.7)
        if i == 0:
            ax_top.set_ylabel(r"$|P_F(k,\mu)|$")
            ax_bot.set_ylabel(r"$P_{model}/P_{data}-1$")
        ax_bot.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _fit_poly_1d(kp: np.ndarray, p_target: np.ndarray, p_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.column_stack([np.ones_like(kp), kp**2, kp**4])
    coeff, *_ = np.linalg.lstsq(A, p_target - p_raw, rcond=None)
    return coeff, p_raw + A @ coeff


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    kmax_fit = float(cfg.fit.kmax_fit if args.kmax_fit is None else args.kmax_fit)

    run_paths = init_run_dir(cfg.run.output_root, tag="sherwood_pk_residual_comparison")
    meta = build_repro_metadata(args.config)
    meta.update({"p3d_path": str(args.p3d), "p1d_path": str(args.p1d), "kmax_fit": kmax_fit})
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

    one_loop_model = IvanovToyModel(lp.k_hmpc, lp.p_lin_h3mpc3, lp.f_growth)
    hybrid_model = HybridToyModel(lp.k_hmpc, lp.p_lin_h3mpc3, lp.f_growth, k_t=cfg.hybrid_toy.k_t)

    mask_fit = (k_all >= cfg.fit.kmin_fit) & (k_all <= kmax_fit)
    kf = k_all[mask_fit]
    muf = mu_all[mask_fit]
    pf = p_all[mask_fit]
    cf = counts_all[mask_fit]

    sigma = _pseudo_sigma(pf, cf, sigma_frac=cfg.fit.sigma_frac, sigma_floor=cfg.fit.sigma_floor)

    # Original/tree fit.
    x0_tree = np.array([cfg.ivanov_toy.b1, cfg.ivanov_toy.b_eta], dtype=float)

    def residual_tree(x: np.ndarray) -> np.ndarray:
        params = IvanovToyParams(
            b1=float(x[0]),
            b_eta=float(x[1]),
            c0=0.0,
            c2=0.0,
            c4=0.0,
            loop_amp=0.0,
            loop_mu2=float(cfg.ivanov_toy.loop_mu2),
            loop_mu4=float(cfg.ivanov_toy.loop_mu4),
            loop_k_nl=float(cfg.ivanov_toy.loop_k_nl),
            stochastic=0.0,
        )
        pred = one_loop_model.evaluate_components(kf, muf, params)["total"]
        return (pred - pf) / sigma

    fit_tree = least_squares(
        residual_tree,
        x0=x0_tree,
        bounds=(np.array([-1.2, -5.0]), np.array([0.5, 5.0])),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=10000,
    )

    params_tree = IvanovToyParams(
        b1=float(fit_tree.x[0]),
        b_eta=float(fit_tree.x[1]),
        c0=0.0,
        c2=0.0,
        c4=0.0,
        loop_amp=0.0,
        loop_mu2=float(cfg.ivanov_toy.loop_mu2),
        loop_mu4=float(cfg.ivanov_toy.loop_mu4),
        loop_k_nl=float(cfg.ivanov_toy.loop_k_nl),
        stochastic=0.0,
    )

    # One-loop fit.
    x0_1l = np.array(
        [
            params_tree.b1,
            params_tree.b_eta,
            cfg.ivanov_toy.c0,
            cfg.ivanov_toy.c2,
            cfg.ivanov_toy.c4,
            cfg.ivanov_toy.loop_amp,
        ],
        dtype=float,
    )

    def residual_oneloop(x: np.ndarray) -> np.ndarray:
        params = IvanovToyParams(
            b1=float(x[0]),
            b_eta=float(x[1]),
            c0=float(x[2]),
            c2=float(x[3]),
            c4=float(x[4]),
            loop_amp=float(x[5]),
            loop_mu2=float(cfg.ivanov_toy.loop_mu2),
            loop_mu4=float(cfg.ivanov_toy.loop_mu4),
            loop_k_nl=float(cfg.ivanov_toy.loop_k_nl),
            stochastic=float(cfg.ivanov_toy.stochastic),
        )
        pred = one_loop_model.evaluate_components(kf, muf, params)["total"]
        return (pred - pf) / sigma

    fit_1l = least_squares(
        residual_oneloop,
        x0=x0_1l,
        bounds=(
            np.array([-1.0, -4.0, -8.0, -8.0, -8.0, -4.0]),
            np.array([0.3, 4.0, 8.0, 8.0, 8.0, 4.0]),
        ),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=10000,
    )

    params_1l = IvanovToyParams(
        b1=float(fit_1l.x[0]),
        b_eta=float(fit_1l.x[1]),
        c0=float(fit_1l.x[2]),
        c2=float(fit_1l.x[3]),
        c4=float(fit_1l.x[4]),
        loop_amp=float(fit_1l.x[5]),
        loop_mu2=float(cfg.ivanov_toy.loop_mu2),
        loop_mu4=float(cfg.ivanov_toy.loop_mu4),
        loop_k_nl=float(cfg.ivanov_toy.loop_k_nl),
        stochastic=float(cfg.ivanov_toy.stochastic),
    )

    # Hybrid fit.
    x0_h = np.array(
        [
            params_1l.b1,
            params_1l.b_eta,
            cfg.hybrid_toy.b_t,
            params_1l.c0,
            params_1l.c2,
            params_1l.c4,
            params_1l.loop_amp,
            cfg.hybrid_toy.sigma_th,
        ],
        dtype=float,
    )

    def residual_hybrid(x: np.ndarray) -> np.ndarray:
        params = HybridToyParams(
            b_delta=float(x[0]),
            b_eta=float(x[1]),
            b_t=float(x[2]),
            c0=float(x[3]),
            c2=float(x[4]),
            c4=float(x[5]),
            loop_amp=float(x[6]),
            loop_mu2=float(cfg.hybrid_toy.loop_mu2),
            loop_mu4=float(cfg.hybrid_toy.loop_mu4),
            loop_k_nl=float(cfg.hybrid_toy.loop_k_nl),
            sigma_th=float(x[7]),
            stochastic=float(cfg.hybrid_toy.stochastic),
        )
        pred = hybrid_model.evaluate_components(kf, muf, params)["total"]
        return (pred - pf) / sigma

    fit_h = least_squares(
        residual_hybrid,
        x0=x0_h,
        bounds=(
            np.array([-1.0, -4.0, -2.0, -8.0, -8.0, -8.0, -4.0, 0.0]),
            np.array([0.3, 4.0, 2.0, 8.0, 8.0, 8.0, 4.0, 0.35]),
        ),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=12000,
    )

    params_h = HybridToyParams(
        b_delta=float(fit_h.x[0]),
        b_eta=float(fit_h.x[1]),
        b_t=float(fit_h.x[2]),
        c0=float(fit_h.x[3]),
        c2=float(fit_h.x[4]),
        c4=float(fit_h.x[5]),
        loop_amp=float(fit_h.x[6]),
        loop_mu2=float(cfg.hybrid_toy.loop_mu2),
        loop_mu4=float(cfg.hybrid_toy.loop_mu4),
        loop_k_nl=float(cfg.hybrid_toy.loop_k_nl),
        sigma_th=float(fit_h.x[7]),
        stochastic=float(cfg.hybrid_toy.stochastic),
    )

    # Predictions over all valid data points.
    p_tree_all = one_loop_model.evaluate_components(k_all, mu_all, params_tree)["total"]
    comp_1l_all = one_loop_model.evaluate_components(k_all, mu_all, params_1l)
    p_1l_all = comp_1l_all["total"]
    comp_h_all = hybrid_model.evaluate_components(k_all, mu_all, params_h)
    p_h_all = comp_h_all["total"]

    # Fit-region chi2 values.
    chi2_tree, chi2dof_tree = _chi2_dof((one_loop_model.evaluate_components(kf, muf, params_tree)["total"] - pf) / sigma, 2)
    chi2_1l, chi2dof_1l = _chi2_dof((one_loop_model.evaluate_components(kf, muf, params_1l)["total"] - pf) / sigma, 6)
    chi2_h, chi2dof_h = _chi2_dof((hybrid_model.evaluate_components(kf, muf, params_h)["total"] - pf) / sigma, 8)

    # Requested: Pk + residual, original vs one-loop.
    f_tree_vs_1l = run_paths.figures_dir / "31_pk_residual_original_vs_oneloop.png"
    _make_pk_residual_figure(
        k=k_all,
        mu=mu_all,
        p_data=p_all,
        p_a=p_tree_all,
        p_b=p_1l_all,
        label_a="original/tree",
        label_b="one-loop",
        title=fr"Sherwood z=2.8: original vs one-loop (fit: $k_\max={kmax_fit:.1f}$)",
        out_path=f_tree_vs_1l,
    )

    # Next: one-loop vs hybrid.
    f_1l_vs_h = run_paths.figures_dir / "32_pk_residual_oneloop_vs_hybrid.png"
    _make_pk_residual_figure(
        k=k_all,
        mu=mu_all,
        p_data=p_all,
        p_a=p_1l_all,
        p_b=p_h_all,
        label_a="one-loop",
        label_b="hybrid (source+LOS)",
        title=fr"Sherwood z=2.8: one-loop vs hybrid (fit: $k_\max={kmax_fit:.1f}$)",
        out_path=f_1l_vs_h,
    )

    # Global pseudo-chi2 comparison.
    f_chi = run_paths.figures_dir / "33_model_pseudo_chi2.png"
    plt.figure(figsize=(7, 4.5))
    labels = ["original", "one-loop", "hybrid"]
    vals = [chi2dof_tree, chi2dof_1l, chi2dof_h]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    plt.bar(labels, vals, color=colors)
    plt.ylabel(r"$\chi^2/\mathrm{dof}$ (pseudo)")
    plt.title(fr"Fit-region comparison ($k \leq {kmax_fit:.1f}\ h\,{{\rm Mpc}}^{{-1}}$)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(f_chi, dpi=160)
    plt.close()

    # 1D projection diagnostic for all three models.
    m1d = p1d_data.valid_mask() & (p1d_data.kp_hmpc > cfg.projection_1d.kpar_min) & (
        p1d_data.kp_hmpc <= cfg.projection_1d.kpar_max
    )
    kp = p1d_data.kp_hmpc[m1d]
    p1d_target = p1d_data.p1d_hmpc[m1d]

    def p3d_tree(kvals: np.ndarray, muvals: np.ndarray) -> np.ndarray:
        return one_loop_model.evaluate_components(kvals, muvals, params_tree)["total"]

    def p3d_1l(kvals: np.ndarray, muvals: np.ndarray) -> np.ndarray:
        return one_loop_model.evaluate_components(kvals, muvals, params_1l)["total"]

    def p3d_hybrid(kvals: np.ndarray, muvals: np.ndarray) -> np.ndarray:
        return hybrid_model.evaluate_components(kvals, muvals, params_h)["total"]

    p1d_tree_raw = project_to_1d(kpar_values=kp, p3d_callable=p3d_tree, kmax_proj=args.kmax_proj, nint=1400)["raw"]
    p1d_1l_raw = project_to_1d(kpar_values=kp, p3d_callable=p3d_1l, kmax_proj=args.kmax_proj, nint=1400)["raw"]
    p1d_h_raw = project_to_1d(kpar_values=kp, p3d_callable=p3d_hybrid, kmax_proj=args.kmax_proj, nint=1400)["raw"]

    coeff_tree, p1d_tree_poly = _fit_poly_1d(kp, p1d_target, p1d_tree_raw)
    coeff_1l, p1d_1l_poly = _fit_poly_1d(kp, p1d_target, p1d_1l_raw)
    coeff_h, p1d_h_poly = _fit_poly_1d(kp, p1d_target, p1d_h_raw)

    f_1d = run_paths.figures_dir / "34_projection_1d_model_comparison.png"
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.06})

    ax[0].loglog(kp, p1d_target, color="black", lw=1.8, label="Sherwood P1D")
    ax[0].loglog(kp, np.abs(p1d_tree_poly), color="#1f77b4", lw=1.8, label="original + poly")
    ax[0].loglog(kp, np.abs(p1d_1l_poly), color="#d62728", lw=1.8, label="one-loop + poly")
    ax[0].loglog(kp, np.abs(p1d_h_poly), color="#2ca02c", lw=1.8, label="hybrid + poly")
    ax[0].set_ylabel(r"$P_{1D}(k_\parallel)$")
    ax[0].grid(alpha=0.25)
    ax[0].legend()

    ax[1].semilogx(kp, p1d_tree_poly / p1d_target - 1.0, color="#1f77b4", lw=1.6)
    ax[1].semilogx(kp, p1d_1l_poly / p1d_target - 1.0, color="#d62728", lw=1.6)
    ax[1].semilogx(kp, p1d_h_poly / p1d_target - 1.0, color="#2ca02c", lw=1.6)
    ax[1].axhline(0.0, color="k", lw=1)
    ax[1].set_ylim(-0.4, 0.4)
    ax[1].set_xlabel(r"$k_\parallel\ [h\,{\rm Mpc}^{-1}]$")
    ax[1].set_ylabel("ratio-1")
    ax[1].grid(alpha=0.25)

    fig.suptitle("Projected 1D comparison (after per-model C0+C2k^2+C4k^4)")
    fig.tight_layout()
    fig.savefig(f_1d, dpi=160)
    plt.close(fig)

    # Hybrid LOS kernel visualization.
    f_los = run_paths.figures_dir / "35_hybrid_los_kernel.png"
    kk = np.logspace(np.log10(0.03), np.log10(6.0), 250)
    plt.figure(figsize=(7, 4.5))
    for mu0 in [0.3, 0.6, 0.9]:
        los = np.exp(-((kk * mu0 * params_h.sigma_th) ** 2))
        plt.semilogx(kk, los, lw=2.0, label=fr"$\mu={mu0:.1f}$")
    plt.ylim(0.0, 1.05)
    plt.xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    plt.ylabel("LOS thermal kernel")
    plt.title(fr"Hybrid fitted thermal width: $\sigma_{{th}}={params_h.sigma_th:.3f}\ h^{{-1}}\,{{\rm Mpc}}$")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f_los, dpi=160)
    plt.close()

    np.savez(
        run_paths.arrays_dir / "sherwood_model_comparison_arrays.npz",
        k_all=k_all,
        mu_all=mu_all,
        p_all=p_all,
        counts_all=counts_all,
        p_tree=p_tree_all,
        p_oneloop=p_1l_all,
        p_hybrid=p_h_all,
        resid_tree=p_tree_all / p_all - 1.0,
        resid_oneloop=p_1l_all / p_all - 1.0,
        resid_hybrid=p_h_all / p_all - 1.0,
        kp=kp,
        p1d_data=p1d_target,
        p1d_tree_raw=p1d_tree_raw,
        p1d_oneloop_raw=p1d_1l_raw,
        p1d_hybrid_raw=p1d_h_raw,
        p1d_tree_poly=p1d_tree_poly,
        p1d_oneloop_poly=p1d_1l_poly,
        p1d_hybrid_poly=p1d_h_poly,
    )

    fig_target = Path("results/figures")
    fig_target.mkdir(parents=True, exist_ok=True)
    for fp in [f_tree_vs_1l, f_1l_vs_h, f_chi, f_1d, f_los]:
        shutil.copy2(fp, fig_target / fp.name)

    summary = {
        "run_dir": str(run_paths.run_dir),
        "kmax_fit": kmax_fit,
        "fit_region_npts": int(mask_fit.sum()),
        "original": {
            "params": {"b1": params_tree.b1, "b_eta": params_tree.b_eta},
            "chi2": chi2_tree,
            "chi2_dof": chi2dof_tree,
        },
        "one_loop": {
            "params": {
                "b1": params_1l.b1,
                "b_eta": params_1l.b_eta,
                "c0": params_1l.c0,
                "c2": params_1l.c2,
                "c4": params_1l.c4,
                "loop_amp": params_1l.loop_amp,
            },
            "chi2": chi2_1l,
            "chi2_dof": chi2dof_1l,
        },
        "hybrid": {
            "params": {
                "b_delta": params_h.b_delta,
                "b_eta": params_h.b_eta,
                "b_t": params_h.b_t,
                "c0": params_h.c0,
                "c2": params_h.c2,
                "c4": params_h.c4,
                "loop_amp": params_h.loop_amp,
                "sigma_th": params_h.sigma_th,
                "k_t_fixed": cfg.hybrid_toy.k_t,
            },
            "chi2": chi2_h,
            "chi2_dof": chi2dof_h,
        },
        "projection_1d_median_abs_frac_error": {
            "original_raw": float(np.median(np.abs(p1d_tree_raw / p1d_target - 1.0))),
            "one_loop_raw": float(np.median(np.abs(p1d_1l_raw / p1d_target - 1.0))),
            "hybrid_raw": float(np.median(np.abs(p1d_h_raw / p1d_target - 1.0))),
            "original_poly": float(np.median(np.abs(p1d_tree_poly / p1d_target - 1.0))),
            "one_loop_poly": float(np.median(np.abs(p1d_1l_poly / p1d_target - 1.0))),
            "hybrid_poly": float(np.median(np.abs(p1d_h_poly / p1d_target - 1.0))),
        },
        "projection_1d_poly_coefficients": {
            "original_c0_c2_c4": [float(x) for x in coeff_tree],
            "one_loop_c0_c2_c4": [float(x) for x in coeff_1l],
            "hybrid_c0_c2_c4": [float(x) for x in coeff_h],
        },
        "figures": [
            str(f_tree_vs_1l),
            str(f_1l_vs_h),
            str(f_chi),
            str(f_1d),
            str(f_los),
        ],
    }
    write_json(run_paths.logs_dir / "summary.json", summary)

    print(f"Sherwood model-comparison run complete: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
