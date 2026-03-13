#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import binned_statistic

from lya_hybrid.config import load_config
from lya_hybrid.io import load_sherwood_flux_p3d
from lya_hybrid.linear_power import compute_linear_power_camb
from lya_hybrid.logging_utils import build_repro_metadata, init_run_dir, write_json
from lya_hybrid.model_hybrid import HybridToyModel, HybridToyParams
from lya_hybrid.model_ivanov import IvanovToyModel, IvanovToyParams

DEFAULT_P3D = Path("data/external/sherwood_p3d/data/flux_p3d/p3d_160_2048_9_0_1024_2048_20_16_20.fits")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="High-kmax Sherwood scan: original vs one-loop vs hybrid.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--p3d", type=Path, default=DEFAULT_P3D)
    p.add_argument("--kmax-list", type=str, default="3.0,3.5,4.0,4.5,5.0")
    return p.parse_args()


def pseudo_sigma(p: np.ndarray, counts: np.ndarray, sigma_frac: float, sigma_floor: float) -> np.ndarray:
    return sigma_frac * np.maximum(np.abs(p), 1.0e-8) + np.maximum(1.0 / np.sqrt(np.maximum(counts, 1.0)), sigma_floor)


def data_chi2_dof(residual_vec: np.ndarray, n_params: int) -> tuple[float, float]:
    chi2 = float(np.sum(residual_vec**2))
    dof = max(int(residual_vec.size - n_params), 1)
    return chi2, chi2 / dof


def binned_curve(k: np.ndarray, y: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yb, edges, _ = binned_statistic(k, y, statistic="median", bins=bins)
    xc = np.sqrt(edges[:-1] * edges[1:])
    keep = np.isfinite(yb)
    return xc[keep], yb[keep]


def make_pk_residual_figure(
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

        kx, pd = binned_curve(k[m], p_data[m], bins=k_bins)
        _, pa = binned_curve(k[m], p_a[m], bins=k_bins)
        _, pb = binned_curve(k[m], p_b[m], bins=k_bins)
        _, ra = binned_curve(k[m], p_a[m] / p_data[m] - 1.0, bins=k_bins)
        _, rb = binned_curve(k[m], p_b[m] / p_data[m] - 1.0, bins=k_bins)

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
        ax_bot.set_ylim(-0.8, 0.8)

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


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    kmax_values = [float(x) for x in args.kmax_list.split(",")]
    if sorted(kmax_values) != kmax_values:
        raise ValueError("kmax-list must be sorted ascending.")

    run_paths = init_run_dir(cfg.run.output_root, tag="sherwood_highk_scan")
    meta = build_repro_metadata(args.config)
    meta.update({"p3d_path": str(args.p3d), "kmax_values": kmax_values})
    write_json(run_paths.logs_dir / "repro_metadata.json", meta)

    p3d_data = load_sherwood_flux_p3d(args.p3d)
    k_all, mu_all, p_all, counts_all = p3d_data.flatten_valid()

    lp = compute_linear_power_camb(
        h=cfg.cosmology.h,
        omega_b=cfg.cosmology.omega_b,
        omega_cdm=cfg.cosmology.omega_cdm,
        ns=cfg.cosmology.ns,
        As=cfg.cosmology.As,
        z=cfg.cosmology.z,
        kmin=cfg.k_grid.kmin,
        kmax=cfg.k_grid.kmax,
        nk=cfg.k_grid.nk,
    )

    one_loop_model = IvanovToyModel(lp.k_hmpc, lp.p_lin_h3mpc3, lp.f_growth)
    hybrid_model = HybridToyModel(lp.k_hmpc, lp.p_lin_h3mpc3, lp.f_growth, k_t=cfg.hybrid_toy.k_t)

    # Warm-start parameter vectors.
    x_tree = np.array([cfg.ivanov_toy.b1, cfg.ivanov_toy.b_eta], dtype=float)
    x_1l = np.array(
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
    x_h = np.array(
        [
            cfg.hybrid_toy.b_delta,
            cfg.hybrid_toy.b_eta,
            cfg.hybrid_toy.b_t,
            cfg.hybrid_toy.c0,
            cfg.hybrid_toy.c2,
            cfg.hybrid_toy.c4,
            cfg.hybrid_toy.loop_amp,
            cfg.hybrid_toy.sigma_th,
        ],
        dtype=float,
    )

    rows: list[dict[str, float]] = []
    params_at_kmax: dict[float, dict[str, object]] = {}

    tree_lo = np.array([-1.2, -2.0])
    tree_hi = np.array([0.4, 2.0])
    one_lo = np.array([-1.0, -2.0, -0.2, -0.2, -0.2, -0.3])
    one_hi = np.array([0.2, 2.0, 0.2, 0.2, 0.2, 0.3])
    hy_lo = np.array([-1.0, -2.0, -0.25, -0.10, -0.10, -0.10, -0.30, 0.0])
    hy_hi = np.array([0.2, 2.0, 0.25, 0.10, 0.10, 0.10, 0.30, 0.25])

    for kmax in kmax_values:
        mf = (k_all >= cfg.fit.kmin_fit) & (k_all <= kmax)
        kf, muf, pf, cf = k_all[mf], mu_all[mf], p_all[mf], counts_all[mf]
        sig = pseudo_sigma(pf, cf, sigma_frac=cfg.fit.sigma_frac, sigma_floor=cfg.fit.sigma_floor)

        def res_tree(x: np.ndarray) -> np.ndarray:
            p = IvanovToyParams(
                b1=float(x[0]),
                b_eta=float(x[1]),
                c0=0.0,
                c2=0.0,
                c4=0.0,
                loop_amp=0.0,
                loop_mu2=cfg.ivanov_toy.loop_mu2,
                loop_mu4=cfg.ivanov_toy.loop_mu4,
                loop_k_nl=cfg.ivanov_toy.loop_k_nl,
                stochastic=0.0,
            )
            pred = one_loop_model.evaluate_components(kf, muf, p)["total"]
            return (pred - pf) / sig

        x_tree = np.clip(x_tree, tree_lo + 1.0e-8, tree_hi - 1.0e-8)
        fit_tree = least_squares(
            res_tree,
            x0=x_tree,
            bounds=(tree_lo, tree_hi),
            method="trf",
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=12000,
        )
        x_tree = fit_tree.x.copy()
        p_tree = IvanovToyParams(
            b1=float(x_tree[0]),
            b_eta=float(x_tree[1]),
            c0=0.0,
            c2=0.0,
            c4=0.0,
            loop_amp=0.0,
            loop_mu2=cfg.ivanov_toy.loop_mu2,
            loop_mu4=cfg.ivanov_toy.loop_mu4,
            loop_k_nl=cfg.ivanov_toy.loop_k_nl,
            stochastic=0.0,
        )

        def res_1l(x: np.ndarray) -> np.ndarray:
            p = IvanovToyParams(
                b1=float(x[0]),
                b_eta=float(x[1]),
                c0=float(x[2]),
                c2=float(x[3]),
                c4=float(x[4]),
                loop_amp=float(x[5]),
                loop_mu2=cfg.ivanov_toy.loop_mu2,
                loop_mu4=cfg.ivanov_toy.loop_mu4,
                loop_k_nl=cfg.ivanov_toy.loop_k_nl,
                stochastic=cfg.ivanov_toy.stochastic,
            )
            pred = one_loop_model.evaluate_components(kf, muf, p)["total"]
            return (pred - pf) / sig

        x_1l = np.clip(x_1l, one_lo + 1.0e-8, one_hi - 1.0e-8)
        fit_1l = least_squares(
            res_1l,
            x0=x_1l,
            bounds=(one_lo, one_hi),
            method="trf",
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=12000,
        )
        x_1l = fit_1l.x.copy()
        p_1l = IvanovToyParams(
            b1=float(x_1l[0]),
            b_eta=float(x_1l[1]),
            c0=float(x_1l[2]),
            c2=float(x_1l[3]),
            c4=float(x_1l[4]),
            loop_amp=float(x_1l[5]),
            loop_mu2=cfg.ivanov_toy.loop_mu2,
            loop_mu4=cfg.ivanov_toy.loop_mu4,
            loop_k_nl=cfg.ivanov_toy.loop_k_nl,
            stochastic=cfg.ivanov_toy.stochastic,
        )

        def res_hybrid(x: np.ndarray) -> np.ndarray:
            p = HybridToyParams(
                b_delta=float(x[0]),
                b_eta=float(x[1]),
                b_t=float(x[2]),
                c0=float(x[3]),
                c2=float(x[4]),
                c4=float(x[5]),
                loop_amp=float(x[6]),
                loop_mu2=cfg.hybrid_toy.loop_mu2,
                loop_mu4=cfg.hybrid_toy.loop_mu4,
                loop_k_nl=cfg.hybrid_toy.loop_k_nl,
                sigma_th=float(x[7]),
                stochastic=cfg.hybrid_toy.stochastic,
            )
            pred = hybrid_model.evaluate_components(kf, muf, p)["total"]
            data_res = (pred - pf) / sig

            # Tight hybrid priors to control high-kmax degeneracy.
            prior_res = np.array(
                [
                    (x[2] - 0.0) / 0.08,   # b_t
                    (x[3] - 0.0) / 0.06,   # c0
                    (x[4] - 0.0) / 0.06,   # c2
                    (x[5] - 0.0) / 0.06,   # c4
                    (x[6] - 0.06) / 0.10,  # loop_amp
                    (x[7] - 0.08) / 0.05,  # sigma_th
                ]
            )
            return np.concatenate([data_res, prior_res])

        x_h = np.clip(x_h, hy_lo + 1.0e-8, hy_hi - 1.0e-8)
        fit_h = least_squares(
            res_hybrid,
            x0=x_h,
            bounds=(hy_lo, hy_hi),
            method="trf",
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=16000,
        )
        x_h = fit_h.x.copy()
        p_h = HybridToyParams(
            b_delta=float(x_h[0]),
            b_eta=float(x_h[1]),
            b_t=float(x_h[2]),
            c0=float(x_h[3]),
            c2=float(x_h[4]),
            c4=float(x_h[5]),
            loop_amp=float(x_h[6]),
            loop_mu2=cfg.hybrid_toy.loop_mu2,
            loop_mu4=cfg.hybrid_toy.loop_mu4,
            loop_k_nl=cfg.hybrid_toy.loop_k_nl,
            sigma_th=float(x_h[7]),
            stochastic=cfg.hybrid_toy.stochastic,
        )

        r_tree = res_tree(x_tree)
        r_1l = res_1l(x_1l)
        pred_h_data = hybrid_model.evaluate_components(kf, muf, p_h)["total"]
        r_h_data = (pred_h_data - pf) / sig

        chi2_tree, chi2d_tree = data_chi2_dof(r_tree, 2)
        chi2_1l, chi2d_1l = data_chi2_dof(r_1l, 6)
        chi2_h, chi2d_h = data_chi2_dof(r_h_data, 8)

        rows.append(
            {
                "kmax": float(kmax),
                "npts": int(kf.size),
                "chi2_tree": chi2_tree,
                "chi2dof_tree": chi2d_tree,
                "chi2_oneloop": chi2_1l,
                "chi2dof_oneloop": chi2d_1l,
                "chi2_hybrid": chi2_h,
                "chi2dof_hybrid": chi2d_h,
                "hybrid_b_delta": p_h.b_delta,
                "hybrid_b_eta": p_h.b_eta,
                "hybrid_b_t": p_h.b_t,
                "hybrid_c0": p_h.c0,
                "hybrid_c2": p_h.c2,
                "hybrid_c4": p_h.c4,
                "hybrid_loop_amp": p_h.loop_amp,
                "hybrid_sigma_th": p_h.sigma_th,
            }
        )

        params_at_kmax[kmax] = {
            "tree": p_tree,
            "one_loop": p_1l,
            "hybrid": p_h,
        }

    kmax_ref = max(kmax_values)
    p_tree_ref = params_at_kmax[kmax_ref]["tree"]
    p_1l_ref = params_at_kmax[kmax_ref]["one_loop"]
    p_h_ref = params_at_kmax[kmax_ref]["hybrid"]

    pred_tree_all = one_loop_model.evaluate_components(k_all, mu_all, p_tree_ref)["total"]
    pred_1l_all = one_loop_model.evaluate_components(k_all, mu_all, p_1l_ref)["total"]
    pred_h_all = hybrid_model.evaluate_components(k_all, mu_all, p_h_ref)["total"]

    np.savez(
        run_paths.arrays_dir / "sherwood_highk_scan_arrays.npz",
        k_all=k_all,
        mu_all=mu_all,
        p_all=p_all,
        pred_tree_all=pred_tree_all,
        pred_oneloop_all=pred_1l_all,
        pred_hybrid_all=pred_h_all,
        kmax_values=np.array(kmax_values),
    )

    # Plot 1: chi2/dof vs kmax.
    f1 = run_paths.figures_dir / "41_highk_chi2dof_vs_kmax.png"
    xx = np.array([r["kmax"] for r in rows])
    plt.figure(figsize=(8, 5))
    plt.plot(xx, [r["chi2dof_tree"] for r in rows], marker="o", lw=2, label="original/tree")
    plt.plot(xx, [r["chi2dof_oneloop"] for r in rows], marker="o", lw=2, label="one-loop")
    plt.plot(xx, [r["chi2dof_hybrid"] for r in rows], marker="o", lw=2, label="hybrid (tight priors)")
    plt.xlabel(r"$k_{\max}\ [h\,{\rm Mpc}^{-1}]$")
    plt.ylabel(r"$\chi^2/\mathrm{dof}$ (data only)")
    plt.title("High-kmax scan: fit quality trend")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f1, dpi=160)
    plt.close()

    # Plot 2: hybrid gain vs one-loop.
    f2 = run_paths.figures_dir / "42_highk_hybrid_gain_vs_oneloop.png"
    gain = np.array([r["chi2dof_oneloop"] - r["chi2dof_hybrid"] for r in rows])
    plt.figure(figsize=(8, 4.8))
    plt.plot(xx, gain, marker="o", lw=2, color="#2ca02c")
    plt.axhline(0.0, color="k", lw=1)
    plt.xlabel(r"$k_{\max}\ [h\,{\rm Mpc}^{-1}]$")
    plt.ylabel(r"$\Delta(\chi^2/\mathrm{dof}) = \mathrm{one\!\!\!\!-\!loop} - \mathrm{hybrid}$")
    plt.title("Positive values indicate hybrid improvement")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f2, dpi=160)
    plt.close()

    # Plot 3: hybrid parameters vs kmax.
    f3 = run_paths.figures_dir / "43_highk_hybrid_params_vs_kmax.png"
    keys = [
        "hybrid_b_delta",
        "hybrid_b_eta",
        "hybrid_b_t",
        "hybrid_c0",
        "hybrid_c2",
        "hybrid_c4",
        "hybrid_loop_amp",
        "hybrid_sigma_th",
    ]
    labels = [
        r"$b_\delta$",
        r"$b_\eta$",
        r"$b_T$",
        r"$c_0$",
        r"$c_2$",
        r"$c_4$",
        r"$A_{loop}$",
        r"$\sigma_{th}$",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True)
    for ax, key, lab in zip(axes.ravel(), keys, labels):
        ax.plot(xx, [r[key] for r in rows], marker="o", lw=1.8)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25)
    for ax in axes[1, :]:
        ax.set_xlabel(r"$k_{\max}$")
    fig.suptitle("Hybrid best-fit parameters under high-kmax scan")
    fig.tight_layout()
    fig.savefig(f3, dpi=160)
    plt.close(fig)

    # Plot 4: Pk+residual original vs one-loop at highest kmax.
    f4 = run_paths.figures_dir / "44_highk_pk_residual_original_vs_oneloop.png"
    make_pk_residual_figure(
        k=k_all,
        mu=mu_all,
        p_data=p_all,
        p_a=pred_tree_all,
        p_b=pred_1l_all,
        label_a="original/tree",
        label_b="one-loop",
        title=fr"Highest-cut comparison at $k_{{\max}}={kmax_ref:.1f}$",
        out_path=f4,
    )

    # Plot 5: Pk+residual one-loop vs hybrid at highest kmax.
    f5 = run_paths.figures_dir / "45_highk_pk_residual_oneloop_vs_hybrid.png"
    make_pk_residual_figure(
        k=k_all,
        mu=mu_all,
        p_data=p_all,
        p_a=pred_1l_all,
        p_b=pred_h_all,
        label_a="one-loop",
        label_b="hybrid (tight priors)",
        title=fr"Highest-cut comparison at $k_{{\max}}={kmax_ref:.1f}$",
        out_path=f5,
    )

    fig_target = Path("results/figures")
    fig_target.mkdir(parents=True, exist_ok=True)
    for fp in [f1, f2, f3, f4, f5]:
        shutil.copy2(fp, fig_target / fp.name)

    rows_out = []
    for r in rows:
        rows_out.append({k: float(v) if isinstance(v, (np.floating, float, int)) else v for k, v in r.items()})

    summary = {
        "run_dir": str(run_paths.run_dir),
        "kmax_values": kmax_values,
        "rows": rows_out,
        "highest_kmax": float(kmax_ref),
        "highest_kmax_params": {
            "tree": asdict(p_tree_ref),
            "one_loop": asdict(p_1l_ref),
            "hybrid": asdict(p_h_ref),
        },
        "hybrid_priors": {
            "b_t_sigma": 0.08,
            "c0_sigma": 0.06,
            "c2_sigma": 0.06,
            "c4_sigma": 0.06,
            "loop_amp_center_sigma": [0.06, 0.10],
            "sigma_th_center_sigma": [0.08, 0.05],
        },
        "hybrid_bounds": {
            "b_delta": [-1.0, 0.2],
            "b_eta": [-2.0, 2.0],
            "b_t": [-0.25, 0.25],
            "c0": [-0.10, 0.10],
            "c2": [-0.10, 0.10],
            "c4": [-0.10, 0.10],
            "loop_amp": [-0.30, 0.30],
            "sigma_th": [0.0, 0.25],
        },
        "figures": [str(f1), str(f2), str(f3), str(f4), str(f5)],
    }
    write_json(run_paths.logs_dir / "summary.json", summary)

    print(f"Sherwood high-k scan complete: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
