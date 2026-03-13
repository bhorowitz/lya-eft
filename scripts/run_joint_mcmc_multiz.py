#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import binned_statistic

from lya_hybrid.config import load_config
from lya_hybrid.io import load_sherwood_flux_p3d
from lya_hybrid.linear_power import compute_linear_power_camb
from lya_hybrid.logging_utils import build_repro_metadata, init_run_dir, write_json

SNAP_TO_Z = {8: 3.2, 9: 2.8, 10: 2.4, 11: 2.0}
DEFAULT_P3D_DIR = Path("data/external/sherwood_p3d/data/flux_p3d")
STAGE1_BIAS_PARAM_ORDER = {
    "one_loop": ["b1", "b_eta", "c0", "c2", "c4", "loop_amp"],
    "hybrid": ["b_delta", "b_eta", "b_t", "c0", "c2", "c4", "loop_amp", "sigma_th"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Joint multi-z MCMC with shared cosmology and per-z bias blocks, with checkpoint diagnostics."
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--p3d-dir", type=Path, default=DEFAULT_P3D_DIR)
    p.add_argument("--stage1-summary", type=Path, default=None)
    p.add_argument("--z-targets", type=str, default="2.0,3.0")
    p.add_argument("--model", choices=["one_loop", "hybrid", "both"], default="both")
    p.add_argument("--parametrization", choices=["As", "sigma8"], default="sigma8")
    p.add_argument("--kmax-fit", type=float, default=10.0)

    p.add_argument("--omega-min", type=float, default=0.20)
    p.add_argument("--omega-max", type=float, default=0.36)
    p.add_argument("--n-omega-grid", type=int, default=65)

    p.add_argument("--as-min", type=float, default=1.0e-9)
    p.add_argument("--as-max", type=float, default=3.5e-9)
    p.add_argument("--sigma8-min", type=float, default=0.55)
    p.add_argument("--sigma8-max", type=float, default=1.05)

    p.add_argument("--nwalkers", type=int, default=56)
    p.add_argument("--burnin", type=int, default=220)
    p.add_argument("--nsteps", type=int, default=1200)
    p.add_argument("--thin", type=int, default=6)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--posterior-band-draws", type=int, default=140)
    p.add_argument("--hybrid-tight-priors", action="store_true")
    p.add_argument("--simulation-based-bias-prior", action="store_true")
    p.add_argument("--sim-bias-prior-from", type=Path, default=None)
    p.add_argument("--sim-bias-prior-inflate", type=float, default=2.0)
    return p.parse_args()


def pseudo_sigma(p: np.ndarray, counts: np.ndarray, sigma_frac: float, sigma_floor: float) -> np.ndarray:
    return sigma_frac * np.maximum(np.abs(p), 1.0e-8) + np.maximum(1.0 / np.sqrt(np.maximum(counts, 1.0)), sigma_floor)


def binned_curve(k: np.ndarray, y: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yb, edges, _ = binned_statistic(k, y, statistic="median", bins=bins)
    xc = np.sqrt(edges[:-1] * edges[1:])
    keep = np.isfinite(yb)
    return xc[keep], yb[keep]


def interp_linear_grid(x: float, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray | float:
    if x < grid_x[0] or x > grid_x[-1]:
        raise ValueError("outside interpolation range")
    i = int(np.searchsorted(grid_x, x) - 1)
    i = max(0, min(i, grid_x.size - 2))
    x0, x1 = grid_x[i], grid_x[i + 1]
    t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    return (1.0 - t) * grid_y[i] + t * grid_y[i + 1]


def nearest_snapshot(z_target: float) -> tuple[int, float]:
    snap = min(SNAP_TO_Z, key=lambda s: abs(SNAP_TO_Z[s] - z_target))
    return snap, SNAP_TO_Z[snap]


def choose_p3d_file_for_snapshot(p3d_dir: Path, snapshot: int) -> Path:
    preferred = p3d_dir / f"p3d_80_1024_{snapshot}_0_512_1024_20_16_20.fits"
    if preferred.exists():
        return preferred

    candidates = sorted(p3d_dir.glob(f"p3d_*_{snapshot}_0_*_20_16_20.fits"))
    if not candidates:
        raise FileNotFoundError(f"No p3d file found for snapshot={snapshot} in {p3d_dir}")
    return candidates[0]


def latest_stage1_summary() -> Path:
    candidates = sorted(Path("results/runs").glob("*_bias_mcmc_stage1/logs/summary.json"))
    if not candidates:
        raise FileNotFoundError(
            "No stage-1 bias MCMC summary found under results/runs/*_bias_mcmc_stage1/logs/summary.json"
        )
    return candidates[-1]


def load_stage1_bias_priors(summary_path: Path, inflate: float) -> dict[str, dict[str, np.ndarray]]:
    if inflate <= 0.0:
        raise ValueError(f"--sim-bias-prior-inflate must be > 0 (got {inflate})")

    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    out: dict[str, dict[str, np.ndarray]] = {}
    for row in payload.get("models", []):
        model = row.get("model")
        if model not in STAGE1_BIAS_PARAM_ORDER:
            continue

        arr_path = Path(row["arrays"])
        arr = np.load(arr_path, allow_pickle=True)
        if "flat_samples" not in arr:
            raise KeyError(f"Missing 'flat_samples' in {arr_path}")
        flat = np.asarray(arr["flat_samples"], dtype=float)

        expected = STAGE1_BIAS_PARAM_ORDER[model]
        names = row.get("param_names")
        if isinstance(names, list) and len(names) == flat.shape[1]:
            names = [str(x) for x in names]
        elif flat.shape[1] == len(expected):
            names = list(expected)
        else:
            raise ValueError(
                f"Could not infer stage-1 parameter order for model '{model}' from {arr_path}; "
                f"ndim={flat.shape[1]}, expected={len(expected)}"
            )

        try:
            perm = [names.index(nm) for nm in expected]
        except ValueError as exc:
            raise ValueError(f"Stage-1 parameter names for model '{model}' do not match expected order {expected}") from exc

        flat = flat[:, perm]
        mean = np.median(flat, axis=0)
        cov = np.cov(flat, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=float)
        cov = (float(inflate) ** 2) * cov

        diag = np.diag(cov)
        diag_scale = float(np.median(np.clip(diag, 1.0e-16, np.inf))) if diag.size else 1.0
        reg = max(1.0e-12, 1.0e-6 * diag_scale)
        cov_reg = cov + reg * np.eye(cov.shape[0], dtype=float)
        inv_cov = np.linalg.pinv(cov_reg, hermitian=True)

        out[model] = {"mean": mean, "inv_cov": inv_cov, "cov": cov_reg}

    if not out:
        raise ValueError(f"No usable model entries found in stage-1 summary: {summary_path}")
    return out


def make_trace_plot(chain: np.ndarray, names: list[str], out_path: Path) -> None:
    nsteps, nwalkers, ndim = chain.shape
    fig, axes = plt.subplots(ndim, 1, figsize=(10, max(4.0, 1.9 * ndim)), sharex=True)
    if ndim == 1:
        axes = [axes]
    x = np.arange(nsteps)
    for i in range(ndim):
        ax = axes[i]
        for w in range(nwalkers):
            ax.plot(x, chain[:, w, i], color="#1f77b4", alpha=0.18, lw=0.7)
        ax.set_ylabel(names[i])
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("step")
    fig.suptitle("Trace")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_corner_like_plot(samples: np.ndarray, names: list[str], out_path: Path) -> None:
    ndim = samples.shape[1]
    fig, axes = plt.subplots(ndim, ndim, figsize=(2.1 * ndim + 1, 2.1 * ndim + 1))

    ns = min(samples.shape[0], 4500)
    sub = samples[np.random.default_rng(0).choice(samples.shape[0], size=ns, replace=False)]

    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]
            if i == j:
                ax.hist(samples[:, j], bins=40, color="#4c78a8", alpha=0.9, density=True)
            elif i > j:
                ax.scatter(sub[:, j], sub[:, i], s=1.8, alpha=0.22, color="#4c78a8", rasterized=True)
            else:
                ax.axis("off")
                continue

            if i == ndim - 1:
                ax.set_xlabel(names[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(names[i])
            elif j != 0:
                ax.set_yticklabels([])
            ax.grid(alpha=0.15)

    fig.suptitle("Corner (selected params)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_overlay_plot(samples_a: np.ndarray, samples_b: np.ndarray, names: list[str], out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(samples_a[:, 0], samples_a[:, 1], s=2.5, alpha=0.15, color="#1f77b4", label="one-loop")
    plt.scatter(samples_b[:, 0], samples_b[:, 1], s=2.5, alpha=0.15, color="#d62728", label="hybrid")
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    plt.title("Cosmology Posterior Overlay")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def make_residual_band_plot_multiz(
    *,
    datasets: list[dict],
    draw_predictions: np.ndarray,
    model_label: str,
    out_path: Path,
) -> None:
    # draw_predictions shape: (ndraw, n_data_total)
    nd = len(datasets)
    fig, axes = plt.subplots(1, nd, figsize=(6.0 * nd, 4.5), sharey=True)
    if nd == 1:
        axes = [axes]

    mu_bins = np.array([0.0, 0.5, 1.01])
    k_bins = np.logspace(np.log10(0.04), np.log10(10.0), 16)

    offset = 0
    for j, d in enumerate(datasets):
        n = d["kf"].size
        k = d["kf"]
        mu = d["muf"]
        p = d["pf"]
        preds = draw_predictions[:, offset : offset + n]
        offset += n

        ax = axes[j]
        for i in range(mu_bins.size - 1):
            mm = (mu >= mu_bins[i]) & (mu < mu_bins[i + 1])
            if np.count_nonzero(mm) < 6:
                continue

            kx, _ = binned_curve(k[mm], p[mm], bins=k_bins)
            curves = []
            for pr in preds:
                _, rr = binned_curve(k[mm], pr[mm] / p[mm] - 1.0, bins=k_bins)
                if rr.size == kx.size:
                    curves.append(rr)
            if not curves:
                continue
            arr = np.asarray(curves)
            r16 = np.nanpercentile(arr, 16, axis=0)
            r50 = np.nanpercentile(arr, 50, axis=0)
            r84 = np.nanpercentile(arr, 84, axis=0)
            color = "#1f77b4" if i == 0 else "#ff7f0e"
            lbl = "68% (low-mu)" if i == 0 else "68% (high-mu)"
            ax.fill_between(kx, r16, r84, alpha=0.25, color=color, label=lbl)
            ax.plot(kx, r50, color=color, lw=1.8)

        ax.axhline(0.0, color="k", lw=1)
        ax.set_xscale("log")
        ax.set_ylim(-0.9, 0.9)
        ax.grid(alpha=0.2)
        ax.set_title(f"z={d['z']:.2f}")
        ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")

    axes[0].set_ylabel(r"$P_{model}/P_{data}-1$")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Residual Posterior Bands ({model_label})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def build_cosmo_grid(*, cfg, omega_grid: np.ndarray, z_values: list[float], datasets: list[dict]) -> dict:
    nz = len(z_values)
    no = omega_grid.size
    p_ref = [[None for _ in range(no)] for _ in range(nz)]
    f_grid = np.zeros((nz, no))
    s8_grid = np.zeros(no)

    for i, om in enumerate(omega_grid):
        for iz, z in enumerate(z_values):
            lp = compute_linear_power_camb(
                h=cfg.cosmology.h,
                omega_b=cfg.cosmology.omega_b,
                omega_cdm=float(om),
                ns=cfg.cosmology.ns,
                As=cfg.cosmology.As,
                z=float(z),
                kmin=cfg.k_grid.kmin,
                kmax=max(cfg.k_grid.kmax, cfg.fit.kmax_fit),
                nk=cfg.k_grid.nk,
            )
            d = datasets[iz]
            p_ref[iz][i] = np.interp(d["kf"], lp.k_hmpc, lp.p_lin_h3mpc3)
            f_grid[iz, i] = lp.f_growth
            s8_grid[i] = float(lp.sigma8_0 if lp.sigma8_0 is not None else np.nan)

    return {
        "omega_grid": omega_grid,
        "p_ref": p_ref,
        "f_grid": f_grid,
        "s8_grid": s8_grid,
    }


def run_model_chain(
    *,
    model_name: str,
    datasets: list[dict],
    z_values: list[float],
    grids: dict,
    cfg,
    args,
    run_paths,
    seed: int,
    bias_prior: dict[str, np.ndarray] | None,
) -> dict:
    nz = len(z_values)
    omega_grid = grids["omega_grid"]
    p_ref = grids["p_ref"]
    f_grid = grids["f_grid"]
    s8_grid = grids["s8_grid"]

    # Flatten all data blocks for fast likelihood
    k_blocks = [d["kf"] for d in datasets]
    mu_blocks = [d["muf"] for d in datasets]
    p_blocks = [d["pf"] for d in datasets]
    s_blocks = [d["sigma"] for d in datasets]

    invvar_blocks = [1.0 / (s**2) for s in s_blocks]
    log_norm = -0.5 * sum(np.sum(np.log(2.0 * np.pi * s**2)) for s in s_blocks)

    if model_name == "one_loop":
        # [amp/σ8, omega] + nz * [b1,b_eta,c0,c2,c4,loop_amp]
        if args.parametrization == "As":
            names = ["As", "omega_cdm"]
            lo = [args.as_min, args.omega_min]
            hi = [args.as_max, args.omega_max]
            center = [cfg.cosmology.As, cfg.cosmology.omega_cdm]
        else:
            s8_init = float(interp_linear_grid(cfg.cosmology.omega_cdm, omega_grid, s8_grid))
            names = ["sigma8", "omega_cdm"]
            lo = [args.sigma8_min, args.omega_min]
            hi = [args.sigma8_max, args.omega_max]
            center = [s8_init, cfg.cosmology.omega_cdm]

        for iz in range(nz):
            ztag = f"z{z_values[iz]:.2f}"
            names += [f"b1_{ztag}", f"b_eta_{ztag}", f"c0_{ztag}", f"c2_{ztag}", f"c4_{ztag}", f"loop_amp_{ztag}"]
            if bias_prior is None:
                center += [
                    cfg.ivanov_toy.b1,
                    cfg.ivanov_toy.b_eta,
                    cfg.ivanov_toy.c0,
                    cfg.ivanov_toy.c2,
                    cfg.ivanov_toy.c4,
                    cfg.ivanov_toy.loop_amp,
                ]
            else:
                center += bias_prior["mean"].tolist()
            lo += [-1.0, -2.0, -0.2, -0.2, -0.2, -0.3]
            hi += [0.2, 2.0, 0.2, 0.2, 0.2, 0.3]

        lo = np.array(lo, dtype=float)
        hi = np.array(hi, dtype=float)
        center = np.array(center, dtype=float)

        def log_prior(theta: np.ndarray) -> float:
            if bias_prior is None:
                return 0.0
            lp = 0.0
            base = 2
            for iz in range(nz):
                block = theta[base + 6 * iz : base + 6 * (iz + 1)]
                delta = block - bias_prior["mean"]
                lp += -0.5 * float(delta @ bias_prior["inv_cov"] @ delta)
            return lp

        def predict_blocks(theta: np.ndarray) -> list[np.ndarray]:
            amp, om = float(theta[0]), float(theta[1])
            if args.parametrization == "As":
                scale_amp = amp / cfg.cosmology.As
            else:
                s8_ref = float(interp_linear_grid(om, omega_grid, s8_grid))
                scale_amp = (amp / s8_ref) ** 2

            preds = []
            base = 2
            for iz in range(nz):
                block = theta[base + 6 * iz : base + 6 * (iz + 1)]
                b1, b_eta, c0, c2, c4, loop_amp = [float(x) for x in block]

                p_lin = scale_amp * interp_linear_grid(om, omega_grid, np.asarray(p_ref[iz]))
                fg = float(interp_linear_grid(om, omega_grid, f_grid[iz]))
                k = k_blocks[iz]
                mu = mu_blocks[iz]
                mu2 = mu**2
                mu4 = mu2**2
                pref = b1 + b_eta * fg * mu2
                tree = pref**2 * p_lin
                loop_shape = (k / cfg.ivanov_toy.loop_k_nl) ** 2 / (1.0 + (k / cfg.ivanov_toy.loop_k_nl) ** 2)
                loop_anis = 1.0 + cfg.ivanov_toy.loop_mu2 * mu2 + cfg.ivanov_toy.loop_mu4 * mu4
                loop = loop_amp * loop_shape * loop_anis * p_lin
                counter = -2.0 * (c0 + c2 * mu2 + c4 * mu4) * (k**2) * p_lin
                preds.append(tree + loop + counter)

            return preds

        # Reduced diagnostic subset for checkpoint plots
        diag_idx = [0, 1]
        diag_names = [names[0], names[1]]
        for iz in range(nz):
            base = 2 + 6 * iz
            diag_idx += [base + 0, base + 1, base + 5]
            diag_names += [names[base + 0], names[base + 1], names[base + 5]]

    else:
        # [amp/σ8, omega] + nz * [b_delta,b_eta,b_t,c0,c2,c4,loop_amp,sigma_th]
        if args.parametrization == "As":
            names = ["As", "omega_cdm"]
            lo = [args.as_min, args.omega_min]
            hi = [args.as_max, args.omega_max]
            center = [cfg.cosmology.As, cfg.cosmology.omega_cdm]
        else:
            s8_init = float(interp_linear_grid(cfg.cosmology.omega_cdm, omega_grid, s8_grid))
            names = ["sigma8", "omega_cdm"]
            lo = [args.sigma8_min, args.omega_min]
            hi = [args.sigma8_max, args.omega_max]
            center = [s8_init, cfg.cosmology.omega_cdm]

        for iz in range(nz):
            ztag = f"z{z_values[iz]:.2f}"
            names += [
                f"b_delta_{ztag}",
                f"b_eta_{ztag}",
                f"b_t_{ztag}",
                f"c0_{ztag}",
                f"c2_{ztag}",
                f"c4_{ztag}",
                f"loop_amp_{ztag}",
                f"sigma_th_{ztag}",
            ]
            if bias_prior is None:
                center += [
                    cfg.hybrid_toy.b_delta,
                    cfg.hybrid_toy.b_eta,
                    cfg.hybrid_toy.b_t,
                    cfg.hybrid_toy.c0,
                    cfg.hybrid_toy.c2,
                    cfg.hybrid_toy.c4,
                    cfg.hybrid_toy.loop_amp,
                    cfg.hybrid_toy.sigma_th,
                ]
            else:
                center += bias_prior["mean"].tolist()
            lo += [-1.0, -2.0, -0.25, -0.10, -0.10, -0.10, -0.30, 0.0]
            hi += [0.2, 2.0, 0.25, 0.10, 0.10, 0.10, 0.30, 0.25]

        lo = np.array(lo, dtype=float)
        hi = np.array(hi, dtype=float)
        center = np.array(center, dtype=float)

        def log_prior(theta: np.ndarray) -> float:
            lp = 0.0
            base = 2
            for iz in range(nz):
                if bias_prior is not None:
                    block = theta[base + 8 * iz : base + 8 * (iz + 1)]
                    delta = block - bias_prior["mean"]
                    lp += -0.5 * float(delta @ bias_prior["inv_cov"] @ delta)
                if args.hybrid_tight_priors:
                    b_t, c0, c2, c4, loop_amp, sigma_th = [
                        float(x) for x in theta[base + 8 * iz + 2 : base + 8 * iz + 8]
                    ]
                    lp += -0.5 * (
                        (b_t / 0.08) ** 2
                        + (c0 / 0.06) ** 2
                        + (c2 / 0.06) ** 2
                        + (c4 / 0.06) ** 2
                        + ((loop_amp - 0.06) / 0.10) ** 2
                        + ((sigma_th - 0.08) / 0.05) ** 2
                    )
            return lp

        def predict_blocks(theta: np.ndarray) -> list[np.ndarray]:
            amp, om = float(theta[0]), float(theta[1])
            if args.parametrization == "As":
                scale_amp = amp / cfg.cosmology.As
            else:
                s8_ref = float(interp_linear_grid(om, omega_grid, s8_grid))
                scale_amp = (amp / s8_ref) ** 2

            preds = []
            base = 2
            for iz in range(nz):
                block = theta[base + 8 * iz : base + 8 * (iz + 1)]
                b_delta, b_eta, b_t, c0, c2, c4, loop_amp, sigma_th = [float(x) for x in block]

                p_lin = scale_amp * interp_linear_grid(om, omega_grid, np.asarray(p_ref[iz]))
                fg = float(interp_linear_grid(om, omega_grid, f_grid[iz]))
                k = k_blocks[iz]
                mu = mu_blocks[iz]
                mu2 = mu**2
                mu4 = mu2**2
                temp_shape = 1.0 / (1.0 + (k / cfg.hybrid_toy.k_t) ** 2)

                source_pref = b_delta + b_eta * fg * mu2 + b_t * temp_shape
                tree = source_pref**2 * p_lin
                loop_shape = (k / cfg.hybrid_toy.loop_k_nl) ** 2 / (1.0 + (k / cfg.hybrid_toy.loop_k_nl) ** 2)
                loop_anis = 1.0 + cfg.hybrid_toy.loop_mu2 * mu2 + cfg.hybrid_toy.loop_mu4 * mu4
                loop = loop_amp * loop_shape * loop_anis * p_lin
                counter = -2.0 * (c0 + c2 * mu2 + c4 * mu4) * (k**2) * p_lin
                source_total = tree + loop + counter
                los = np.exp(-((k * mu * sigma_th) ** 2))
                preds.append(los * source_total)

            return preds

        diag_idx = [0, 1]
        diag_names = [names[0], names[1]]
        for iz in range(nz):
            base = 2 + 8 * iz
            diag_idx += [base + 0, base + 1, base + 7]
            diag_names += [names[base + 0], names[base + 1], names[base + 7]]

    def residual(theta: np.ndarray) -> np.ndarray:
        preds = predict_blocks(theta)
        res = [((preds[i] - p_blocks[i]) / s_blocks[i]) for i in range(nz)]
        return np.concatenate(res)

    center = np.clip(center, lo + 1.0e-8, hi - 1.0e-8)
    fit = least_squares(
        residual,
        x0=center,
        bounds=(lo, hi),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=20000,
    )
    map_theta = fit.x.copy()

    def log_prob(theta: np.ndarray) -> float:
        if np.any(theta <= lo) or np.any(theta >= hi):
            return -np.inf
        lp = log_prior(theta)
        preds = predict_blocks(theta)
        ll = log_norm
        for i in range(nz):
            diff = preds[i] - p_blocks[i]
            ll += -0.5 * np.sum(diff * diff * invvar_blocks[i])
        return lp + ll

    nwalkers = max(int(args.nwalkers), 2 * len(names))
    rng = np.random.default_rng(seed)
    spread = 0.012 * (hi - lo)
    p0 = map_theta + rng.normal(scale=spread, size=(nwalkers, len(names)))
    p0 = np.clip(p0, lo + 1.0e-8, hi - 1.0e-8)

    sampler = emcee.EnsembleSampler(nwalkers, len(names), log_prob)
    state = sampler.run_mcmc(p0, int(args.burnin), progress=False)
    sampler.reset()

    checkpoints = run_paths.figures_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    checkpoint_stats = []
    steps_done = 0

    while steps_done < int(args.nsteps):
        n_chunk = min(int(args.checkpoint_every), int(args.nsteps) - steps_done)
        state = sampler.run_mcmc(state, n_chunk, progress=False)
        steps_done += n_chunk

        chain = sampler.get_chain()
        flat = sampler.get_chain(flat=True, thin=max(1, int(args.thin)))

        q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
        cp = {
            "steps_done": int(steps_done),
            "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
            "total_samples": int(flat.shape[0]),
            "q50": q50.tolist(),
        }
        checkpoint_stats.append(cp)

        tag = f"step{steps_done:05d}"
        f_trace = checkpoints / f"{model_name}_trace_{tag}.png"
        f_corner = checkpoints / f"{model_name}_corner_{tag}.png"

        make_trace_plot(chain[:, :, diag_idx], diag_names, f_trace)
        make_corner_like_plot(flat[:, diag_idx], diag_names, f_corner)
        write_json(checkpoints / f"{model_name}_summary_{tag}.json", cp)

    chain = sampler.get_chain()
    flat = sampler.get_chain(flat=True, thin=max(1, int(args.thin)))
    flat_lp = sampler.get_log_prob(flat=True, thin=max(1, int(args.thin)))

    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
    best_idx = int(np.argmax(flat_lp))
    best_theta = flat[best_idx]

    preds_best = predict_blocks(best_theta)
    chi2 = 0.0
    n_data = 0
    for i in range(nz):
        chi2 += float(np.sum(((preds_best[i] - p_blocks[i]) / s_blocks[i]) ** 2))
        n_data += p_blocks[i].size
    dof = max(int(n_data - len(names)), 1)

    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_out = [float(x) for x in tau]
    except Exception:
        tau_out = []

    arr_path = run_paths.arrays_dir / f"joint_multiz_{model_name}.npz"
    np.savez(
        arr_path,
        chain=chain,
        flat_samples=flat,
        flat_log_prob=flat_lp,
        map_theta=map_theta,
        best_theta=best_theta,
        q16=q16,
        q50=q50,
        q84=q84,
        param_names=np.array(names, dtype=object),
        diag_idx=np.array(diag_idx),
        diag_names=np.array(diag_names, dtype=object),
        z_values=np.array(z_values),
    )

    f_trace_final = run_paths.figures_dir / f"101_{model_name}_multiz_trace_final.png"
    f_corner_final = run_paths.figures_dir / f"102_{model_name}_multiz_corner_final.png"
    make_trace_plot(chain[:, :, diag_idx], diag_names, f_trace_final)
    make_corner_like_plot(flat[:, diag_idx], diag_names, f_corner_final)

    ndraw = min(int(args.posterior_band_draws), flat.shape[0])
    idx = rng.choice(flat.shape[0], size=ndraw, replace=False)
    draw_preds = []
    for ii in idx:
        pred_list = predict_blocks(flat[ii])
        draw_preds.append(np.concatenate(pred_list))
    draw_preds = np.asarray(draw_preds)

    f_band = run_paths.figures_dir / f"103_{model_name}_multiz_residual_band_final.png"
    make_residual_band_plot_multiz(
        datasets=datasets,
        draw_predictions=draw_preds,
        model_label=f"{model_name} multiz",
        out_path=f_band,
    )

    return {
        "model": model_name,
        "param_names": names,
        "diag_param_names": diag_names,
        "bounds": {"lower": lo.tolist(), "upper": hi.tolist()},
        "map_theta": map_theta.tolist(),
        "best_theta": best_theta.tolist(),
        "q16": q16.tolist(),
        "q50": q50.tolist(),
        "q84": q84.tolist(),
        "fit_chi2": float(chi2),
        "fit_chi2_dof": float(chi2 / dof),
        "n_data_total": int(n_data),
        "n_dim": len(names),
        "nwalkers": int(nwalkers),
        "nsteps": int(args.nsteps),
        "burnin": int(args.burnin),
        "thin": int(args.thin),
        "checkpoint_every": int(args.checkpoint_every),
        "checkpoint_stats": checkpoint_stats,
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
        "autocorr_time": tau_out,
        "arrays": str(arr_path),
        "final_figures": [str(f_trace_final), str(f_corner_final), str(f_band)],
        "checkpoint_dir": str(checkpoints),
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    prior_summary_path = None
    bias_priors = {}
    if args.simulation_based_bias_prior:
        if args.sim_bias_prior_from is not None:
            prior_summary_path = args.sim_bias_prior_from
        elif args.stage1_summary is not None:
            prior_summary_path = args.stage1_summary
        else:
            prior_summary_path = latest_stage1_summary()
        bias_priors = load_stage1_bias_priors(prior_summary_path, inflate=float(args.sim_bias_prior_inflate))

    z_targets = [float(x) for x in args.z_targets.split(",") if x.strip()]
    if len(z_targets) < 2:
        raise ValueError("Please provide at least two target redshifts, e.g. --z-targets 2.0,3.0")

    resolved = []
    for zt in z_targets:
        snap, z_data = nearest_snapshot(zt)
        fp = choose_p3d_file_for_snapshot(args.p3d_dir, snap)
        resolved.append({"z_target": zt, "snapshot": snap, "z_data": z_data, "file": fp})

    run_paths = init_run_dir(cfg.run.output_root, tag="joint_mcmc_multiz")
    meta = build_repro_metadata(args.config)
    meta.update(
        {
            "z_targets": z_targets,
            "resolved": [{**r, "file": str(r["file"])} for r in resolved],
            "model": args.model,
            "parametrization": args.parametrization,
            "kmax_fit": float(args.kmax_fit),
            "nwalkers": int(args.nwalkers),
            "burnin": int(args.burnin),
            "nsteps": int(args.nsteps),
            "thin": int(args.thin),
            "checkpoint_every": int(args.checkpoint_every),
            "omega_grid": [float(args.omega_min), float(args.omega_max), int(args.n_omega_grid)],
            "hybrid_tight_priors": bool(args.hybrid_tight_priors),
            "simulation_based_bias_prior": {
                "enabled": bool(args.simulation_based_bias_prior),
                "source": (str(prior_summary_path) if args.simulation_based_bias_prior else None),
                "inflate": float(args.sim_bias_prior_inflate),
            },
        }
    )
    write_json(run_paths.logs_dir / "repro_metadata.json", meta)

    datasets = []
    z_values = []
    for r in resolved:
        d = load_sherwood_flux_p3d(r["file"])
        k, mu, p, c = d.flatten_valid()
        m = (k >= cfg.fit.kmin_fit) & (k <= float(args.kmax_fit))
        kf, muf, pf, cf = k[m], mu[m], p[m], c[m]
        datasets.append(
            {
                "z": float(r["z_data"]),
                "z_target": float(r["z_target"]),
                "snapshot": int(r["snapshot"]),
                "file": str(r["file"]),
                "kf": kf,
                "muf": muf,
                "pf": pf,
                "sigma": pseudo_sigma(
                    pf,
                    cf,
                    sigma_frac=cfg.fit.sigma_frac,
                    sigma_floor=cfg.fit.sigma_floor,
                ),
            }
        )
        z_values.append(float(r["z_data"]))

    omega_grid = np.linspace(float(args.omega_min), float(args.omega_max), int(args.n_omega_grid))
    grids = build_cosmo_grid(cfg=cfg, omega_grid=omega_grid, z_values=z_values, datasets=datasets)

    requested_models = [args.model] if args.model in {"one_loop", "hybrid"} else ["one_loop", "hybrid"]
    if args.simulation_based_bias_prior:
        for m in requested_models:
            if m not in bias_priors:
                raise ValueError(f"Simulation based prior source lacks model '{m}': {prior_summary_path}")
    results = []
    flat_by_model = {}

    for i, m in enumerate(requested_models):
        out = run_model_chain(
            model_name=m,
            datasets=datasets,
            z_values=z_values,
            grids=grids,
            cfg=cfg,
            args=args,
            run_paths=run_paths,
            seed=int(args.seed + 37 * i),
            bias_prior=(bias_priors[m] if args.simulation_based_bias_prior else None),
        )
        results.append(out)

        arr = np.load(Path(out["arrays"]), allow_pickle=True)
        flat_by_model[m] = arr["flat_samples"]

    overlay = None
    if set(requested_models) == {"one_loop", "hybrid"}:
        overlay = run_paths.figures_dir / "104_multiz_cosmo_overlay.png"
        make_overlay_plot(
            samples_a=flat_by_model["one_loop"][:, :2],
            samples_b=flat_by_model["hybrid"][:, :2],
            names=results[0]["param_names"][:2],
            out_path=overlay,
        )

    fig_target = Path("results/figures")
    fig_target.mkdir(parents=True, exist_ok=True)

    for res in results:
        for fp in res["final_figures"]:
            shutil.copy2(fp, fig_target / Path(fp).name)
        cdir = Path(res["checkpoint_dir"])
        for fp in sorted(cdir.glob(f"{res['model']}_trace_step*.png")):
            shutil.copy2(fp, fig_target / fp.name)
        for fp in sorted(cdir.glob(f"{res['model']}_corner_step*.png")):
            shutil.copy2(fp, fig_target / fp.name)

    if overlay is not None:
        shutil.copy2(overlay, fig_target / overlay.name)

    summary = {
        "run_dir": str(run_paths.run_dir),
        "z_targets": z_targets,
        "resolved": [{**r, "file": str(r["file"])} for r in resolved],
        "parametrization": args.parametrization,
        "kmax_fit": float(args.kmax_fit),
        "requested_models": requested_models,
        "simulation_based_bias_prior": {
            "enabled": bool(args.simulation_based_bias_prior),
            "source": (str(prior_summary_path) if args.simulation_based_bias_prior else None),
            "inflate": float(args.sim_bias_prior_inflate),
        },
        "results": results,
        "overlay_figure": (str(overlay) if overlay is not None else None),
    }
    write_json(run_paths.logs_dir / "summary.json", summary)

    print(f"Joint multi-z MCMC complete: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
