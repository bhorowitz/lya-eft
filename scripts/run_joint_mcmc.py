#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
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

DEFAULT_P3D = Path("data/external/sherwood_p3d/data/flux_p3d/p3d_80_1024_9_0_384_1024_20_16_20.fits")
STAGE1_BIAS_PARAM_ORDER = {
    "one_loop": ["b1", "b_eta", "c0", "c2", "c4", "loop_amp"],
    "hybrid": ["b_delta", "b_eta", "b_t", "c0", "c2", "c4", "loop_amp", "sigma_th"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Joint bias+cosmology MCMC with periodic checkpoint diagnostics (trace/corner)."
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--p3d", type=Path, default=DEFAULT_P3D)
    p.add_argument("--stage1-summary", type=Path, default=None)
    p.add_argument("--model", choices=["one_loop", "hybrid", "both"], default="both")
    p.add_argument("--parametrization", choices=["As", "sigma8"], default=None)
    p.add_argument("--kmax-fit", type=float, default=10.0)
    p.add_argument("--nwalkers", type=int, default=None)
    p.add_argument("--nsteps", type=int, default=None)
    p.add_argument("--burnin", type=int, default=None)
    p.add_argument("--thin", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--n-omega-grid", type=int, default=None)
    p.add_argument("--omega-min", type=float, default=None)
    p.add_argument("--omega-max", type=float, default=None)
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


def latest_stage1_summary() -> Path:
    candidates = sorted(Path("results/runs").glob("*_bias_mcmc_stage1/logs/summary.json"))
    if not candidates:
        raise FileNotFoundError(
            "No stage-1 bias MCMC summary found under results/runs/*_bias_mcmc_stage1/logs/summary.json"
        )
    return candidates[-1]


def load_stage1_best_biases(summary_path: Path) -> dict[str, dict[str, float]]:
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    out: dict[str, dict[str, float]] = {}
    for row in payload.get("models", []):
        out[row["model"]] = row["best_params_dict"]
    if not out:
        raise ValueError(f"No model entries found in stage-1 summary: {summary_path}")
    return out


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


def interp_linear_grid(x: float, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray | float:
    if x < grid_x[0] or x > grid_x[-1]:
        raise ValueError("Requested interpolation point outside grid.")
    i = int(np.searchsorted(grid_x, x) - 1)
    i = max(0, min(i, grid_x.size - 2))
    x0, x1 = grid_x[i], grid_x[i + 1]
    t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    return (1.0 - t) * grid_y[i] + t * grid_y[i + 1]


def make_trace_plot(chain: np.ndarray, names: list[str], out_path: Path) -> None:
    nsteps, nwalkers, ndim = chain.shape
    fig, axes = plt.subplots(ndim, 1, figsize=(10, max(4.0, 2.1 * ndim)), sharex=True)
    if ndim == 1:
        axes = [axes]
    x = np.arange(nsteps)
    for i in range(ndim):
        ax = axes[i]
        for w in range(nwalkers):
            ax.plot(x, chain[:, w, i], color="#1f77b4", alpha=0.18, lw=0.7)
        ax.set_ylabel(names[i])
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("MCMC step")
    fig.suptitle("Trace Plot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_corner_like_plot(samples: np.ndarray, names: list[str], out_path: Path, max_scatter: int = 5000) -> None:
    ndim = samples.shape[1]
    fig, axes = plt.subplots(ndim, ndim, figsize=(2.0 * ndim + 1.5, 2.0 * ndim + 1.5))

    if samples.shape[0] > max_scatter:
        idx = np.random.default_rng(0).choice(samples.shape[0], size=max_scatter, replace=False)
        subs = samples[idx]
    else:
        subs = samples

    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]
            if i == j:
                ax.hist(samples[:, j], bins=40, color="#4c78a8", alpha=0.85, density=True)
            elif i > j:
                ax.scatter(subs[:, j], subs[:, i], s=1.8, alpha=0.2, color="#4c78a8", rasterized=True)
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

    fig.suptitle("Posterior Pair Plot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_residual_band_plot(
    *,
    k_all: np.ndarray,
    mu_all: np.ndarray,
    p_data: np.ndarray,
    draw_predictions: np.ndarray,
    model_label: str,
    out_path: Path,
) -> None:
    mu_bins = np.array([0.0, 0.25, 0.5, 0.75, 1.01])
    k_bins = np.logspace(np.log10(0.04), np.log10(10.0), 18)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

    for i in range(4):
        lo, hi = mu_bins[i], mu_bins[i + 1]
        m = (mu_all >= lo) & (mu_all < hi)
        ax = axes[i]

        if np.count_nonzero(m) < 6:
            ax.set_visible(False)
            continue

        kx, _ = binned_curve(k_all[m], p_data[m], bins=k_bins)
        band_curves = []
        for pred in draw_predictions:
            _, rr = binned_curve(k_all[m], pred[m] / p_data[m] - 1.0, bins=k_bins)
            if rr.size == kx.size:
                band_curves.append(rr)

        if not band_curves:
            ax.set_visible(False)
            continue

        arr = np.asarray(band_curves)
        r16 = np.nanpercentile(arr, 16, axis=0)
        r50 = np.nanpercentile(arr, 50, axis=0)
        r84 = np.nanpercentile(arr, 84, axis=0)

        ax.fill_between(kx, r16, r84, color="#4c78a8", alpha=0.30, label="68% posterior")
        ax.plot(kx, r50, color="#1f77b4", lw=2.0, label="posterior median")
        ax.axhline(0.0, color="k", lw=1)
        ax.set_xscale("log")
        ax.set_title(fr"$\mu\in[{lo:.2f},{hi:.2f})$")
        ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
        ax.grid(alpha=0.2)
        ax.set_ylim(-0.9, 0.9)

    axes[0].set_ylabel(r"$P_{model}/P_{data}-1$")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Residual Posterior Band: {model_label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_overlay_plot(
    *,
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    names: list[str],
    label_a: str,
    label_b: str,
    out_path: Path,
) -> None:
    rng = np.random.default_rng(123)
    na = min(5500, samples_a.shape[0])
    nb = min(5500, samples_b.shape[0])
    sa = samples_a[rng.choice(samples_a.shape[0], size=na, replace=False)]
    sb = samples_b[rng.choice(samples_b.shape[0], size=nb, replace=False)]

    plt.figure(figsize=(7, 6))
    plt.scatter(sa[:, 0], sa[:, 1], s=2.5, alpha=0.15, color="#1f77b4", label=label_a)
    plt.scatter(sb[:, 0], sb[:, 1], s=2.5, alpha=0.15, color="#d62728", label=label_b)
    ma = np.median(samples_a, axis=0)
    mb = np.median(samples_b, axis=0)
    plt.scatter([ma[0]], [ma[1]], s=90, marker="x", color="#1f77b4")
    plt.scatter([mb[0]], [mb[1]], s=90, marker="x", color="#d62728")
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    plt.title("Joint Posterior Overlay")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def build_cosmo_grids(*, cfg, omega_grid: np.ndarray, kf: np.ndarray) -> dict[str, np.ndarray]:
    n_omega = omega_grid.size
    p_ref = np.zeros((n_omega, kf.size))
    f_grid = np.zeros(n_omega)
    sigma8_grid = np.zeros(n_omega)

    for i, om in enumerate(omega_grid):
        lp = compute_linear_power_camb(
            h=cfg.cosmology.h,
            omega_b=cfg.cosmology.omega_b,
            omega_cdm=float(om),
            ns=cfg.cosmology.ns,
            As=cfg.cosmology.As,
            z=cfg.cosmology.z,
            kmin=cfg.k_grid.kmin,
            kmax=max(cfg.k_grid.kmax, cfg.fit.kmax_fit),
            nk=cfg.k_grid.nk,
        )
        p_ref[i, :] = np.interp(kf, lp.k_hmpc, lp.p_lin_h3mpc3)
        f_grid[i] = lp.f_growth
        sigma8_grid[i] = float(lp.sigma8_0 if lp.sigma8_0 is not None else np.nan)

    return {
        "omega_grid": omega_grid,
        "p_ref_grid": p_ref,
        "f_growth_grid": f_grid,
        "sigma8_grid": sigma8_grid,
    }


def run_joint_model(
    *,
    model_name: str,
    stage1_bias: dict[str, float],
    grids: dict[str, np.ndarray],
    kf: np.ndarray,
    muf: np.ndarray,
    pf: np.ndarray,
    sigma_data: np.ndarray,
    cfg,
    args,
    run_paths,
    seed: int,
    use_tight_priors: bool,
    bias_prior: dict[str, np.ndarray] | None,
) -> dict:
    omega_grid = grids["omega_grid"]
    p_ref_grid = grids["p_ref_grid"]
    f_growth_grid = grids["f_growth_grid"]
    sigma8_grid = grids["sigma8_grid"]

    mu2 = muf**2
    mu4 = mu2**2
    k2 = kf**2
    loop_shape = (kf / cfg.ivanov_toy.loop_k_nl) ** 2 / (1.0 + (kf / cfg.ivanov_toy.loop_k_nl) ** 2)
    loop_anis = 1.0 + cfg.ivanov_toy.loop_mu2 * mu2 + cfg.ivanov_toy.loop_mu4 * mu4
    temp_shape = 1.0 / (1.0 + (kf / cfg.hybrid_toy.k_t) ** 2)

    invvar = 1.0 / (sigma_data**2)
    log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * sigma_data**2))

    param_mode = args.parametrization

    if model_name == "one_loop":
        if param_mode == "As":
            names = ["As", "omega_cdm", "b1", "b_eta", "c0", "c2", "c4", "loop_amp"]
            lo = np.array([args.as_min, args.omega_min, -1.0, -2.0, -0.2, -0.2, -0.2, -0.3], dtype=float)
            hi = np.array([args.as_max, args.omega_max, 0.2, 2.0, 0.2, 0.2, 0.2, 0.3], dtype=float)
            center = np.array(
                [
                    cfg.cosmology.As,
                    cfg.cosmology.omega_cdm,
                    stage1_bias["b1"],
                    stage1_bias["b_eta"],
                    stage1_bias["c0"],
                    stage1_bias["c2"],
                    stage1_bias["c4"],
                    stage1_bias["loop_amp"],
                ],
                dtype=float,
            )
        else:
            s8_init = float(interp_linear_grid(cfg.cosmology.omega_cdm, omega_grid, sigma8_grid))
            names = ["sigma8", "omega_cdm", "b1", "b_eta", "c0", "c2", "c4", "loop_amp"]
            lo = np.array([args.sigma8_min, args.omega_min, -1.0, -2.0, -0.2, -0.2, -0.2, -0.3], dtype=float)
            hi = np.array([args.sigma8_max, args.omega_max, 0.2, 2.0, 0.2, 0.2, 0.2, 0.3], dtype=float)
            center = np.array(
                [
                    s8_init,
                    cfg.cosmology.omega_cdm,
                    stage1_bias["b1"],
                    stage1_bias["b_eta"],
                    stage1_bias["c0"],
                    stage1_bias["c2"],
                    stage1_bias["c4"],
                    stage1_bias["loop_amp"],
                ],
                dtype=float,
            )

        def predict(theta: np.ndarray) -> np.ndarray:
            amp = float(theta[0])
            om = float(theta[1])
            p_ref = interp_linear_grid(om, omega_grid, p_ref_grid)
            fg = float(interp_linear_grid(om, omega_grid, f_growth_grid))

            if param_mode == "As":
                scale = amp / cfg.cosmology.As
            else:
                s8_ref = float(interp_linear_grid(om, omega_grid, sigma8_grid))
                scale = (amp / s8_ref) ** 2

            p_lin = scale * p_ref
            b1, b_eta, c0, c2, c4, loop_amp = [float(x) for x in theta[2:]]

            pref = b1 + b_eta * fg * mu2
            tree = pref**2 * p_lin
            loop = loop_amp * loop_shape * loop_anis * p_lin
            counter = -2.0 * (c0 + c2 * mu2 + c4 * mu4) * k2 * p_lin
            return tree + loop + counter

        def log_prior(theta: np.ndarray) -> float:
            if bias_prior is None:
                return 0.0
            delta = theta[2:8] - bias_prior["mean"]
            return -0.5 * float(delta @ bias_prior["inv_cov"] @ delta)

    else:
        if param_mode == "As":
            names = [
                "As",
                "omega_cdm",
                "b_delta",
                "b_eta",
                "b_t",
                "c0",
                "c2",
                "c4",
                "loop_amp",
                "sigma_th",
            ]
            lo = np.array([args.as_min, args.omega_min, -1.0, -2.0, -0.25, -0.10, -0.10, -0.10, -0.30, 0.0], dtype=float)
            hi = np.array([args.as_max, args.omega_max, 0.2, 2.0, 0.25, 0.10, 0.10, 0.10, 0.30, 0.25], dtype=float)
            center = np.array(
                [
                    cfg.cosmology.As,
                    cfg.cosmology.omega_cdm,
                    stage1_bias["b_delta"],
                    stage1_bias["b_eta"],
                    stage1_bias["b_t"],
                    stage1_bias["c0"],
                    stage1_bias["c2"],
                    stage1_bias["c4"],
                    stage1_bias["loop_amp"],
                    stage1_bias["sigma_th"],
                ],
                dtype=float,
            )
        else:
            s8_init = float(interp_linear_grid(cfg.cosmology.omega_cdm, omega_grid, sigma8_grid))
            names = [
                "sigma8",
                "omega_cdm",
                "b_delta",
                "b_eta",
                "b_t",
                "c0",
                "c2",
                "c4",
                "loop_amp",
                "sigma_th",
            ]
            lo = np.array([args.sigma8_min, args.omega_min, -1.0, -2.0, -0.25, -0.10, -0.10, -0.10, -0.30, 0.0], dtype=float)
            hi = np.array([args.sigma8_max, args.omega_max, 0.2, 2.0, 0.25, 0.10, 0.10, 0.10, 0.30, 0.25], dtype=float)
            center = np.array(
                [
                    s8_init,
                    cfg.cosmology.omega_cdm,
                    stage1_bias["b_delta"],
                    stage1_bias["b_eta"],
                    stage1_bias["b_t"],
                    stage1_bias["c0"],
                    stage1_bias["c2"],
                    stage1_bias["c4"],
                    stage1_bias["loop_amp"],
                    stage1_bias["sigma_th"],
                ],
                dtype=float,
            )

        def predict(theta: np.ndarray) -> np.ndarray:
            amp = float(theta[0])
            om = float(theta[1])
            p_ref = interp_linear_grid(om, omega_grid, p_ref_grid)
            fg = float(interp_linear_grid(om, omega_grid, f_growth_grid))

            if param_mode == "As":
                scale = amp / cfg.cosmology.As
            else:
                s8_ref = float(interp_linear_grid(om, omega_grid, sigma8_grid))
                scale = (amp / s8_ref) ** 2

            p_lin = scale * p_ref
            b_delta, b_eta, b_t, c0, c2, c4, loop_amp, sigma_th = [float(x) for x in theta[2:]]

            source_pref = b_delta + b_eta * fg * mu2 + b_t * temp_shape
            tree = source_pref**2 * p_lin
            loop = loop_amp * loop_shape * loop_anis * p_lin
            counter = -2.0 * (c0 + c2 * mu2 + c4 * mu4) * k2 * p_lin
            source_total = tree + loop + counter
            los = np.exp(-((kf * muf * sigma_th) ** 2))
            return los * source_total

        def log_prior(theta: np.ndarray) -> float:
            lp = 0.0
            if bias_prior is not None:
                delta = theta[2:10] - bias_prior["mean"]
                lp += -0.5 * float(delta @ bias_prior["inv_cov"] @ delta)
            if use_tight_priors:
                b_t, c0, c2, c4, loop_amp, sigma_th = [float(x) for x in theta[4:10]]
                lp += -0.5 * (
                    (b_t / 0.08) ** 2
                    + (c0 / 0.06) ** 2
                    + (c2 / 0.06) ** 2
                    + (c4 / 0.06) ** 2
                    + ((loop_amp - 0.06) / 0.10) ** 2
                    + ((sigma_th - 0.08) / 0.05) ** 2
                )
            return lp

    def residual(theta: np.ndarray) -> np.ndarray:
        pred = predict(theta)
        return (pred - pf) / sigma_data

    center = np.clip(center, lo + 1.0e-8, hi - 1.0e-8)
    fit = least_squares(
        residual,
        x0=center,
        bounds=(lo, hi),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=18000,
    )
    map_theta = fit.x.copy()

    def log_prob(theta: np.ndarray) -> float:
        if np.any(theta <= lo) or np.any(theta >= hi):
            return -np.inf
        lp = log_prior(theta)
        pred = predict(theta)
        diff = pred - pf
        return lp + log_norm - 0.5 * np.sum(diff * diff * invvar)

    nwalkers = max(int(args.nwalkers), 2 * len(names))
    rng = np.random.default_rng(seed)
    spread = 0.015 * (hi - lo)
    p0 = map_theta + rng.normal(scale=spread, size=(nwalkers, len(names)))
    p0 = np.clip(p0, lo + 1.0e-8, hi - 1.0e-8)

    sampler = emcee.EnsembleSampler(nwalkers, len(names), log_prob)

    # Burn-in
    state = sampler.run_mcmc(p0, int(args.burnin), progress=False)
    sampler.reset()

    checkpoints_dir = run_paths.figures_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    steps_done = 0
    checkpoint_stats = []
    t0 = time.perf_counter()

    while steps_done < int(args.nsteps):
        n_chunk = min(int(args.checkpoint_every), int(args.nsteps) - steps_done)
        state = sampler.run_mcmc(state, n_chunk, progress=False)
        steps_done += n_chunk

        chain = sampler.get_chain()
        flat = sampler.get_chain(flat=True, thin=max(1, int(args.thin)))

        q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
        cp = {
            "steps_done": int(steps_done),
            "total_samples": int(flat.shape[0]),
            "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
            "q16": q16.tolist(),
            "q50": q50.tolist(),
            "q84": q84.tolist(),
        }
        checkpoint_stats.append(cp)

        tag = f"step{steps_done:05d}"
        f_trace = checkpoints_dir / f"{model_name}_trace_{tag}.png"
        f_corner = checkpoints_dir / f"{model_name}_corner_{tag}.png"
        make_trace_plot(chain, names, f_trace)
        make_corner_like_plot(flat, names, f_corner)

        write_json(checkpoints_dir / f"{model_name}_summary_{tag}.json", cp)

    elapsed = time.perf_counter() - t0

    chain = sampler.get_chain()
    flat = sampler.get_chain(flat=True, thin=max(1, int(args.thin)))
    flat_lp = sampler.get_log_prob(flat=True, thin=max(1, int(args.thin)))

    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
    best_idx = int(np.argmax(flat_lp))
    best_theta = flat[best_idx]

    pred_best = predict(best_theta)
    chi2 = float(np.sum(((pred_best - pf) / sigma_data) ** 2))
    dof = max(int(pf.size - len(names)), 1)

    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_out = [float(x) for x in tau]
    except Exception:
        tau_out = []

    arr_path = run_paths.arrays_dir / f"joint_mcmc_{model_name}.npz"
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
        bounds_lo=lo,
        bounds_hi=hi,
    )

    # Final diagnostics.
    f_trace_final = run_paths.figures_dir / f"91_{model_name}_joint_trace_final.png"
    f_corner_final = run_paths.figures_dir / f"92_{model_name}_joint_corner_final.png"
    make_trace_plot(chain, names, f_trace_final)
    make_corner_like_plot(flat, names, f_corner_final)

    ndraw = min(int(args.posterior_band_draws), flat.shape[0])
    idx = rng.choice(flat.shape[0], size=ndraw, replace=False)
    pred_draws = np.asarray([predict(th) for th in flat[idx]])
    f_band_final = run_paths.figures_dir / f"93_{model_name}_joint_residual_band_final.png"
    make_residual_band_plot(
        k_all=kf,
        mu_all=muf,
        p_data=pf,
        draw_predictions=pred_draws,
        model_label=f"{model_name} joint",
        out_path=f_band_final,
    )

    return {
        "model": model_name,
        "param_names": names,
        "bounds": {"lower": lo.tolist(), "upper": hi.tolist()},
        "map_theta": map_theta.tolist(),
        "best_theta": best_theta.tolist(),
        "q16": q16.tolist(),
        "q50": q50.tolist(),
        "q84": q84.tolist(),
        "fit_chi2": chi2,
        "fit_chi2_dof": float(chi2 / dof),
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
        "autocorr_time": tau_out,
        "runtime_sec": float(elapsed),
        "n_data": int(pf.size),
        "n_dim": len(names),
        "nwalkers": int(nwalkers),
        "nsteps": int(args.nsteps),
        "burnin": int(args.burnin),
        "thin": int(args.thin),
        "checkpoint_every": int(args.checkpoint_every),
        "checkpoint_stats": checkpoint_stats,
        "arrays": str(arr_path),
        "final_figures": [str(f_trace_final), str(f_corner_final), str(f_band_final)],
        "checkpoint_dir": str(checkpoints_dir),
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    if args.parametrization is None:
        args.parametrization = cfg.joint_mcmc.parametrization
    if args.nwalkers is None:
        args.nwalkers = cfg.joint_mcmc.nwalkers
    if args.nsteps is None:
        args.nsteps = cfg.joint_mcmc.nsteps
    if args.burnin is None:
        args.burnin = cfg.joint_mcmc.burnin
    if args.thin is None:
        args.thin = cfg.joint_mcmc.thin
    if args.seed is None:
        args.seed = cfg.joint_mcmc.seed
    if args.checkpoint_every is None:
        args.checkpoint_every = cfg.joint_mcmc.checkpoint_every
    if args.n_omega_grid is None:
        args.n_omega_grid = cfg.joint_mcmc.n_omega_grid
    if args.omega_min is None:
        args.omega_min = cfg.joint_mcmc.omega_min
    if args.omega_max is None:
        args.omega_max = cfg.joint_mcmc.omega_max
    if not args.hybrid_tight_priors:
        args.hybrid_tight_priors = bool(cfg.joint_mcmc.hybrid_tight_priors)

    args.as_min = cfg.joint_mcmc.as_min
    args.as_max = cfg.joint_mcmc.as_max
    args.sigma8_min = cfg.joint_mcmc.sigma8_min
    args.sigma8_max = cfg.joint_mcmc.sigma8_max
    args.posterior_band_draws = cfg.joint_mcmc.posterior_band_draws

    summary_path = latest_stage1_summary() if args.stage1_summary is None else args.stage1_summary
    best_biases = load_stage1_best_biases(summary_path)
    prior_summary_path = args.sim_bias_prior_from if args.sim_bias_prior_from is not None else summary_path
    bias_priors = (
        load_stage1_bias_priors(prior_summary_path, inflate=float(args.sim_bias_prior_inflate))
        if args.simulation_based_bias_prior
        else {}
    )

    requested_models = [args.model] if args.model in {"one_loop", "hybrid"} else ["one_loop", "hybrid"]
    for m in requested_models:
        if m not in best_biases:
            raise ValueError(f"Stage-1 summary lacks model '{m}': {summary_path}")
        if args.simulation_based_bias_prior and m not in bias_priors:
            raise ValueError(f"Simulation based prior source lacks model '{m}': {prior_summary_path}")

    run_paths = init_run_dir(cfg.run.output_root, tag="joint_mcmc")
    meta = build_repro_metadata(args.config)
    meta.update(
        {
            "p3d_path": str(args.p3d),
            "stage1_summary": str(summary_path),
            "requested_models": requested_models,
            "parametrization": args.parametrization,
            "kmax_fit": float(args.kmax_fit),
            "joint_mcmc": {
                "nwalkers": int(args.nwalkers),
                "nsteps": int(args.nsteps),
                "burnin": int(args.burnin),
                "thin": int(args.thin),
                "seed": int(args.seed),
                "checkpoint_every": int(args.checkpoint_every),
            },
            "omega_grid": {
                "min": float(args.omega_min),
                "max": float(args.omega_max),
                "n": int(args.n_omega_grid),
            },
            "hybrid_tight_priors": bool(args.hybrid_tight_priors),
            "simulation_based_bias_prior": {
                "enabled": bool(args.simulation_based_bias_prior),
                "source": (str(prior_summary_path) if args.simulation_based_bias_prior else None),
                "inflate": float(args.sim_bias_prior_inflate),
            },
        }
    )
    write_json(run_paths.logs_dir / "repro_metadata.json", meta)

    p3d_data = load_sherwood_flux_p3d(args.p3d)
    k_all, mu_all, p_all, counts_all = p3d_data.flatten_valid()
    fit_mask = (k_all >= cfg.fit.kmin_fit) & (k_all <= float(args.kmax_fit))
    kf, muf, pf, cf = k_all[fit_mask], mu_all[fit_mask], p_all[fit_mask], counts_all[fit_mask]
    sigma_data = pseudo_sigma(pf, cf, sigma_frac=cfg.fit.sigma_frac, sigma_floor=cfg.fit.sigma_floor)

    omega_grid = np.linspace(float(args.omega_min), float(args.omega_max), int(args.n_omega_grid))
    grids = build_cosmo_grids(cfg=cfg, omega_grid=omega_grid, kf=kf)

    results = []
    flat_by_model: dict[str, np.ndarray] = {}

    for i, m in enumerate(requested_models):
        out = run_joint_model(
            model_name=m,
            stage1_bias=best_biases[m],
            grids=grids,
            kf=kf,
            muf=muf,
            pf=pf,
            sigma_data=sigma_data,
            cfg=cfg,
            args=args,
            run_paths=run_paths,
            seed=int(args.seed + 31 * i),
            use_tight_priors=bool(args.hybrid_tight_priors),
            bias_prior=(bias_priors[m] if args.simulation_based_bias_prior else None),
        )
        results.append(out)

        arr = np.load(Path(out["arrays"]))
        flat_by_model[m] = arr["flat_samples"]

    overlay = None
    if set(requested_models) == {"one_loop", "hybrid"}:
        overlay = run_paths.figures_dir / "94_joint_overlay_cosmo_params.png"
        make_overlay_plot(
            samples_a=flat_by_model["one_loop"][:, :2],
            samples_b=flat_by_model["hybrid"][:, :2],
            names=results[0]["param_names"][:2],
            label_a="one-loop",
            label_b="hybrid",
            out_path=overlay,
        )

    fig_target = Path("results/figures")
    fig_target.mkdir(parents=True, exist_ok=True)

    for res in results:
        for fp in res["final_figures"]:
            shutil.copy2(fp, fig_target / Path(fp).name)

        cdir = Path(res["checkpoint_dir"])
        # Copy checkpoint diagnostic plots so progress is visible from shared figures dir.
        for fp in sorted(cdir.glob(f"{res['model']}_trace_step*.png")):
            shutil.copy2(fp, fig_target / fp.name)
        for fp in sorted(cdir.glob(f"{res['model']}_corner_step*.png")):
            shutil.copy2(fp, fig_target / fp.name)

    if overlay is not None:
        shutil.copy2(overlay, fig_target / overlay.name)

    summary = {
        "run_dir": str(run_paths.run_dir),
        "stage1_summary": str(summary_path),
        "fit_data_npts": int(pf.size),
        "kmax_fit": float(args.kmax_fit),
        "parametrization": args.parametrization,
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

    print(f"Joint MCMC complete: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
