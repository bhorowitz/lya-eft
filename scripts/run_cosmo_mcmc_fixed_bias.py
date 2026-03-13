#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic

from lya_hybrid.config import load_config
from lya_hybrid.io import load_sherwood_flux_p3d
from lya_hybrid.linear_power import compute_linear_power_camb
from lya_hybrid.logging_utils import build_repro_metadata, init_run_dir, write_json
from lya_hybrid.model_hybrid import HybridToyModel, HybridToyParams
from lya_hybrid.model_ivanov import IvanovToyModel, IvanovToyParams

DEFAULT_P3D = Path("data/external/sherwood_p3d/data/flux_p3d/p3d_80_1024_9_0_384_1024_20_16_20.fits")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cosmology MCMC test with fixed biases from Stage-1 (sample As/omega_cdm or sigma8/omega_cdm)."
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--p3d", type=Path, default=DEFAULT_P3D)
    p.add_argument("--stage1-summary", type=Path, default=None)
    p.add_argument("--model", choices=["one_loop", "hybrid", "both"], default="both")
    p.add_argument("--parametrization", choices=["As", "sigma8"], default="As")
    p.add_argument("--kmax-fit", type=float, default=10.0)

    p.add_argument("--omega-min", type=float, default=0.20)
    p.add_argument("--omega-max", type=float, default=0.33)
    p.add_argument("--n-omega-grid", type=int, default=55)

    p.add_argument("--as-min", type=float, default=1.0e-9)
    p.add_argument("--as-max", type=float, default=3.5e-9)
    p.add_argument("--sigma8-min", type=float, default=0.60)
    p.add_argument("--sigma8-max", type=float, default=1.05)

    p.add_argument("--nwalkers", type=int, default=24)
    p.add_argument("--nsteps", type=int, default=700)
    p.add_argument("--burnin", type=int, default=220)
    p.add_argument("--thin", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--posterior-band-draws", type=int, default=180)
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


def one_loop_params_from_dict(d: dict[str, float], cfg) -> IvanovToyParams:
    return IvanovToyParams(
        b1=float(d.get("b1", cfg.ivanov_toy.b1)),
        b_eta=float(d.get("b_eta", cfg.ivanov_toy.b_eta)),
        c0=float(d.get("c0", cfg.ivanov_toy.c0)),
        c2=float(d.get("c2", cfg.ivanov_toy.c2)),
        c4=float(d.get("c4", cfg.ivanov_toy.c4)),
        loop_amp=float(d.get("loop_amp", cfg.ivanov_toy.loop_amp)),
        loop_mu2=float(d.get("loop_mu2", cfg.ivanov_toy.loop_mu2)),
        loop_mu4=float(d.get("loop_mu4", cfg.ivanov_toy.loop_mu4)),
        loop_k_nl=float(d.get("loop_k_nl", cfg.ivanov_toy.loop_k_nl)),
        stochastic=float(d.get("stochastic", cfg.ivanov_toy.stochastic)),
    )


def hybrid_params_from_dict(d: dict[str, float], cfg) -> HybridToyParams:
    return HybridToyParams(
        b_delta=float(d.get("b_delta", cfg.hybrid_toy.b_delta)),
        b_eta=float(d.get("b_eta", cfg.hybrid_toy.b_eta)),
        b_t=float(d.get("b_t", cfg.hybrid_toy.b_t)),
        c0=float(d.get("c0", cfg.hybrid_toy.c0)),
        c2=float(d.get("c2", cfg.hybrid_toy.c2)),
        c4=float(d.get("c4", cfg.hybrid_toy.c4)),
        loop_amp=float(d.get("loop_amp", cfg.hybrid_toy.loop_amp)),
        loop_mu2=float(d.get("loop_mu2", cfg.hybrid_toy.loop_mu2)),
        loop_mu4=float(d.get("loop_mu4", cfg.hybrid_toy.loop_mu4)),
        loop_k_nl=float(d.get("loop_k_nl", cfg.hybrid_toy.loop_k_nl)),
        sigma_th=float(d.get("sigma_th", cfg.hybrid_toy.sigma_th)),
        stochastic=float(d.get("stochastic", cfg.hybrid_toy.stochastic)),
    )


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
    fig, axes = plt.subplots(ndim, 1, figsize=(9, max(4.0, 2.3 * ndim)), sharex=True)
    if ndim == 1:
        axes = [axes]
    x = np.arange(nsteps)
    for i in range(ndim):
        ax = axes[i]
        for w in range(nwalkers):
            ax.plot(x, chain[:, w, i], color="#1f77b4", alpha=0.25, lw=0.6)
        ax.set_ylabel(names[i])
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("MCMC step")
    fig.suptitle("Trace Plot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_corner_like_plot(samples: np.ndarray, names: list[str], out_path: Path, max_scatter: int = 4500) -> None:
    ndim = samples.shape[1]
    fig, axes = plt.subplots(ndim, ndim, figsize=(2.5 * ndim + 1, 2.5 * ndim + 1))

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
                ax.scatter(subs[:, j], subs[:, i], s=2.0, alpha=0.2, color="#4c78a8", rasterized=True)
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
    fig.savefig(out_path, dpi=160)
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

        ax.fill_between(kx, r16, r84, color="#4c78a8", alpha=0.3, label="68% posterior")
        ax.plot(kx, r50, color="#1f77b4", lw=2.0, label="posterior median")
        ax.axhline(0.0, color="k", lw=1)
        ax.set_xscale("log")
        ax.set_title(fr"$\mu\in[{lo:.2f},{hi:.2f})$")
        ax.grid(alpha=0.2)
        ax.set_ylim(-0.9, 0.9)
        ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")

    axes[0].set_ylabel(r"$P_{model}/P_{data}-1$")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Residual Posterior Band: {model_label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
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
    na = min(4000, samples_a.shape[0])
    nb = min(4000, samples_b.shape[0])
    sa = samples_a[rng.choice(samples_a.shape[0], size=na, replace=False)]
    sb = samples_b[rng.choice(samples_b.shape[0], size=nb, replace=False)]

    plt.figure(figsize=(7, 6))
    plt.scatter(sa[:, 0], sa[:, 1], s=3, alpha=0.15, color="#1f77b4", label=label_a)
    plt.scatter(sb[:, 0], sb[:, 1], s=3, alpha=0.15, color="#d62728", label=label_b)

    ma = np.median(samples_a, axis=0)
    mb = np.median(samples_b, axis=0)
    plt.scatter([ma[0]], [ma[1]], s=90, marker="x", color="#1f77b4")
    plt.scatter([mb[0]], [mb[1]], s=90, marker="x", color="#d62728")

    plt.xlabel(names[0])
    plt.ylabel(names[1])
    plt.title("Cosmology Posterior Overlay")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def build_prediction_grids(
    *,
    model_names: list[str],
    biases: dict[str, dict[str, float]],
    cfg,
    kf: np.ndarray,
    muf: np.ndarray,
    omega_grid: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    pred_grids: dict[str, np.ndarray] = {name: np.zeros((omega_grid.size, kf.size)) for name in model_names}
    sigma8_grid = np.zeros(omega_grid.size)

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
        sigma8_grid[i] = float(lp.sigma8_0 if lp.sigma8_0 is not None else np.nan)

        if "one_loop" in model_names:
            p = one_loop_params_from_dict(biases["one_loop"], cfg)
            mdl = IvanovToyModel(lp.k_hmpc, lp.p_lin_h3mpc3, lp.f_growth)
            pred_grids["one_loop"][i, :] = mdl.evaluate_components(kf, muf, p)["total"]

        if "hybrid" in model_names:
            p = hybrid_params_from_dict(biases["hybrid"], cfg)
            mdl = HybridToyModel(lp.k_hmpc, lp.p_lin_h3mpc3, lp.f_growth, k_t=cfg.hybrid_toy.k_t)
            pred_grids["hybrid"][i, :] = mdl.evaluate_components(kf, muf, p)["total"]

    return pred_grids, sigma8_grid


def run_cosmo_chain(
    *,
    model_name: str,
    pred_grid_ref_as: np.ndarray,
    sigma8_grid_ref_as: np.ndarray,
    omega_grid: np.ndarray,
    pf: np.ndarray,
    sigma_data: np.ndarray,
    args,
    cfg,
    run_paths,
    seed: int,
) -> dict:
    invvar = 1.0 / (sigma_data**2)
    log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * sigma_data**2))

    if args.parametrization == "As":
        names = ["As", "omega_cdm"]
        lo = np.array([args.as_min, args.omega_min], dtype=float)
        hi = np.array([args.as_max, args.omega_max], dtype=float)
        theta0 = np.array([cfg.cosmology.As, cfg.cosmology.omega_cdm], dtype=float)
    else:
        names = ["sigma8", "omega_cdm"]
        lo = np.array([args.sigma8_min, args.omega_min], dtype=float)
        hi = np.array([args.sigma8_max, args.omega_max], dtype=float)
        sig8_init = float(interp_linear_grid(cfg.cosmology.omega_cdm, omega_grid, sigma8_grid_ref_as))
        theta0 = np.array([sig8_init, cfg.cosmology.omega_cdm], dtype=float)

    def log_prob(theta: np.ndarray) -> float:
        amp, omega = float(theta[0]), float(theta[1])
        if np.any(theta <= lo) or np.any(theta >= hi):
            return -np.inf

        try:
            pred_ref = interp_linear_grid(omega, omega_grid, pred_grid_ref_as)
        except ValueError:
            return -np.inf

        if args.parametrization == "As":
            scale = amp / cfg.cosmology.As
        else:
            try:
                sigma8_ref = float(interp_linear_grid(omega, omega_grid, sigma8_grid_ref_as))
            except ValueError:
                return -np.inf
            if not np.isfinite(sigma8_ref) or sigma8_ref <= 0.0:
                return -np.inf
            scale = (amp / sigma8_ref) ** 2

        pred = scale * pred_ref
        diff = pred - pf
        return log_norm - 0.5 * np.sum(diff * diff * invvar)

    nwalkers = max(args.nwalkers, 2 * len(names))
    rng = np.random.default_rng(seed)
    theta0 = np.clip(theta0, lo + 1.0e-10, hi - 1.0e-10)
    spread = 0.015 * (hi - lo)
    p0 = theta0 + rng.normal(scale=spread, size=(nwalkers, len(names)))
    p0 = np.clip(p0, lo + 1.0e-10, hi - 1.0e-10)

    sampler = emcee.EnsembleSampler(nwalkers, len(names), log_prob)
    state = sampler.run_mcmc(p0, args.burnin, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, args.nsteps, progress=False)

    chain = sampler.get_chain()
    flat = sampler.get_chain(flat=True, thin=args.thin)
    flat_lp = sampler.get_log_prob(flat=True, thin=args.thin)

    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
    best_idx = int(np.argmax(flat_lp))
    best_theta = flat[best_idx]

    # best-fit chi2/dof in fit data space
    amp_best, omega_best = float(best_theta[0]), float(best_theta[1])
    pred_ref_best = interp_linear_grid(omega_best, omega_grid, pred_grid_ref_as)
    if args.parametrization == "As":
        scale_best = amp_best / cfg.cosmology.As
    else:
        s8_ref_best = float(interp_linear_grid(omega_best, omega_grid, sigma8_grid_ref_as))
        scale_best = (amp_best / s8_ref_best) ** 2
    pred_best = scale_best * pred_ref_best
    chi2 = float(np.sum(((pred_best - pf) / sigma_data) ** 2))
    dof = max(int(pf.size - len(names)), 1)

    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_out = [float(x) for x in tau]
    except Exception:
        tau_out = []

    arr_path = run_paths.arrays_dir / f"cosmo_mcmc_fixed_bias_{model_name}.npz"
    np.savez(
        arr_path,
        chain=chain,
        flat_samples=flat,
        flat_log_prob=flat_lp,
        q16=q16,
        q50=q50,
        q84=q84,
        best_theta=best_theta,
        param_names=np.array(names, dtype=object),
    )

    f_trace = run_paths.figures_dir / f"71_{model_name}_cosmo_trace.png"
    make_trace_plot(chain, names, f_trace)
    f_corner = run_paths.figures_dir / f"72_{model_name}_cosmo_corner_like.png"
    make_corner_like_plot(flat, names, f_corner)

    # Residual posterior band
    ndraw = min(args.posterior_band_draws, flat.shape[0])
    take = rng.choice(flat.shape[0], size=ndraw, replace=False)
    pred_draws = []
    for th in flat[take]:
        amp_i, om_i = float(th[0]), float(th[1])
        pref_i = interp_linear_grid(om_i, omega_grid, pred_grid_ref_as)
        if args.parametrization == "As":
            scale_i = amp_i / cfg.cosmology.As
        else:
            s8_ref_i = float(interp_linear_grid(om_i, omega_grid, sigma8_grid_ref_as))
            scale_i = (amp_i / s8_ref_i) ** 2
        pred_draws.append(scale_i * pref_i)
    pred_draws = np.asarray(pred_draws)

    # Need k and mu for residual-band plot (saved externally in summary call)
    return {
        "model": model_name,
        "param_names": names,
        "bounds": {"lower": lo.tolist(), "upper": hi.tolist()},
        "q16": q16.tolist(),
        "q50": q50.tolist(),
        "q84": q84.tolist(),
        "best_theta": best_theta.tolist(),
        "fit_chi2": chi2,
        "fit_chi2_dof": float(chi2 / dof),
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
        "autocorr_time": tau_out,
        "nwalkers": int(nwalkers),
        "nsteps": int(args.nsteps),
        "burnin": int(args.burnin),
        "thin": int(args.thin),
        "arrays": str(arr_path),
        "figures": [str(f_trace), str(f_corner)],
        "posterior_prediction_draws": pred_draws,
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    summary_path = latest_stage1_summary() if args.stage1_summary is None else args.stage1_summary
    best_biases = load_stage1_best_biases(summary_path)

    requested_models = [args.model] if args.model in {"one_loop", "hybrid"} else ["one_loop", "hybrid"]
    for m in requested_models:
        if m not in best_biases:
            raise ValueError(
                f"Stage-1 summary {summary_path} does not contain best-fit bias block for model='{m}'."
            )

    run_paths = init_run_dir(cfg.run.output_root, tag="cosmo_mcmc_fixed_bias")
    meta = build_repro_metadata(args.config)
    meta.update(
        {
            "p3d_path": str(args.p3d),
            "stage1_summary": str(summary_path),
            "requested_models": requested_models,
            "parametrization": args.parametrization,
            "kmax_fit": float(args.kmax_fit),
            "omega_grid": [float(args.omega_min), float(args.omega_max), int(args.n_omega_grid)],
            "mcmc": {
                "nwalkers": int(args.nwalkers),
                "nsteps": int(args.nsteps),
                "burnin": int(args.burnin),
                "thin": int(args.thin),
                "seed": int(args.seed),
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
    pred_grids, sigma8_grid = build_prediction_grids(
        model_names=requested_models,
        biases=best_biases,
        cfg=cfg,
        kf=kf,
        muf=muf,
        omega_grid=omega_grid,
    )

    results = []
    flat_by_model: dict[str, np.ndarray] = {}

    for i, m in enumerate(requested_models):
        out = run_cosmo_chain(
            model_name=m,
            pred_grid_ref_as=pred_grids[m],
            sigma8_grid_ref_as=sigma8_grid,
            omega_grid=omega_grid,
            pf=pf,
            sigma_data=sigma_data,
            args=args,
            cfg=cfg,
            run_paths=run_paths,
            seed=int(args.seed + 19 * i),
        )

        # Residual band plot (needs fit-space k,mu,p)
        f_band = run_paths.figures_dir / f"73_{m}_cosmo_residual_band.png"
        make_residual_band_plot(
            k_all=kf,
            mu_all=muf,
            p_data=pf,
            draw_predictions=out["posterior_prediction_draws"],
            model_label=f"{m} (fixed bias)",
            out_path=f_band,
        )
        out["figures"].append(str(f_band))

        # Remove large array from summary blob; keep on disk in chain file only.
        del out["posterior_prediction_draws"]
        results.append(out)

        arr = np.load(Path(out["arrays"]))
        flat_by_model[m] = arr["flat_samples"]

    # Overlay posterior comparison if both models were sampled.
    overlay_fig = None
    if set(requested_models) == {"one_loop", "hybrid"}:
        overlay_fig = run_paths.figures_dir / "74_cosmo_posterior_overlay.png"
        names = results[0]["param_names"]
        make_overlay_plot(
            samples_a=flat_by_model["one_loop"],
            samples_b=flat_by_model["hybrid"],
            names=names,
            label_a="one-loop",
            label_b="hybrid",
            out_path=overlay_fig,
        )

    fig_target = Path("results/figures")
    fig_target.mkdir(parents=True, exist_ok=True)
    for res in results:
        for fp in res["figures"]:
            shutil.copy2(fp, fig_target / Path(fp).name)
    if overlay_fig is not None:
        shutil.copy2(overlay_fig, fig_target / overlay_fig.name)

    summary = {
        "run_dir": str(run_paths.run_dir),
        "stage1_summary": str(summary_path),
        "fit_data_npts": int(pf.size),
        "kmax_fit": float(args.kmax_fit),
        "parametrization": args.parametrization,
        "omega_grid": {
            "min": float(args.omega_min),
            "max": float(args.omega_max),
            "n": int(args.n_omega_grid),
        },
        "requested_models": requested_models,
        "models": results,
        "overlay_figure": (str(overlay_fig) if overlay_fig is not None else None),
    }
    write_json(run_paths.logs_dir / "summary.json", summary)

    print(f"Cosmology MCMC (fixed bias) complete: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
