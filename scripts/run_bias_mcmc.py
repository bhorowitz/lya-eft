#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict
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
from lya_hybrid.model_hybrid import HybridToyModel, HybridToyParams
from lya_hybrid.model_ivanov import IvanovToyModel, IvanovToyParams

DEFAULT_P3D = Path("data/external/sherwood_p3d/data/flux_p3d/p3d_80_1024_9_0_384_1024_20_16_20.fits")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-1 bias-only MCMC for one-loop and/or hybrid models.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--p3d", type=Path, default=DEFAULT_P3D)
    p.add_argument("--model", choices=["one_loop", "hybrid", "both"], default="both")
    p.add_argument("--kmax-fit", type=float, default=None)
    p.add_argument("--nwalkers", type=int, default=None)
    p.add_argument("--nsteps", type=int, default=None)
    p.add_argument("--burnin", type=int, default=None)
    p.add_argument("--thin", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--hybrid-tight-priors", action="store_true")
    return p.parse_args()


def pseudo_sigma(p: np.ndarray, counts: np.ndarray, sigma_frac: float, sigma_floor: float) -> np.ndarray:
    return sigma_frac * np.maximum(np.abs(p), 1.0e-8) + np.maximum(1.0 / np.sqrt(np.maximum(counts, 1.0)), sigma_floor)


def binned_curve(k: np.ndarray, y: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yb, edges, _ = binned_statistic(k, y, statistic="median", bins=bins)
    xc = np.sqrt(edges[:-1] * edges[1:])
    keep = np.isfinite(yb)
    return xc[keep], yb[keep]


def make_trace_plot(chain: np.ndarray, names: list[str], out_path: Path) -> None:
    # chain shape: (nsteps, nwalkers, ndim)
    nsteps, nwalkers, ndim = chain.shape
    fig, axes = plt.subplots(ndim, 1, figsize=(9, max(2.0 * ndim, 4.5)), sharex=True)
    if ndim == 1:
        axes = [axes]
    x = np.arange(nsteps)
    for i in range(ndim):
        ax = axes[i]
        for w in range(nwalkers):
            ax.plot(x, chain[:, w, i], color="#1f77b4", alpha=0.15, lw=0.8)
        ax.set_ylabel(names[i])
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("MCMC step")
    fig.suptitle("Trace Plot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
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
                ax.scatter(subs[:, j], subs[:, i], s=2, alpha=0.18, color="#4c78a8", rasterized=True)
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
    # draw_predictions shape: (ndraw, npts)
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

        # Data-space median curve for reference
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
        ax.set_ylim(-0.8, 0.8)
        ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")

    axes[0].set_ylabel(r"$P_{model}/P_{data}-1$")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Residual Posterior Band: {model_label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_one_loop_mcmc(
    *,
    model: IvanovToyModel,
    kf: np.ndarray,
    muf: np.ndarray,
    pf: np.ndarray,
    sigma: np.ndarray,
    cfg,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    thin: int,
    seed: int,
    run_paths,
) -> dict:
    names = ["b1", "b_eta", "c0", "c2", "c4", "loop_amp"]
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
        p = unpack(theta)
        pred = model.evaluate_components(kf, muf, p)["total"]
        return (pred - pf) / sigma

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
    x0 = np.clip(x0, lo + 1.0e-8, hi - 1.0e-8)
    fit = least_squares(
        residual,
        x0=x0,
        bounds=(lo, hi),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=12000,
    )
    x_map = fit.x.copy()

    invvar = 1.0 / (sigma**2)
    log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * sigma**2))

    def log_prob(theta: np.ndarray) -> float:
        if np.any(theta <= lo) or np.any(theta >= hi):
            return -np.inf
        p = unpack(theta)
        pred = model.evaluate_components(kf, muf, p)["total"]
        diff = pred - pf
        return log_norm - 0.5 * np.sum(diff * diff * invvar)

    if nwalkers < 2 * len(names):
        nwalkers = 2 * len(names)

    rng = np.random.default_rng(seed)
    spread = cfg.mcmc.init_jitter_frac * (hi - lo)
    p0 = x_map + rng.normal(scale=spread, size=(nwalkers, len(names)))
    p0 = np.clip(p0, lo + 1.0e-7, hi - 1.0e-7)

    sampler = emcee.EnsembleSampler(nwalkers, len(names), log_prob)
    state = sampler.run_mcmc(p0, burnin, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, nsteps, progress=False)

    chain = sampler.get_chain()  # (nsteps, nwalkers, ndim)
    flat = sampler.get_chain(flat=True, thin=thin)
    flat_lp = sampler.get_log_prob(flat=True, thin=thin)

    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
    best_idx = int(np.argmax(flat_lp))
    best_theta = flat[best_idx]

    p_best = unpack(best_theta)
    pred_best = model.evaluate_components(kf, muf, p_best)["total"]
    chi2 = float(np.sum(((pred_best - pf) / sigma) ** 2))
    dof = max(int(kf.size - len(names)), 1)

    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_out = [float(x) for x in tau]
    except Exception:
        tau_out = []

    arr_path = run_paths.arrays_dir / "bias_mcmc_one_loop_chain.npz"
    np.savez(
        arr_path,
        chain=chain,
        flat_samples=flat,
        flat_log_prob=flat_lp,
        map_theta=x_map,
        best_theta=best_theta,
        q16=q16,
        q50=q50,
        q84=q84,
    )

    f_trace = run_paths.figures_dir / "51_oneloop_trace.png"
    make_trace_plot(chain, names, f_trace)
    f_corner = run_paths.figures_dir / "52_oneloop_corner_like.png"
    make_corner_like_plot(flat, names, f_corner)

    # Posterior residual band on fit points.
    ndraw = min(cfg.mcmc.posterior_band_draws, flat.shape[0])
    choose = rng.choice(flat.shape[0], size=ndraw, replace=False)
    draws = []
    for th in flat[choose]:
        p = unpack(th)
        draws.append(model.evaluate_components(kf, muf, p)["total"])
    draws_arr = np.asarray(draws)
    f_band = run_paths.figures_dir / "53_oneloop_residual_band.png"
    make_residual_band_plot(
        k_all=kf,
        mu_all=muf,
        p_data=pf,
        draw_predictions=draws_arr,
        model_label="One-loop",
        out_path=f_band,
    )

    return {
        "model": "one_loop",
        "param_names": names,
        "bounds": {"lower": lo.tolist(), "upper": hi.tolist()},
        "map_theta": x_map.tolist(),
        "best_theta": best_theta.tolist(),
        "q16": q16.tolist(),
        "q50": q50.tolist(),
        "q84": q84.tolist(),
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
        "autocorr_time": tau_out,
        "fit_chi2": chi2,
        "fit_chi2_dof": float(chi2 / dof),
        "n_data": int(kf.size),
        "n_dim": len(names),
        "nwalkers": int(nwalkers),
        "nsteps": int(nsteps),
        "burnin": int(burnin),
        "thin": int(thin),
        "best_params_dict": asdict(p_best),
        "arrays": str(arr_path),
        "figures": [str(f_trace), str(f_corner), str(f_band)],
    }


def run_hybrid_mcmc(
    *,
    model: HybridToyModel,
    kf: np.ndarray,
    muf: np.ndarray,
    pf: np.ndarray,
    sigma: np.ndarray,
    cfg,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    thin: int,
    seed: int,
    run_paths,
    use_tight_priors: bool,
) -> dict:
    names = ["b_delta", "b_eta", "b_t", "c0", "c2", "c4", "loop_amp", "sigma_th"]
    lo = np.array([-1.0, -2.0, -0.25, -0.10, -0.10, -0.10, -0.30, 0.0], dtype=float)
    hi = np.array([0.2, 2.0, 0.25, 0.10, 0.10, 0.10, 0.30, 0.25], dtype=float)

    def unpack(theta: np.ndarray) -> HybridToyParams:
        return HybridToyParams(
            b_delta=float(theta[0]),
            b_eta=float(theta[1]),
            b_t=float(theta[2]),
            c0=float(theta[3]),
            c2=float(theta[4]),
            c4=float(theta[5]),
            loop_amp=float(theta[6]),
            loop_mu2=float(cfg.hybrid_toy.loop_mu2),
            loop_mu4=float(cfg.hybrid_toy.loop_mu4),
            loop_k_nl=float(cfg.hybrid_toy.loop_k_nl),
            sigma_th=float(theta[7]),
            stochastic=float(cfg.hybrid_toy.stochastic),
        )

    def prior_penalty(theta: np.ndarray) -> float:
        if not use_tight_priors:
            return 0.0
        # Gaussian priors to stabilize high-k directions.
        return -0.5 * (
            (theta[2] / 0.08) ** 2
            + (theta[3] / 0.06) ** 2
            + (theta[4] / 0.06) ** 2
            + (theta[5] / 0.06) ** 2
            + ((theta[6] - 0.06) / 0.10) ** 2
            + ((theta[7] - 0.08) / 0.05) ** 2
        )

    def residual(theta: np.ndarray) -> np.ndarray:
        p = unpack(theta)
        pred = model.evaluate_components(kf, muf, p)["total"]
        res = (pred - pf) / sigma
        if not use_tight_priors:
            return res
        prior = np.array(
            [
                theta[2] / 0.08,
                theta[3] / 0.06,
                theta[4] / 0.06,
                theta[5] / 0.06,
                (theta[6] - 0.06) / 0.10,
                (theta[7] - 0.08) / 0.05,
            ]
        )
        return np.concatenate([res, prior])

    x0 = np.array(
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
    x_map = fit.x.copy()

    invvar = 1.0 / (sigma**2)
    log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * sigma**2))

    def log_prob(theta: np.ndarray) -> float:
        if np.any(theta <= lo) or np.any(theta >= hi):
            return -np.inf
        lp = prior_penalty(theta)
        p = unpack(theta)
        pred = model.evaluate_components(kf, muf, p)["total"]
        diff = pred - pf
        return lp + log_norm - 0.5 * np.sum(diff * diff * invvar)

    if nwalkers < 2 * len(names):
        nwalkers = 2 * len(names)

    rng = np.random.default_rng(seed)
    spread = cfg.mcmc.init_jitter_frac * (hi - lo)
    p0 = x_map + rng.normal(scale=spread, size=(nwalkers, len(names)))
    p0 = np.clip(p0, lo + 1.0e-7, hi - 1.0e-7)

    sampler = emcee.EnsembleSampler(nwalkers, len(names), log_prob)
    state = sampler.run_mcmc(p0, burnin, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, nsteps, progress=False)

    chain = sampler.get_chain()
    flat = sampler.get_chain(flat=True, thin=thin)
    flat_lp = sampler.get_log_prob(flat=True, thin=thin)

    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
    best_idx = int(np.argmax(flat_lp))
    best_theta = flat[best_idx]

    p_best = unpack(best_theta)
    pred_best = model.evaluate_components(kf, muf, p_best)["total"]
    chi2 = float(np.sum(((pred_best - pf) / sigma) ** 2))
    dof = max(int(kf.size - len(names)), 1)

    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_out = [float(x) for x in tau]
    except Exception:
        tau_out = []

    arr_path = run_paths.arrays_dir / "bias_mcmc_hybrid_chain.npz"
    np.savez(
        arr_path,
        chain=chain,
        flat_samples=flat,
        flat_log_prob=flat_lp,
        map_theta=x_map,
        best_theta=best_theta,
        q16=q16,
        q50=q50,
        q84=q84,
    )

    f_trace = run_paths.figures_dir / "61_hybrid_trace.png"
    make_trace_plot(chain, names, f_trace)
    f_corner = run_paths.figures_dir / "62_hybrid_corner_like.png"
    make_corner_like_plot(flat, names, f_corner)

    ndraw = min(cfg.mcmc.posterior_band_draws, flat.shape[0])
    choose = rng.choice(flat.shape[0], size=ndraw, replace=False)
    draws = []
    for th in flat[choose]:
        p = unpack(th)
        draws.append(model.evaluate_components(kf, muf, p)["total"])
    draws_arr = np.asarray(draws)
    f_band = run_paths.figures_dir / "63_hybrid_residual_band.png"
    make_residual_band_plot(
        k_all=kf,
        mu_all=muf,
        p_data=pf,
        draw_predictions=draws_arr,
        model_label="Hybrid",
        out_path=f_band,
    )

    return {
        "model": "hybrid",
        "tight_priors": bool(use_tight_priors),
        "param_names": names,
        "bounds": {"lower": lo.tolist(), "upper": hi.tolist()},
        "map_theta": x_map.tolist(),
        "best_theta": best_theta.tolist(),
        "q16": q16.tolist(),
        "q50": q50.tolist(),
        "q84": q84.tolist(),
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
        "autocorr_time": tau_out,
        "fit_chi2": chi2,
        "fit_chi2_dof": float(chi2 / dof),
        "n_data": int(kf.size),
        "n_dim": len(names),
        "nwalkers": int(nwalkers),
        "nsteps": int(nsteps),
        "burnin": int(burnin),
        "thin": int(thin),
        "best_params_dict": asdict(p_best),
        "arrays": str(arr_path),
        "figures": [str(f_trace), str(f_corner), str(f_band)],
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    kmax_fit = float(cfg.fit.kmax_fit if args.kmax_fit is None else args.kmax_fit)
    nwalkers = int(cfg.mcmc.nwalkers if args.nwalkers is None else args.nwalkers)
    nsteps = int(cfg.mcmc.nsteps if args.nsteps is None else args.nsteps)
    burnin = int(cfg.mcmc.burnin if args.burnin is None else args.burnin)
    thin = int(cfg.mcmc.thin if args.thin is None else args.thin)
    seed = int(cfg.mcmc.seed if args.seed is None else args.seed)

    run_paths = init_run_dir(cfg.run.output_root, tag="bias_mcmc_stage1")
    meta = build_repro_metadata(args.config)
    meta.update(
        {
            "p3d_path": str(args.p3d),
            "kmax_fit": kmax_fit,
            "model": args.model,
            "nwalkers": nwalkers,
            "nsteps": nsteps,
            "burnin": burnin,
            "thin": thin,
            "seed": seed,
            "hybrid_tight_priors": bool(args.hybrid_tight_priors),
        }
    )
    write_json(run_paths.logs_dir / "repro_metadata.json", meta)

    p3d_data = load_sherwood_flux_p3d(args.p3d)
    k_all, mu_all, p_all, counts_all = p3d_data.flatten_valid()
    mask_fit = (k_all >= cfg.fit.kmin_fit) & (k_all <= kmax_fit)
    kf, muf, pf, cf = k_all[mask_fit], mu_all[mask_fit], p_all[mask_fit], counts_all[mask_fit]
    sigma = pseudo_sigma(pf, cf, sigma_frac=cfg.fit.sigma_frac, sigma_floor=cfg.fit.sigma_floor)

    lp = compute_linear_power_camb(
        h=cfg.cosmology.h,
        omega_b=cfg.cosmology.omega_b,
        omega_cdm=cfg.cosmology.omega_cdm,
        ns=cfg.cosmology.ns,
        As=cfg.cosmology.As,
        z=cfg.cosmology.z,
        kmin=cfg.k_grid.kmin,
        kmax=max(cfg.k_grid.kmax, kmax_fit),
        nk=cfg.k_grid.nk,
    )

    one_loop_model = IvanovToyModel(lp.k_hmpc, lp.p_lin_h3mpc3, lp.f_growth)
    hybrid_model = HybridToyModel(lp.k_hmpc, lp.p_lin_h3mpc3, lp.f_growth, k_t=cfg.hybrid_toy.k_t)

    results = []
    if args.model in {"one_loop", "both"}:
        results.append(
            run_one_loop_mcmc(
                model=one_loop_model,
                kf=kf,
                muf=muf,
                pf=pf,
                sigma=sigma,
                cfg=cfg,
                nwalkers=nwalkers,
                nsteps=nsteps,
                burnin=burnin,
                thin=thin,
                seed=seed,
                run_paths=run_paths,
            )
        )

    if args.model in {"hybrid", "both"}:
        results.append(
            run_hybrid_mcmc(
                model=hybrid_model,
                kf=kf,
                muf=muf,
                pf=pf,
                sigma=sigma,
                cfg=cfg,
                nwalkers=nwalkers,
                nsteps=nsteps,
                burnin=burnin,
                thin=thin,
                seed=seed + 7,
                run_paths=run_paths,
                use_tight_priors=bool(args.hybrid_tight_priors),
            )
        )

    fig_target = Path("results/figures")
    fig_target.mkdir(parents=True, exist_ok=True)
    for res in results:
        for fp in res["figures"]:
            shutil.copy2(fp, fig_target / Path(fp).name)

    summary = {
        "run_dir": str(run_paths.run_dir),
        "kmax_fit": kmax_fit,
        "n_data_fit": int(kf.size),
        "models": results,
    }
    write_json(run_paths.logs_dir / "summary.json", summary)

    print(f"Bias MCMC stage-1 complete: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
