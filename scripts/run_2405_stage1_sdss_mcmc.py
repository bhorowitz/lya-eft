#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lya_hybrid.config import load_config
from lya_hybrid.linear_power import compute_linear_power_camb
from lya_hybrid.logging_utils import build_repro_metadata, init_run_dir, write_json
from lya_hybrid.model_hybrid import HybridToyModel
from lya_hybrid.model_ivanov import IvanovToyModel
from lya_hybrid.sdss_p1d import load_chabanier2019_blocks, load_eboss_mock_blocks
from scripts.run_2405_stage1_sdss_baseline import (
    DEFAULT_MOCK_DIR,
    DEFAULT_SDSS_DIR,
    THEORY_CHOICES,
    FitBlock,
    build_fit_blocks,
    fit_mock_counterterm_relations,
    latest_stage1_prior_csv,
    load_prior_relations,
    make_fit_plot,
    p1d_prediction,
    run_sdss_fit,
)

_POOL_LIKELIHOOD = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Full SDSS Stage-1 MCMC for the 2405-style baseline likelihood. "
            "Samples the full parameter set for each mode and makes Figure-S3-style corner contours."
        )
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--theory", choices=THEORY_CHOICES, default="hybrid")
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
    p.add_argument("--counterterm-mode", choices=["proxy", "paper_baseline", "paper_rescaled"], default="proxy")
    p.add_argument("--apply-paper-systematics", action="store_true")
    p.add_argument("--b-eta-min", type=float, default=-2.0)
    p.add_argument("--b-eta-max", type=float, default=2.0)
    p.add_argument("--map-summary", type=Path, default=None)
    p.add_argument("--map-n-starts", type=int, default=6)
    p.add_argument("--nwalkers", type=int, default=48)
    p.add_argument("--nsteps", type=int, default=1200)
    p.add_argument("--burnin", type=int, default=400)
    p.add_argument("--thin", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=400)
    p.add_argument("--posterior-band-draws", type=int, default=160)
    p.add_argument("--max-nfev", type=int, default=2600)
    p.add_argument("--n-procs", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260313)
    return p.parse_args()


def _pool_log_prob(theta: np.ndarray) -> float:
    global _POOL_LIKELIHOOD
    if _POOL_LIKELIHOOD is None:
        raise RuntimeError("Pool likelihood state is not initialized.")
    return _POOL_LIKELIHOOD.log_prob(theta)


@dataclass
class ParamLayout:
    names: list[str]
    lower: np.ndarray
    upper: np.ndarray
    display_idx: np.ndarray
    display_names: list[str]


def build_param_layout(
    *,
    theory: Literal["one_loop", "hybrid"],
    zvals: list[float],
    mode: str,
    b_eta_min: float,
    b_eta_max: float,
) -> ParamLayout:
    nz = len(zvals)
    use_offsets = mode == "conservative"
    bias_label = "b1" if theory == "one_loop" else "b_delta"

    names = ["sigma8"] + [f"{bias_label}_z{z:.1f}" for z in zvals] + [f"b_eta_z{z:.1f}" for z in zvals]
    if use_offsets:
        names += ["d_c0_3d", "d_c2_3d", "d_c4_3d", "d_loop_amp", "d_C0_1d", "d_C2_1d"]
        if theory == "hybrid":
            names += ["d_b_t", "d_sigma_th"]

    lo = np.full(len(names), -np.inf, dtype=float)
    hi = np.full(len(names), np.inf, dtype=float)
    lo[0], hi[0] = 0.6, 1.1
    lo[1 : 1 + nz], hi[1 : 1 + nz] = -1.5, 0.2
    lo[1 + nz : 1 + 2 * nz], hi[1 + nz : 1 + 2 * nz] = float(b_eta_min), float(b_eta_max)
    if use_offsets:
        if theory == "one_loop":
            lo[-6:], hi[-6:] = -1.5, 1.5
        else:
            lo[-8:-2], hi[-8:-2] = -1.5, 1.5
            lo[-2:], hi[-2:] = [-0.5, -0.2], [0.5, 0.2]

    display_idx = np.arange(1 + 2 * nz, dtype=int)
    display_names = names[: 1 + 2 * nz]
    return ParamLayout(names=names, lower=lo, upper=hi, display_idx=display_idx, display_names=display_names)


def load_map_thetas(summary_path: Path) -> dict[str, np.ndarray]:
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    out: dict[str, np.ndarray] = {}
    for row in payload.get("fit_results", []):
        mode = str(row.get("mode"))
        best_theta = row.get("best_theta")
        if isinstance(best_theta, list):
            out[mode] = np.asarray(best_theta, dtype=float)
    if not out:
        raise ValueError(f"No fit_results/best_theta entries found in {summary_path}")
    return out


def load_calibration_from_sdss_summary(
    summary_path: Path,
) -> tuple[tuple[float, float], tuple[float, float], dict[str, float]] | None:
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    c0 = payload.get("c0_rel_proxy")
    c2 = payload.get("c2_rel_proxy")
    conservative_sigmas = payload.get("conservative_sigmas")
    if not isinstance(c0, dict) or not isinstance(c2, dict) or not isinstance(conservative_sigmas, dict):
        return None

    c0_rel = (float(c0["A"]), float(c0["B"]))
    c2_rel = (float(c2["A"]), float(c2["B"]))
    sigmas = {str(k): float(v) for k, v in conservative_sigmas.items()}
    return c0_rel, c2_rel, sigmas


class SDSSLikelihood:
    def __init__(
        self,
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
        cfg,
        args,
    ) -> None:
        self.theory = theory
        self.mode = mode
        self.sdss_blocks = sdss_blocks
        self.models_by_z = models_by_z
        self.sigma8_ref = float(sigma8_ref)
        self.prior3d = prior3d
        self.c0_rel = c0_rel
        self.c2_rel = c2_rel
        self.conservative_sigmas = conservative_sigmas
        self.cfg = cfg
        self.args = args
        self.zvals = [b.z for b in sdss_blocks]
        self.nz = len(sdss_blocks)
        self.use_offsets = mode == "conservative"
        self.layout = build_param_layout(
            theory=theory,
            zvals=self.zvals,
            mode=mode,
            b_eta_min=float(args.b_eta_min),
            b_eta_max=float(args.b_eta_max),
        )

    def unpack(self, theta: np.ndarray) -> tuple[float, list[float], list[float], dict[str, float]]:
        sigma8 = float(theta[0])
        b1 = [float(x) for x in theta[1 : 1 + self.nz]]
        b_eta = [float(x) for x in theta[1 + self.nz : 1 + 2 * self.nz]]
        offsets: dict[str, float] = {}
        if self.use_offsets:
            if self.theory == "one_loop":
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

    def residual(self, theta: np.ndarray) -> np.ndarray:
        sigma8, b1_list, beta_list, offsets = self.unpack(theta)
        chunks = []
        for i, block in enumerate(self.sdss_blocks):
            pred = p1d_prediction(
                theory=self.theory,
                model=self.models_by_z[block.z],
                z=float(block.z),
                kpar_hmpc=block.k_hmpc,
                b1=b1_list[i],
                b_eta=beta_list[i],
                sigma8=sigma8,
                sigma8_ref=self.sigma8_ref,
                prior3d=self.prior3d,
                c0_1d_rel=self.c0_rel,
                c2_1d_rel=self.c2_rel,
                apply_paper_systematics=bool(self.args.apply_paper_systematics),
                kmax_proj=float(self.args.kmax_proj),
                nint_proj=int(self.args.nint_proj),
                cfg=self.cfg,
                offsets=offsets,
            )
            chunks.append(np.linalg.solve(block.chol, pred - block.p_hmpc))

        if self.use_offsets:
            chunks.append(np.array([offsets["c0_3d"] / self.conservative_sigmas["c0_3d"]], dtype=float))
            chunks.append(np.array([offsets["c2_3d"] / self.conservative_sigmas["c2_3d"]], dtype=float))
            chunks.append(np.array([offsets["c4_3d"] / self.conservative_sigmas["c4_3d"]], dtype=float))
            chunks.append(np.array([offsets["loop_amp"] / self.conservative_sigmas["loop_amp"]], dtype=float))
            chunks.append(np.array([offsets["C0_1d"] / self.conservative_sigmas["C0_1d"]], dtype=float))
            chunks.append(np.array([offsets["C2_1d"] / self.conservative_sigmas["C2_1d"]], dtype=float))
            if self.theory == "hybrid":
                chunks.append(np.array([offsets["b_t"] / self.conservative_sigmas["b_t"]], dtype=float))
                chunks.append(np.array([offsets["sigma_th"] / self.conservative_sigmas["sigma_th"]], dtype=float))
        return np.concatenate(chunks)

    def log_prob(self, theta: np.ndarray) -> float:
        if np.any(theta <= self.layout.lower) or np.any(theta >= self.layout.upper):
            return -np.inf
        res = self.residual(theta)
        return -0.5 * float(np.sum(res * res))

    def predict_by_z(self, theta: np.ndarray) -> dict[float, np.ndarray]:
        sigma8, b1_list, beta_list, offsets = self.unpack(theta)
        out: dict[float, np.ndarray] = {}
        for i, block in enumerate(self.sdss_blocks):
            out[block.z] = p1d_prediction(
                theory=self.theory,
                model=self.models_by_z[block.z],
                z=float(block.z),
                kpar_hmpc=block.k_hmpc,
                b1=b1_list[i],
                b_eta=beta_list[i],
                sigma8=sigma8,
                sigma8_ref=self.sigma8_ref,
                prior3d=self.prior3d,
                c0_1d_rel=self.c0_rel,
                c2_1d_rel=self.c2_rel,
                apply_paper_systematics=bool(self.args.apply_paper_systematics),
                kmax_proj=float(self.args.kmax_proj),
                nint_proj=int(self.args.nint_proj),
                cfg=self.cfg,
                offsets=offsets,
            )
        return out

    def data_chi2(self, theta: np.ndarray) -> float:
        sigma8, b1_list, beta_list, offsets = self.unpack(theta)
        chunks = []
        for i, block in enumerate(self.sdss_blocks):
            pred = p1d_prediction(
                theory=self.theory,
                model=self.models_by_z[block.z],
                z=float(block.z),
                kpar_hmpc=block.k_hmpc,
                b1=b1_list[i],
                b_eta=beta_list[i],
                sigma8=sigma8,
                sigma8_ref=self.sigma8_ref,
                prior3d=self.prior3d,
                c0_1d_rel=self.c0_rel,
                c2_1d_rel=self.c2_rel,
                apply_paper_systematics=bool(self.args.apply_paper_systematics),
                kmax_proj=float(self.args.kmax_proj),
                nint_proj=int(self.args.nint_proj),
                cfg=self.cfg,
                offsets=offsets,
            )
            chunks.append(np.linalg.solve(block.chol, pred - block.p_hmpc))
        data_res = np.concatenate(chunks)
        return float(np.sum(data_res * data_res))


def make_trace_plot(chain: np.ndarray, names: list[str], out_path: Path, title: str) -> None:
    nsteps, nwalkers, ndim = chain.shape
    fig, axes = plt.subplots(ndim, 1, figsize=(10, max(4.0, 1.7 * ndim)), sharex=True)
    if ndim == 1:
        axes = [axes]
    x = np.arange(nsteps)
    for i in range(ndim):
        ax = axes[i]
        for w in range(nwalkers):
            ax.plot(x, chain[:, w, i], color="#1f77b4", alpha=0.16, lw=0.6)
        ax.set_ylabel(names[i], fontsize=8)
        ax.grid(alpha=0.18)
    axes[-1].set_xlabel("MCMC step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def contour_levels_from_hist(hist: np.ndarray, levels: tuple[float, ...] = (0.393, 0.865)) -> list[float]:
    h = np.asarray(hist, dtype=float)
    if not np.any(np.isfinite(h)) or np.nanmax(h) <= 0.0:
        return []
    flat = h[np.isfinite(h)].ravel()
    flat = flat[flat > 0]
    if flat.size == 0:
        return []
    flat = np.sort(flat)[::-1]
    cdf = np.cumsum(flat)
    cdf /= cdf[-1]
    out = []
    for lev in levels:
        idx = int(np.searchsorted(cdf, lev, side="left"))
        idx = min(max(idx, 0), flat.size - 1)
        out.append(float(flat[idx]))
    return out


def plot_hist_contours(ax, x: np.ndarray, y: np.ndarray, color: str, bins: int = 35, smooth: float = 1.0) -> None:
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    if smooth > 0:
        h = gaussian_filter(h, smooth)
    levels = contour_levels_from_hist(h)
    if not levels:
        return
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    # contour wants strictly increasing levels
    levels = sorted(set(levels))
    if len(levels) == 1:
        ax.contour(xc, yc, h.T, levels=levels, colors=[color], linewidths=1.2)
    else:
        ax.contour(xc, yc, h.T, levels=levels, colors=[color] * len(levels), linewidths=1.2)


def make_corner_contour_plot(
    samples: np.ndarray,
    names: list[str],
    out_path: Path,
    *,
    color: str,
    title: str,
    max_points: int = 25000,
) -> None:
    rng = np.random.default_rng(0)
    ndim = samples.shape[1]
    if samples.shape[0] > max_points:
        idx = rng.choice(samples.shape[0], size=max_points, replace=False)
        samples = samples[idx]

    fig, axes = plt.subplots(ndim, ndim, figsize=(1.9 * ndim + 2.0, 1.9 * ndim + 2.0))
    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(samples[:, j], bins=35, density=True, histtype="step", lw=1.5, color=color)
            else:
                plot_hist_contours(ax, samples[:, j], samples[:, i], color=color)
            if i == ndim - 1:
                ax.set_xlabel(names[j], fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(names[i], fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.grid(alpha=0.12)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_corner_overlay_plot(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    names: list[str],
    out_path: Path,
    *,
    label_a: str,
    label_b: str,
    color_a: str = "#1f77b4",
    color_b: str = "#d62728",
    max_points: int = 25000,
) -> None:
    rng = np.random.default_rng(1)
    if samples_a.shape[0] > max_points:
        samples_a = samples_a[rng.choice(samples_a.shape[0], size=max_points, replace=False)]
    if samples_b.shape[0] > max_points:
        samples_b = samples_b[rng.choice(samples_b.shape[0], size=max_points, replace=False)]

    ndim = samples_a.shape[1]
    fig, axes = plt.subplots(ndim, ndim, figsize=(1.9 * ndim + 2.0, 1.9 * ndim + 2.0))
    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(samples_a[:, j], bins=35, density=True, histtype="step", lw=1.5, color=color_a, label=label_a)
                ax.hist(samples_b[:, j], bins=35, density=True, histtype="step", lw=1.5, color=color_b, label=label_b)
            else:
                plot_hist_contours(ax, samples_a[:, j], samples_a[:, i], color=color_a)
                plot_hist_contours(ax, samples_b[:, j], samples_b[:, i], color=color_b)
            if i == ndim - 1:
                ax.set_xlabel(names[j], fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(names[i], fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.grid(alpha=0.12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Linear-Parameter Posterior Overlay")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize_samples(samples: np.ndarray, names: list[str]) -> dict[str, dict[str, float]]:
    q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
    out = {}
    for i, name in enumerate(names):
        out[name] = {
            "q16": float(q16[i]),
            "q50": float(q50[i]),
            "q84": float(q84[i]),
            "minus": float(q50[i] - q16[i]),
            "plus": float(q84[i] - q50[i]),
        }
    return out


def initial_walker_positions(
    *,
    x_map: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nwalkers: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = 0.02 * np.maximum(upper - lower, 1.0e-3)
    p0 = x_map + rng.normal(scale=scale, size=(nwalkers, x_map.size))
    p0 = np.clip(p0, lower + 1.0e-8, upper - 1.0e-8)
    return p0


def run_chain_for_mode(
    *,
    likelihood: SDSSLikelihood,
    mode: str,
    map_theta: np.ndarray,
    run_paths,
    args,
    seed: int,
) -> dict[str, object]:
    layout = likelihood.layout
    ndim = len(layout.names)
    nwalkers = max(int(args.nwalkers), 2 * ndim)
    p0 = initial_walker_positions(
        x_map=np.clip(map_theta, layout.lower + 1.0e-8, layout.upper - 1.0e-8),
        lower=layout.lower,
        upper=layout.upper,
        nwalkers=nwalkers,
        seed=seed,
    )

    checkpoints = run_paths.figures_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    global _POOL_LIKELIHOOD
    _POOL_LIKELIHOOD = likelihood

    t0 = time.perf_counter()
    flat = None
    flat_lp = None
    checkpoint_stats = []

    if int(args.n_procs) > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=int(args.n_procs)) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, _pool_log_prob, pool=pool)
            state = sampler.run_mcmc(p0, int(args.burnin), progress=False)
            sampler.reset()

            steps_done = 0
            while steps_done < int(args.nsteps):
                n_chunk = min(int(args.checkpoint_every), int(args.nsteps) - steps_done)
                state = sampler.run_mcmc(state, n_chunk, progress=False)
                steps_done += n_chunk
                chain = sampler.get_chain()
                flat = sampler.get_chain(flat=True, thin=max(1, int(args.thin)))
                flat_lp = sampler.get_log_prob(flat=True, thin=max(1, int(args.thin)))
                q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
                checkpoint_stats.append(
                    {
                        "steps_done": int(steps_done),
                        "samples_flat": int(flat.shape[0]),
                        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
                        "q16": q16.tolist(),
                        "q50": q50.tolist(),
                        "q84": q84.tolist(),
                    }
                )
                tag = f"{mode}_step{steps_done:05d}"
                make_trace_plot(
                    chain[:, :, layout.display_idx],
                    layout.display_names,
                    checkpoints / f"{tag}_trace.png",
                    title=f"{mode.title()} trace",
                )
                make_corner_contour_plot(
                    flat[:, layout.display_idx],
                    layout.display_names,
                    checkpoints / f"{tag}_corner_linear.png",
                    color="#1f77b4" if mode == "informative" else "#d62728",
                    title=f"{mode.title()} linear posterior",
                )
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood.log_prob)
        state = sampler.run_mcmc(p0, int(args.burnin), progress=False)
        sampler.reset()

        steps_done = 0
        while steps_done < int(args.nsteps):
            n_chunk = min(int(args.checkpoint_every), int(args.nsteps) - steps_done)
            state = sampler.run_mcmc(state, n_chunk, progress=False)
            steps_done += n_chunk
            chain = sampler.get_chain()
            flat = sampler.get_chain(flat=True, thin=max(1, int(args.thin)))
            flat_lp = sampler.get_log_prob(flat=True, thin=max(1, int(args.thin)))
            q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
            checkpoint_stats.append(
                {
                    "steps_done": int(steps_done),
                    "samples_flat": int(flat.shape[0]),
                    "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
                    "q16": q16.tolist(),
                    "q50": q50.tolist(),
                    "q84": q84.tolist(),
                }
            )
            tag = f"{mode}_step{steps_done:05d}"
            make_trace_plot(
                chain[:, :, layout.display_idx],
                layout.display_names,
                checkpoints / f"{tag}_trace.png",
                title=f"{mode.title()} trace",
            )
            make_corner_contour_plot(
                flat[:, layout.display_idx],
                layout.display_names,
                checkpoints / f"{tag}_corner_linear.png",
                color="#1f77b4" if mode == "informative" else "#d62728",
                title=f"{mode.title()} linear posterior",
            )

    runtime = time.perf_counter() - t0
    chain = sampler.get_chain()
    flat = sampler.get_chain(flat=True, thin=max(1, int(args.thin)))
    flat_lp = sampler.get_log_prob(flat=True, thin=max(1, int(args.thin)))
    best_idx = int(np.argmax(flat_lp))
    best_theta = np.asarray(flat[best_idx], dtype=float)
    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
    median_theta = np.asarray(q50, dtype=float)

    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_out = [float(x) for x in tau]
    except Exception:
        tau_out = []

    pred_best = likelihood.predict_by_z(best_theta)
    pred_median = likelihood.predict_by_z(median_theta)
    fit_plot = run_paths.figures_dir / f"{mode}_p1d_fit_median.png"
    make_fit_plot(
        sdss_blocks=likelihood.sdss_blocks,
        pred_by_mode={mode: pred_median},
        theory=likelihood.theory,
        out_path=fit_plot,
    )

    arr_path = run_paths.arrays_dir / f"sdss_stage1_mcmc_{mode}_{likelihood.theory}.npz"
    np.savez(
        arr_path,
        chain=chain,
        flat_samples=flat,
        flat_log_prob=flat_lp,
        map_theta=np.asarray(map_theta, dtype=float),
        best_theta=best_theta,
        q16=q16,
        q50=q50,
        q84=q84,
        param_names=np.array(layout.names, dtype=object),
        display_indices=layout.display_idx,
        display_names=np.array(layout.display_names, dtype=object),
    )

    trace_final = run_paths.figures_dir / f"{mode}_trace_linear_final.png"
    corner_final = run_paths.figures_dir / f"{mode}_corner_linear_final.png"
    make_trace_plot(chain[:, :, layout.display_idx], layout.display_names, trace_final, title=f"{mode.title()} trace")
    make_corner_contour_plot(
        flat[:, layout.display_idx],
        layout.display_names,
        corner_final,
        color="#1f77b4" if mode == "informative" else "#d62728",
        title=f"{mode.title()} linear posterior",
    )

    summary_all = summarize_samples(flat, layout.names)
    summary_display = summarize_samples(flat[:, layout.display_idx], layout.display_names)
    data_chi2 = likelihood.data_chi2(best_theta)
    dof = max(int(sum(b.k_hmpc.size for b in likelihood.sdss_blocks) - ndim), 1)

    return {
        "mode": mode,
        "theory": likelihood.theory,
        "param_names": layout.names,
        "display_names": layout.display_names,
        "bounds": {"lower": layout.lower.tolist(), "upper": layout.upper.tolist()},
        "map_theta": np.asarray(map_theta, dtype=float).tolist(),
        "best_theta": best_theta.tolist(),
        "q16": q16.tolist(),
        "q50": q50.tolist(),
        "q84": q84.tolist(),
        "summary_all": summary_all,
        "summary_display": summary_display,
        "fit_chi2": float(data_chi2),
        "fit_chi2_dof": float(data_chi2 / dof),
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
        "autocorr_time": tau_out,
        "runtime_sec": float(runtime),
        "n_data": int(sum(b.k_hmpc.size for b in likelihood.sdss_blocks)),
        "n_dim": int(ndim),
        "nwalkers": int(nwalkers),
        "nsteps": int(args.nsteps),
        "burnin": int(args.burnin),
        "thin": int(args.thin),
        "checkpoint_every": int(args.checkpoint_every),
        "checkpoint_stats": checkpoint_stats,
        "arrays": str(arr_path),
        "figures": [str(trace_final), str(corner_final), str(fit_plot)],
        "pred_median_by_z": {f"{z:.1f}": pred_median[z].tolist() for z in likelihood.zvals},
        "pred_best_by_z": {f"{z:.1f}": pred_best[z].tolist() for z in likelihood.zvals},
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    prior_csv = latest_stage1_prior_csv(args.theory) if args.prior_csv is None else args.prior_csv
    prior3d = load_prior_relations(prior_csv, args.theory)

    run_paths = init_run_dir(cfg.run.output_root, tag=f"repro_2405_stage1_sdss_mcmc_{args.theory}")
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
            "b_eta_min": float(args.b_eta_min),
            "b_eta_max": float(args.b_eta_max),
            "map_summary": (str(args.map_summary) if args.map_summary is not None else None),
            "map_n_starts": int(args.map_n_starts),
            "calibration_reuse_from_summary": bool(args.map_summary is not None),
            "mcmc": {
                "nwalkers": int(args.nwalkers),
                "nsteps": int(args.nsteps),
                "burnin": int(args.burnin),
                "thin": int(args.thin),
                "checkpoint_every": int(args.checkpoint_every),
                "posterior_band_draws": int(args.posterior_band_draws),
                "max_nfev": int(args.max_nfev),
                "n_procs": int(args.n_procs),
                "seed": int(args.seed),
            },
        }
    )
    write_json(run_paths.logs_dir / "repro_metadata.json", meta)

    sdss_raw = load_chabanier2019_blocks(
        data_dir=args.sdss_dir,
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        h=cfg.cosmology.h,
        omega_b=cfg.cosmology.omega_b,
        omega_cdm=cfg.cosmology.omega_cdm,
        include_syst=True,
    )
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
    if not mock_raw:
        raise ValueError(f"No mock blocks found in z range [{args.z_min}, {args.z_max}]")

    sdss_blocks = build_fit_blocks(sdss_raw, kmin_fit_hmpc=float(args.kmin_fit_hmpc), kmax_fit_hmpc=float(args.kmax_fit_hmpc))
    mock_blocks = build_fit_blocks(mock_raw, kmin_fit_hmpc=float(args.kmin_fit_hmpc), kmax_fit_hmpc=float(args.kmax_fit_hmpc))

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

    calibration_source = "recomputed"
    cached_cal = load_calibration_from_sdss_summary(args.map_summary) if args.map_summary is not None else None
    if cached_cal is not None:
        c0_rel, c2_rel, conservative_sigmas = cached_cal
        calibration_source = f"summary:{args.map_summary}"
    else:
        c0_rel, c2_rel, mock_rows, mock_fit_rows = fit_mock_counterterm_relations(
            theory=args.theory,
            mock_blocks=mock_blocks,
            models_by_z=models_by_z,
            sigma8_ref=sigma8_ref,
            prior3d=prior3d,
            cfg=cfg,
            args=args,
        )

        conservative_sigmas = {
            "c0_3d": float(args.conservative_inflate) * max(prior3d["c0_3d"]["rmse"], 1.0e-4),
            "c2_3d": float(args.conservative_inflate) * max(prior3d["c2_3d"]["rmse"], 1.0e-4),
            "c4_3d": float(args.conservative_inflate) * max(prior3d["c4_3d"]["rmse"], 1.0e-4),
            "loop_amp": float(args.conservative_inflate) * max(prior3d["loop_amp"]["rmse"], 1.0e-4),
            "C0_1d": float(args.conservative_inflate) * max(mock_fit_rows[0]["fit_rmse"], 1.0e-4),
            "C2_1d": float(args.conservative_inflate) * max(mock_fit_rows[1]["fit_rmse"], 1.0e-4),
        }
        if args.theory == "hybrid":
            conservative_sigmas["b_t"] = float(args.conservative_inflate) * max(prior3d["b_t"]["rmse"], 1.0e-4)
            conservative_sigmas["sigma_th"] = float(args.conservative_inflate) * max(prior3d["sigma_th"]["rmse"], 1.0e-4)

    map_thetas = load_map_thetas(args.map_summary) if args.map_summary is not None else {}
    modes = [args.mode] if args.mode in {"informative", "conservative"} else ["informative", "conservative"]
    results = []
    samples_by_mode: dict[str, np.ndarray] = {}

    for i, mode in enumerate(modes):
        like = SDSSLikelihood(
            theory=args.theory,
            mode=mode,
            sdss_blocks=sdss_blocks,
            models_by_z=models_by_z,
            sigma8_ref=sigma8_ref,
            prior3d=prior3d,
            c0_rel=c0_rel,
            c2_rel=c2_rel,
            conservative_sigmas=conservative_sigmas,
            cfg=cfg,
            args=args,
        )
        if mode in map_thetas:
            map_theta = map_thetas[mode]
        else:
            fit_args = argparse.Namespace(**vars(args))
            fit_args.n_starts = int(args.map_n_starts)
            fit_args.max_nfev = 2600
            map_res = run_sdss_fit(
                theory=args.theory,
                mode=mode,
                sdss_blocks=sdss_blocks,
                models_by_z=models_by_z,
                sigma8_ref=sigma8_ref,
                prior3d=prior3d,
                c0_rel=c0_rel,
                c2_rel=c2_rel,
                conservative_sigmas=conservative_sigmas,
                linear_start_hint=None,
                cfg=cfg,
                args=fit_args,
            )
            map_theta = np.asarray(map_res["best_theta"], dtype=float)

        out = run_chain_for_mode(
            likelihood=like,
            mode=mode,
            map_theta=np.asarray(map_theta, dtype=float),
            run_paths=run_paths,
            args=args,
            seed=int(args.seed + 97 * i),
        )
        results.append(out)
        arr = np.load(Path(out["arrays"]), allow_pickle=True)
        samples_by_mode[mode] = np.asarray(arr["flat_samples"][:, like.layout.display_idx], dtype=float)

    overlay = None
    if set(samples_by_mode.keys()) == {"informative", "conservative"}:
        overlay = run_paths.figures_dir / "linear_corner_overlay.png"
        display_names = results[0]["display_names"]
        make_corner_overlay_plot(
            samples_by_mode["informative"],
            samples_by_mode["conservative"],
            display_names,
            overlay,
            label_a="informative",
            label_b="conservative",
        )

    fig_target = Path("results/figures")
    fig_target.mkdir(parents=True, exist_ok=True)
    for res in results:
        for fp in res["figures"]:
            shutil.copy2(fp, fig_target / Path(fp).name)
    if overlay is not None:
        shutil.copy2(overlay, fig_target / overlay.name)

    summary = {
        "run_dir": str(run_paths.run_dir),
        "theory": str(args.theory),
        "prior_csv": str(prior_csv),
        "calibration_source": calibration_source,
        "sdss_z_bins": [float(b.z) for b in sdss_blocks],
        "sigma8_ref": float(sigma8_ref),
        "mode": args.mode,
        "results": results,
        "overlay_figure": (str(overlay) if overlay is not None else None),
    }
    write_json(run_paths.logs_dir / "summary.json", summary)
    print(f"2405 Stage-1 SDSS MCMC ({args.theory}) complete: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
