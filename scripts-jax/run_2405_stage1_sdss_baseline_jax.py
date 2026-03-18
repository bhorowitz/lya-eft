#!/usr/bin/env python3
"""
JAX/GPU-accelerated version of scripts/run_2405_stage1_sdss_baseline.py.

CPU (unchanged):  CAMB linear power, data I/O, scipy least_squares optimizer.
GPU (JAX):        IvanovFullModel loop integrals (IvanovFullModelJAX),
                  P1D projection (project_to_1d_jax),
                  paper systematics factor.

Usage is identical to the numpy script; add --compare to also run the numpy
model and print accuracy/timing comparisons.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import csv
import shutil
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

# --- lya_hybrid_jax package (src-jax/) ---
sys.path.insert(0, str(Path(__file__).parent.parent / "src-jax"))
from lya_hybrid_jax import (
    IvanovFullModelJAX,
    JaxP3DGrid,
    make_jax_p3d_grid,
    project_to_1d_jax,
    paper_systematics_factor_jnp,
)

# --- lya_hybrid package (src/) ---
from lya_hybrid.config import load_config
from lya_hybrid.linear_power import compute_linear_power_camb
from lya_hybrid.logging_utils import build_repro_metadata, init_run_dir, write_json
from lya_hybrid.model_ivanov_full import IvanovFullParams
from lya_hybrid.sdss_p1d import P1DBlock, load_chabanier2019_blocks, load_eboss_mock_blocks

# Import shared constants and pure-numpy helpers from the numpy script.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_2405_stage1_sdss_baseline import (  # noqa: E402
    DEFAULT_SDSS_DIR,
    DEFAULT_MOCK_DIR,
    PAPER_COUNTERTERM_RELATIONS,
    PAPER_C0_SIGMA,
    PAPER_C2_SIGMA,
    FitBlock,
    build_fit_blocks,
    latest_stage1_prior_csv,
    read_csv_rows,
    write_csv,
    published_counterterm_relations,
    make_fit_plot,
    make_sigma8_plot,
)


# ============================================================================
# Argument parsing
# ============================================================================
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="JAX/GPU 2405 SDSS Stage-1 baseline")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--theory", choices=["one_loop", "hybrid"], default="one_loop")
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
    p.add_argument("--counterterm-mode", choices=["proxy", "paper_baseline", "paper_rescaled"],
                   default="proxy")
    p.add_argument("--apply-paper-systematics", action="store_true")
    p.add_argument("--b-eta-min", type=float, default=-2.0)
    p.add_argument("--b-eta-max", type=float, default=2.0)
    p.add_argument("--n-zbins", type=int, default=0,
                   help="Limit to first N z-bins (0 = all, use for quick tests)")
    p.add_argument("--n-starts", type=int, default=12)
    p.add_argument("--max-nfev", type=int, default=2600)
    p.add_argument("--seed", type=int, default=20260313)
    p.add_argument("--compare", action="store_true",
                   help="Also run numpy model and print accuracy/timing comparisons.")
    return p.parse_args()


# ============================================================================
# Prior helpers
# ============================================================================
def load_prior_relations(path: Path, theory: str) -> dict:
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))
    out = {
        r["operator_name"]: {
            "A": float(r["A_O"]),
            "B": float(r["B_O"]),
            "rmse": float(r["fit_rmse"]),
        }
        for r in rows
    }
    needed = {"c0_3d", "c2_3d", "c4_3d", "loop_amp"}
    if theory == "hybrid":
        needed |= {"b_t", "sigma_th"}
    missing = needed.difference(out.keys())
    if missing:
        raise ValueError(f"Missing operators in prior CSV {path}: {sorted(missing)}")
    return out


def relation_eval(rel: dict, name: str, b1: float) -> float:
    return rel[name]["A"] * b1 + rel[name]["B"]


# ============================================================================
# JAX P1D prediction  —  drop-in replacement for the numpy version
# ============================================================================
def p1d_prediction(
    *,
    jax_model: IvanovFullModelJAX,
    grid: JaxP3DGrid,
    z: float,
    kpar_hmpc: np.ndarray,
    b1: float,
    b_eta: float,
    sigma8: float,
    sigma8_ref: float,
    prior3d: dict,
    c0_1d_rel: tuple,
    c2_1d_rel: tuple,
    apply_paper_systematics: bool,
    kmax_proj: float,
    nint_proj: int,
    cfg,
    offsets: dict | None = None,
):
    """
    Mirrors the numpy p1d_prediction from run_2405_stage1_sdss_baseline.py.
    P3D loop integrals and projection run on GPU; data transfers are minimised.
    """
    if offsets is None:
        offsets = {}
    d = lambda key: float(offsets.get(key, 0.0))

    params = IvanovFullParams(
        b1=float(b1), b_eta=float(b_eta),
        b_delta2=relation_eval(prior3d, "b_delta2", b1),
        b_G2=relation_eval(prior3d, "b_G2", b1),
        b_KK_par=relation_eval(prior3d, "b_KK_par", b1),
        b_delta_eta=relation_eval(prior3d, "b_delta_eta", b1),
        b_eta2=relation_eval(prior3d, "b_eta2", b1),
        b_Pi2_par=relation_eval(prior3d, "b_Pi2_par", b1),
        b_Pi3_par=relation_eval(prior3d, "b_Pi3_par", b1),
        b_delta_Pi2_par=relation_eval(prior3d, "b_delta_Pi2_par", b1),
        b_KPi2_par=relation_eval(prior3d, "b_KPi2_par", b1),
        b_eta_Pi2_par=relation_eval(prior3d, "b_eta_Pi2_par", b1),
        c0_ct=relation_eval(prior3d, "c0_3d", b1) + d("c0_3d"),
        c2_ct=relation_eval(prior3d, "c2_3d", b1) + d("c2_3d"),
        c4_ct=relation_eval(prior3d, "c4_3d", b1) + d("c4_3d"),
    )
    amp = (float(sigma8) / float(sigma8_ref)) ** 2

    # GPU: P3D on grid
    p3d_flat = amp * jax_model.evaluate_components(grid.kk_flat, grid.mm_flat, params)["total"]
    p3d_g    = p3d_flat.reshape(grid.nk_g, grid.nmu_g)

    # GPU: P1D projection
    raw = project_to_1d_jax(grid.kpar, grid.k_g, grid.mu_g, p3d_g, kmax_proj, nint_proj)

    # GPU: counterterms and optional systematics
    c0_1d = c0_1d_rel[0] * b1 + c0_1d_rel[1] + d("C0_1d")
    c2_1d = c2_1d_rel[0] * b1 + c2_1d_rel[1] + d("C2_1d")
    pred  = raw + c0_1d + c2_1d * jnp.asarray(kpar_hmpc) ** 2
    if apply_paper_systematics:
        pred *= paper_systematics_factor_jnp(
            z, jnp.asarray(kpar_hmpc),
            cfg.cosmology.h, cfg.cosmology.omega_b, cfg.cosmology.omega_cdm,
        )
    return pred


# ============================================================================
# fit_mock_counterterm_relations
# ============================================================================
def fit_mock_counterterm_relations(
    *,
    mock_blocks: list[FitBlock],
    jax_models_by_z: dict,
    jax_grids_by_z: dict,
    sigma8_ref: float,
    prior3d: dict,
    cfg,
    args,
) -> tuple:
    rows = []
    x0   = np.array([-0.5, 0.0, 0.0, 0.0], dtype=float)

    for i, block in enumerate(mock_blocks):
        print(f"[fit_mock] block {i+1}/{len(mock_blocks)} z={block.z:.2f}")
        jax_m = jax_models_by_z[block.z]
        grid  = jax_grids_by_z[block.z]

        def residual(theta):
            b1_, b_eta_, c0_, c2_ = [float(x) for x in theta]
            pred = p1d_prediction(
                jax_model=jax_m, grid=grid, z=float(block.z),
                kpar_hmpc=block.k_hmpc, b1=b1_, b_eta=b_eta_,
                sigma8=sigma8_ref, sigma8_ref=sigma8_ref,
                prior3d=prior3d,
                c0_1d_rel=(0.0, c0_), c2_1d_rel=(0.0, c2_),
                apply_paper_systematics=bool(args.apply_paper_systematics),
                kmax_proj=float(args.kmax_proj), nint_proj=int(args.nint_proj), cfg=cfg,
            )
            return np.linalg.solve(block.chol, np.asarray(pred) - block.p_hmpc)

        lo = np.array([-1.5, -2.0, -1.5, -1.5])
        hi = np.array([ 0.2,  2.0,  1.5,  1.5])
        x0 = np.clip(x0, lo + 1e-8, hi - 1e-8)
        fit = least_squares(residual, x0=x0, bounds=(lo, hi), method="trf",
                            loss="soft_l1", f_scale=1.0, max_nfev=int(args.max_nfev))
        print(f"[fit_mock] done z={block.z:.2f}  status={fit.status}  "
              f"nfev={fit.nfev}  cost={fit.cost:.4g}")
        x0 = fit.x.copy()
        b1_, b_eta_, c0_, c2_ = [float(x) for x in fit.x]
        chi2 = float(np.sum(residual(fit.x) ** 2))
        dof  = max(int(block.k_hmpc.size - 4), 1)
        rows.append({
            "z": float(block.z), "b1": b1_, "b_eta": b_eta_,
            "C0_1d_fit": c0_, "C2_1d_fit": c2_, "chi2_dof": chi2 / dof,
        })

    b1_a = np.array([r["b1"]        for r in rows])
    c0_a = np.array([r["C0_1d_fit"] for r in rows])
    c2_a = np.array([r["C2_1d_fit"] for r in rows])

    def linfit(x, y):
        A = np.column_stack([x, np.ones_like(x)])
        coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
        rmse = float(np.sqrt(np.mean((y - A @ coeff) ** 2)))
        return float(coeff[0]), float(coeff[1]), rmse

    c0A, c0B, c0r = linfit(b1_a, c0_a)
    c2A, c2B, c2r = linfit(b1_a, c2_a)
    fit_rows = [
        {"operator_name": "C0_1d", "A_O": c0A, "B_O": c0B, "fit_rmse": c0r,
         "n_points": int(b1_a.size)},
        {"operator_name": "C2_1d", "A_O": c2A, "B_O": c2B, "fit_rmse": c2r,
         "n_points": int(b1_a.size)},
    ]
    return (c0A, c0B), (c2A, c2B), rows, fit_rows


# ============================================================================
# run_sdss_fit
# ============================================================================
def run_sdss_fit(
    *,
    sdss_blocks: list[FitBlock],
    jax_models_by_z: dict,
    jax_grids_by_z: dict,
    sigma8_ref: float,
    prior3d: dict,
    c0_rel: tuple,
    c2_rel: tuple,
    conservative_sigmas: dict,
    mode: str,
    linear_start_hint: np.ndarray | None,
    cfg,
    args,
) -> dict:
    zvals = [b.z for b in sdss_blocks]
    nz    = len(sdss_blocks)
    use_offsets = (mode == "conservative")

    names = ["sigma8"] + [f"b1_z{z:.1f}" for z in zvals] + [f"b_eta_z{z:.1f}" for z in zvals]
    if use_offsets:
        names += ["d_c0_3d", "d_c2_3d", "d_c4_3d", "d_loop_amp", "d_C0_1d", "d_C2_1d"]

    n_dim = len(names)
    lo = np.full(n_dim, -np.inf)
    hi = np.full(n_dim,  np.inf)
    lo[0], hi[0] = 0.6, 1.1
    lo[1:1+nz],    hi[1:1+nz]    = -1.5, 0.2
    lo[1+nz:1+2*nz], hi[1+nz:1+2*nz] = float(args.b_eta_min), float(args.b_eta_max)
    if use_offsets:
        lo[-6:], hi[-6:] = -1.5, 1.5

    x0 = np.zeros(n_dim, dtype=float)
    x0[0] = sigma8_ref
    for i, z in enumerate(zvals):
        x0[1+i]    = -0.45 - 0.18 * (z - min(zvals)) / max(max(zvals) - min(zvals), 1e-6)
        x0[1+nz+i] = 0.0
    if linear_start_hint is not None and linear_start_hint.size >= 1 + 2 * nz:
        x0[:1+2*nz] = linear_start_hint[:1+2*nz]

    def unpack(theta):
        s8   = float(theta[0])
        b1_l = [float(theta[1+i])    for i in range(nz)]
        be_l = [float(theta[1+nz+i]) for i in range(nz)]
        off  = {}
        if use_offsets:
            off = {
                "c0_3d":    float(theta[-6]),
                "c2_3d":    float(theta[-5]),
                "c4_3d":    float(theta[-4]),
                "loop_amp": float(theta[-3]),
                "C0_1d":    float(theta[-2]),
                "C2_1d":    float(theta[-1]),
            }
        return s8, b1_l, be_l, off

    def residual(theta):
        s8, b1_l, be_l, off = unpack(theta)
        chunks = []
        for i, block in enumerate(sdss_blocks):
            pred = p1d_prediction(
                jax_model=jax_models_by_z[block.z],
                grid=jax_grids_by_z[block.z],
                z=float(block.z), kpar_hmpc=block.k_hmpc,
                b1=b1_l[i], b_eta=be_l[i],
                sigma8=s8, sigma8_ref=sigma8_ref,
                prior3d=prior3d, c0_1d_rel=c0_rel, c2_1d_rel=c2_rel,
                apply_paper_systematics=bool(args.apply_paper_systematics),
                kmax_proj=float(args.kmax_proj), nint_proj=int(args.nint_proj),
                cfg=cfg, offsets=off,
            )
            chunks.append(np.linalg.solve(block.chol, np.asarray(pred) - block.p_hmpc))
        if use_offsets:
            chunks += [
                np.array([off["c0_3d"]    / conservative_sigmas["c0_3d"]]),
                np.array([off["c2_3d"]    / conservative_sigmas["c2_3d"]]),
                np.array([off["c4_3d"]    / conservative_sigmas["c4_3d"]]),
                np.array([off["loop_amp"] / conservative_sigmas["loop_amp"]]),
                np.array([off["C0_1d"]    / conservative_sigmas["C0_1d"]]),
                np.array([off["C2_1d"]    / conservative_sigmas["C2_1d"]]),
            ]
        return np.concatenate(chunks)

    x0 = np.clip(x0, lo + 1e-8, hi - 1e-8)
    n_starts    = max(1, int(args.n_starts))
    seed_offset = 0 if mode == "informative" else 1000
    rng = np.random.default_rng(int(args.seed) + seed_offset)

    starts = [x0]
    if n_starts > 1:
        trend = x0.copy()
        trend[1:1+nz]      = np.linspace(-0.30, -0.60, nz)
        trend[1+nz:1+2*nz] = np.linspace(-0.20, -0.55, nz)
        starts.append(np.clip(trend, lo + 1e-8, hi - 1e-8))
    if n_starts > 2:
        mid = 0.5 * (lo + hi);  mid[0] = sigma8_ref
        starts.append(np.clip(mid, lo + 1e-8, hi - 1e-8))
    while len(starts) < n_starts:
        r = rng.uniform(lo, hi)
        r[0] = np.clip(rng.normal(sigma8_ref, 0.08), lo[0] + 1e-8, hi[0] - 1e-8)
        starts.append(np.clip(r, lo + 1e-8, hi - 1e-8))

    best_fit, best_obj, best_idx = None, float("inf"), -1
    start_objs, start_nfevs = [], []
    for i, start in enumerate(starts):
        fi = least_squares(residual, x0=start, bounds=(lo, hi), method="trf",
                           loss="soft_l1", f_scale=1.0, max_nfev=int(args.max_nfev))
        obj = float(np.sum(fi.fun ** 2))
        start_objs.append(obj);  start_nfevs.append(int(fi.nfev))
        if obj < best_obj:
            best_obj, best_idx, best_fit = obj, i, fi

    theta = best_fit.x.copy()
    s8, b1_l, be_l, off = unpack(theta)

    data_res, pred_by_z = [], {}
    for i, block in enumerate(sdss_blocks):
        pred = p1d_prediction(
            jax_model=jax_models_by_z[block.z],
            grid=jax_grids_by_z[block.z],
            z=float(block.z), kpar_hmpc=block.k_hmpc,
            b1=b1_l[i], b_eta=be_l[i], sigma8=s8, sigma8_ref=sigma8_ref,
            prior3d=prior3d, c0_1d_rel=c0_rel, c2_1d_rel=c2_rel,
            apply_paper_systematics=bool(args.apply_paper_systematics),
            kmax_proj=float(args.kmax_proj), nint_proj=int(args.nint_proj),
            cfg=cfg, offsets=off,
        )
        pred_by_z[block.z] = np.asarray(pred)
        data_res.append(np.linalg.solve(block.chol, np.asarray(pred) - block.p_hmpc))

    data_res_vec = np.concatenate(data_res)
    data_chi2    = float(np.sum(data_res_vec ** 2))
    n_data       = int(data_res_vec.size)
    dof          = max(n_data - n_dim, 1)

    s8_std = float("nan")
    try:
        cov = np.linalg.pinv(best_fit.jac.T @ best_fit.jac) * max(data_chi2 / dof, 1.0)
        if cov[0, 0] > 0:
            s8_std = float(np.sqrt(cov[0, 0]))
    except Exception:
        pass

    return {
        "mode": mode, "param_names": names, "best_theta": theta.tolist(),
        "sigma8": float(s8), "sigma8_std_approx": s8_std,
        "chi2": data_chi2, "chi2_dof": float(data_chi2 / dof),
        "n_data": n_data, "n_dim": n_dim, "bias_label": "b1",
        "bias_linear_by_z": {f"{z:.1f}": float(b1_l[i]) for i, z in enumerate(zvals)},
        "b1_by_z":          {f"{z:.1f}": float(b1_l[i]) for i, z in enumerate(zvals)},
        "b_delta_by_z":     {},
        "b_eta_by_z":       {f"{z:.1f}": float(be_l[i]) for i, z in enumerate(zvals)},
        "offsets": {k: float(v) for k, v in off.items()},
        "pred_by_z": {f"{z:.1f}": pred_by_z[z].tolist() for z in zvals},
        "multistart": {
            "n_starts": n_starts, "seed": int(args.seed) + seed_offset,
            "best_start_index": best_idx, "best_objective": best_obj,
            "start_objectives": start_objs, "start_nfev": start_nfevs,
        },
    }


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    args = parse_args()
    cfg  = load_config(args.config)
    _    = np.random.default_rng(args.seed).integers(0, 10)

    prior_csv = (
        latest_stage1_prior_csv(args.theory) if args.prior_csv is None else args.prior_csv
    )
    prior3d = load_prior_relations(prior_csv, args.theory)

    run_paths = init_run_dir(cfg.run.output_root, tag=f"repro_2405_stage1_sdss_{args.theory}_jax")
    meta = build_repro_metadata(args.config)
    meta.update({
        "theory": str(args.theory), "prior_csv": str(prior_csv),
        "jax_device": f"GPU:{os.environ['CUDA_VISIBLE_DEVICES']}",
    })
    write_json(run_paths.logs_dir / "repro_metadata.json", meta)

    # --- Load data ---
    sdss_raw = load_chabanier2019_blocks(
        data_dir=args.sdss_dir, z_min=float(args.z_min), z_max=float(args.z_max),
        h=cfg.cosmology.h, omega_b=cfg.cosmology.omega_b, omega_cdm=cfg.cosmology.omega_cdm,
        include_syst=True)
    mock_raw: list[P1DBlock] = []
    if args.counterterm_mode == "proxy":
        mock_raw = load_eboss_mock_blocks(
            data_dir=args.mock_dir, z_min=float(args.z_min), z_max=float(args.z_max),
            h=cfg.cosmology.h, omega_b=cfg.cosmology.omega_b, omega_cdm=cfg.cosmology.omega_cdm)

    if not sdss_raw:
        raise ValueError(f"No SDSS blocks in z=[{args.z_min},{args.z_max}]")
    if args.counterterm_mode == "proxy" and not mock_raw:
        raise ValueError(f"No mock blocks in z=[{args.z_min},{args.z_max}]")

    sdss_blocks = build_fit_blocks(
        sdss_raw, kmin_fit_hmpc=float(args.kmin_fit_hmpc), kmax_fit_hmpc=float(args.kmax_fit_hmpc))
    mock_blocks = (
        build_fit_blocks(mock_raw, kmin_fit_hmpc=float(args.kmin_fit_hmpc),
                         kmax_fit_hmpc=float(args.kmax_fit_hmpc))
        if args.counterterm_mode == "proxy" else []
    )

    zvals = sorted({b.z for b in sdss_blocks})
    if args.n_zbins > 0:
        zvals = zvals[:args.n_zbins]
    sdss_blocks = [b for b in sdss_blocks if b.z in zvals]
    mock_blocks = [b for b in mock_blocks if b.z in zvals]

    # --- Build JAX models (CPU CAMB, GPU loop integrals) ---
    jax_models_by_z: dict = {}
    sigma8_vals = []
    for z in zvals:
        print(f"[main] Building JAX model z={z:.2f} ...")
        lp = compute_linear_power_camb(
            h=cfg.cosmology.h, omega_b=cfg.cosmology.omega_b, omega_cdm=cfg.cosmology.omega_cdm,
            ns=cfg.cosmology.ns, As=cfg.cosmology.As, z=float(z),
            kmin=cfg.k_grid.kmin, kmax=max(cfg.k_grid.kmax, float(args.kmax_fit_hmpc)),
            nk=cfg.k_grid.nk)
        nq = min(cfg.k_grid.nk // 10 if cfg.k_grid.nk >= 40 else 8, 14)
        jax_models_by_z[z] = IvanovFullModelJAX(
            lp.k_hmpc, lp.p_lin_h3mpc3, lp.f_growth,
            qmin=cfg.k_grid.kmin, qmax=cfg.k_grid.kmax, nq=nq, nmuq=6, nphi=6)
        if lp.sigma8_0 is not None:
            sigma8_vals.append(float(lp.sigma8_0))
    sigma8_ref = float(np.median(sigma8_vals)) if sigma8_vals else 0.83

    # --- Pre-allocate JAX P3D grids per z-bin ---
    jax_grids_by_z: dict = {}
    for block in sdss_blocks + mock_blocks:
        if block.z not in jax_grids_by_z:
            jax_grids_by_z[block.z] = make_jax_p3d_grid(block.k_hmpc, float(args.kmax_proj))

    # --- JIT warmup ---
    print("[main] Warming up JAX JIT ...")
    _dummy = sdss_blocks[0]
    _dummy_pred = p1d_prediction(
        jax_model=jax_models_by_z[_dummy.z], grid=jax_grids_by_z[_dummy.z],
        z=float(_dummy.z), kpar_hmpc=_dummy.k_hmpc,
        b1=-0.45, b_eta=0.0, sigma8=sigma8_ref, sigma8_ref=sigma8_ref,
        prior3d=prior3d, c0_1d_rel=(0.0, 0.0), c2_1d_rel=(0.0, 0.0),
        apply_paper_systematics=False,
        kmax_proj=float(args.kmax_proj), nint_proj=int(args.nint_proj), cfg=cfg,
    )
    print(f"[main] Warmup done. Dummy P1D shape: {np.asarray(_dummy_pred).shape}")

    # --- Counterterm relations ---
    if args.counterterm_mode == "proxy":
        c0_rel, c2_rel, mock_rows, mock_fit_rows = fit_mock_counterterm_relations(
            mock_blocks=mock_blocks, jax_models_by_z=jax_models_by_z,
            jax_grids_by_z=jax_grids_by_z, sigma8_ref=sigma8_ref,
            prior3d=prior3d, cfg=cfg, args=args)
        counterterm_label = "Mock-based eBOSS proxy for C0/C2 (JAX)"
    else:
        c0_rel, c2_rel, mock_rows, mock_fit_rows = published_counterterm_relations(
            args.counterterm_mode)
        counterterm_label = PAPER_COUNTERTERM_RELATIONS[args.counterterm_mode]["label"]

    # --- Conservative prior widths ---
    conservative_sigmas = {
        "c0_3d":    float(args.conservative_inflate) * max(prior3d["c0_3d"]["rmse"],    1e-4),
        "c2_3d":    float(args.conservative_inflate) * max(prior3d["c2_3d"]["rmse"],    1e-4),
        "c4_3d":    float(args.conservative_inflate) * max(prior3d["c4_3d"]["rmse"],    1e-4),
        "loop_amp": float(args.conservative_inflate) * max(prior3d["loop_amp"]["rmse"], 1e-4),
        "C0_1d": (PAPER_C0_SIGMA if args.counterterm_mode != "proxy"
                  else float(args.conservative_inflate) * max(mock_fit_rows[0]["fit_rmse"], 1e-4)),
        "C2_1d": (PAPER_C2_SIGMA if args.counterterm_mode != "proxy"
                  else float(args.conservative_inflate) * max(mock_fit_rows[1]["fit_rmse"], 1e-4)),
    }

    # --- SDSS fits ---
    modes = ([args.mode] if args.mode in {"informative", "conservative"}
             else ["conservative", "informative"])
    fit_results, pred_by_mode = [], {}
    linear_start_hint = None
    for mode in modes:
        print(f"[main] Running SDSS fit: mode={mode} ...")
        t0 = time.time()
        res = run_sdss_fit(
            sdss_blocks=sdss_blocks, jax_models_by_z=jax_models_by_z,
            jax_grids_by_z=jax_grids_by_z, sigma8_ref=sigma8_ref,
            prior3d=prior3d, c0_rel=c0_rel, c2_rel=c2_rel,
            conservative_sigmas=conservative_sigmas, mode=mode,
            linear_start_hint=linear_start_hint, cfg=cfg, args=args)
        print(f"[main] {mode} done in {time.time()-t0:.1f}s  "
              f"sigma8={res['sigma8']:.4f}±{res['sigma8_std_approx']:.4f}  "
              f"chi2/dof={res['chi2_dof']:.2f}")
        fit_results.append(res)
        pred_by_mode[mode] = {b.z: np.array(res["pred_by_z"][f"{b.z:.1f}"]) for b in sdss_blocks}
        linear_start_hint = np.array(res["best_theta"])

    # --- Plots ---
    f_fit = run_paths.figures_dir / "71_sdss_p1d_fit_multiz.png"
    make_fit_plot(sdss_blocks=sdss_blocks, pred_by_mode=pred_by_mode,
                  theory=args.theory, out_path=f_fit)
    f_s8 = run_paths.figures_dir / "72_sdss_sigma8_comparison.png"
    make_sigma8_plot(fit_results, f_s8)

    # --- Tables ---
    summary_rows = [
        {
            "model_name": f"{args.theory}_2405_stage1_{args.counterterm_mode}_jax",
            "k_max": float(args.kmax_fit_hmpc),
            "chi2": float(r["chi2"]), "sigma8_mean": float(r["sigma8"]),
            "sigma8_std": float(r["sigma8_std_approx"]), "prior_variant": r["mode"],
            "notes": f"JAX/GPU {args.theory}; paper_systematics={bool(args.apply_paper_systematics)}",
        }
        for r in fit_results
    ]
    write_csv(run_paths.logs_dir / "sdss_comparison_summary.csv", summary_rows,
              fieldnames=["model_name", "k_max", "chi2", "sigma8_mean", "sigma8_std",
                          "prior_variant", "notes"])
    write_csv(run_paths.logs_dir / "lace_counterterm_fits_proxy.csv", mock_fit_rows,
              fieldnames=["operator_name", "A_O", "B_O", "fit_rmse", "n_points"])
    write_csv(run_paths.logs_dir / "lace_counterterm_points_proxy.csv", mock_rows,
              fieldnames=["z", "b1", "b_eta", "C0_1d_fit", "C2_1d_fit", "chi2_dof"])

    for tgt, flist in [
        (Path("results/figures"), [f_fit, f_s8]),
        (Path("results/tables"), [
            run_paths.logs_dir / "sdss_comparison_summary.csv",
            run_paths.logs_dir / "lace_counterterm_fits_proxy.csv",
            run_paths.logs_dir / "lace_counterterm_points_proxy.csv",
        ]),
    ]:
        tgt.mkdir(parents=True, exist_ok=True)
        for fp in flist:
            shutil.copy2(fp, tgt / f"{fp.stem}_{args.theory}_jax{fp.suffix}")

    summary = {
        "run_dir": str(run_paths.run_dir), "theory": str(args.theory),
        "prior_csv": str(prior_csv), "sdss_z_bins": [float(b.z) for b in sdss_blocks],
        "sigma8_ref": sigma8_ref, "counterterm_mode": str(args.counterterm_mode),
        "counterterm_label": counterterm_label,
        "jax_device": f"GPU:{os.environ['CUDA_VISIBLE_DEVICES']}",
        "fit_results": fit_results,
    }
    write_json(run_paths.logs_dir / "summary.json", summary)
    print(f"2405 Stage-1 SDSS baseline (JAX, {args.theory}) complete: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
