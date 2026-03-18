#!/usr/bin/env python3
"""
JAX/GPU-accelerated version of scripts/run_2405_stage1_sherwood_full.py.

CPU (unchanged): CAMB linear power, data I/O, stage-A bias pre-fit, scipy optimiser.
GPU (JAX):       IvanovFullModelJAX for stage-B P3D residuals (the dominant cost),
                 project_to_1d_jax for the final P1D projection check.

Outputs the same files as the numpy version:
  - sherwood_prior_linear_fits.csv  (ingested by the SDSS stage-1 scripts)
  - z_bin_fit_summary.csv
  - per-snapshot diagnostic figures

Usage:
    conda run -n jax-gpu python scripts-jax/run_2405_stage1_sherwood_jax.py \\
        --config configs/repro_2405.yaml
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import csv
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import binned_statistic

# --- lya_hybrid_jax package (src-jax/) ---
sys.path.insert(0, str(Path(__file__).parent.parent / "src-jax"))
from lya_hybrid_jax import (
    IvanovFullModelJAX,
    make_jax_p3d_grid,
    project_to_1d_jax,
)

# --- lya_hybrid package (src/) ---
from lya_hybrid.config import load_config
from lya_hybrid.io import load_sherwood_flux_p1d, load_sherwood_flux_p3d
from lya_hybrid.linear_power import compute_linear_power_camb
from lya_hybrid.logging_utils import build_repro_metadata, init_run_dir, write_json
from lya_hybrid.model_ivanov_full import IvanovFullModel, IvanovFullParams

# Reuse snapshot / published-relation constants from the numpy Sherwood script.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_2405_stage1_sherwood_full import (  # noqa: E402
    SNAP_TO_Z,
    DEFAULT_P3D_DIR,
    DEFAULT_P1D_DIR,
    PUBLISHED_RELATIONS,
    choose_p3d_file,
    choose_p1d_file,
    pseudo_sigma,
    binned_curve,
    published_guess,
    params_to_theta,
    theta_to_params,
    fit_linear_relation,
    write_csv,
    make_p3d_residual_plot,
    make_p1d_projection_plot,
    make_linear_prior_plot,
)


# ============================================================================
# Argument parsing
# ============================================================================
def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description=(
            "JAX/GPU stage-1 Sherwood prior extraction with the full Ivanov basis: "
            "fit P3D in four redshift bins, project to P1D, fit linear b_O(b1) relations."
        )
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--p3d-dir", type=Path, default=DEFAULT_P3D_DIR)
    p.add_argument("--p1d-dir", type=Path, default=DEFAULT_P1D_DIR)
    p.add_argument("--snapshots", type=str, default="11,10,9,8")
    p.add_argument("--kmax-fit", type=float, default=2.0)
    p.add_argument("--kpar-max-fit", type=float, default=2.5)
    p.add_argument("--kmax-proj", type=float, default=3.0)
    p.add_argument("--nint-proj", type=int, default=180)
    p.add_argument("--qmax-loop", type=float, default=4.0)
    p.add_argument("--nq-loop", type=int, default=10)
    p.add_argument("--nmuq-loop", type=int, default=6)
    p.add_argument("--nphi-loop", type=int, default=6)
    p.add_argument("--stage-a-max-nfev", type=int, default=12)
    p.add_argument("--stage-b-max-nfev", type=int, default=20)
    p.add_argument("--n-starts", type=int, default=1)
    p.add_argument("--start-spread", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=20260313)
    return p.parse_args()


# ============================================================================
# Per-snapshot fitting  (JAX stage-B + JAX projection)
# ============================================================================
def fit_single_bin_jax(
    *,
    cfg,
    z: float,
    p3d_path: Path,
    p1d_path: Path,
    args,
    rng: np.random.Generator,
) -> dict:
    """
    Mirrors fit_single_bin from run_2405_stage1_sherwood_full.py but uses
    IvanovFullModelJAX for the stage-B P3D residuals and project_to_1d_jax
    for the final P1D projection.
    """
    # --- Load Sherwood data ---
    p3d = load_sherwood_flux_p3d(p3d_path)
    p1d = load_sherwood_flux_p1d(p1d_path)
    k_all, mu_all, p_all, counts_all = p3d.flatten_valid()
    mask_3d = (k_all >= cfg.fit.kmin_fit) & (k_all <= float(args.kmax_fit))
    k_fit      = k_all[mask_3d]
    mu_fit     = mu_all[mask_3d]
    p_fit      = p_all[mask_3d]
    counts_fit = counts_all[mask_3d]
    sigma_3d   = pseudo_sigma(p_fit, counts_fit,
                               sigma_frac=cfg.fit.sigma_frac,
                               sigma_floor=cfg.fit.sigma_floor)

    # --- Build linear power + JAX model (once per snapshot) ---
    lp = compute_linear_power_camb(
        h=cfg.cosmology.h,
        omega_b=cfg.cosmology.omega_b,
        omega_cdm=cfg.cosmology.omega_cdm,
        ns=cfg.cosmology.ns,
        As=cfg.cosmology.As,
        z=float(z),
        kmin=cfg.k_grid.kmin,
        kmax=max(cfg.k_grid.kmax, float(args.qmax_loop) + float(args.kmax_fit)),
        nk=max(cfg.k_grid.nk, 1200),
    )
    jax_model = IvanovFullModelJAX(
        lp.k_hmpc, lp.p_lin_h3mpc3, lp.f_growth,
        qmin=float(cfg.k_grid.kmin),
        qmax=float(args.qmax_loop),
        nq=int(args.nq_loop),
        nmuq=int(args.nmuq_loop),
        nphi=int(args.nphi_loop),
    )

    # --- Stage A: fit (b1, b_eta) only with published-relation priors (numpy, fast) ---
    # Stage A uses the numpy IvanovFullModel because we only do ~12 evaluations
    # and the scipy Jacobian finite-difference step benefits from numpy's lower overhead.
    numpy_model = IvanovFullModel(
        lp.k_hmpc, lp.p_lin_h3mpc3, lp.f_growth,
        qmin=float(cfg.k_grid.kmin),
        qmax=float(args.qmax_loop),
        nq=int(args.nq_loop),
        nmuq=int(args.nmuq_loop),
        nphi=int(args.nphi_loop),
    )

    def stage_a_residual(theta):
        b1, b_eta = [float(x) for x in theta]
        params = published_guess(b1=b1, b_eta=b_eta)
        pred = numpy_model.evaluate_components(k_fit, mu_fit, params)["total"]
        return (pred - p_fit) / sigma_3d

    x0_a = np.array([cfg.ivanov_toy.b1, cfg.ivanov_toy.b_eta], dtype=float)
    lo_a = np.array([-1.2, -2.5], dtype=float)
    hi_a = np.array([ 0.05, 1.0], dtype=float)
    fit_a = least_squares(
        stage_a_residual,
        x0=np.clip(x0_a, lo_a + 1e-8, hi_a - 1e-8),
        bounds=(lo_a, hi_a), method="trf", loss="soft_l1", f_scale=1.0,
        max_nfev=int(args.stage_a_max_nfev),
    )
    params_a = published_guess(b1=float(fit_a.x[0]), b_eta=float(fit_a.x[1]))

    # --- Stage B: fit 13 nonlinear parameters with b1, b_eta pinned to stage-A values ---
    #
    # Releasing b1 and b_eta in a 15-parameter joint fit creates a catastrophic
    # degeneracy: many (b1≈0, large b_O) combinations give similar P3D chi2, so
    # the optimizer wanders along a flat valley and returns physically wrong slopes.
    # Pinning b1/b_eta to the stage-A values avoids this: the 13 remaining nonlinear
    # parameters are then well-constrained by the higher-order P3D structure.  The
    # b_O(b1) linear relations emerge naturally from how these fitted nonlinear biases
    # vary with the stage-A b1 across the 4 Sherwood redshift snapshots.
    #
    # JIT warmup: compile for the shape of (k_fit, mu_fit) before the optimizer loop.
    print(f"  [z={z:.1f}] Stage-B JIT warmup ({k_fit.size} P3D points) ...")
    _ = np.asarray(jax_model.evaluate_components(k_fit, mu_fit, params_a)["total"])
    print(f"  [z={z:.1f}] JIT warmup done.")

    x0_b = params_to_theta(params_a)
    lo_b = x0_b.copy();  hi_b = x0_b.copy()
    # Pin b1 and b_eta tightly to stage-A values (effectively frozen).
    b1_a, beta_a = float(fit_a.x[0]), float(fit_a.x[1])
    lo_b[0], hi_b[0] = b1_a   - 5e-4, b1_a   + 5e-4
    lo_b[1], hi_b[1] = beta_a - 5e-4, beta_a + 5e-4
    # Wide search range for nonlinear bias params and counterterms.
    spreads = np.array([0.0, 0.0, 2.5, 2.5, 2.5, 6.0, 4.0, 6.0, 8.0, 20.0, 8.0, 8.0,
                        12.0, 30.0, 30.0], dtype=float)
    lo_b[2:] = x0_b[2:] - spreads[2:]
    hi_b[2:] = x0_b[2:] + spreads[2:]

    def stage_b_residual(theta):
        params = theta_to_params(theta)
        # evaluate_components returns JAX arrays; pull to numpy for scipy
        pred = np.asarray(jax_model.evaluate_components(k_fit, mu_fit, params)["total"])
        return (pred - p_fit) / sigma_3d

    starts = [np.clip(x0_b, lo_b + 1e-8, hi_b - 1e-8)]
    n_starts = max(int(args.n_starts), 1)
    spread   = float(args.start_spread)
    for _ in range(n_starts - 1):
        trial = x0_b.copy()
        # Do not perturb b1/b_eta; only perturb the nonlinear block.
        trial[2:] += rng.normal(
            scale=spread * np.maximum(np.abs(x0_b[2:]), 0.2),
            size=x0_b.size - 2,
        )
        starts.append(np.clip(trial, lo_b + 1e-8, hi_b - 1e-8))

    best_fit_b = None
    best_cost  = np.inf
    for start in starts:
        trial_fit = least_squares(
            stage_b_residual, x0=start, bounds=(lo_b, hi_b),
            method="trf", loss="soft_l1", f_scale=1.0,
            max_nfev=int(args.stage_b_max_nfev),
        )
        if trial_fit.cost < best_cost:
            best_cost  = float(trial_fit.cost)
            best_fit_b = trial_fit

    if best_fit_b is None:
        raise RuntimeError("Stage-B multistart produced no result.")

    fit_b    = best_fit_b
    params_b = theta_to_params(fit_b.x)

    pred_p3d = np.asarray(jax_model.evaluate_components(k_all,  mu_all,  params_b)["total"])
    pred_fit = np.asarray(jax_model.evaluate_components(k_fit,  mu_fit,  params_b)["total"])
    chi2_3d  = float(np.sum(((pred_fit - p_fit) / sigma_3d) ** 2))
    dof_3d   = max(int(k_fit.size - fit_b.x.size), 1)

    # --- P1D projection via JAX ---
    kp_valid = p1d.kp_hmpc[p1d.valid_mask()]
    p1d_valid = p1d.p1d_hmpc[p1d.valid_mask()]
    mask_1d  = (kp_valid >= 0.03) & (kp_valid <= float(args.kpar_max_fit))
    kp_fit   = kp_valid[mask_1d]
    p1d_fit  = p1d_valid[mask_1d]

    grid = make_jax_p3d_grid(kp_fit, float(args.kmax_proj))
    p3d_flat = jax_model.evaluate_components(grid.kk_flat, grid.mm_flat, params_b)["total"]
    p3d_g    = p3d_flat.reshape(grid.nk_g, grid.nmu_g)
    raw1d    = np.asarray(project_to_1d_jax(
        grid.kpar, grid.k_g, grid.mu_g, p3d_g,
        float(args.kmax_proj), int(args.nint_proj),
    ))

    # --- 1D counterterm fit (numpy, 2 free params: C0, C2) ---
    sigma_1d = 0.05 * np.maximum(np.abs(p1d_fit), 1e-8) + 0.02

    def residual_1d(theta):
        c0, c2 = [float(x) for x in theta]
        return (raw1d + c0 + c2 * kp_fit ** 2 - p1d_fit) / sigma_1d

    fit_1d = least_squares(
        residual_1d, x0=np.array([0.0, 0.0]),
        bounds=(np.array([-4.0, -4.0]), np.array([4.0, 4.0])),
        method="trf", loss="soft_l1", f_scale=1.0, max_nfev=8000,
    )
    c0_1d, c2_1d = [float(x) for x in fit_1d.x]
    pred1d = raw1d + c0_1d + c2_1d * kp_fit ** 2
    chi2_1d = float(np.sum(((pred1d - p1d_fit) / sigma_1d) ** 2))
    dof_1d  = max(int(kp_fit.size - 2), 1)

    return {
        "params_stage_a": params_a,
        "params_stage_b": params_b,
        "theta_stage_a":  fit_a.x.copy(),
        "theta_stage_b":  fit_b.x.copy(),
        "p3d_pred_all":   pred_p3d,
        "p1d_k_fit":      kp_fit,
        "p1d_data_fit":   p1d_fit,
        "p1d_raw":        raw1d,
        "p1d_fit":        pred1d,
        "p3d_diag": {
            "chi2": chi2_3d, "chi2_dof": float(chi2_3d / dof_3d),
            "n_data": int(k_fit.size), "n_dim": int(fit_b.x.size),
            "nfev_stage_a": int(fit_a.nfev), "nfev_stage_b": int(fit_b.nfev),
        },
        "p1d_diag": {
            "chi2": chi2_1d, "chi2_dof": float(chi2_1d / dof_1d),
            "n_data": int(kp_fit.size), "C0_1d": c0_1d, "C2_1d": c2_1d,
        },
        "data": {
            "k_all": k_all, "mu_all": mu_all, "p3d_all": p_all,
            "kp_valid": kp_valid, "p1d_valid": p1d_valid,
        },
    }


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    args = parse_args()
    cfg  = load_config(args.config)
    snapshots = [int(x) for x in args.snapshots.split(",") if x.strip()]
    unknown = [s for s in snapshots if s not in SNAP_TO_Z]
    if unknown:
        raise ValueError(f"Unsupported snapshots {unknown}; supported: {sorted(SNAP_TO_Z)}")

    run_paths = init_run_dir(cfg.run.output_root, tag="repro_2405_stage1_sherwood_full_jax")
    meta = build_repro_metadata(args.config)
    meta.update({
        "p3d_dir": str(args.p3d_dir), "p1d_dir": str(args.p1d_dir),
        "snapshots": snapshots,
        "kmax_fit": float(args.kmax_fit), "kpar_max_fit": float(args.kpar_max_fit),
        "kmax_proj": float(args.kmax_proj), "nint_proj": int(args.nint_proj),
        "qmax_loop": float(args.qmax_loop), "nq_loop": int(args.nq_loop),
        "nmuq_loop": int(args.nmuq_loop), "nphi_loop": int(args.nphi_loop),
        "stage_a_max_nfev": int(args.stage_a_max_nfev),
        "stage_b_max_nfev": int(args.stage_b_max_nfev),
        "n_starts": int(args.n_starts), "start_spread": float(args.start_spread),
        "seed": int(args.seed),
        "jax_device": f"GPU:{os.environ['CUDA_VISIBLE_DEVICES']}",
    })
    write_json(run_paths.logs_dir / "repro_metadata.json", meta)

    rng = np.random.default_rng(args.seed)
    rows_z: list[dict] = []
    b1_values: list[float] = []

    op_names = [
        "b_eta", "b_delta2", "b_G2", "b_KK_par", "b_delta_eta", "b_eta2",
        "b_Pi2_par", "b_Pi3_par", "b_delta_Pi2_par", "b_KPi2_par", "b_eta_Pi2_par",
        "c0_ct", "c2_ct", "c4_ct", "C0_1d", "C2_1d",
    ]
    op_data = {name: [] for name in op_names}

    t0 = time.perf_counter()
    for snapshot in snapshots:
        z = SNAP_TO_Z[snapshot]
        p3d_path = choose_p3d_file(args.p3d_dir, snapshot)
        p1d_path = choose_p1d_file(args.p1d_dir, snapshot)
        print(f"[main] Fitting snapshot {snapshot} (z={z:.1f}) ...")
        fit = fit_single_bin_jax(
            cfg=cfg, z=z, p3d_path=p3d_path, p1d_path=p1d_path, args=args, rng=rng,
        )
        print(f"[main] z={z:.1f}: P3D chi2/dof={fit['p3d_diag']['chi2_dof']:.3f}  "
              f"P1D chi2/dof={fit['p1d_diag']['chi2_dof']:.3f}  "
              f"nfev_a={fit['p3d_diag']['nfev_stage_a']}  "
              f"nfev_b={fit['p3d_diag']['nfev_stage_b']}")

        z_tag = f"z{z:.1f}".replace(".", "p")
        f_p3d = run_paths.figures_dir / f"51_{z_tag}_p3d_residuals.png"
        f_p1d = run_paths.figures_dir / f"52_{z_tag}_p1d_projection.png"
        make_p3d_residual_plot(
            k_all=fit["data"]["k_all"], mu_all=fit["data"]["mu_all"],
            p_data=fit["data"]["p3d_all"], p_model=fit["p3d_pred_all"],
            z=z, out_path=f_p3d,
        )
        make_p1d_projection_plot(
            kp=fit["p1d_k_fit"], p_data=fit["p1d_data_fit"],
            p_raw=fit["p1d_raw"], p_fit=fit["p1d_fit"],
            z=z, out_path=f_p1d,
        )

        params = fit["params_stage_b"]
        row = {
            "snapshot": int(snapshot), "redshift": float(z),
            "p3d_file": str(p3d_path), "p1d_file": str(p1d_path),
            "b1": float(params.b1), "b_eta": float(params.b_eta),
            "b_delta2": float(params.b_delta2), "b_G2": float(params.b_G2),
            "b_KK_par": float(params.b_KK_par), "b_delta_eta": float(params.b_delta_eta),
            "b_eta2": float(params.b_eta2), "b_Pi2_par": float(params.b_Pi2_par),
            "b_Pi3_par": float(params.b_Pi3_par),
            "b_delta_Pi2_par": float(params.b_delta_Pi2_par),
            "b_KPi2_par": float(params.b_KPi2_par),
            "b_eta_Pi2_par": float(params.b_eta_Pi2_par),
            "c0_ct": float(params.c0_ct), "c2_ct": float(params.c2_ct),
            "c4_ct": float(params.c4_ct),
            "C0_1d": float(fit["p1d_diag"]["C0_1d"]),
            "C2_1d": float(fit["p1d_diag"]["C2_1d"]),
            "p3d_chi2_dof": float(fit["p3d_diag"]["chi2_dof"]),
            "p1d_chi2_dof": float(fit["p1d_diag"]["chi2_dof"]),
            "params_stage_a": asdict(fit["params_stage_a"]),
            "params_stage_b": asdict(params),
            "figure_p3d": str(f_p3d), "figure_p1d": str(f_p1d),
        }
        rows_z.append(row)

        b1_values.append(float(params.b1))
        for name in op_names[:-2]:
            op_data[name].append(float(getattr(params, name)))
        op_data["C0_1d"].append(float(fit["p1d_diag"]["C0_1d"]))
        op_data["C2_1d"].append(float(fit["p1d_diag"]["C2_1d"]))

        np.savez(
            run_paths.arrays_dir / f"{z_tag}_fit_arrays.npz",
            k_all=fit["data"]["k_all"], mu_all=fit["data"]["mu_all"],
            p3d_data=fit["data"]["p3d_all"], p3d_pred=fit["p3d_pred_all"],
            kp_fit=fit["p1d_k_fit"], p1d_data=fit["p1d_data_fit"],
            p1d_raw=fit["p1d_raw"], p1d_fit=fit["p1d_fit"],
            theta_stage_a=fit["theta_stage_a"], theta_stage_b=fit["theta_stage_b"],
        )

    b1     = np.asarray(b1_values, dtype=float)
    y_by_name = {name: np.asarray(vals, dtype=float) for name, vals in op_data.items()}

    prior_rows: list[dict] = []
    for name, yy in y_by_name.items():
        out_name = {"c0_ct": "c0_3d", "c2_ct": "c2_3d", "c4_ct": "c4_3d"}.get(name, name)
        slope, intercept, rmse = fit_linear_relation(b1, yy)
        prior_rows.append({
            "operator_name": out_name, "A_O": slope, "B_O": intercept,
            "fit_rmse": rmse, "n_points": int(b1.size),
            "fit_variant": "full_ivanov_stage1_repro_jax",
        })

    if not any(r["operator_name"] == "loop_amp" for r in prior_rows):
        prior_rows.append({
            "operator_name": "loop_amp", "A_O": 0.0, "B_O": 1.0,
            "fit_rmse": 0.0, "n_points": int(b1.size),
            "fit_variant": "full_ivanov_stage1_repro_jax_compat",
        })

    f_prior = run_paths.figures_dir / "61_linear_prior_relations.png"
    make_linear_prior_plot(b1=b1, y_by_name=y_by_name, fit_rows=prior_rows, out_path=f_prior)

    write_csv(
        run_paths.logs_dir / "z_bin_fit_summary.csv", rows_z,
        fieldnames=[
            "snapshot", "redshift", "p3d_file", "p1d_file", "b1", "b_eta",
            "b_delta2", "b_G2", "b_KK_par", "b_delta_eta", "b_eta2",
            "b_Pi2_par", "b_Pi3_par", "b_delta_Pi2_par", "b_KPi2_par", "b_eta_Pi2_par",
            "c0_ct", "c2_ct", "c4_ct", "C0_1d", "C2_1d",
            "p3d_chi2_dof", "p1d_chi2_dof",
            "params_stage_a", "params_stage_b", "figure_p3d", "figure_p1d",
        ],
    )
    write_csv(
        run_paths.logs_dir / "sherwood_prior_linear_fits.csv", prior_rows,
        fieldnames=["operator_name", "A_O", "B_O", "fit_rmse", "n_points", "fit_variant"],
    )

    fig_target   = Path("results/figures");   fig_target.mkdir(parents=True, exist_ok=True)
    table_target = Path("results/tables");    table_target.mkdir(parents=True, exist_ok=True)
    for row in rows_z:
        shutil.copy2(Path(row["figure_p3d"]), fig_target / Path(row["figure_p3d"]).name)
        shutil.copy2(Path(row["figure_p1d"]), fig_target / Path(row["figure_p1d"]).name)
    shutil.copy2(f_prior, fig_target / f"{f_prior.stem}_full_jax{f_prior.suffix}")
    shutil.copy2(
        run_paths.logs_dir / "z_bin_fit_summary.csv",
        table_target / "z_bin_fit_summary_full_jax.csv",
    )
    shutil.copy2(
        run_paths.logs_dir / "sherwood_prior_linear_fits.csv",
        table_target / "sherwood_prior_linear_fits_full.csv",  # same name as numpy version
    )

    summary = {
        "run_dir": str(run_paths.run_dir),
        "runtime_sec": float(time.perf_counter() - t0),
        "snapshots": snapshots,
        "redshifts": [float(SNAP_TO_Z[s]) for s in snapshots],
        "kmax_fit": float(args.kmax_fit), "kpar_max_fit": float(args.kpar_max_fit),
        "kmax_proj": float(args.kmax_proj),
        "integration_grid": {
            "qmax_loop": float(args.qmax_loop), "nq": int(args.nq_loop),
            "nmuq": int(args.nmuq_loop), "nphi": int(args.nphi_loop),
            "nint_proj": int(args.nint_proj),
        },
        "jax_device": f"GPU:{os.environ['CUDA_VISIBLE_DEVICES']}",
        "rows_z": rows_z, "linear_prior_fits": prior_rows,
        "figures": ([str(f_prior)]
                    + [str(r["figure_p3d"]) for r in rows_z]
                    + [str(r["figure_p1d"]) for r in rows_z]),
        "tables": [
            str(run_paths.logs_dir / "z_bin_fit_summary.csv"),
            str(run_paths.logs_dir / "sherwood_prior_linear_fits.csv"),
        ],
        "notes": [
            "JAX/GPU stage-B P3D residuals via IvanovFullModelJAX.",
            "P1D projection via project_to_1d_jax (JAX vmap + trapz).",
            "Stage-A pre-fit uses numpy IvanovFullModel (fast, ~12 evaluations).",
            "Output sherwood_prior_linear_fits.csv is compatible with the SDSS stage-1 scripts.",
        ],
    }
    write_json(run_paths.logs_dir / "summary.json", summary)
    print(f"2405 Stage-1 full-basis Sherwood (JAX) complete: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
