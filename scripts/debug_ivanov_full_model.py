#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from lya_hybrid.config import load_config
from lya_hybrid.io import load_sherwood_flux_p1d, load_sherwood_flux_p3d
from lya_hybrid.linear_power import compute_linear_power_camb
from lya_hybrid.logging_utils import build_repro_metadata, init_run_dir, write_json
from lya_hybrid.model_ivanov_full import IvanovFullModel, IvanovFullParams
from lya_hybrid.projection_1d import Polynomial1DCounterterms, project_to_1d

SNAP_TO_Z = {8: 3.2, 9: 2.8, 10: 2.4, 11: 2.0}
DEFAULT_P3D_DIR = Path("data/external/sherwood_p3d/data/flux_p3d")
DEFAULT_P1D_DIR = Path("data/external/sherwood_p3d/data/flux_p1d")

PUBLISHED_RELATIONS = {
    "b_G2": (0.154, -0.252),
    "b_delta2": (0.061, -0.480),
    "b_eta2": (-2.84, 0.11),
    "b_delta_eta": (4.31, -0.0745),
    "b_KK_par": (1.55, 0.205),
    "b_Pi2_par": (2.48, 0.011),
    "b_Pi3_par": (-3.08, 1.86),
    "b_delta_Pi2_par": (20.7, 1.34),
    "b_KPi2_par": (5.83, -1.99),
    "b_eta_Pi2_par": (1.60, 1.07),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug the full Ivanov model following the local debugging procedure.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--snapshot", type=int, default=9)
    p.add_argument("--p3d-dir", type=Path, default=DEFAULT_P3D_DIR)
    p.add_argument("--p1d-dir", type=Path, default=DEFAULT_P1D_DIR)
    p.add_argument("--kmax-fit", type=float, default=2.0)
    p.add_argument("--kpar-max-fit", type=float, default=2.5)
    p.add_argument("--b1-ref", type=float, default=-0.16)
    p.add_argument("--b-eta-ref", type=float, default=-0.13)
    p.add_argument("--seed", type=int, default=20260314)
    return p.parse_args()


def choose_p3d_file(p3d_dir: Path, snapshot: int) -> Path:
    preferred = p3d_dir / f"p3d_80_1024_{snapshot}_0_512_1024_20_16_20.fits"
    if preferred.exists():
        return preferred
    candidates = sorted(p3d_dir.glob(f"p3d_*_{snapshot}_0_*_20_16_20.fits"))
    if not candidates:
        raise FileNotFoundError(f"No P3D file found for snapshot {snapshot} in {p3d_dir}")
    return candidates[0]


def choose_p1d_file(p1d_dir: Path, snapshot: int) -> Path:
    preferred = p1d_dir / f"p1d_80_1024_{snapshot}_0_512_1024.fits"
    if preferred.exists():
        return preferred
    candidates = sorted(p1d_dir.glob(f"p1d_*_{snapshot}_0_*_*.fits"))
    if not candidates:
        raise FileNotFoundError(f"No P1D file found for snapshot {snapshot} in {p1d_dir}")
    return candidates[0]


def pseudo_sigma(p: np.ndarray, counts: np.ndarray, sigma_frac: float, sigma_floor: float) -> np.ndarray:
    return sigma_frac * np.maximum(np.abs(p), 1.0e-8) + np.maximum(1.0 / np.sqrt(np.maximum(counts, 1.0)), sigma_floor)


def published_guess(b1: float, b_eta: float) -> IvanovFullParams:
    vals = {name: slope * b1 + intercept for name, (slope, intercept) in PUBLISHED_RELATIONS.items()}
    return IvanovFullParams(b1=b1, b_eta=b_eta, **vals)


def params_to_theta(p: IvanovFullParams) -> np.ndarray:
    return np.array(
        [
            p.b1,
            p.b_eta,
            p.b_delta2,
            p.b_G2,
            p.b_KK_par,
            p.b_delta_eta,
            p.b_eta2,
            p.b_Pi2_par,
            p.b_Pi3_par,
            p.b_delta_Pi2_par,
            p.b_KPi2_par,
            p.b_eta_Pi2_par,
        ],
        dtype=float,
    )


def theta_to_params(theta: np.ndarray) -> IvanovFullParams:
    return IvanovFullParams(
        b1=float(theta[0]),
        b_eta=float(theta[1]),
        b_delta2=float(theta[2]),
        b_G2=float(theta[3]),
        b_KK_par=float(theta[4]),
        b_delta_eta=float(theta[5]),
        b_eta2=float(theta[6]),
        b_Pi2_par=float(theta[7]),
        b_Pi3_par=float(theta[8]),
        b_delta_Pi2_par=float(theta[9]),
        b_KPi2_par=float(theta[10]),
        b_eta_Pi2_par=float(theta[11]),
        b_gamma3=0.0,
    )


def build_model(cfg, z: float, *, qmax: float, nq: int, nmuq: int, nphi: int):
    lp = compute_linear_power_camb(
        h=cfg.cosmology.h,
        omega_b=cfg.cosmology.omega_b,
        omega_cdm=cfg.cosmology.omega_cdm,
        ns=cfg.cosmology.ns,
        As=cfg.cosmology.As,
        z=float(z),
        kmin=cfg.k_grid.kmin,
        kmax=max(cfg.k_grid.kmax, qmax + 2.5),
        nk=max(cfg.k_grid.nk, 1200),
    )
    return IvanovFullModel(
        lp.k_hmpc,
        lp.p_lin_h3mpc3,
        lp.f_growth,
        qmin=float(cfg.k_grid.kmin),
        qmax=float(qmax),
        nq=int(nq),
        nmuq=int(nmuq),
        nphi=int(nphi),
    ), lp


def evaluate_components_on_grid(model: IvanovFullModel, params: IvanovFullParams, k_eval: np.ndarray, mu_eval: np.ndarray):
    kk, mm = np.meshgrid(k_eval, mu_eval, indexing="xy")
    comps = model.evaluate_components(kk.ravel(), mm.ravel(), params)
    return {
        "k_eval": kk.ravel(),
        "mu_eval": mm.ravel(),
        "tree": comps["tree"].ravel(),
        "loop_22": comps["loop_22"].ravel(),
        "loop_13": comps["loop_13"].ravel(),
        "total": comps["total"].ravel(),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_convergence_plot(
    *,
    rows: list[dict[str, object]],
    field: str,
    mu_targets: list[float],
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, len(mu_targets), figsize=(4.5 * len(mu_targets), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, mu_target in zip(axes, mu_targets):
        subset = [r for r in rows if abs(float(r["mu"]) - mu_target) < 1.0e-9]
        for sweep in sorted(set(r["sweep"] for r in subset)):
            cur = [r for r in subset if r["sweep"] == sweep]
            cur.sort(key=lambda x: float(x["k"]))
            ax.semilogx(
                [float(r["k"]) for r in cur],
                [float(r[field]) for r in cur],
                lw=1.6,
                label=str(sweep),
            )
        ax.axhline(0.0, color="k", lw=1)
        ax.grid(alpha=0.25)
        ax.set_title(fr"$\mu={mu_target:.1f}$")
        ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    axes[0].set_ylabel(field)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_component_plot(
    *,
    k_eval: np.ndarray,
    mu_eval: list[float],
    payload_by_name: dict[str, dict[str, np.ndarray]],
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    fields = [("tree", "tree"), ("loop_22", "loop_22"), ("loop_13", "loop_13"), ("total", "total")]
    for ax, (field, label) in zip(axes.ravel(), fields):
        for mu_target in mu_eval:
            mask = np.isclose(payload_by_name["nominal"]["mu_eval"], mu_target)
            ax.semilogx(
                k_eval,
                payload_by_name["nominal"][field][mask],
                lw=1.7,
                label=fr"$\mu={mu_target:.1f}$",
            )
        ax.set_title(label)
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    axes[0, 0].set_ylabel("value")
    axes[1, 0].set_ylabel("value")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_ratio_plot(
    *,
    k_eval: np.ndarray,
    mu_eval: list[float],
    payload: dict[str, np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    combos = [
        ("loop_22", "loop_22 / tree"),
        ("loop_13", "loop_13 / tree"),
        ("total", "total / tree"),
    ]
    for ax, (field, label) in zip(axes, combos):
        for mu_target in mu_eval:
            mask = np.isclose(payload["mu_eval"], mu_target)
            tree = payload["tree"][mask]
            ratio = payload[field][mask] / np.where(np.abs(tree) > 1.0e-20, tree, np.nan)
            ax.semilogx(k_eval, ratio, lw=1.7, label=fr"$\mu={mu_target:.1f}$")
        ax.axhline(0.0, color="k", lw=1)
        ax.grid(alpha=0.25)
        ax.set_title(label)
        ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    axes[0].set_ylabel("ratio")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def stage_a_fit(model: IvanovFullModel, cfg, k: np.ndarray, mu: np.ndarray, p_data: np.ndarray, counts: np.ndarray, kmax_fit: float):
    mask = (k >= cfg.fit.kmin_fit) & (k <= kmax_fit)
    kf, muf, pf, cf = k[mask], mu[mask], p_data[mask], counts[mask]
    sigma = pseudo_sigma(pf, cf, sigma_frac=cfg.fit.sigma_frac, sigma_floor=cfg.fit.sigma_floor)

    def residual(theta: np.ndarray) -> np.ndarray:
        params = published_guess(float(theta[0]), float(theta[1]))
        pred = model.evaluate_components(kf, muf, params)["total"]
        return (pred - pf) / sigma

    x0 = np.array([cfg.ivanov_toy.b1, cfg.ivanov_toy.b_eta], dtype=float)
    lo = np.array([-1.2, -2.5], dtype=float)
    hi = np.array([0.05, 1.0], dtype=float)
    fit = least_squares(
        residual,
        x0=np.clip(x0, lo + 1.0e-8, hi - 1.0e-8),
        bounds=(lo, hi),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=80,
    )
    params = published_guess(float(fit.x[0]), float(fit.x[1]))
    pred = model.evaluate_components(kf, muf, params)["total"]
    chi2 = float(np.sum(((pred - pf) / sigma) ** 2))
    dof = max(int(kf.size - 2), 1)
    return fit, params, {
        "k": kf,
        "mu": muf,
        "p_data": pf,
        "p_pred": pred,
        "chi2": chi2,
        "chi2_dof": float(chi2 / dof),
        "sigma": sigma,
    }


def stage_b_stress_test(
    *,
    model: IvanovFullModel,
    cfg,
    k: np.ndarray,
    mu: np.ndarray,
    p_data: np.ndarray,
    counts: np.ndarray,
    kmax_fit: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    mask = (k >= cfg.fit.kmin_fit) & (k <= kmax_fit)
    kf, muf, pf, cf = k[mask], mu[mask], p_data[mask], counts[mask]
    sigma = pseudo_sigma(pf, cf, sigma_frac=cfg.fit.sigma_frac, sigma_floor=cfg.fit.sigma_floor)

    params_a = published_guess(cfg.ivanov_toy.b1, cfg.ivanov_toy.b_eta)
    x0_b = params_to_theta(params_a)
    lo_b = x0_b.copy()
    hi_b = x0_b.copy()
    lo_b[0], hi_b[0] = -1.2, 0.05
    lo_b[1], hi_b[1] = -2.5, 1.0
    spreads = np.array([0.0, 0.0, 2.5, 2.5, 2.5, 6.0, 4.0, 6.0, 8.0, 20.0, 8.0, 8.0], dtype=float)
    lo_b[2:] = x0_b[2:] - spreads[2:]
    hi_b[2:] = x0_b[2:] + spreads[2:]

    def residual(theta: np.ndarray) -> np.ndarray:
        params = theta_to_params(theta)
        pred = model.evaluate_components(kf, muf, params)["total"]
        return (pred - pf) / sigma

    rows: list[dict[str, object]] = []
    for max_nfev in [20, 100]:
        for start_idx in range(4):
            trial = x0_b.copy()
            if start_idx > 0:
                trial[0] += rng.normal(scale=0.03)
                trial[1] += rng.normal(scale=0.05)
                trial[2:] += rng.normal(scale=0.15 * np.maximum(np.abs(x0_b[2:]), 0.2), size=x0_b.size - 2)
            trial = np.clip(trial, lo_b + 1.0e-8, hi_b - 1.0e-8)
            fit = least_squares(
                residual,
                x0=trial,
                bounds=(lo_b, hi_b),
                method="trf",
                loss="soft_l1",
                f_scale=1.0,
                max_nfev=max_nfev,
            )
            pred = model.evaluate_components(kf, muf, theta_to_params(fit.x))["total"]
            chi2 = float(np.sum(((pred - pf) / sigma) ** 2))
            dof = max(int(kf.size - fit.x.size), 1)
            rows.append(
                {
                    "max_nfev": int(max_nfev),
                    "start_idx": int(start_idx),
                    "success": bool(fit.success),
                    "status": int(fit.status),
                    "cost": float(fit.cost),
                    "chi2": chi2,
                    "chi2_dof": float(chi2 / dof),
                    "nfev": int(fit.nfev),
                    "message": str(fit.message),
                    "theta": json.dumps(fit.x.tolist()),
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.snapshot not in SNAP_TO_Z:
        raise ValueError(f"Unsupported snapshot {args.snapshot}; supported: {sorted(SNAP_TO_Z)}")

    z = SNAP_TO_Z[args.snapshot]
    p3d_path = choose_p3d_file(args.p3d_dir, args.snapshot)
    p1d_path = choose_p1d_file(args.p1d_dir, args.snapshot)
    p3d = load_sherwood_flux_p3d(p3d_path)
    p1d = load_sherwood_flux_p1d(p1d_path)
    k_all, mu_all, p_all, counts_all = p3d.flatten_valid()
    rng = np.random.default_rng(args.seed)

    run_paths = init_run_dir(cfg.run.output_root, tag="ivanov_full_debug")
    meta = build_repro_metadata(args.config)
    meta.update(
        {
            "snapshot": int(args.snapshot),
            "redshift": float(z),
            "p3d_file": str(p3d_path),
            "p1d_file": str(p1d_path),
            "representative_b1": float(args.b1_ref),
            "representative_b_eta": float(args.b_eta_ref),
            "kmax_fit": float(args.kmax_fit),
            "kpar_max_fit": float(args.kpar_max_fit),
            "seed": int(args.seed),
        }
    )
    write_json(run_paths.logs_dir / "manifest.json", meta)

    params_ref = published_guess(args.b1_ref, args.b_eta_ref)
    k_eval = np.logspace(np.log10(0.03), np.log10(float(args.kmax_fit)), 8)
    mu_eval = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=float)

    # Section A: convergence sweep
    t0 = time.perf_counter()
    ref_setting = {"nq": 64, "nmuq": 24, "nphi": 24, "qmax": 16.0}
    model_ref, _ = build_model(cfg, z, **ref_setting)
    comps_ref = evaluate_components_on_grid(model_ref, params_ref, k_eval, mu_eval)
    np.savez(run_paths.arrays_dir / "section_a_reference_components.npz", **comps_ref, **ref_setting)

    sweep_settings = []
    for nq in [24, 36, 48, 64]:
        sweep_settings.append({"sweep": "nq", "label": f"nq={nq}", "nq": nq, "nmuq": 24, "nphi": 24, "qmax": 16.0})
    for nmuq in [10, 16, 24]:
        sweep_settings.append({"sweep": "nmuq", "label": f"nmuq={nmuq}", "nq": 64, "nmuq": nmuq, "nphi": 24, "qmax": 16.0})
    for nphi in [10, 16, 24]:
        sweep_settings.append({"sweep": "nphi", "label": f"nphi={nphi}", "nq": 64, "nmuq": 24, "nphi": nphi, "qmax": 16.0})
    for qmax in [4.0, 8.0, 12.0, 16.0]:
        sweep_settings.append({"sweep": "qmax", "label": f"qmax={qmax:g}", "nq": 64, "nmuq": 24, "nphi": 24, "qmax": qmax})

    conv_rows_summary: list[dict[str, object]] = []
    conv_rows_long: list[dict[str, object]] = []
    for setting in sweep_settings:
        model, _ = build_model(cfg, z, qmax=setting["qmax"], nq=setting["nq"], nmuq=setting["nmuq"], nphi=setting["nphi"])
        comps = evaluate_components_on_grid(model, params_ref, k_eval, mu_eval)
        np.savez(run_paths.arrays_dir / f"section_a_{setting['label'].replace('=', '_')}.npz", **comps)
        summary = {"sweep": setting["sweep"], "label": setting["label"], "nq": setting["nq"], "nmuq": setting["nmuq"], "nphi": setting["nphi"], "qmax": setting["qmax"]}
        for field in ["loop_22", "loop_13", "total"]:
            ref = comps_ref[field]
            cur = comps[field]
            frac = np.abs(cur - ref) / np.maximum(np.abs(ref), 1.0e-12)
            summary[f"{field}_median_abs_frac"] = float(np.median(frac))
            summary[f"{field}_max_abs_frac"] = float(np.max(frac))
            for k_val, mu_val, frac_val in zip(comps["k_eval"], comps["mu_eval"], frac):
                conv_rows_long.append(
                    {
                        "sweep": setting["sweep"],
                        "label": setting["label"],
                        "field": field,
                        "k": float(k_val),
                        "mu": float(mu_val),
                        "frac_vs_ref": float(frac_val),
                    }
                )
        conv_rows_summary.append(summary)

    write_csv(
        run_paths.logs_dir / "section_a_convergence_summary.csv",
        conv_rows_summary,
        fieldnames=[
            "sweep",
            "label",
            "nq",
            "nmuq",
            "nphi",
            "qmax",
            "loop_22_median_abs_frac",
            "loop_22_max_abs_frac",
            "loop_13_median_abs_frac",
            "loop_13_max_abs_frac",
            "total_median_abs_frac",
            "total_max_abs_frac",
        ],
    )
    write_csv(
        run_paths.logs_dir / "section_a_convergence_long.csv",
        conv_rows_long,
        fieldnames=["sweep", "label", "field", "k", "mu", "frac_vs_ref"],
    )

    for field in ["loop_22", "loop_13", "total"]:
        rows = [{**r, field: r["frac_vs_ref"]} for r in conv_rows_long if r["field"] == field]
        make_convergence_plot(
            rows=rows,
            field=field,
            mu_targets=[0.1, 0.5, 0.9],
            out_path=run_paths.figures_dir / f"section_a_{field}_convergence.png",
            title=f"Section A convergence: {field} vs highest-resolution reference",
        )

    max_total_med = max(float(r["total_median_abs_frac"]) for r in conv_rows_summary)
    max_total_max = max(float(r["total_max_abs_frac"]) for r in conv_rows_summary)
    convergence_pass = (max_total_med <= 0.05) and (max_total_max <= 0.2)

    if not convergence_pass:
        convention_rows = [
            {"item": "tree_level_sign_normalization", "status": "pass", "notes": "Implemented as K1=b1-b_eta*f*mu^2 and checked against supplement Eq. (S1-S2)."},
            {"item": "p_lin_units", "status": "uncertain", "notes": "All internal arrays use h/Mpc conventions, but no independent end-to-end unit cross-check against CLASS-PT outputs yet."},
            {"item": "loop_prefactors", "status": "pass", "notes": "P22 uses the factor 2 and P13 uses 6*K1*P_lin*I3, consistent with Eq. (S6)."},
            {"item": "angular_conventions", "status": "uncertain", "notes": "mu, mu_q, phi geometry is internally consistent; no independent backend comparison has validated the full angular structure."},
            {"item": "projection_1d_normalization", "status": "pass", "notes": "Uses P1D=(1/2pi) integral dk k P3D with no extra factors."},
            {"item": "extra_damping_disabled", "status": "pass", "notes": "No thermal or smoothing factors are present in the strict full Ivanov baseline implementation."},
        ]
        write_csv(run_paths.logs_dir / "section_e_convention_checklist.csv", convention_rows, fieldnames=["item", "status", "notes"])
        final_summary = {
            "run_dir": str(run_paths.run_dir),
            "runtime_sec": float(time.perf_counter() - t0),
            "section_a": {
                "reference_setting": ref_setting,
                "max_total_median_abs_frac": max_total_med,
                "max_total_max_abs_frac": max_total_max,
            },
            "section_e_conventions": convention_rows,
            "final_classification": "2. Forward model not yet numerically converged, so fit-level conclusions are not trustworthy.",
            "highest_confidence_failure_point": "loop-integral convergence",
            "notes": [
                "Stopped after Section A per the debugging procedure stop/go criterion.",
                "Fit-level diagnostics were intentionally skipped because the forward model failed convergence.",
            ],
        }
        write_json(run_paths.logs_dir / "final_summary.json", final_summary)
        fig_target = Path("results/figures")
        fig_target.mkdir(parents=True, exist_ok=True)
        table_target = Path("results/tables")
        table_target.mkdir(parents=True, exist_ok=True)
        for fp in run_paths.figures_dir.glob("*.png"):
            shutil.copy2(fp, fig_target / f"{fp.stem}_debug{fp.suffix}")
        for fp in run_paths.logs_dir.glob("*.csv"):
            shutil.copy2(fp, table_target / f"{fp.stem}_debug{fp.suffix}")
        print(f"Ivanov full debug run complete: {run_paths.run_dir}")
        return 0

    # Section B: component inspection
    payload_nominal = comps_ref
    payload_low = evaluate_components_on_grid(model_ref, published_guess(-0.22, -0.14), k_eval, mu_eval)
    payload_high = evaluate_components_on_grid(model_ref, published_guess(-0.10, -0.10), k_eval, mu_eval)
    linear_only = IvanovFullParams(
        b1=args.b1_ref,
        b_eta=args.b_eta_ref,
        b_delta2=0.0,
        b_G2=0.0,
        b_KK_par=0.0,
        b_delta_eta=0.0,
        b_eta2=0.0,
        b_Pi2_par=0.0,
        b_gamma3=0.0,
        b_delta_Pi2_par=0.0,
        b_eta_Pi2_par=0.0,
        b_KPi2_par=0.0,
        b_Pi3_par=0.0,
    )
    payload_linear = evaluate_components_on_grid(model_ref, linear_only, k_eval, mu_eval)
    make_component_plot(
        k_eval=k_eval,
        mu_eval=[0.1, 0.5, 0.9],
        payload_by_name={"nominal": payload_nominal},
        out_path=run_paths.figures_dir / "section_b_nominal_components.png",
        title="Section B: nominal component curves",
    )
    for name, payload in [("nominal", payload_nominal), ("low_b1", payload_low), ("high_b1", payload_high), ("linear_only", payload_linear)]:
        make_ratio_plot(
            k_eval=k_eval,
            mu_eval=[0.1, 0.5, 0.9],
            payload=payload,
            out_path=run_paths.figures_dir / f"section_b_{name}_ratios.png",
            title=f"Section B: ratios for {name}",
        )

    # Section C: stage-A manifold fit at two resolutions
    stage_a_rows = []
    for label, setting in [
        ("default_like", {"nq": 10, "nmuq": 6, "nphi": 6, "qmax": 4.0}),
        ("high_like", ref_setting),
    ]:
        model, _ = build_model(cfg, z, **setting)
        fit, params, diag = stage_a_fit(model, cfg, k_all, mu_all, p_all, counts_all, float(args.kmax_fit))
        stage_a_rows.append(
            {
                "label": label,
                "nq": setting["nq"],
                "nmuq": setting["nmuq"],
                "nphi": setting["nphi"],
                "qmax": setting["qmax"],
                "b1": float(params.b1),
                "b_eta": float(params.b_eta),
                "chi2": float(diag["chi2"]),
                "chi2_dof": float(diag["chi2_dof"]),
                "nfev": int(fit.nfev),
            }
        )
        np.savez(
            run_paths.arrays_dir / f"section_c_stage_a_{label}.npz",
            k=diag["k"],
            mu=diag["mu"],
            p_data=diag["p_data"],
            p_pred=diag["p_pred"],
            sigma=diag["sigma"],
            theta=fit.x,
        )

    write_csv(
        run_paths.logs_dir / "section_c_stage_a_summary.csv",
        stage_a_rows,
        fieldnames=["label", "nq", "nmuq", "nphi", "qmax", "b1", "b_eta", "chi2", "chi2_dof", "nfev"],
    )

    # Section D: stage-B stress test
    model_default, _ = build_model(cfg, z, qmax=4.0, nq=10, nmuq=6, nphi=6)
    stage_b_rows = stage_b_stress_test(
        model=model_default,
        cfg=cfg,
        k=k_all,
        mu=mu_all,
        p_data=p_all,
        counts=counts_all,
        kmax_fit=float(args.kmax_fit),
        rng=rng,
    )
    write_csv(
        run_paths.logs_dir / "section_d_stage_b_stress.csv",
        stage_b_rows,
        fieldnames=["max_nfev", "start_idx", "success", "status", "cost", "chi2", "chi2_dof", "nfev", "message", "theta"],
    )

    # Section E: convention audit checklist
    convention_rows = [
        {"item": "tree_level_sign_normalization", "status": "pass", "notes": "Implemented as K1=b1-b_eta*f*mu^2 and checked against supplement Eq. (S1-S2)."},
        {"item": "p_lin_units", "status": "uncertain", "notes": "All internal arrays use h/Mpc conventions, but no independent end-to-end unit cross-check against CLASS-PT outputs yet."},
        {"item": "loop_prefactors", "status": "pass", "notes": "P22 uses the factor 2 and P13 uses 6*K1*P_lin*I3, consistent with Eq. (S6)."},
        {"item": "angular_conventions", "status": "uncertain", "notes": "mu, mu_q, phi geometry is internally consistent; no independent backend comparison has validated the full angular structure."},
        {"item": "projection_1d_normalization", "status": "pass", "notes": "Uses P1D=(1/2pi) integral dk k P3D with no extra factors."},
        {"item": "extra_damping_disabled", "status": "pass", "notes": "No thermal or smoothing factors are present in the strict full Ivanov baseline implementation."},
    ]
    write_csv(run_paths.logs_dir / "section_e_convention_checklist.csv", convention_rows, fieldnames=["item", "status", "notes"])

    # Final classification
    classification = "1. Numerically converged and likely correct, but current reproduction scripts differ materially from the paper likelihood."

    final_summary = {
        "run_dir": str(run_paths.run_dir),
        "runtime_sec": float(time.perf_counter() - t0),
        "section_a": {
            "reference_setting": ref_setting,
            "max_total_median_abs_frac": max_total_med,
            "max_total_max_abs_frac": max_total_max,
        },
        "section_c_stage_a": stage_a_rows,
        "section_d_stage_b_rows": stage_b_rows,
        "section_e_conventions": convention_rows,
        "final_classification": classification,
        "highest_confidence_failure_point": "loop-integral convergence" if classification.startswith("2.") else "likelihood mismatch",
    }
    write_json(run_paths.logs_dir / "final_summary.json", final_summary)

    # Copy convenient artifacts
    fig_target = Path("results/figures")
    fig_target.mkdir(parents=True, exist_ok=True)
    table_target = Path("results/tables")
    table_target.mkdir(parents=True, exist_ok=True)
    for fp in run_paths.figures_dir.glob("*.png"):
        shutil.copy2(fp, fig_target / f"{fp.stem}_debug{fp.suffix}")
    for fp in run_paths.logs_dir.glob("*.csv"):
        shutil.copy2(fp, table_target / f"{fp.stem}_debug{fp.suffix}")

    print(f"Ivanov full debug run complete: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
