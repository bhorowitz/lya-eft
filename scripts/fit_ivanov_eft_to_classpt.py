#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from lya_hybrid.class_pt_backend import ClassPTBackend
from lya_hybrid.config import load_config
from lya_hybrid.linear_power import compute_linear_power_camb
from lya_hybrid.logging_utils import build_repro_metadata, init_run_dir, write_json
from lya_hybrid.model_ivanov_full import IvanovFullModel, IvanovFullParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit minimal (c0 + c2 mu^2 + c4 mu^4) k^2 P_lin EFT counterterms in IvanovFullModel to CLASS-PT.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--z", type=float, default=2.8)
    p.add_argument("--kmax", type=float, default=0.6)
    p.add_argument("--nk", type=int, default=16)
    p.add_argument("--mu-targets", type=str, default="0.1,0.5,0.9")
    p.add_argument("--b-eta", type=float, default=-1.0)
    p.add_argument("--qmax-list", type=str, default="4.0,8.0,12.0")
    p.add_argument("--nq", type=int, default=80)
    p.add_argument("--nmuq", type=int, default=18)
    p.add_argument("--nphi", type=int, default=12)
    return p.parse_args()


def reconstruct_pkmu(p0: np.ndarray, p2: np.ndarray, p4: np.ndarray, mu: float) -> np.ndarray:
    l2 = 0.5 * (3.0 * mu**2 - 1.0)
    l4 = (35.0 * mu**4 - 30.0 * mu**2 + 3.0) / 8.0
    return p0 + p2 * l2 + p4 * l4


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    run_paths = init_run_dir(cfg.run.output_root, tag="ivanov_eft_fit_classpt")
    meta = build_repro_metadata(args.config)
    meta.update({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()})
    write_json(run_paths.logs_dir / "manifest.json", meta)

    k_eval = np.logspace(np.log10(0.03), np.log10(float(args.kmax)), int(args.nk))
    mu_targets = [float(x) for x in str(args.mu_targets).split(",") if x.strip()]
    qmax_list = [float(x) for x in str(args.qmax_list).split(",") if x.strip()]

    lp = compute_linear_power_camb(
        h=cfg.cosmology.h,
        omega_b=cfg.cosmology.omega_b,
        omega_cdm=cfg.cosmology.omega_cdm,
        ns=cfg.cosmology.ns,
        As=cfg.cosmology.As,
        z=float(args.z),
        kmin=cfg.k_grid.kmin,
        kmax=max(cfg.k_grid.kmax, float(args.kmax) + 4.0),
        nk=max(cfg.k_grid.nk, 1600),
    )

    with ClassPTBackend(
        h=cfg.cosmology.h,
        omega_b=cfg.cosmology.omega_b,
        omega_cdm=cfg.cosmology.omega_cdm,
        ns=cfg.cosmology.ns,
        As=cfg.cosmology.As,
        z=float(args.z),
        P_k_max_hmpc=max(4.0, float(args.kmax) + 0.4),
    ) as backend:
        backend.initialize_output(k_eval)
        _, p0_class, p2_class, p4_class = backend.matter_multipoles()
    class_pkmu_by_mu = {mu: reconstruct_pkmu(p0_class, p2_class, p4_class, mu) for mu in mu_targets}

    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_payload: dict[str, np.ndarray] | None = None

    for qmax in qmax_list:
        model = IvanovFullModel(
            lp.k_hmpc,
            lp.p_lin_h3mpc3,
            lp.f_growth,
            qmin=float(cfg.k_grid.kmin),
            qmax=qmax,
            nq=int(args.nq),
            nmuq=int(args.nmuq),
            nphi=int(args.nphi),
        )
        base_params = dict(
            b1=1.0,
            b_eta=float(args.b_eta),
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

        # Build a linear least-squares system across all requested mu targets.
        y_blocks: list[np.ndarray] = []
        x0_blocks: list[np.ndarray] = []
        a_blocks: list[np.ndarray] = []
        payload_by_mu: dict[float, dict[str, np.ndarray]] = {}

        for mu in mu_targets:
            comps0 = model.evaluate_components(k_eval, np.full_like(k_eval, mu), IvanovFullParams(**base_params))
            base_total = comps0["tree"] + comps0["loop_22"] + comps0["loop_13"]
            pk = np.asarray(model._interp(k_eval), dtype=float)
            k2pk = (k_eval**2) * pk
            basis = np.column_stack([
                k2pk,
                (mu**2) * k2pk,
                (mu**4) * k2pk,
            ])
            target = class_pkmu_by_mu[mu]
            y_blocks.append(target)
            x0_blocks.append(base_total)
            a_blocks.append(basis)
            payload_by_mu[mu] = {
                "class_pkmu": target.copy(),
                "direct_base": base_total.copy(),
                "tree": comps0["tree"].copy(),
                "loop_22": comps0["loop_22"].copy(),
                "loop_13": comps0["loop_13"].copy(),
                "k2pk": k2pk.copy(),
            }

        y = np.concatenate(y_blocks)
        x0 = np.concatenate(x0_blocks)
        A = np.vstack(a_blocks)
        coeffs, *_ = np.linalg.lstsq(A, y - x0, rcond=None)
        c0_ct, c2_ct, c4_ct = [float(v) for v in coeffs]
        pred = x0 + A @ coeffs
        frac = np.abs(pred - y) / np.maximum(np.abs(y), 1.0e-10)
        score = float(np.mean(frac**2))
        row = {
            "qmax": qmax,
            "c0_ct": c0_ct,
            "c2_ct": c2_ct,
            "c4_ct": c4_ct,
            "mean_sq_frac": score,
            "median_abs_frac": float(np.median(frac)),
            "max_abs_frac": float(np.max(frac)),
        }
        rows.append(row)
        if best is None or score < float(best["mean_sq_frac"]):
            best = row
            best_payload = {"k": k_eval.copy()}
            for mu in mu_targets:
                basis_ct = (c0_ct + c2_ct * mu**2 + c4_ct * mu**4) * payload_by_mu[mu]["k2pk"]
                best_payload[f"class_pkmu_mu{mu:.1f}"] = payload_by_mu[mu]["class_pkmu"]
                best_payload[f"direct_base_mu{mu:.1f}"] = payload_by_mu[mu]["direct_base"]
                best_payload[f"direct_with_ct_mu{mu:.1f}"] = payload_by_mu[mu]["direct_base"] + basis_ct
                best_payload[f"tree_mu{mu:.1f}"] = payload_by_mu[mu]["tree"]
                best_payload[f"loop_22_mu{mu:.1f}"] = payload_by_mu[mu]["loop_22"]
                best_payload[f"loop_13_mu{mu:.1f}"] = payload_by_mu[mu]["loop_13"]
                best_payload[f"counterterm_mu{mu:.1f}"] = basis_ct

    assert best is not None and best_payload is not None

    write_csv(
        run_paths.logs_dir / "eft_fit_scan.csv",
        rows,
        fieldnames=["qmax", "c0_ct", "c2_ct", "c4_ct", "mean_sq_frac", "median_abs_frac", "max_abs_frac"],
    )
    np.savez(run_paths.arrays_dir / "eft_fit_best_arrays.npz", **best_payload)
    write_json(run_paths.logs_dir / "summary.json", {"best": best, "run_dir": str(run_paths.run_dir)})

    print("Best fit:")
    print(best)
    print(f"Run dir: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
