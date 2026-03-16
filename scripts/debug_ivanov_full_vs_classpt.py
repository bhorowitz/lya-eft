#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from lya_hybrid.class_pt_backend import ClassPTBackend
from lya_hybrid.config import load_config
from lya_hybrid.linear_power import compute_linear_power_camb
from lya_hybrid.logging_utils import build_repro_metadata, init_run_dir, write_json
from lya_hybrid.model_ivanov_full import IvanovFullModel, IvanovFullParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-check the direct Ivanov full model against CLASS-PT in the matter limit.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--z", type=float, default=2.8)
    p.add_argument("--kmax", type=float, default=0.6)
    p.add_argument("--nk", type=int, default=16)
    return p.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def legendre_l2(mu: np.ndarray) -> np.ndarray:
    return 0.5 * (3.0 * mu**2 - 1.0)


def legendre_l4(mu: np.ndarray) -> np.ndarray:
    return (35.0 * mu**4 - 30.0 * mu**2 + 3.0) / 8.0


def multipoles_from_direct(
    model: IvanovFullModel,
    params: IvanovFullParams,
    k_eval: np.ndarray,
    *,
    nmu: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu_nodes, mu_w = np.polynomial.legendre.leggauss(nmu)
    kk = np.repeat(k_eval[:, None], nmu, axis=1)
    mm = np.repeat(mu_nodes[None, :], k_eval.size, axis=0)
    total = model.evaluate_components(kk.ravel(), mm.ravel(), params)["total"].reshape(k_eval.size, nmu)
    p0 = 0.5 * np.sum(mu_w[None, :] * total, axis=1)
    p2 = 2.5 * np.sum(mu_w[None, :] * total * legendre_l2(mu_nodes)[None, :], axis=1)
    p4 = 4.5 * np.sum(mu_w[None, :] * total * legendre_l4(mu_nodes)[None, :], axis=1)
    return p0, p2, p4


def reconstruct_pkmu(p0: np.ndarray, p2: np.ndarray, p4: np.ndarray, mu: float) -> np.ndarray:
    return p0 + p2 * legendre_l2(np.asarray(mu)) + p4 * legendre_l4(np.asarray(mu))


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    run_paths = init_run_dir(cfg.run.output_root, tag="ivanov_full_classpt_debug")
    meta = build_repro_metadata(args.config)
    meta.update({"z": float(args.z), "kmax": float(args.kmax), "nk": int(args.nk)})
    write_json(run_paths.logs_dir / "manifest.json", meta)

    k_eval = np.logspace(np.log10(0.03), np.log10(float(args.kmax)), int(args.nk))
    params_matter = IvanovFullParams(
        b1=1.0,
        b_eta=-1.0,
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

    lp = compute_linear_power_camb(
        h=cfg.cosmology.h,
        omega_b=cfg.cosmology.omega_b,
        omega_cdm=cfg.cosmology.omega_cdm,
        ns=cfg.cosmology.ns,
        As=cfg.cosmology.As,
        z=float(args.z),
        kmin=cfg.k_grid.kmin,
        kmax=max(cfg.k_grid.kmax, float(args.kmax) + 2.0),
        nk=max(cfg.k_grid.nk, 1200),
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
        k_class, p0_class, p2_class, p4_class = backend.matter_multipoles()

    settings = [
        {"label": "direct_qmax4", "nq": 20, "nmuq": 8, "nphi": 8, "qmax": 4.0},
        {"label": "direct_qmax12", "nq": 24, "nmuq": 10, "nphi": 10, "qmax": 12.0},
    ]

    rows: list[dict[str, object]] = []
    spectra_by_label: dict[str, dict[str, np.ndarray]] = {
        "class_pt": {"k": k_class, "p0": p0_class, "p2": p2_class, "p4": p4_class}
    }
    for setting in settings:
        print(f"Evaluating {setting['label']}...")
        model = IvanovFullModel(
            lp.k_hmpc,
            lp.p_lin_h3mpc3,
            lp.f_growth,
            qmin=float(cfg.k_grid.kmin),
            qmax=float(setting["qmax"]),
            nq=int(setting["nq"]),
            nmuq=int(setting["nmuq"]),
            nphi=int(setting["nphi"]),
        )
        p0_dir, p2_dir, p4_dir = multipoles_from_direct(model, params_matter, k_eval)
        spectra_by_label[setting["label"]] = {"k": k_eval, "p0": p0_dir, "p2": p2_dir, "p4": p4_dir}
        for ell, p_dir, p_ref in [(0, p0_dir, p0_class), (2, p2_dir, p2_class), (4, p4_dir, p4_class)]:
            frac = np.abs(p_dir - p_ref) / np.maximum(np.abs(p_ref), 1.0e-10)
            rows.append(
                {
                    "label": setting["label"],
                    "ell": ell,
                    "nq": setting["nq"],
                    "nmuq": setting["nmuq"],
                    "nphi": setting["nphi"],
                    "qmax": setting["qmax"],
                    "median_abs_frac_vs_classpt": float(np.median(frac)),
                    "max_abs_frac_vs_classpt": float(np.max(frac)),
                }
            )

    write_csv(
        run_paths.logs_dir / "classpt_crosscheck_summary.csv",
        rows,
        fieldnames=["label", "ell", "nq", "nmuq", "nphi", "qmax", "median_abs_frac_vs_classpt", "max_abs_frac_vs_classpt"],
    )

    np.savez(
        run_paths.arrays_dir / "classpt_crosscheck_arrays.npz",
        k=k_eval,
        p0_class=p0_class,
        p2_class=p2_class,
        p4_class=p4_class,
        **{f"p0_{label}": payload["p0"] for label, payload in spectra_by_label.items() if label != "class_pt"},
        **{f"p2_{label}": payload["p2"] for label, payload in spectra_by_label.items() if label != "class_pt"},
        **{f"p4_{label}": payload["p4"] for label, payload in spectra_by_label.items() if label != "class_pt"},
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for ax, ell, ref in zip(axes, [0, 2, 4], [p0_class, p2_class, p4_class]):
        ax.semilogx(k_eval, ref, color="k", lw=2.0, label="CLASS-PT")
        for setting in settings:
            payload = spectra_by_label[setting["label"]]
            ax.semilogx(k_eval, payload[f"p{ell}"], lw=1.4, label=setting["label"])
        ax.set_title(fr"$P_{{{ell}}}(k)$ matter limit")
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    axes[0].set_ylabel(r"$P_\ell(k)\ [h^{-3}{\rm Mpc}^3]$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(run_paths.figures_dir / "01_classpt_multipole_crosscheck.png", dpi=160)
    plt.close(fig)

    mu_targets = [0.1, 0.5, 0.9]
    fig, axes = plt.subplots(1, len(mu_targets), figsize=(5 * len(mu_targets), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, mu in zip(axes, mu_targets):
        class_pk = reconstruct_pkmu(p0_class, p2_class, p4_class, mu)
        ax.semilogx(k_eval, class_pk, color="k", lw=2.0, label="CLASS-PT reconstructed")
        for setting in settings:
            payload = spectra_by_label[setting["label"]]
            ax.semilogx(k_eval, reconstruct_pkmu(payload["p0"], payload["p2"], payload["p4"], mu), lw=1.4, label=setting["label"])
        ax.set_title(fr"$\mu={mu:.1f}$")
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    axes[0].set_ylabel(r"$P(k,\mu)$ from multipoles")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(run_paths.figures_dir / "02_classpt_reconstructed_pkmu.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    for ax, ell, ref in zip(axes, [0, 2, 4], [p0_class, p2_class, p4_class]):
        for setting in settings:
            payload = spectra_by_label[setting["label"]]
            frac = np.abs(payload[f"p{ell}"] - ref) / np.maximum(np.abs(ref), 1.0e-10)
            ax.semilogx(k_eval, frac, lw=1.4, label=setting["label"])
        ax.axhline(0.05, color="k", ls="--", lw=1)
        ax.set_title(fr"$|P_{{{ell}}}^{{dir}}/P_{{{ell}}}^{{CLASS-PT}}-1|$")
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    axes[0].set_ylabel("fractional difference")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(run_paths.figures_dir / "03_classpt_fractional_difference.png", dpi=160)
    plt.close(fig)

    summary = {
        "run_dir": str(run_paths.run_dir),
        "z": float(args.z),
        "kmax": float(args.kmax),
        "matter_limit_params": {
            "b1": 1.0,
            "b_eta": -1.0,
            "all_higher_biases": 0.0,
        },
        "best_setting_by_p0": min((r for r in rows if int(r["ell"]) == 0), key=lambda r: float(r["median_abs_frac_vs_classpt"])),
        "best_setting_by_p2": min((r for r in rows if int(r["ell"]) == 2), key=lambda r: float(r["median_abs_frac_vs_classpt"])),
        "best_setting_by_p4": min((r for r in rows if int(r["ell"]) == 4), key=lambda r: float(r["median_abs_frac_vs_classpt"])),
        "notes": [
            "CLASS-PT is used here as an independent one-loop backend in the matter limit.",
            "This is a backend cross-check, not a full LyA full-basis replacement.",
        ],
    }
    write_json(run_paths.logs_dir / "summary.json", summary)

    fig_target = Path("results/figures")
    fig_target.mkdir(parents=True, exist_ok=True)
    table_target = Path("results/tables")
    table_target.mkdir(parents=True, exist_ok=True)
    for fp in run_paths.figures_dir.glob("*.png"):
        shutil.copy2(fp, fig_target / f"classpt_{fp.name}")
    shutil.copy2(run_paths.logs_dir / "classpt_crosscheck_summary.csv", table_target / "classpt_crosscheck_summary.csv")

    print(f"CLASS-PT cross-check run complete: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
