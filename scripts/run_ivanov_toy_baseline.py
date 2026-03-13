#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lya_hybrid.config import load_config
from lya_hybrid.diagnostics import projection_convergence_scan
from lya_hybrid.grids import log_k_grid, mu_grid
from lya_hybrid.linear_power import compute_linear_power_camb
from lya_hybrid.logging_utils import build_repro_metadata, init_run_dir, write_json
from lya_hybrid.model_ivanov import IvanovToyModel, IvanovToyParams
from lya_hybrid.projection_1d import Polynomial1DCounterterms, project_to_1d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Ivanov-style toy baseline and projection diagnostics.")
    p.add_argument("--config", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    run_paths = init_run_dir(cfg.run.output_root, tag="ivanov_toy_baseline")
    meta = build_repro_metadata(args.config)
    write_json(run_paths.logs_dir / "repro_metadata.json", meta)

    lp = compute_linear_power_camb(
        h=cfg.cosmology.h,
        omega_b=cfg.cosmology.omega_b,
        omega_cdm=cfg.cosmology.omega_cdm,
        ns=cfg.cosmology.ns,
        As=cfg.cosmology.As,
        z=cfg.cosmology.z,
        kmin=cfg.k_grid.kmin,
        kmax=cfg.k_grid.kmax,
        nk=cfg.k_grid.nk,
    )

    k = log_k_grid(cfg.k_grid.kmin, cfg.k_grid.kmax, cfg.k_grid.nk)
    mu = mu_grid(cfg.mu_grid.nmu)

    model = IvanovToyModel(lp.k_hmpc, lp.p_lin_h3mpc3, lp.f_growth)
    params = IvanovToyParams(**cfg.ivanov_toy.model_dump())
    components = model.evaluate_grid(k, mu, params)

    kpar = np.logspace(
        np.log10(cfg.projection_1d.kpar_min),
        np.log10(cfg.projection_1d.kpar_max),
        cfg.projection_1d.nkpar,
    )

    def p3d_callable(kvals: np.ndarray, muvals: np.ndarray) -> np.ndarray:
        out = model.evaluate_components(kvals, muvals, params)
        return out["total"]

    projection = project_to_1d(
        kpar_values=kpar,
        p3d_callable=p3d_callable,
        kmax_proj=cfg.projection_1d.kmax_proj,
        nint=1600,
        method="trapz",
        counterterms=Polynomial1DCounterterms(
            c0=cfg.projection_1d.poly_c0,
            c2=cfg.projection_1d.poly_c2,
            c4=cfg.projection_1d.poly_c4,
        ),
    )

    conv = projection_convergence_scan(
        kpar_values=kpar,
        p3d_callable=p3d_callable,
        kmax_proj=cfg.projection_1d.kmax_proj,
        nint_values=[200, 400, 800, 1600],
        counterterms=Polynomial1DCounterterms(
            c0=cfg.projection_1d.poly_c0,
            c2=cfg.projection_1d.poly_c2,
            c4=cfg.projection_1d.poly_c4,
        ),
    )

    np.savez(
        run_paths.arrays_dir / "ivanov_toy_arrays.npz",
        k=k,
        mu=mu,
        tree=components["tree"],
        loop=components["loop"],
        counterterm=components["counterterm"],
        total=components["total"],
        kpar=kpar,
        p1d_raw=projection["raw"],
        p1d_total=projection["total"],
        p1d_n200=conv[200],
        p1d_n400=conv[400],
        p1d_n800=conv[800],
        p1d_n1600=conv[1600],
    )

    # Plot 1: 3D component breakdown for representative mu bins.
    mu_targets = [0.1, 0.5, 0.9]
    plt.figure(figsize=(9, 6))
    for m in mu_targets:
        i = int(np.argmin(np.abs(mu - m)))
        plt.loglog(k, np.abs(components["total"][i]), label=fr"total, $\mu={mu[i]:.2f}$")
        plt.loglog(k, np.abs(components["tree"][i]), ls="--", alpha=0.6)
    plt.xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    plt.ylabel(r"$|P_F(k,\mu)|$")
    plt.title("Ivanov-style toy total+tree in representative mu bins")
    plt.legend()
    plt.tight_layout()
    f1 = run_paths.figures_dir / "11_toy_pkmu_components.png"
    plt.savefig(f1, dpi=160)
    plt.close()

    # Plot 2: component decomposition at fixed mu.
    i = int(np.argmin(np.abs(mu - 0.8)))
    plt.figure(figsize=(9, 6))
    plt.semilogx(k, components["tree"][i], label="tree")
    plt.semilogx(k, components["loop"][i], label="loop")
    plt.semilogx(k, components["counterterm"][i], label="counterterm")
    plt.semilogx(k, components["total"][i], label="total", lw=2)
    plt.xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    plt.ylabel(r"$P_F(k,\mu=0.8)$")
    plt.title("Toy operator components at fixed LOS angle")
    plt.legend()
    plt.tight_layout()
    f2 = run_paths.figures_dir / "12_toy_operator_breakdown_mu08.png"
    plt.savefig(f2, dpi=160)
    plt.close()

    # Plot 3: 1D projection.
    plt.figure(figsize=(8, 5))
    plt.loglog(kpar, np.abs(projection["raw"]), label="projected raw")
    plt.loglog(kpar, np.abs(projection["total"]), label="with polynomial", ls="--")
    plt.xlabel(r"$k_\parallel\ [h\,{\rm Mpc}^{-1}]$")
    plt.ylabel(r"$|P_{1D}(k_\parallel)|$")
    plt.title("3D -> 1D projection (toy baseline)")
    plt.legend()
    plt.tight_layout()
    f3 = run_paths.figures_dir / "13_toy_projection_1d.png"
    plt.savefig(f3, dpi=160)
    plt.close()

    # Plot 4: projection convergence.
    base = conv[1600]
    plt.figure(figsize=(8, 5))
    for nint in [200, 400, 800]:
        frac = (conv[nint] - base) / np.maximum(np.abs(base), 1e-30)
        plt.semilogx(kpar, frac, label=f"nint={nint} vs 1600")
    plt.axhline(0.0, color="k", lw=1)
    plt.xlabel(r"$k_\parallel\ [h\,{\rm Mpc}^{-1}]$")
    plt.ylabel("fractional difference")
    plt.title("Projection grid convergence diagnostic")
    plt.legend()
    plt.tight_layout()
    f4 = run_paths.figures_dir / "14_projection_convergence.png"
    plt.savefig(f4, dpi=160)
    plt.close()

    fig_target = Path("results/figures")
    fig_target.mkdir(parents=True, exist_ok=True)
    for fp in [f1, f2, f3, f4]:
        shutil.copy2(fp, fig_target / fp.name)

    max_frac_err_n800 = float(np.max(np.abs((conv[800] - conv[1600]) / np.maximum(np.abs(conv[1600]), 1e-30))))

    summary = {
        "run_dir": str(run_paths.run_dir),
        "figures": [str(f1), str(f2), str(f3), str(f4)],
        "projection_max_frac_error_n800_vs_n1600": max_frac_err_n800,
    }
    write_json(run_paths.logs_dir / "summary.json", summary)

    print(f"Toy baseline run complete: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
