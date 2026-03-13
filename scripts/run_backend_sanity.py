#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lya_hybrid.config import load_config
from lya_hybrid.grids import log_k_grid
from lya_hybrid.linear_power import compute_linear_power_camb
from lya_hybrid.logging_utils import build_repro_metadata, init_run_dir, write_json
from lya_hybrid.velocileptors_backend import EPTReducedParams, VelocileptorsEPTBackend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run velocileptors backend sanity checks.")
    p.add_argument("--config", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    run_paths = init_run_dir(cfg.run.output_root, tag="backend_sanity")
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

    backend = VelocileptorsEPTBackend(
        lp.k_hmpc,
        lp.p_lin_h3mpc3,
        lp.p_nw_h3mpc3,
        kmin=cfg.backend.kmin,
        kmax=cfg.backend.kmax,
        nk=cfg.backend.nk,
        threads=cfg.backend.threads,
        beyond_gauss=cfg.backend.beyond_gauss,
    )

    params = EPTReducedParams()
    mus = [0.0, 0.5, 0.9]

    p_mu = {}
    for mu in mus:
        k_native, p_native = backend.power_at_mu(mu=mu, f_growth=lp.f_growth, params=params)
        p_mu[mu] = (k_native, p_native)

    kmp, p0, p2, p4 = backend.multipoles(f_growth=lp.f_growth, params=params)

    np.savez(
        run_paths.arrays_dir / "backend_sanity_arrays.npz",
        k_lin=lp.k_hmpc,
        p_lin=lp.p_lin_h3mpc3,
        p_nw=lp.p_nw_h3mpc3,
        k_multipole=kmp,
        p0=p0,
        p2=p2,
        p4=p4,
        f_growth=lp.f_growth,
    )

    # Plot 1: linear and no-wiggle spectra
    plt.figure(figsize=(8, 5))
    plt.loglog(lp.k_hmpc, lp.p_lin_h3mpc3, label=r"$P_{lin}$")
    plt.loglog(lp.k_hmpc, lp.p_nw_h3mpc3, label=r"$P_{nw}$", ls="--")
    plt.xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    plt.ylabel(r"$P(k)\ [h^{-3}\,{\rm Mpc}^3]$")
    plt.title("Linear matter spectrum (CAMB) at z=2.8")
    plt.legend()
    plt.tight_layout()
    f1 = run_paths.figures_dir / "01_linear_vs_nowiggle.png"
    plt.savefig(f1, dpi=160)
    plt.close()

    # Plot 2: P(k,mu) sanity curves
    plt.figure(figsize=(8, 5))
    for mu in mus:
        kk, pp = p_mu[mu]
        plt.loglog(kk, np.abs(pp), label=fr"$\mu={mu:.1f}$")
    plt.xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    plt.ylabel(r"$|P_s(k,\mu)|$")
    plt.title("velocileptors EPT redshift-space sanity curves")
    plt.legend()
    plt.tight_layout()
    f2 = run_paths.figures_dir / "02_pkmu_sanity.png"
    plt.savefig(f2, dpi=160)
    plt.close()

    # Plot 3: multipoles
    plt.figure(figsize=(8, 5))
    plt.plot(kmp, kmp * p0, label=r"$kP_0$")
    plt.plot(kmp, kmp * p2, label=r"$kP_2$")
    plt.plot(kmp, kmp * p4, label=r"$kP_4$")
    plt.xscale("log")
    plt.xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    plt.ylabel(r"$kP_\ell(k)$")
    plt.title("velocileptors EPT multipoles (sanity)")
    plt.legend()
    plt.tight_layout()
    f3 = run_paths.figures_dir / "03_multipoles_sanity.png"
    plt.savefig(f3, dpi=160)
    plt.close()

    fig_target = Path("results/figures")
    fig_target.mkdir(parents=True, exist_ok=True)
    for fp in [f1, f2, f3]:
        shutil.copy2(fp, fig_target / fp.name)

    summary = {
        "run_dir": str(run_paths.run_dir),
        "figures": [str(f1), str(f2), str(f3)],
        "f_growth": float(lp.f_growth),
        "k_range": [float(lp.k_hmpc.min()), float(lp.k_hmpc.max())],
    }
    write_json(run_paths.logs_dir / "summary.json", summary)

    print(f"Backend sanity run complete: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
