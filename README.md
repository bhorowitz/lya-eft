# Lyman-alpha Hybrid EFT Research Scaffold

This workspace bootstraps Stage A research for reproducing Ivanov-style correlator-level modeling and monitoring progress with reproducible plots.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
bash scripts/setup_env.sh
```

## Run progress diagnostics

```bash
source .venv/bin/activate
python scripts/run_backend_sanity.py --config configs/baseline_ivanov.yaml
python scripts/run_ivanov_toy_baseline.py --config configs/baseline_ivanov.yaml
python scripts/run_sherwood_toy_fit.py --config configs/baseline_ivanov.yaml
python scripts/run_sherwood_pk_residual_comparison.py --config configs/baseline_ivanov.yaml
python scripts/run_sherwood_highk_scan.py --config configs/baseline_ivanov.yaml
python scripts/run_bias_mcmc.py --config configs/baseline_ivanov.yaml --model both --kmax-fit 10.0 --hybrid-tight-priors
python scripts/run_cosmo_mcmc_fixed_bias.py --config configs/baseline_ivanov.yaml --model both --parametrization As --kmax-fit 10.0
python scripts/run_cosmo_mcmc_fixed_bias.py --config configs/baseline_ivanov.yaml --model both --parametrization sigma8 --kmax-fit 10.0
python scripts/run_joint_mcmc.py --config configs/baseline_ivanov.yaml --model both --kmax-fit 10.0
python scripts/run_joint_mcmc_multiz.py --config configs/baseline_ivanov.yaml --z-targets 2.0,3.0 --model both --kmax-fit 10.0
```

Outputs are written to `results/runs/<timestamp>_<tag>/` and summary figures are copied into `results/figures/`.

## Current status

- [x] Environment/bootstrap scripts
- [x] Linear-power + velocileptors backend sanity run
- [x] Ivanov-style toy operator scaffold
- [x] 3D->1D projection module with analytic regression test
- [x] Sherwood z=2.8 public data ingestion + first toy fit diagnostics
- [x] Original/tree vs one-loop P(k,mu)+residual comparison plots
- [x] First hybrid source+LOS comparison against one-loop
- [x] High-kmax scan with tightened hybrid bounds/priors
- [x] Stage-1 bias-only MCMC pipeline (one-loop + hybrid)
- [x] Short cosmology MCMC with fixed Stage-1 biases (As/omega_cdm and sigma8/omega_cdm)
- [x] Joint bias+cosmology MCMC with checkpoint diagnostics every 1000 steps
- [x] Joint multi-redshift bias+cosmology MCMC (z-targets 2.0 and 3.0->3.2)
- [ ] Real Ivanov data ingestion and fit reproduction
- [ ] Hybrid source+LOS extension
