from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    tag: str = "run"
    output_root: Path = Path("results/runs")


class CosmologyConfig(BaseModel):
    h: float = 0.6774
    omega_b: float = 0.0486
    omega_cdm: float = 0.2589
    ns: float = 0.9667
    As: float = Field(default=2.1e-9)
    z: float = 2.8


class KGridConfig(BaseModel):
    kmin: float = 1.0e-3
    kmax: float = 6.0
    nk: int = 800


class MuGridConfig(BaseModel):
    nmu: int = 121


class BackendConfig(BaseModel):
    kmin: float = 1.0e-2
    kmax: float = 0.6
    nk: int = 220
    threads: int = 1
    beyond_gauss: bool = False


class IvanovToyConfig(BaseModel):
    b1: float = -0.14
    b_eta: float = -0.2
    c0: float = 0.45
    c2: float = 0.8
    c4: float = 0.3
    loop_amp: float = 0.35
    loop_mu2: float = 0.6
    loop_mu4: float = 0.2
    loop_k_nl: float = 1.7
    stochastic: float = 0.0


class HybridToyConfig(BaseModel):
    b_delta: float = -0.14
    b_eta: float = -0.2
    b_t: float = 0.03
    c0: float = 0.45
    c2: float = 0.8
    c4: float = 0.3
    loop_amp: float = 0.35
    loop_mu2: float = 0.6
    loop_mu4: float = 0.2
    loop_k_nl: float = 1.7
    sigma_th: float = 0.08
    k_t: float = 0.5
    stochastic: float = 0.0


class FitConfig(BaseModel):
    kmin_fit: float = 0.03
    kmax_fit: float = 3.0
    sigma_frac: float = 0.05
    sigma_floor: float = 0.02


class MCMCConfig(BaseModel):
    nwalkers: int = 48
    nsteps: int = 2500
    burnin: int = 700
    thin: int = 10
    seed: int = 42
    init_jitter_frac: float = 0.02
    posterior_band_draws: int = 220


class JointMCMCConfig(BaseModel):
    parametrization: str = "sigma8"
    nwalkers: int = 40
    nsteps: int = 3000
    burnin: int = 500
    thin: int = 8
    seed: int = 2026
    checkpoint_every: int = 1000
    posterior_band_draws: int = 220
    omega_min: float = 0.20
    omega_max: float = 0.33
    n_omega_grid: int = 65
    as_min: float = 1.0e-9
    as_max: float = 3.5e-9
    sigma8_min: float = 0.60
    sigma8_max: float = 1.05
    hybrid_tight_priors: bool = True


class Projection1DConfig(BaseModel):
    kpar_min: float = 0.03
    kpar_max: float = 4.0
    nkpar: int = 150
    kmax_proj: float = 6.0
    poly_c0: float = 0.0
    poly_c2: float = 0.0
    poly_c4: float = 0.0


class BaselineConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    cosmology: CosmologyConfig = Field(default_factory=CosmologyConfig)
    k_grid: KGridConfig = Field(default_factory=KGridConfig)
    mu_grid: MuGridConfig = Field(default_factory=MuGridConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    ivanov_toy: IvanovToyConfig = Field(default_factory=IvanovToyConfig)
    hybrid_toy: HybridToyConfig = Field(default_factory=HybridToyConfig)
    fit: FitConfig = Field(default_factory=FitConfig)
    mcmc: MCMCConfig = Field(default_factory=MCMCConfig)
    joint_mcmc: JointMCMCConfig = Field(default_factory=JointMCMCConfig)
    projection_1d: Projection1DConfig = Field(default_factory=Projection1DConfig)


def load_config(config_path: str | Path) -> BaselineConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        payload: dict[str, Any] = yaml.safe_load(f)
    return BaselineConfig.model_validate(payload)
