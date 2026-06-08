# VitalRateModels.jl

[![Build Status](https://github.com/ecorecipes/VitalRateModels.jl/workflows/CI/badge.svg)](https://github.com/ecorecipes/VitalRateModels.jl/actions)

Statistical fitting and model selection for demographic vital rates in structured population models.

## Overview

VitalRateModels.jl provides a unified interface for fitting survival, growth, fecundity, and recruitment models from longitudinal demographic data. It bridges raw field data with the [ecorecipes](https://github.com/ecorecipes) Julia population modeling ecosystem:

- **[IntegralProjectionModels.jl](https://github.com/ecorecipes/IntegralProjectionModels.jl)** — fitted vital rates → IPM kernels
- **[MatrixProjectionModels.jl](https://github.com/ecorecipes/MatrixProjectionModels.jl)** — fitted vital rates → MPM matrix elements
- **[PhysiologicallyBasedDemographicModels.jl](https://github.com/ecorecipes/PhysiologicallyBasedDemographicModels.jl)** — fitted vital rates → PBDM rate functions

## Features

- **Vital rate fitting** — survival (logistic), growth (Gaussian/gamma), fecundity (Poisson/NB), recruitment
- **Model selection** — automated candidate comparison with AIC/BIC ranking and Akaike weights
- **Data preparation** — `verticalize()` wide → long format, `create_stageframe()` for stage classification
- **Kernel construction** — direct conversion of fitted rates to P/F kernels on continuous domains
- **Extensible** — GAM.jl and MixedModels.jl backends via Julia weak-dependency extensions

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/ecorecipes/VitalRateModels.jl")
```

## Quick Example

```julia
using VitalRateModels, DataFrames, StatsModels

# Simulated demographic data
data = DataFrame(
    size_t = randn(200) .+ 5,
    size_t1 = randn(200) .+ 5.5,
    survived = rand(Bool, 200),
    fecundity = rand(0:10, 200)
)

# Fit vital rates
surv = fit_vital_rate(SurvivalModel, data, @formula(survived ~ size_t))
growth = fit_vital_rate(GrowthModel, data, @formula(size_t1 ~ size_t))

# Model selection
formulas = [
    @formula(survived ~ size_t),
    @formula(survived ~ size_t + size_t^2),
]
result = modelsearch(SurvivalModel, data, formulas)
best = best_model(result)

# Build IPM kernel
using StructuredPopulationCore: ContinuousDomain
dom = ContinuousDomain(0.0, 10.0, 100)
P = vital_rates_to_kernel(surv, growth, dom)
```

## Architecture

```
StructuredPopulationCore.jl (shared abstractions)
         ↑
VitalRateModels.jl ← GLM.jl (core) + GAM.jl (ext) + MixedModels.jl (ext)
    ↓         ↓         ↓
IntegralPM  MatrixPM   PBDM
```

## Supported Distributions

| Vital Rate | Default | Alternatives |
|-----------|---------|-------------|
| Survival | Binomial (logistic) | — |
| Growth | Gaussian | Gamma |
| Fecundity | Poisson | Negative binomial, zero-inflated, truncated |
| Recruitment | Gaussian | Log-normal |

## See Also

- [StructuredPopulationCore.jl](https://github.com/ecorecipes/ProjectionModels.jl) — shared abstractions
- [GAM.jl](https://github.com/ecorecipes/GAM.jl) — generalized additive models for non-linear vital rates
