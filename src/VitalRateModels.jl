"""
    VitalRateModels

Statistical fitting and model selection for demographic vital rates.

Provides a unified interface for fitting survival, growth, fecundity, and
recruitment models from longitudinal demographic data. Supports GLMs, GAMs
(via extension), and mixed-effects models (via extension) with automated
model selection.

Fitted vital rates can be directly consumed by IntegralProjectionModels.jl,
MatrixProjectionModels.jl, and PhysiologicallyBasedDemographicModels.jl.

# Architecture
- Define vital rate model structures (survival, growth, fecundity, recruitment)
- Fit models from DataFrames using GLM.jl (core), GAM.jl (ext), MixedModels.jl (ext)
- Predict vital rates on arbitrary domains (meshpoints, stage classes)
- Export to IPM kernels or MPM matrix elements

# Example
```julia
using VitalRateModels, DataFrames

data = DataFrame(size_t=rand(100), size_t1=rand(100),
                 survived=rand(Bool, 100), fecundity=rand(100))

# Fit vital rates
surv = fit(SurvivalModel, data, @formula(survived ~ size_t))
growth = fit(GrowthModel, data, @formula(size_t1 ~ size_t))
fec = fit(FecundityModel, data, @formula(fecundity ~ size_t))

# Predict on a domain
using StructuredPopulationCore: ContinuousDomain, meshpoints
dom = ContinuousDomain(0.0, 1.0, 50)
surv_pred = predict(surv, meshpoints(dom))
```
"""
module VitalRateModels

using LinearAlgebra
using Statistics
using DataFrames
using Distributions
using GLM
using StatsBase
using StatsModels
using StructuredPopulationCore: ContinuousDomain, meshpoints, step_size

# --- Abstract types ---

include("types.jl")
export AbstractVitalRateModel, AbstractFittedVitalRate
export SurvivalModel, GrowthModel, FecundityModel, RecruitmentModel
export FittedSurvival, FittedGrowth, FittedFecundity, FittedRecruitment
export VitalRateDistribution
export Gaussian, Binomial_, NegativeBinomial_, Poisson_
export ZeroInflatedPoisson, ZeroInflatedNegBin, TruncatedPoisson, TruncatedNegBin

# --- Fitting ---

include("fitting.jl")
export fit_vital_rate, predict_vital_rate

# --- Model selection ---

include("model_selection.jl")
export modelsearch, ModelComparisonResult, best_model

# --- Kernel construction ---

include("kernels.jl")
export vital_rates_to_kernel, vital_rates_to_matrix

# --- Data preparation ---

include("data_prep.jl")
export verticalize, create_stageframe, StageFrame
export validate_demographic_data, summarize_transitions

end # module VitalRateModels
