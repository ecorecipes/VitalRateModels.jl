"""
Type hierarchy for vital rate models.
"""

# --- Model type tags ---

"""Abstract supertype for all vital rate model specifications."""
abstract type AbstractVitalRateModel end

"""Survival model: binary outcome (alive/dead)."""
struct SurvivalModel <: AbstractVitalRateModel end

"""Growth model: continuous size at t+1 given size at t."""
struct GrowthModel <: AbstractVitalRateModel end

"""Fecundity model: reproductive output (count or continuous)."""
struct FecundityModel <: AbstractVitalRateModel end

"""Recruitment model: size distribution of new individuals."""
struct RecruitmentModel <: AbstractVitalRateModel end

# --- Distribution specifications ---

"""Distribution family for vital rate responses."""
abstract type VitalRateDistribution end

struct Gaussian <: VitalRateDistribution end
struct Binomial_ <: VitalRateDistribution end
struct NegativeBinomial_ <: VitalRateDistribution end
struct Poisson_ <: VitalRateDistribution end
struct ZeroInflatedPoisson <: VitalRateDistribution end
struct ZeroInflatedNegBin <: VitalRateDistribution end
struct TruncatedPoisson <: VitalRateDistribution end
struct TruncatedNegBin <: VitalRateDistribution end

# --- Fitted vital rate types ---

"""Abstract supertype for fitted vital rate objects."""
abstract type AbstractFittedVitalRate end

"""
    FittedSurvival

A fitted survival model. Predicts P(survival | size, covariates).

# Fields
- `model`: The underlying fitted GLM/GAM object
- `formula`: The model formula
- `distribution`: Distribution family used
- `aic`: AIC of the fitted model
- `n_obs`: Number of observations
"""
struct FittedSurvival{M} <: AbstractFittedVitalRate
    model::M
    formula::FormulaTerm
    distribution::VitalRateDistribution
    aic::Float64
    n_obs::Int
end

"""
    FittedGrowth

A fitted growth model. Predicts size(t+1) | size(t), covariates.

# Fields
- `model`: Fitted model for mean response
- `sigma`: Residual standard deviation (for Gaussian)
- `formula`: Model formula
- `distribution`: Distribution family
- `aic`: AIC
- `n_obs`: Number of observations
"""
struct FittedGrowth{M} <: AbstractFittedVitalRate
    model::M
    sigma::Float64
    formula::FormulaTerm
    distribution::VitalRateDistribution
    aic::Float64
    n_obs::Int
end

"""
    FittedFecundity

A fitted fecundity model. Predicts reproductive output | size, covariates.

# Fields
- `model`: Fitted GLM/GAM
- `formula`: Model formula
- `distribution`: Distribution family
- `aic`: AIC
- `n_obs`: Number of observations
- `prob_repro`: Optional separate model for P(reproducing)
"""
struct FittedFecundity{M, P} <: AbstractFittedVitalRate
    model::M
    formula::FormulaTerm
    distribution::VitalRateDistribution
    aic::Float64
    n_obs::Int
    prob_repro::P  # Nothing or a fitted model for P(reproducing)
end

"""
    FittedRecruitment

A fitted recruitment/offspring size distribution.

# Fields
- `model`: Fitted model for offspring size distribution
- `formula`: Model formula
- `distribution`: Distribution family
- `aic`: AIC
- `n_obs`: Number of observations
"""
struct FittedRecruitment{M} <: AbstractFittedVitalRate
    model::M
    formula::FormulaTerm
    distribution::VitalRateDistribution
    aic::Float64
    n_obs::Int
end
