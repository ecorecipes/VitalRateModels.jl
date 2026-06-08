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

"""Gaussian response model, typically used for growth or recruitment."""
struct Gaussian <: VitalRateDistribution end
"""Binomial response model with a logit link, typically used for survival."""
struct Binomial_ <: VitalRateDistribution end
"""Negative-binomial count model for overdispersed fecundity responses."""
struct NegativeBinomial_ <: VitalRateDistribution end
"""Poisson count model for fecundity or recruitment counts."""
struct Poisson_ <: VitalRateDistribution end
"""Zero-inflated Poisson count model placeholder for extension backends."""
struct ZeroInflatedPoisson <: VitalRateDistribution end
"""Zero-inflated negative-binomial model placeholder for extension backends."""
struct ZeroInflatedNegBin <: VitalRateDistribution end
"""Truncated Poisson count model placeholder for extension backends."""
struct TruncatedPoisson <: VitalRateDistribution end
"""Truncated negative-binomial model placeholder for extension backends."""
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
