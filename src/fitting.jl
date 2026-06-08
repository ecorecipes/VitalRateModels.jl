"""
Vital rate fitting using GLM.jl.

Core fitting functionality for survival, growth, fecundity, and recruitment
models. GAM and mixed-effects extensions are loaded via weak dependencies.
"""

"""
    fit_vital_rate(::Type{SurvivalModel}, data::DataFrame, formula::FormulaTerm;
                   distribution=Binomial_())

Fit a survival model from demographic data.

# Arguments
- `data`: DataFrame with response and predictor columns
- `formula`: StatsModels formula (e.g., `@formula(survived ~ size_t)`)
- `distribution`: Distribution family (default: Binomial_ for logistic regression)

# Returns
A `FittedSurvival` object.
"""
function fit_vital_rate(::Type{SurvivalModel}, data::DataFrame, formula::FormulaTerm;
                        distribution::VitalRateDistribution=Binomial_())
    glm_dist, link = _get_glm_family(distribution)
    m = glm(formula, data, glm_dist, link)
    return FittedSurvival(m, formula, distribution, aic(m), Int(nobs(m)))
end

"""
    fit_vital_rate(::Type{GrowthModel}, data::DataFrame, formula::FormulaTerm;
                   distribution=Gaussian())

Fit a growth model (size at t+1 given size at t).
"""
function fit_vital_rate(::Type{GrowthModel}, data::DataFrame, formula::FormulaTerm;
                        distribution::VitalRateDistribution=Gaussian())
    glm_dist, link = _get_glm_family(distribution)
    m = glm(formula, data, glm_dist, link)
    σ = sqrt(deviance(m) / dof_residual(m))
    return FittedGrowth(m, σ, formula, distribution, aic(m), Int(nobs(m)))
end

"""
    fit_vital_rate(::Type{FecundityModel}, data::DataFrame, formula::FormulaTerm;
                   distribution=Poisson_(), repro_formula=nothing)

Fit a fecundity model with optional separate P(reproducing) model.
"""
function fit_vital_rate(::Type{FecundityModel}, data::DataFrame, formula::FormulaTerm;
                        distribution::VitalRateDistribution=Poisson_(),
                        repro_formula::Union{Nothing,FormulaTerm}=nothing)
    m = if distribution isa NegativeBinomial_
        negbin(formula, data, LogLink())
    else
        glm_dist, link = _get_glm_family(distribution)
        glm(formula, data, glm_dist, link)
    end

    # Optional probability-of-reproducing sub-model
    prob_repro = if repro_formula !== nothing
        glm(repro_formula, data, Bernoulli(), LogitLink())
    else
        nothing
    end

    return FittedFecundity(m, formula, distribution, aic(m), Int(nobs(m)), prob_repro)
end

"""
    fit_vital_rate(::Type{RecruitmentModel}, data::DataFrame, formula::FormulaTerm;
                   distribution=Gaussian())

Fit an offspring/recruit size distribution model.
"""
function fit_vital_rate(::Type{RecruitmentModel}, data::DataFrame, formula::FormulaTerm;
                        distribution::VitalRateDistribution=Gaussian())
    glm_dist, link = _get_glm_family(distribution)
    m = glm(formula, data, glm_dist, link)
    σ = if distribution isa Gaussian
        sqrt(deviance(m) / dof_residual(m))
    else
        0.0
    end
    return FittedRecruitment(m, formula, distribution, aic(m), Int(nobs(m)))
end

# --- Prediction ---

"""
    predict_vital_rate(fitted::AbstractFittedVitalRate, newdata::DataFrame)

Predict vital rate values for new data.
"""
function predict_vital_rate(fitted::AbstractFittedVitalRate, newdata::DataFrame)
    return GLM.predict(fitted.model, newdata)
end

"""
    predict_vital_rate(fitted::AbstractFittedVitalRate, x::AbstractVector;
                       predictor_name::Symbol=:size_t)

Predict vital rate values for a vector of predictor values.
Convenience method that wraps values into a DataFrame.
"""
function predict_vital_rate(fitted::AbstractFittedVitalRate, x::AbstractVector;
                            predictor_name::Symbol=:size_t)
    df = DataFrame(predictor_name => x)
    return predict_vital_rate(fitted, df)
end

"""
    predict_vital_rate(fitted::FittedGrowth, x::AbstractVector, y::AbstractVector;
                       predictor_name::Symbol=:size_t)

Predict growth kernel: P(size_t+1 = y | size_t = x) for Gaussian growth.
Returns a matrix of size (length(y), length(x)).
"""
function predict_vital_rate(fitted::FittedGrowth, x::AbstractVector, y::AbstractVector;
                            predictor_name::Symbol=:size_t)
    μ = predict_vital_rate(fitted, x; predictor_name=predictor_name)
    σ = fitted.sigma
    # Gaussian kernel: P(y | x) = dnorm(y, mean=μ(x), sd=σ)
    kernel = zeros(length(y), length(x))
    for (j, μj) in enumerate(μ)
        d = Normal(μj, σ)
        for (i, yi) in enumerate(y)
            kernel[i, j] = pdf(d, yi)
        end
    end
    return kernel
end

# --- Internal helpers ---

"""Map VitalRateDistribution to GLM.jl distribution and link."""
function _get_glm_family(dist::VitalRateDistribution)
    if dist isa Gaussian
        return Normal(), IdentityLink()
    elseif dist isa Binomial_
        return Bernoulli(), LogitLink()
    elseif dist isa Poisson_
        return Poisson(), LogLink()
    elseif dist isa NegativeBinomial_
        return NegativeBinomial(), LogLink()
    else
        # Default fallback for extended distributions
        return Normal(), IdentityLink()
    end
end
