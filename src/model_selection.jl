"""
Automated model selection for vital rates.

Fits multiple candidate models and ranks by information criteria.
Inspired by lefko3's modelsearch() function.
"""

"""
    ModelComparisonResult

Result of model comparison/selection.

# Fields
- `models`: Vector of fitted models
- `formulas`: Corresponding formulas
- `aic_values`: AIC for each model
- `delta_aic`: ΔAIC relative to best model
- `weights`: Akaike weights
- `best_idx`: Index of the best model
"""
struct ModelComparisonResult{M<:AbstractFittedVitalRate}
    models::Vector{M}
    formulas::Vector{FormulaTerm}
    aic_values::Vector{Float64}
    delta_aic::Vector{Float64}
    weights::Vector{Float64}
    best_idx::Int
end

"""
    best_model(result::ModelComparisonResult)

Return the best-fitting model from a comparison result.
"""
best_model(result::ModelComparisonResult) = result.models[result.best_idx]

"""
    modelsearch(::Type{T}, data::DataFrame, formulas::AbstractVector{<:FormulaTerm};
                distribution=nothing, criterion=:aic) where T<:AbstractVitalRateModel

Fit multiple candidate models and rank by information criterion.

# Arguments
- `T`: Model type (SurvivalModel, GrowthModel, etc.)
- `data`: DataFrame with demographic data
- `formulas`: Vector of candidate formulas to compare
- `distribution`: Distribution family (auto-selected if nothing)
- `criterion`: `:aic` or `:bic`

# Returns
A `ModelComparisonResult` with ranked models.

# Example
```julia
formulas = [
    @formula(survived ~ size_t),
    @formula(survived ~ size_t + size_t^2),
    @formula(survived ~ size_t + size_t^2 + size_t^3),
]
result = modelsearch(SurvivalModel, data, formulas)
best = best_model(result)
```
"""
function modelsearch(::Type{T}, data::DataFrame, formulas::AbstractVector{<:FormulaTerm};
                     distribution::Union{Nothing,VitalRateDistribution}=nothing,
                     criterion::Symbol=:aic) where T<:AbstractVitalRateModel
    # Auto-select distribution
    dist = if distribution !== nothing
        distribution
    elseif T == SurvivalModel
        Binomial_()
    elseif T == GrowthModel
        Gaussian()
    elseif T == FecundityModel
        Poisson_()
    else
        Gaussian()
    end

    # Fit all candidate models
    fitted_models = AbstractFittedVitalRate[]
    valid_formulas = FormulaTerm[]
    aic_vals = Float64[]

    for f in formulas
        try
            m = fit_vital_rate(T, data, f; distribution=dist)
            push!(fitted_models, m)
            push!(valid_formulas, f)
            push!(aic_vals, m.aic)
        catch e
            @warn "Failed to fit model with formula $f: $e"
        end
    end

    isempty(fitted_models) && error("All candidate models failed to fit")

    # Compute ΔAIC and Akaike weights
    min_aic = minimum(aic_vals)
    delta = aic_vals .- min_aic
    raw_weights = exp.(-0.5 .* delta)
    weights = raw_weights ./ sum(raw_weights)
    best_idx = argmin(aic_vals)

    return ModelComparisonResult(fitted_models, valid_formulas,
                                aic_vals, delta, weights, best_idx)
end

"""
    modelsearch(::Type{T}, data::DataFrame, response::Symbol, predictors::Vector{Symbol};
                max_order::Int=2, distribution=nothing) where T<:AbstractVitalRateModel

Automated model search with polynomial terms up to `max_order`.
Generates candidate formulas automatically from predictor combinations.
"""
function modelsearch(::Type{T}, data::DataFrame, response::Symbol,
                     predictors::Vector{Symbol};
                     max_order::Int=2,
                     distribution::Union{Nothing,VitalRateDistribution}=nothing) where T<:AbstractVitalRateModel
    # Generate candidate formulas with polynomial terms
    formulas = FormulaTerm[]

    # Linear models with each predictor
    for p in predictors
        push!(formulas, term(response) ~ term(p))
    end

    # All predictors linear
    if length(predictors) > 1
        rhs = foldl(+, term.(predictors))
        push!(formulas, term(response) ~ rhs)
    end

    # Add polynomial terms
    for p in predictors
        for order in 2:max_order
            terms_list = [term(p)]
            for o in 2:order
                push!(terms_list, FunctionTerm(^, [term(p), ConstantTerm(o)]))
            end
            rhs = foldl(+, terms_list)
            push!(formulas, term(response) ~ rhs)
        end
    end

    return modelsearch(T, data, formulas; distribution=distribution)
end
