# Model selection

```@docs
ModelComparisonResult
best_model
modelsearch(::Type{T}, data::DataFrames.DataFrame, formulas::AbstractVector{<:StatsModels.FormulaTerm}; distribution=nothing, criterion=:aic) where T<:AbstractVitalRateModel
modelsearch(::Type{T}, data::DataFrames.DataFrame, response::Symbol, predictors::Vector{Symbol}; max_order=2, distribution=nothing) where T<:AbstractVitalRateModel
```

```@example vrm
using VitalRateModels, DataFrames, StatsModels, Distributions, Plots
using Random

Random.seed!(2103)

n = 220
size_t = clamp.(rand(Normal(5.0, 1.4), n), 0.5, 10.0)
logit_survival = @. -2.7 + 0.9 * size_t - 0.08 * size_t^2
survived = rand.(Bernoulli.(1 ./(1 .+ exp.(-logit_survival)))) .== 1

demo = DataFrame(size_t=size_t, survived=survived)

formulas = [
    @formula(survived ~ size_t),
    @formula(survived ~ size_t + size_t^2),
]

comparison = modelsearch(SurvivalModel, demo, formulas)
```

```@example vrm
DataFrame(
    formula=string.(comparison.formulas),
    aic=round.(comparison.aic_values, digits=2),
    delta_aic=round.(comparison.delta_aic, digits=2),
    weight=round.(comparison.weights, digits=3),
)
```

```@example vrm
best_fit = best_model(comparison)
size_grid = collect(range(0.5, 10.0; length=150))

p = plot(
    size_grid,
    predict_vital_rate(best_fit, size_grid),
    linewidth=3,
    xlabel="Size at time t",
    ylabel="Survival probability",
    title="Best-supported survival model",
    label="best model",
)
savefig(p, "api_model_selection.svg") # hide
p
```
