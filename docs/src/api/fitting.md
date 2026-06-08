# Fitting and prediction

## Fitting functions

```@docs
fit_vital_rate(::Type{SurvivalModel}, data::DataFrames.DataFrame, formula::StatsModels.FormulaTerm; distribution=Binomial_())
fit_vital_rate(::Type{GrowthModel}, data::DataFrames.DataFrame, formula::StatsModels.FormulaTerm; distribution=Gaussian())
fit_vital_rate(::Type{FecundityModel}, data::DataFrames.DataFrame, formula::StatsModels.FormulaTerm; distribution=Poisson_(), repro_formula=nothing)
fit_vital_rate(::Type{RecruitmentModel}, data::DataFrames.DataFrame, formula::StatsModels.FormulaTerm; distribution=Gaussian())
```

## Prediction functions

```@docs
predict_vital_rate(fitted::AbstractFittedVitalRate, newdata::DataFrames.DataFrame)
predict_vital_rate(fitted::AbstractFittedVitalRate, x::AbstractVector; predictor_name=:size_t)
predict_vital_rate(fitted::FittedGrowth, x::AbstractVector, y::AbstractVector; predictor_name=:size_t)
```

```@example vrm
using VitalRateModels, DataFrames, StatsModels, Distributions, Plots
using Random

Random.seed!(2102)

n = 160
size_t = clamp.(rand(Normal(4.8, 1.2), n), 0.5, 9.5)
surv_prob = @. 1 / (1 + exp(-(-2.4 + 0.55 * size_t)))
survived = rand.(Bernoulli.(surv_prob)) .== 1
size_t1 = clamp.(1.1 .+ 0.85 .* size_t .+ rand(Normal(0, 0.6), n), 0.0, 11.0)
fecundity = rand.(Poisson.(exp.(-3.2 .+ 0.35 .* size_t)))
reproduced = fecundity .> 0

demo = DataFrame(
    size_t=size_t,
    size_t1=size_t1,
    survived=survived,
    fecundity=fecundity,
    reproduced=reproduced,
)

survival_fit = fit_vital_rate(SurvivalModel, demo, @formula(survived ~ size_t))
growth_fit = fit_vital_rate(GrowthModel, demo, @formula(size_t1 ~ size_t))
fecundity_fit = fit_vital_rate(
    FecundityModel,
    demo,
    @formula(fecundity ~ size_t),
    distribution=Poisson_(),
    repro_formula=@formula(reproduced ~ size_t),
)
```

```@example vrm
size_grid = collect(range(0.5, 9.5; length=6))

DataFrame(
    size_t=size_grid,
    survival=predict_vital_rate(survival_fit, size_grid),
    growth_mean=predict_vital_rate(growth_fit, size_grid),
    fecundity_mean=predict_vital_rate(fecundity_fit, size_grid),
)
```

```@example vrm
kernel_grid = collect(range(0.5, 9.5; length=5))
growth_kernel = predict_vital_rate(growth_fit, kernel_grid, kernel_grid)
size(growth_kernel)
```
