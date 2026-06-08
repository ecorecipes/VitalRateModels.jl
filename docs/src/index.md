# VitalRateModels.jl

`VitalRateModels.jl` fits survival, growth, fecundity, and recruitment models from demographic data, then turns those fitted relationships into ingredients for structured population models.

```@docs
VitalRateModels
```

## Where it fits

```text
field or census data
        |
        v
VitalRateModels.jl
  ├─ fit_vital_rate
  ├─ modelsearch
  ├─ verticalize / create_stageframe
  └─ vital_rates_to_kernel / vital_rates_to_matrix
        |
        v
StructuredPopulationCore.jl
        |
        +--> IntegralProjectionModels.jl
        +--> MatrixProjectionModels.jl
        +--> other projection-model packages
```

## Quick example

```@example vrm
using VitalRateModels, DataFrames, StatsModels, Distributions, Plots
using Random
using StructuredPopulationCore: ContinuousDomain, meshpoints

Random.seed!(2026)

n = 150
size_t = clamp.(rand(Normal(5.0, 1.1), n), 0.5, 9.5)
surv_prob = @. 1 / (1 + exp(-(-2.0 + 0.5 * size_t)))
survived = rand.(Bernoulli.(surv_prob)) .== 1
size_t1 = clamp.(1.0 .+ 0.9 .* size_t .+ rand(Normal(0, 0.6), n), 0.0, 11.0)

data = DataFrame(size_t=size_t, size_t1=size_t1, survived=survived)

survival_fit = fit_vital_rate(SurvivalModel, data, @formula(survived ~ size_t))
growth_fit = fit_vital_rate(GrowthModel, data, @formula(size_t1 ~ size_t))

domain = ContinuousDomain(0.5, 10.0, 40)
mesh = meshpoints(domain)

DataFrame(
    size_t = mesh[1:5],
    survival = predict_vital_rate(survival_fit, mesh)[1:5],
    expected_size_t1 = predict_vital_rate(growth_fit, mesh)[1:5],
)
```

```@example vrm
p = plot(
    mesh,
    predict_vital_rate(survival_fit, mesh),
    linewidth=3,
    xlabel="Size at time t",
    ylabel="Survival probability",
    title="Predicted survival across the mesh",
    label="survival",
)
savefig(p, "index_quick_example.svg") # hide
p
```

## Learn more

- [Tutorials](tutorials/01_introduction.md) walk through full workflows with executable examples.
- [API Reference](api/types.md) documents the exported types and core functions.
