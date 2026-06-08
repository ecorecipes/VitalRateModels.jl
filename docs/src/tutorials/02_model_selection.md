## Overview

Vital-rate regressions are rarely known a priori. Following information-theoretic practice in demography, it is common to compare biologically plausible alternatives and select among them with AIC, $\Delta$AIC, and Akaike weights. `VitalRateModels.jl` provides `modelsearch()` for exactly this purpose.

Here we compare linear, quadratic, and cubic survival models for a size-structured perennial.

## Setup

```@example vrm
using VitalRateModels, DataFrames, StatsModels, Distributions, Plots
using Random
using Statistics

Random.seed!(2027)
```

## Simulated survival data

We generate data from a slightly curved survival relationship so that the model-selection exercise has a meaningful signal.

```@example vrm
n = 450
size_t = clamp.(rand(Normal(4.8, 1.5), n), 0.5, 10.0)
logit_survival = @. -3.0 + 1.0 * size_t - 0.09 * size_t^2 + 0.003 * size_t^3
survival_prob = @. 1 / (1 + exp(-logit_survival))
survived = rand.(Bernoulli.(survival_prob)) .== 1

data = DataFrame(size_t=size_t, survived=survived)
println("Observed survival = ", round(mean(data.survived), digits=3))
```

## Candidate models

The candidate set includes a linear trend, a quadratic curvature term, and a cubic polynomial. In real analyses these choices should come from biological expectations rather than arbitrary model fishing.

```@example vrm
formulas = [
    @formula(survived ~ size_t),
    @formula(survived ~ size_t + size_t^2),
    @formula(survived ~ size_t + size_t^2 + size_t^3),
]

comparison = modelsearch(SurvivalModel, data, formulas)
best_fit = best_model(comparison)
```

## AIC table

`modelsearch()` returns the fitted models, the original formulas, their AIC scores, the $\Delta$AIC values relative to the best model, and the corresponding Akaike weights.

```@example vrm
labels = ["linear", "quadratic", "cubic"]
aic_table = DataFrame(
    model = labels,
    formula = string.(comparison.formulas),
    AIC = round.(comparison.aic_values, digits=2),
    delta_AIC = round.(comparison.delta_aic, digits=2),
    akaike_weight = round.(comparison.weights, digits=3),
)

aic_table
```

```@example vrm
println("Best model formula: ", comparison.formulas[comparison.best_idx])
println("Best model AIC:     ", round(comparison.aic_values[comparison.best_idx], digits=2))
```

## Comparing fitted curves

To see what the ranking means biologically, we predict survival across the observed size range and overlay the candidate fits.

```@example vrm
size_domain = collect(range(0.5, 10.0; length=200))
predictions = [predict_vital_rate(model, size_domain) for model in comparison.models]
colors = [:black, :royalblue, :firebrick]
```

```@example vrm
p = scatter(
    data.size_t,
    Float64.(data.survived) .+ rand(Normal(0, 0.03), n),
    alpha=0.18,
    ms=2.5,
    color=:gray60,
    label="Observed data",
    xlabel="Size at time t",
    ylabel="Survival probability",
    title="Candidate survival models",
)

for i in eachindex(predictions)
    plot!(p, size_domain, predictions[i], linewidth=3, color=colors[i], label=labels[i])
end

savefig(p, "tutorial_02_model_selection_curves.svg") # hide
p
```

In this simulated example the quadratic or cubic model typically outperforms the linear fit because the true relationship bends upward for intermediate sizes and then saturates. When multiple models have similar support, the Akaike weights are often more informative than a single "winner".

## Interpreting AIC weights

Akaike weights can be read as relative support for each candidate within the candidate set. They are especially helpful when the top two models have small $\Delta$AIC values.

```@example vrm
p = bar(
    labels,
    comparison.weights,
    xlabel="Candidate model",
    ylabel="Akaike weight",
    title="Relative support across the candidate set",
    legend=false,
    color=[:black, :royalblue, :firebrick],
    alpha=0.8,
)
savefig(p, "tutorial_02_model_selection_weights.svg") # hide
p
```

## Summary

In this vignette we:

1. built a biologically motivated candidate set of survival models,
2. ranked those models with `modelsearch()`,
3. inspected the AIC table, $\Delta$AIC values, and Akaike weights, and
4. compared the fitted curves directly on the survival scale.

The same workflow applies to growth, fecundity, or recruitment models whenever several alternative formulas are scientifically plausible.

## References

- Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference*. Springer.
- Caswell, H. (2001). *Matrix Population Models*. Sinauer.
- Ellner, S. P., Childs, D. Z., & Rees, M. (2016). *Data-Driven Modelling of Structured Populations*. Springer.
