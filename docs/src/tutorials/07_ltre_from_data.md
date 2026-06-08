## Overview

A life table response experiment (LTRE) decomposes differences in population growth between environments, treatments, or populations into contributions from underlying demographic processes. Here we fit vital rates for a "good year" and a "bad year", construct kernels from those fitted rates, and then compare classical and exact LTRE decompositions.

To keep the exact fANOVA decomposition tractable, we use a deliberately coarse discretization of the size domain.

## Setup

```@example vrm
using VitalRateModels, DataFrames, StatsModels, Distributions, Plots
using Random
using StructuredPopulationCore: ContinuousDomain, meshpoints, lambda, ltre, exact_ltre

Random.seed!(2032)
```

## Simulating two environments

The good year has higher survival, more growth, and higher fecundity than the bad year.

```@example vrm
function simulate_year(n; surv_int, surv_slope, grow_int, grow_slope, grow_sd, fec_int, fec_slope, recruit_mean)
    size_t = clamp.(rand(Normal(5.0, 1.3), n), 0.8, 9.5)
    surv_prob = @. 1 / (1 + exp(-(surv_int + surv_slope * size_t)))
    survived = rand.(Bernoulli.(surv_prob)) .== 1

    size_t1 = similar(size_t)
    for i in eachindex(size_t)
        μ = grow_int + grow_slope * size_t[i]
        size_t1[i] = survived[i] ? rand(Normal(μ, grow_sd)) : 0.0
    end
    size_t1 = clamp.(size_t1, 0.0, 11.0)

    fec_mean = @. exp(fec_int + fec_slope * size_t)
    fecundity = [survived[i] ? rand(Poisson(fec_mean[i])) : 0 for i in eachindex(size_t)]

    repro_idx = findall(>(0), fecundity)
    parent_size = size_t[repro_idx]
    recruit_size = clamp.(rand.(Normal.(fill(recruit_mean, length(parent_size)), 0.35)), 0.2, 3.0)

    demography = DataFrame(size_t=size_t, size_t1=size_t1, survived=survived, fecundity=fecundity)
    recruits = DataFrame(size_t=parent_size, recruit_size=recruit_size)
    return demography, recruits
end

good_demo, good_recruits = simulate_year(
    500;
    surv_int=-1.5,
    surv_slope=0.52,
    grow_int=1.2,
    grow_slope=0.90,
    grow_sd=0.7,
    fec_int=-3.1,
    fec_slope=0.36,
    recruit_mean=1.3,
)

bad_demo, bad_recruits = simulate_year(
    500;
    surv_int=-2.6,
    surv_slope=0.42,
    grow_int=0.7,
    grow_slope=0.82,
    grow_sd=0.9,
    fec_int=-4.0,
    fec_slope=0.28,
    recruit_mean=1.0,
)
```

## Fitting vital rates for each population

```@example vrm
function fit_all_vital_rates(demo, recruits)
    survival = fit_vital_rate(SurvivalModel, demo, @formula(survived ~ size_t))
    growth = fit_vital_rate(GrowthModel, filter(:survived => identity, demo), @formula(size_t1 ~ size_t))
    fecundity = fit_vital_rate(FecundityModel, demo, @formula(fecundity ~ size_t), distribution=Poisson_())
    recruitment = fit_vital_rate(RecruitmentModel, recruits, @formula(recruit_size ~ size_t))
    return survival, growth, fecundity, recruitment
end

good_surv, good_grow, good_fec, good_rec = fit_all_vital_rates(good_demo, good_recruits)
bad_surv, bad_grow, bad_fec, bad_rec = fit_all_vital_rates(bad_demo, bad_recruits)
```

## Building kernels

We use only three meshpoints here so that `exact_ltre()` remains computationally feasible.

```@example vrm
domain = ContinuousDomain(1.0, 9.0, 3)
z = meshpoints(domain)

K_good = vital_rates_to_kernel(good_surv, good_grow, domain) +
         vital_rates_to_kernel(good_fec, good_rec, domain)
K_bad = vital_rates_to_kernel(bad_surv, bad_grow, domain) +
        vital_rates_to_kernel(bad_fec, bad_rec, domain)

λ_good = lambda(K_good)
λ_bad = lambda(K_bad)
Δλ = λ_good - λ_bad

println("λ_good = ", round(λ_good, digits=4))
println("λ_bad  = ", round(λ_bad, digits=4))
println("Δλ     = ", round(Δλ, digits=4))
```

## Classical LTRE

The classical LTRE uses a first-order Taylor approximation around the midpoint or reference matrix.

```@example vrm
classical = ltre(K_good, K_bad)
println("Classical LTRE sum = ", round(sum(classical.contributions), digits=6))
println("Approximation error = ", round(sum(classical.contributions) - classical.delta_lambda, digits=6))
```

```@example vrm
p = heatmap(
    z,
    z,
    classical.contributions,
    xlabel="Parent size class",
    ylabel="Offspring/next size class",
    title="Classical LTRE contributions",
    color=:balance,
)
savefig(p, "tutorial_07_ltre_classical.svg") # hide
p
```

## Exact LTRE

The exact LTRE of Hernandez et al. (2023) decomposes $\Delta\lambda$ into main effects and interactions without relying on the linear approximation.

```@example vrm
exact = exact_ltre(K_good, K_bad)
println("Exact LTRE sum = ", round(sum(exact.effects), digits=6))
println("Recovery error = ", round(sum(exact.effects) - Δλ, digits=10))
```

### Main effects versus interactions

```@example vrm
orders = length.(exact.effect_indices)
main_idx = findall(==(1), orders)
interaction_idx = findall(>(1), orders)

main_effects = exact.effects[main_idx]
main_labels = String[]
for idx in main_idx
    varying_slot = exact.effect_indices[idx][1]
    linear_idx = exact.indices_varying[varying_slot]
    I = CartesianIndices(K_bad)[linear_idx]
    push!(main_labels, "(" * string(I[1]) * ", " * string(I[2]) * ")")
end

order = sortperm(abs.(main_effects), rev=true)
keep = order[1:min(6, length(order))]

println("Total interaction contribution = ", round(sum(exact.effects[interaction_idx]), digits=6))
```

```@example vrm
p = bar(
    main_labels[keep],
    main_effects[keep],
    xlabel="Matrix element (row, column)",
    ylabel="Exact main-effect contribution",
    title="Largest exact LTRE main effects",
    legend=false,
    color=:slateblue,
    alpha=0.8,
)
savefig(p, "tutorial_07_ltre_exact.svg") # hide
p
```

## Comparing classical and exact decompositions

```@example vrm
comparison = DataFrame(
    method=["Observed Δλ", "Classical LTRE sum", "Exact LTRE sum"],
    value=round.([Δλ, sum(classical.contributions), sum(exact.effects)], digits=6),
)
comparison
```

The classical LTRE is usually close when the two kernels are similar, but the exact method reveals whether non-additive interactions among matrix elements matter. This is especially useful when multiple vital rates shift together between good and bad environments.

## Summary

In this vignette we:

1. fit vital rates separately for two environments,
2. converted those fits into coarse IPM kernels,
3. decomposed the difference in $\lambda$ with `ltre()`, and
4. compared that linear approximation with `exact_ltre()`.

This workflow links statistical vital-rate estimation directly to demographic explanation.

## References

- Caswell, H. (2001). *Matrix Population Models*. Sinauer.
- Hernandez, M. J., et al. (2023). exactLTRE: exact life table response experiments. *Methods in Ecology and Evolution*, 14, 1065-1078.
- Ellner, S. P., Childs, D. Z., & Rees, M. (2016). *Data-Driven Modelling of Structured Populations*. Springer.
