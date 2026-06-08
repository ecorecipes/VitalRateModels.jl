# Kernel construction

```@docs
vital_rates_to_kernel(survival::FittedSurvival, growth::FittedGrowth, domain::StructuredPopulationCore.ContinuousDomain; predictor_name=:size_t)
vital_rates_to_kernel(fecundity::FittedFecundity, recruitment::FittedRecruitment, domain::StructuredPopulationCore.ContinuousDomain; predictor_name=:size_t)
vital_rates_to_matrix
```

```@example vrm
using VitalRateModels, DataFrames, StatsModels, Distributions, Plots
using Random
using StructuredPopulationCore: ContinuousDomain, meshpoints

Random.seed!(2104)

n = 180
size_t = clamp.(rand(Normal(5.2, 1.1), n), 0.7, 10.0)
survived = rand.(Bernoulli.(1 ./(1 .+ exp.(-(-2.0 .+ 0.5 .* size_t))))) .== 1
size_t1 = clamp.(1.0 .+ 0.9 .* size_t .+ rand(Normal(0, 0.55), n), 0.0, 11.0)
fecundity = rand.(Poisson.(exp.(-3.0 .+ 0.3 .* size_t)))
recruit_size = clamp.(rand.(Normal.(1.1 .+ 0.02 .* size_t, 0.2)), 0.2, 2.5)

demo = DataFrame(
    size_t=size_t,
    size_t1=size_t1,
    survived=survived,
    fecundity=fecundity,
    recruit_size=recruit_size,
)

survival_fit = fit_vital_rate(SurvivalModel, demo, @formula(survived ~ size_t))
growth_fit = fit_vital_rate(GrowthModel, demo, @formula(size_t1 ~ size_t))
fecundity_fit = fit_vital_rate(FecundityModel, demo, @formula(fecundity ~ size_t), distribution=Poisson_())
recruitment_fit = fit_vital_rate(RecruitmentModel, demo, @formula(recruit_size ~ size_t))

domain = ContinuousDomain(0.5, 10.0, 30)
mesh = meshpoints(domain)

P = vital_rates_to_kernel(survival_fit, growth_fit, domain)
F = vital_rates_to_kernel(fecundity_fit, recruitment_fit, domain)
```

```@example vrm
(size(P), size(F), size(P + F))
```

```@example vrm
stage_survival = vital_rates_to_matrix(survival_fit, [1.0, 3.0, 6.0, 9.0])
stage_survival
```

```@example vrm
p = heatmap(
    mesh,
    mesh,
    P + F,
    xlabel="Size at time t",
    ylabel="Size at time t+1",
    title="Combined kernel",
    color=:viridis,
)
savefig(p, "api_kernels.svg") # hide
p
```
