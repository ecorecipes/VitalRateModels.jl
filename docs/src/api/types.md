# Types

## Model tags and abstract interfaces

```@docs
AbstractVitalRateModel
AbstractFittedVitalRate
VitalRateDistribution
SurvivalModel
GrowthModel
FecundityModel
RecruitmentModel
```

```@example vrm
using VitalRateModels, DataFrames, StatsModels, Distributions, Plots
using Random

Random.seed!(2101)

models = (
    SurvivalModel(),
    GrowthModel(),
    FecundityModel(),
    RecruitmentModel(),
)

distributions = (
    Gaussian(),
    Binomial_(),
    NegativeBinomial_(),
    Poisson_(),
    ZeroInflatedPoisson(),
    ZeroInflatedNegBin(),
    TruncatedPoisson(),
    TruncatedNegBin(),
)

(models=models, distributions=distributions)
```

## Distribution subtypes

```@docs
Gaussian
Binomial_
NegativeBinomial_
Poisson_
ZeroInflatedPoisson
ZeroInflatedNegBin
TruncatedPoisson
TruncatedNegBin
```

## Fitted model containers

```@docs
FittedSurvival
FittedGrowth
FittedFecundity
FittedRecruitment
```

```@example vrm
n = 120
size_t = clamp.(rand(Normal(5.0, 1.0), n), 0.5, 9.0)
surv_prob = @. 1 / (1 + exp(-(-1.8 + 0.45 * size_t)))
survived = rand.(Bernoulli.(surv_prob)) .== 1
size_t1 = clamp.(0.9 .+ 0.95 .* size_t .+ rand(Normal(0, 0.5), n), 0.0, 10.5)
fecundity = rand.(Poisson.(exp.(-2.5 .+ 0.25 .* size_t)))
recruit_size = clamp.(rand.(Normal.(1.0 .+ 0.03 .* size_t, 0.25)), 0.1, 2.5)

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

(typeof(survival_fit), typeof(growth_fit), typeof(fecundity_fit), typeof(recruitment_fit))
```

## Stage frames

```@docs
StageFrame
```

```@example vrm
stageframe = create_stageframe(
    stage_names=[:seedling, :small, :large, :flowering],
    sizes=[0.5, 2.5, 6.0, 9.0],
    reproductive=[false, false, false, true],
    observable=[true, true, true, true],
)

stageframe
```
