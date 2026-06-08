# Data preparation

```@docs
verticalize
create_stageframe
validate_demographic_data
summarize_transitions
```

```@example vrm
using VitalRateModels, DataFrames, StatsModels, Distributions, Plots
using Random

Random.seed!(2105)

wide = DataFrame(
    id=1:6,
    size_y1=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    size_y2=[1.4, 1.7, 2.5, 2.8, 3.2, 3.7],
    size_y3=[1.8, 2.1, 2.8, 3.0, 3.5, 4.0],
    fate_y1_y2=[1, 1, 1, 1, 0, 1],
    fate_y2_y3=[1, 1, 1, 0, 0, 1],
    fec_y1=[0, 0, 1, 2, 0, 3],
    fec_y2=[0, 1, 2, 0, 0, 4],
)

long = verticalize(
    wide;
    id_col=:id,
    size_cols=[:size_y1, :size_y2, :size_y3],
    fate_cols=[:fate_y1_y2, :fate_y2_y3],
    fec_cols=[:fec_y1, :fec_y2],
)

first(long, 6)
```

```@example vrm
stageframe = create_stageframe(
    stage_names=[:seedling, :small, :medium, :large],
    sizes=[0.5, 1.5, 2.5, 3.5],
    reproductive=[false, false, false, true],
)

stageframe
```

```@example vrm
validate_demographic_data(long)
```

```@example vrm
summarize_transitions(long)
```
