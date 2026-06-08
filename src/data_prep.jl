"""
Demographic data preparation utilities.

Converts raw longitudinal data into formats suitable for vital rate estimation.
Inspired by lefko3's verticalize3() and sf_create() functions.
"""

"""
    StageFrame

A stage classification framework for structured population models.

# Fields
- `stage_names`: Names of life stages
- `sizes`: Representative size for each stage (midpoint or mean)
- `bin_halfwidths`: Half-width of size bins for IPM-style classification
- `reproductive`: Whether each stage is reproductive
- `observable`: Whether each stage is observable in the field
- `mature`: Whether each stage is mature
- `propagule`: Whether each stage is a propagule/seed stage
- `dormant`: Whether each stage is dormant/unobservable
"""
struct StageFrame
    stage_names::Vector{Symbol}
    sizes::Vector{Float64}
    bin_halfwidths::Vector{Float64}
    reproductive::Vector{Bool}
    observable::Vector{Bool}
    mature::Vector{Bool}
    propagule::Vector{Bool}
    dormant::Vector{Bool}
end

"""
    create_stageframe(; stage_names, sizes, bin_halfwidths=nothing,
                       reproductive=nothing, observable=nothing, mature=nothing,
                       propagule=nothing, dormant=nothing)

Create a StageFrame for stage classification.

# Example
```julia
sf = create_stageframe(
    stage_names = [:seed, :seedling, :small, :large, :flowering],
    sizes = [0.0, 5.0, 20.0, 100.0, 200.0],
    reproductive = [false, false, false, false, true],
    observable = [false, true, true, true, true],
)
```
"""
function create_stageframe(;
        stage_names::Vector{Symbol},
        sizes::Vector{<:Real},
        bin_halfwidths::Union{Nothing,Vector{<:Real}}=nothing,
        reproductive::Union{Nothing,Vector{Bool}}=nothing,
        observable::Union{Nothing,Vector{Bool}}=nothing,
        mature::Union{Nothing,Vector{Bool}}=nothing,
        propagule::Union{Nothing,Vector{Bool}}=nothing,
        dormant::Union{Nothing,Vector{Bool}}=nothing)

    n = length(stage_names)
    length(sizes) == n || throw(ArgumentError("sizes must have same length as stage_names"))

    bw = bin_halfwidths !== nothing ? Float64.(bin_halfwidths) : diff(vcat([0.0], Float64.(sizes))) ./ 2
    length(bw) == n || (bw = fill(mean(diff(Float64.(sizes))) / 2, n))

    return StageFrame(
        stage_names,
        Float64.(sizes),
        bw,
        reproductive !== nothing ? reproductive : fill(false, n),
        observable !== nothing ? observable : fill(true, n),
        mature !== nothing ? mature : fill(true, n),
        propagule !== nothing ? propagule : fill(false, n),
        dormant !== nothing ? dormant : fill(false, n),
    )
end

"""
    verticalize(data::DataFrame; id_col::Symbol, year_cols::Vector{Symbol},
                size_cols::Vector{Symbol}, fate_cols::Vector{Symbol}=Symbol[],
                fec_cols::Vector{Symbol}=Symbol[])

Convert wide-format longitudinal data to vertical (one-row-per-transition) format.

Takes data where each row is an individual and columns represent measurements
across years, and converts to a long format where each row represents a
single demographic transition (t → t+1).

# Arguments
- `data`: Wide-format DataFrame
- `id_col`: Column identifying individuals
- `year_cols`: Columns for year/time identifiers (in order)
- `size_cols`: Size measurement columns (one per year, in order)
- `fate_cols`: Survival/fate columns (optional, one per transition)
- `fec_cols`: Fecundity columns (optional, one per year)

# Returns
A DataFrame with columns: `individual`, `year`, `size_t`, `size_t1`,
`survived` (if fate available), `fecundity` (if fec available).
"""
function verticalize(data::DataFrame;
                     id_col::Symbol,
                     size_cols::Vector{Symbol},
                     fate_cols::Vector{Symbol}=Symbol[],
                     fec_cols::Vector{Symbol}=Symbol[])
    n_years = length(size_cols)
    n_transitions = n_years - 1
    n_transitions > 0 || throw(ArgumentError("Need at least 2 size columns for transitions"))

    rows = NamedTuple[]
    for row in eachrow(data)
        for t in 1:n_transitions
            size_t = row[size_cols[t]]
            size_t1 = row[size_cols[t+1]]

            # Skip if either size is missing
            (ismissing(size_t) || ismissing(size_t1)) && continue

            entry = (individual = row[id_col],
                     year = t,
                     size_t = Float64(size_t),
                     size_t1 = Float64(size_t1))

            # Add survival if available
            if !isempty(fate_cols) && t <= length(fate_cols)
                survived = row[fate_cols[t]]
                entry = merge(entry, (survived = !ismissing(survived) && survived != 0,))
            end

            # Add fecundity if available
            if !isempty(fec_cols) && t <= length(fec_cols)
                fec = row[fec_cols[t]]
                entry = merge(entry, (fecundity = ismissing(fec) ? 0.0 : Float64(fec),))
            end

            push!(rows, entry)
        end
    end

    return DataFrame(rows)
end

"""
    validate_demographic_data(data::DataFrame; size_col=:size_t,
                              survival_col=:survived, fecundity_col=:fecundity)

Run quality checks on demographic data and print a summary.
"""
function validate_demographic_data(data::DataFrame;
                                   size_col::Symbol=:size_t,
                                   survival_col::Symbol=:survived,
                                   fecundity_col::Symbol=:fecundity)
    issues = String[]
    cols = Set(propertynames(data))

    if size_col in cols
        s = data[!, size_col]
        any(ismissing, s) && push!(issues, "$(count(ismissing, s)) missing values in $size_col")
        non_missing = collect(skipmissing(s))
        any(x -> x < 0, non_missing) && push!(issues, "Negative sizes found in $size_col")
    else
        push!(issues, "Column $size_col not found")
    end

    if survival_col in cols
        surv = data[!, survival_col]
        unique_vals = unique(collect(skipmissing(surv)))
        if !all(v -> v in (0, 1, true, false), unique_vals)
            push!(issues, "Non-binary values in $survival_col: $(unique_vals)")
        end
    end

    if fecundity_col in cols
        fec = data[!, fecundity_col]
        non_missing = collect(skipmissing(fec))
        any(x -> x < 0, non_missing) && push!(issues, "Negative fecundity values")
    end

    if isempty(issues)
        println("✓ Data validation passed ($(nrow(data)) observations)")
    else
        println("⚠ Data validation issues:")
        for issue in issues
            println("  - $issue")
        end
    end

    return isempty(issues)
end

"""
    summarize_transitions(data::DataFrame; size_col=:size_t, size_t1_col=:size_t1)

Summarize demographic transitions in the data.
"""
function summarize_transitions(data::DataFrame;
                               size_col::Symbol=:size_t,
                               size_t1_col::Symbol=:size_t1)
    n = nrow(data)
    cols = Set(propertynames(data))
    println("Demographic transition summary:")
    println("  N transitions: $n")

    if size_col in cols
        s = collect(skipmissing(data[!, size_col]))
        println("  Size at t:   min=$(minimum(s)), max=$(maximum(s)), mean=$(round(mean(s), digits=2))")
    end

    if size_t1_col in cols
        s1 = collect(skipmissing(data[!, size_t1_col]))
        println("  Size at t+1: min=$(minimum(s1)), max=$(maximum(s1)), mean=$(round(mean(s1), digits=2))")
    end

    if :survived in cols
        surv = collect(skipmissing(data[!, :survived]))
        println("  Survival:    $(round(mean(surv), digits=3)) ($(sum(surv))/$(length(surv)))")
    end

    if :fecundity in cols
        fec = collect(skipmissing(data[!, :fecundity]))
        println("  Fecundity:   mean=$(round(mean(fec), digits=2)), max=$(maximum(fec))")
    end
end
