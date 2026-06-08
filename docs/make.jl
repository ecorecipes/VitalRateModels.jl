using Documenter
using DataFrames
using StatsModels
using StructuredPopulationCore
using VitalRateModels

ENV["GKSwstype"] = "100"

DocMeta.setdocmeta!(VitalRateModels, :DocTestSetup, :(using VitalRateModels), recursive=true)

makedocs(; 
    modules = [VitalRateModels],
    checkdocs = :exports,
    warnonly = true,
    authors = "Simon Frost",
    sitename = "VitalRateModels.jl",
    format = Documenter.HTML(; 
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://ecorecipes.github.io/VitalRateModels.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Introduction" => "tutorials/01_introduction.md",
            "Model Selection" => "tutorials/02_model_selection.md",
            "Fecundity Models" => "tutorials/03_fecundity.md",
            "Growth Kernels" => "tutorials/04_growth_kernels.md",
            "Data Preparation" => "tutorials/05_data_preparation.md",
            "Density Dependence" => "tutorials/06_density_dependence.md",
            "LTRE from Data" => "tutorials/07_ltre_from_data.md",
        ],
        "API Reference" => [
            "Types" => "api/types.md",
            "Fitting" => "api/fitting.md",
            "Model Selection" => "api/model_selection.md",
            "Kernel Construction" => "api/kernels.md",
            "Data Preparation" => "api/data_prep.md",
        ],
    ],
)

deploydocs(; 
    repo = "github.com/ecorecipes/VitalRateModels.jl.git",
    push_preview = true,
)
