using Test
using VitalRateModels
using DataFrames
using StatsModels

@testset "VitalRateModels" begin

    @testset "Types" begin
        @test SurvivalModel() isa AbstractVitalRateModel
        @test GrowthModel() isa AbstractVitalRateModel
        @test FecundityModel() isa AbstractVitalRateModel
        @test RecruitmentModel() isa AbstractVitalRateModel
        @test Gaussian() isa VitalRateDistribution
        @test Binomial_() isa VitalRateDistribution
        @test Poisson_() isa VitalRateDistribution
    end

    # Generate test data
    n = 200
    size_t = randn(n) .* 2 .+ 5
    survived = [rand() < 1 / (1 + exp(-(s - 5))) for s in size_t]
    size_t1 = size_t .+ randn(n) .* 0.5 .+ 0.3
    fecundity = [max(0, round(Int, exp(0.5 * s - 2) + randn())) for s in size_t]

    data = DataFrame(
        size_t = size_t,
        size_t1 = size_t1,
        survived = survived,
        fecundity = fecundity
    )

    @testset "Fitting" begin
        surv = fit_vital_rate(SurvivalModel, data, @formula(survived ~ size_t))
        @test surv isa FittedSurvival
        @test surv.n_obs == n
        @test isfinite(surv.aic)

        growth = fit_vital_rate(GrowthModel, data, @formula(size_t1 ~ size_t))
        @test growth isa FittedGrowth
        @test growth.sigma > 0

        fec = fit_vital_rate(FecundityModel, data, @formula(fecundity ~ size_t);
                             distribution=Poisson_())
        @test fec isa FittedFecundity
    end

    @testset "Prediction" begin
        surv = fit_vital_rate(SurvivalModel, data, @formula(survived ~ size_t))
        pred = predict_vital_rate(surv, [3.0, 5.0, 7.0])
        @test length(pred) == 3
        @test all(0 .<= pred .<= 1)

        growth = fit_vital_rate(GrowthModel, data, @formula(size_t1 ~ size_t))
        kernel = predict_vital_rate(growth, [4.0, 5.0, 6.0], [4.0, 5.0, 6.0])
        @test size(kernel) == (3, 3)
        @test all(kernel .>= 0)
    end

    @testset "Model selection" begin
        formulas = [
            @formula(survived ~ size_t),
            @formula(survived ~ size_t + size_t^2),
        ]
        result = modelsearch(SurvivalModel, data, formulas)
        @test result isa ModelComparisonResult
        @test length(result.models) == 2
        @test result.best_idx in 1:2
        @test sum(result.weights) ≈ 1.0 atol=1e-10
        @test all(result.delta_aic .>= 0)

        best = best_model(result)
        @test best isa FittedSurvival
    end

    @testset "Data preparation" begin
        # Test StageFrame creation
        sf = create_stageframe(
            stage_names = [:seed, :small, :large],
            sizes = [0.0, 5.0, 20.0],
            reproductive = [false, false, true],
        )
        @test sf isa StageFrame
        @test length(sf.stage_names) == 3
        @test sf.reproductive == [false, false, true]

        # Test verticalize
        wide = DataFrame(
            id = 1:5,
            size_y1 = [1.0, 2.0, 3.0, 4.0, 5.0],
            size_y2 = [1.5, 2.5, 3.5, 4.5, 5.5],
            size_y3 = [2.0, 3.0, 4.0, 5.0, 6.0],
        )
        long = verticalize(wide; id_col=:id,
                           size_cols=[:size_y1, :size_y2, :size_y3])
        @test nrow(long) == 10  # 5 individuals × 2 transitions
        @test :size_t in names(long)
        @test :size_t1 in names(long)

        # Test validation
        @test validate_demographic_data(data) == true
    end
end
