using Test
using VitalRateModels
using DataFrames
using StatsModels
using Random
using GLM: coef, dof, loglikelihood, nobs
using StructuredPopulationCore: ContinuousDomain, meshpoints

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
    size_t = clamp.(randn(n) .* 2 .+ 5, 0.1, Inf)
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

        recruitment = fit_vital_rate(RecruitmentModel, data, @formula(size_t1 ~ size_t))
        @test recruitment isa FittedRecruitment
        @test recruitment.sigma > 0

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
        @test length(result.aic_values) == 2
        @test length(result.bic_values) == 2
        @test result.best_idx in 1:2
        @test sum(result.weights) ≈ 1.0 atol=1e-10
        @test all(result.delta_aic .>= 0)

        best = best_model(result)
        @test best isa FittedSurvival

        rng = MersenneTwister(1)
        n_bic = 25
        x_bic = collect(range(-1.0, 1.0; length=n_bic))
        y_bic = 0.05 .* x_bic .+ 0.02 .* x_bic.^2 .+ 0.45 .* randn(rng, n_bic)
        bic_data = DataFrame(x = x_bic, y = y_bic)
        bic_formulas = [
            @formula(y ~ x),
            @formula(y ~ x + x^2),
            @formula(y ~ x + x^2 + x^3),
            @formula(y ~ x + x^2 + x^3 + x^4),
        ]

        aic_result = modelsearch(GrowthModel, bic_data, bic_formulas; criterion=:aic)
        bic_result = modelsearch(GrowthModel, bic_data, bic_formulas; criterion=:bic)
        @test aic_result.criterion == :aic
        @test bic_result.criterion == :bic
        @test aic_result.criterion_values == aic_result.aic_values
        @test bic_result.criterion_values == bic_result.bic_values
        @test sortperm(aic_result.criterion_values) != sortperm(bic_result.criterion_values)
        first_model = bic_result.models[1].model
        k = dof(first_model)
        logL = loglikelihood(first_model)
        nobs_model = nobs(first_model)
        @test bic_result.bic_values[1] ≈ (k * log(nobs_model) - 2 * logL) atol=1e-8
    end


    @testset "Unsupported distribution errors" begin
        @test_throws ArgumentError fit_vital_rate(FecundityModel, data, @formula(fecundity ~ size_t);
                                                 distribution=ZeroInflatedPoisson())
        @test_throws ArgumentError fit_vital_rate(FecundityModel, data, @formula(fecundity ~ size_t);
                                                 distribution=ZeroInflatedNegBin())
        @test_throws ArgumentError fit_vital_rate(FecundityModel, data, @formula(fecundity ~ size_t);
                                                 distribution=TruncatedPoisson())
        @test_throws ArgumentError fit_vital_rate(FecundityModel, data, @formula(fecundity ~ size_t);
                                                 distribution=TruncatedNegBin())
    end

    @testset "Kernel and matrix construction" begin
        surv = fit_vital_rate(SurvivalModel, data, @formula(survived ~ size_t))
        growth_low = fit_vital_rate(GrowthModel, data, @formula(size_t1 ~ size_t))

        recruit_rng = MersenneTwister(11)
        recruit_x = repeat([1.0, 2.0, 3.0, 4.0], inner=80)
        recruit_df_narrow = DataFrame(size_t = recruit_x, recruit_size = 1.5 .+ 0.1 .* randn(recruit_rng, length(recruit_x)))
        recruit_df_wide = DataFrame(size_t = recruit_x, recruit_size = 1.5 .+ 1.5 .* randn(recruit_rng, length(recruit_x)))
        fec_df = DataFrame(size_t = recruit_x, fecundity = round.(Int, 2 .+ recruit_x .+ abs.(randn(recruit_rng, length(recruit_x)))))
        fec = fit_vital_rate(FecundityModel, fec_df, @formula(fecundity ~ size_t); distribution=Poisson_())
        recruitment_narrow = fit_vital_rate(RecruitmentModel, recruit_df_narrow, @formula(recruit_size ~ size_t))
        recruitment_wide = fit_vital_rate(RecruitmentModel, recruit_df_wide, @formula(recruit_size ~ size_t))
        dom = ContinuousDomain(minimum(recruit_x), maximum(recruit_x), 40)
        F_narrow = vital_rates_to_kernel(fec, recruitment_narrow, dom)
        F_wide = vital_rates_to_kernel(fec, recruitment_wide, dom)
        mesh = meshpoints(dom)
        col = size(F_narrow, 2) ÷ 2
        mean_narrow = sum(mesh .* F_narrow[:, col]) / sum(F_narrow[:, col])
        var_narrow = sum(((mesh .- mean_narrow) .^ 2) .* F_narrow[:, col]) / sum(F_narrow[:, col])
        mean_wide = sum(mesh .* F_wide[:, col]) / sum(F_wide[:, col])
        var_wide = sum(((mesh .- mean_wide) .^ 2) .* F_wide[:, col]) / sum(F_wide[:, col])
        @test var_wide > var_narrow

        stages = [2.0, 3.5, 5.0]
        U = vital_rates_to_matrix(surv, growth_low, stages)
        A = vital_rates_to_matrix(surv, growth_low, fec, recruitment_narrow, stages)
        @test size(U) == (3, 3)
        @test size(A) == (3, 3)
        @test all(A .>= 0)
        @test any(A[1, :] .> U[1, :])
        @test all(vec(sum(U; dims=1)) .<= predict_vital_rate(surv, stages) .+ 1e-8)
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
        @test :size_t in propertynames(long)
        @test :size_t1 in propertynames(long)

        # Test validation
        @test validate_demographic_data(data) == true
    end
end
