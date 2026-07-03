"""
Kernel and matrix construction from fitted vital rates.

Bridges fitted vital rate models to IPM kernels and MPM matrix elements.
"""

"""
    vital_rates_to_kernel(survival::FittedSurvival, growth::FittedGrowth,
                          domain::ContinuousDomain; predictor_name=:size_t)

Construct a P-kernel (survival × growth) from fitted vital rates on a domain.

Returns a matrix of size (n_meshpoints × n_meshpoints).
"""
function vital_rates_to_kernel(survival::FittedSurvival, growth::FittedGrowth,
                               domain::ContinuousDomain;
                               predictor_name::Symbol=:size_t)
    x = meshpoints(domain)
    h = step_size(domain)
    n = length(x)

    # Survival probability at each meshpoint
    s = predict_vital_rate(survival, x; predictor_name=predictor_name)

    # Growth kernel: P(y | x)
    G = predict_vital_rate(growth, x, x; predictor_name=predictor_name)

    # P-kernel = s(x) * g(y|x) * h
    P = zeros(n, n)
    for j in 1:n
        P[:, j] .= s[j] .* G[:, j] .* h
    end

    return P
end

"""
    vital_rates_to_kernel(fecundity::FittedFecundity, recruitment::FittedRecruitment,
                          domain::ContinuousDomain; predictor_name=:size_t)

Construct an F-kernel (fecundity × recruit size) from fitted vital rates.
"""
function vital_rates_to_kernel(fecundity::FittedFecundity,
                               recruitment::FittedRecruitment,
                               domain::ContinuousDomain;
                               predictor_name::Symbol=:size_t)
    x = meshpoints(domain)
    h = step_size(domain)
    n = length(x)

    f = predict_vital_rate(fecundity, x; predictor_name=predictor_name)
    recruit_dists = _prediction_distributions(recruitment, x; predictor_name=predictor_name)

    F = zeros(n, n)
    c_y = [mean(pdf(dist, xi) for dist in recruit_dists) for xi in x]
    c_y ./= (sum(c_y) * h)

    for j in 1:n
        F[:, j] .= f[j] .* c_y .* h
    end

    return F
end

"""
    vital_rates_to_matrix(survival::FittedSurvival, stage_sizes::AbstractVector;
                          predictor_name=:size_t)

Construct a diagonal stage-survival matrix from fitted survival probabilities
at representative stage sizes.
"""
function vital_rates_to_matrix(survival::FittedSurvival, stage_sizes::AbstractVector;
                               predictor_name::Symbol=:size_t)
    s = predict_vital_rate(survival, Float64.(stage_sizes);
                           predictor_name=predictor_name)
    return Matrix(Diagonal(s))
end

"""
    vital_rates_to_matrix(survival::FittedSurvival, growth::FittedGrowth,
                          stage_sizes::AbstractVector; predictor_name=:size_t)

Construct a square stage-transition matrix by combining fitted survival and
stage-to-stage growth probabilities at representative stage sizes.
"""
function vital_rates_to_matrix(survival::FittedSurvival, growth::FittedGrowth,
                               stage_sizes::AbstractVector;
                               predictor_name::Symbol=:size_t)
    sizes = Float64.(stage_sizes)
    s = predict_vital_rate(survival, sizes; predictor_name=predictor_name)
    G = predict_vital_rate(growth, sizes, sizes; predictor_name=predictor_name)
    G = _normalize_columns(G)
    return G .* reshape(s, 1, :)
end

"""
    vital_rates_to_matrix(survival::FittedSurvival, growth::FittedGrowth,
                          fecundity::FittedFecundity, recruitment::FittedRecruitment,
                          stage_sizes::AbstractVector; predictor_name=:size_t)

Construct a square stage-based projection matrix approximation ``A = U + F``
from fitted survival, growth, fecundity, and recruitment vital rates.
"""
function vital_rates_to_matrix(survival::FittedSurvival, growth::FittedGrowth,
                               fecundity::FittedFecundity,
                               recruitment::FittedRecruitment,
                               stage_sizes::AbstractVector;
                               predictor_name::Symbol=:size_t)
    sizes = Float64.(stage_sizes)
    U = vital_rates_to_matrix(survival, growth, sizes; predictor_name=predictor_name)
    fec = predict_vital_rate(fecundity, sizes; predictor_name=predictor_name)
    recruit_dists = _prediction_distributions(recruitment, sizes; predictor_name=predictor_name)
    recruit_probs = [mean(pdf(dist, size_i) for dist in recruit_dists) for size_i in sizes]
    recruit_probs ./= sum(recruit_probs)
    F = recruit_probs .* reshape(fec, 1, :)
    return U + F
end

function _prediction_distributions(fitted::FittedRecruitment, x::AbstractVector;
                                   predictor_name::Symbol=:size_t)
    μ = predict_vital_rate(fitted, x; predictor_name=predictor_name)
    return _distribution_family(fitted.distribution, μ, fitted.sigma, fitted.model)
end

function _distribution_family(::Gaussian, μ::AbstractVector, σ::Float64, model)
    return [Normal(μi, max(σ, sqrt(eps(Float64)))) for μi in μ]
end

function _distribution_family(::Poisson_, μ::AbstractVector, σ::Float64, model)
    return [Poisson(max(μi, eps(Float64))) for μi in μ]
end

function _distribution_family(::NegativeBinomial_, μ::AbstractVector, σ::Float64, model)
    r = model.model.rr.d.r
    return [NegativeBinomial(r, r / (r + max(μi, eps(Float64)))) for μi in μ]
end

function _normalize_columns(mat::AbstractMatrix)
    out = copy(mat)
    for j in axes(out, 2)
        colsum = sum(@view out[:, j])
        if colsum > 0
            @views out[:, j] ./= colsum
        end
    end
    return out
end
