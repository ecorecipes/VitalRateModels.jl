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

    # Fecundity at each parental size
    f = predict_vital_rate(fecundity, x; predictor_name=predictor_name)

    # Recruitment size distribution (assumed independent of parent size for now)
    # Use model intercept to get mean offspring size
    r_pred = predict_vital_rate(recruitment, x; predictor_name=predictor_name)

    # Simple F-kernel: f(x) * c(y) * h
    # where c(y) is the recruit size PDF
    F = zeros(n, n)
    recruit_dist = Normal(mean(r_pred), recruitment isa FittedRecruitment ? 1.0 : 0.1)
    c_y = pdf.(recruit_dist, x)
    c_y ./= (sum(c_y) * h)  # normalize to integrate to 1

    for j in 1:n
        F[:, j] .= f[j] .* c_y .* h
    end

    return F
end

"""
    vital_rates_to_matrix(survival::FittedSurvival, data::DataFrame,
                          stages::Vector{Symbol}; predictor_name=:size_t)

Construct stage-based survival probabilities for an MPM from fitted vital rates.
Returns a vector of survival probabilities, one per stage.
"""
function vital_rates_to_matrix(survival::FittedSurvival, stage_sizes::AbstractVector;
                               predictor_name::Symbol=:size_t)
    return predict_vital_rate(survival, Float64.(stage_sizes);
                              predictor_name=predictor_name)
end
