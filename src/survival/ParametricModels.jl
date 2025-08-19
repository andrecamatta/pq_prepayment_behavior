"""
Modelos paramétricos de sobrevivência para análise de pré-pagamento
Inclui Weibull, Log-normal, Log-logístico com covariáveis
"""

using LinearAlgebra: dot
using SpecialFunctions: loggamma, logbeta, erf, erfc, beta_inc
using Optim: optimize, BFGS, NelderMead, converged, minimizer
using Optim
using Statistics: mean, std, quantile
using Dates: Month

abstract type ParametricSurvivalModel end

function add_l2_regularization(likelihood::Float64, coefficients::Vector{Float64}, 
                              regularization::Float64, exclude_indices::Vector{Int}=[1])::Float64
    """
    Add L2 regularization penalty to likelihood.
    
    Args:
        likelihood: Base log-likelihood value
        coefficients: Model coefficients to regularize
        regularization: Regularization strength (λ)
        exclude_indices: Indices to exclude from regularization (default: [1] for intercept)
    
    Returns:
        Regularized log-likelihood
    """
    if regularization <= 0.0
        return likelihood
    end
    
    # Apply regularization only to specified coefficients (exclude intercept by default)
    reg_indices = setdiff(1:length(coefficients), exclude_indices)
    reg_penalty = regularization * sum(coefficients[reg_indices].^2)
    
    return likelihood - reg_penalty
end

struct WeibullPrepaymentModel <: ParametricSurvivalModel
    shape::Float64
    scale_coefficients::Vector{Float64}
    covariate_names::Vector{Symbol}
    loglikelihood::Float64
    aic::Float64
    n_observations::Int
    feature_transformer::FeatureTransformer  # Centralized transformer
end

struct LogNormalPrepaymentModel <: ParametricSurvivalModel
    location_coefficients::Vector{Float64}
    scale::Float64
    covariate_names::Vector{Symbol}
    loglikelihood::Float64
    aic::Float64
    n_observations::Int
    feature_transformer::FeatureTransformer  # Centralized transformer
end


struct OptimizedBernoulliBetaModel <: ParametricSurvivalModel
    # Componentes do modelo
    bernoulli_coefficients::Vector{Float64}
    beta_alpha_coefficients::Vector{Float64} 
    beta_beta_coefficients::Vector{Float64}
    
    # Metadados
    covariate_names::Vector{Symbol}
    loglikelihood::Float64
    aic::Float64
    bic::Float64  # BIC para penalizar complexidade
    n_observations::Int
    n_events::Int
    
    # Parâmetros de regularização
    regularization_strength::Float64
    feature_transformer::FeatureTransformer  # Centralized transformer
end

function fit_parametric_model(data::LoanData;
                             distribution::Symbol=:weibull,
                             covariates::Vector{Symbol}=Symbol[],
                             regularization::Float64=0.01)::ParametricSurvivalModel
    
    @assert distribution in [:weibull, :lognormal, :loglogistic, :bernoulli_beta_optimized]
    
    # Initialize and fit feature transformer
    transformer = FeatureTransformer(covariates)
    fitted_transformer = fit!(transformer, data)
    
    # Transform data using the fitted transformer
    survival_df = _prepare_survival_data_parametric(data, fitted_transformer)
    
    if distribution == :weibull
        return _fit_weibull_model(survival_df, covariates, fitted_transformer, regularization)
    elseif distribution == :lognormal
        return _fit_lognormal_model(survival_df, covariates, fitted_transformer, regularization)
    elseif distribution == :bernoulli_beta_optimized
        return _fit_optimized_bernoulli_beta(data, covariates, regularization, fitted_transformer)
    else
        return _fit_loglogistic_model(survival_df, covariates, fitted_transformer)
    end
end

function _fit_weibull_model(data::DataFrame, covariates::Vector{Symbol}, 
                            fitted_transformer::FeatureTransformer, regularization::Float64=0.01)::WeibullPrepaymentModel
    # Weibull AFT model: log(T) = μ + σ * ε
    # where μ = X'β (linear predictor) and ε ~ Gumbel(0,1)
    
    n = nrow(data)
    X = _build_design_matrix(data, covariates)
    p = size(X, 2)
    
    # Inicialização simples (reverter para valores que funcionavam)
    β₀ = zeros(p)
    β₀[1] = 3.0  # Intercept inicial razoável para scale ~20 meses  
    if p > 1
        β₀[2:end] .= -0.001  # Coeficientes pequenos para covariáveis
    end
    
    σ₀ = 1.0  # Scale parameter inicial
    θ₀ = vcat(β₀, σ₀)  # [β..., σ]
    
    # Maximum Likelihood Estimation usando Optim.jl
    println("      🔧 Estimando parâmetros via MLE...")
    
    objective(θ) = begin
        try
            if θ[end] <= 0.01   # Evitar σ muito pequeno (relaxado)
                return 1e6
            end
            ll = _weibull_loglikelihood(θ, data, X)
            regularized_ll = add_l2_regularization(ll, θ[1:p], regularization)
            return -regularized_ll
        catch e
            return 1e6  # Retornar valor alto se houver erro numérico
        end
    end
    
    # Otimização com constraints mais relaxados
    lower_bounds = vcat(fill(-Inf, p), 0.01)   # σ > 0.01 (menos restritivo)
    upper_bounds = fill(Inf, p + 1)
    
    # Inicializar variáveis de resultado
    β_hat = β₀
    σ_hat = σ₀
    ll = _weibull_loglikelihood(θ₀, data, X)
    
    try
        # Múltiplas estratégias de otimização (Nível 2)
        best_ll = ll
        best_θ = θ₀
        
        # Strategy 1: Nelder-Mead (robusto)
        result1 = optimize(objective, θ₀, NelderMead(), Optim.Options(iterations=300))
        if converged(result1) && -minimum(result1) > best_ll
            best_ll = -minimum(result1)
            best_θ = minimizer(result1)
            println("      ✅ Nelder-Mead convergiu! LL=$(round(best_ll, digits=2))")
        end
        
        # Strategy 2: BFGS como backup (se Nelder-Mead falhou)
        if best_ll <= ll + 1.0  # Se não melhorou significativamente
            try
                result2 = optimize(objective, θ₀, BFGS(), Optim.Options(iterations=100))
                if converged(result2) && -minimum(result2) > best_ll
                    best_ll = -minimum(result2)
                    best_θ = minimizer(result2)
                    println("      ✅ BFGS backup convergiu! LL=$(round(best_ll, digits=2))")
                end
            catch
                # BFGS pode falhar, continuar com Nelder-Mead result
            end
        end
        
        # Strategy 3: L-BFGS-B como último recurso
        if best_ll <= ll + 1.0
            try
                result3 = optimize(objective, lower_bounds, upper_bounds, θ₀, Fminbox(LBFGS()), 
                                 Optim.Options(iterations=150))
                if converged(result3) && -minimum(result3) > best_ll
                    best_ll = -minimum(result3)
                    best_θ = minimizer(result3)
                    println("      ✅ L-BFGS-B convergiu! LL=$(round(best_ll, digits=2))")
                end
            catch
                # L-BFGS-B pode falhar também
            end
        end
        
        # Aplicar melhor resultado encontrado
        if best_ll > ll + 0.1  # Melhoria mínima requerida
            β_hat = best_θ[1:p]
            σ_hat = max(0.01, best_θ[end])
            ll = best_ll
        else
            println("      ⚠️  MLE não convergiu adequadamente, usando valores iniciais melhorados")
        end
        
    catch e
        println("      ❌ Erro na otimização: $e")
        println("      🔄 Usando valores iniciais")
    end
    
    shape = 1.0 / σ_hat
    aic = -2 * ll + 2 * (p + 1)
    
    # Store expanded covariate names instead of original ones
    expanded_covariates = PrepaymentModels.get_expanded_covariate_names(covariates)
    
    return WeibullPrepaymentModel(
        shape, β_hat, expanded_covariates, ll, aic, n, fitted_transformer
    )
end

function _fit_lognormal_model(data::DataFrame, covariates::Vector{Symbol}, 
                              fitted_transformer::FeatureTransformer, regularization::Float64=0.01)::LogNormalPrepaymentModel
    # Log-normal AFT model: log(T) = μ + σ * ε
    # where μ = X'β and ε ~ N(0,1)
    
    n = nrow(data)
    X = _build_design_matrix(data, covariates)
    p = size(X, 2)
    
    # Better initial parameter guess using method of moments
    log_times = log.(data.time)
    
    # Initial intercept based on mean log time
    β₀ = zeros(p)
    β₀[1] = mean(log_times)
    
    # Initial covariate effects using simple linear regression on log times
    if p > 1
        try
            # Simple OLS on log times as starting point
            y = log_times
            β_ols = (X' * X) \ (X' * y)
            β₀ = β_ols
        catch
            β₀[2:end] .= 0.01 * randn(p-1)
        end
    end
    
    # Initial scale parameter from residuals
    σ₀ = std(log_times - X * β₀)
    σ₀ = max(0.1, min(2.0, σ₀))  # Reasonable bounds
    
    θ₀ = vcat(β₀, σ₀)  # [β..., σ]
    
    println("      🔧 Estimando parâmetros Log-Normal via MLE...")
    println("      📊 Inicialização: β₀[1]=$(round(β₀[1], digits=3)), σ₀=$(round(σ₀, digits=3))")
    
    # Maximum Likelihood Estimation with better objective function
    objective(θ) = begin
        try
            if θ[end] <= 0.01 || θ[end] > 5.0  # More reasonable bounds for σ
                return 1e6
            end
            
            # Check for extreme coefficients
            if any(abs.(θ[1:p]) .> 10.0)
                return 1e6
            end
            
            ll = _lognormal_loglikelihood(θ, data, X)
            
            # Check for valid likelihood
            if !isfinite(ll) || isnan(ll)
                return 1e6
            end
            
            regularized_ll = add_l2_regularization(ll, θ[1:p], regularization)
            return -regularized_ll
        catch e
            return 1e6
        end
    end
    
    β_hat = β₀
    σ_hat = σ₀  
    ll = _lognormal_loglikelihood(θ₀, data, X)
    
    try
        # Use multiple optimization strategies
        best_ll = ll
        best_θ = θ₀
        
        # Strategy 1: BFGS with relaxed tolerance
        result1 = optimize(objective, θ₀, BFGS(), Optim.Options(iterations=150, g_tol=1e-4))
        if converged(result1) && -minimum(result1) > best_ll
            best_ll = -minimum(result1)
            best_θ = minimizer(result1)
        end
        
        # Strategy 2: Nelder-Mead with more iterations
        result2 = optimize(objective, θ₀, NelderMead(), Optim.Options(iterations=300))
        if converged(result2) && -minimum(result2) > best_ll
            best_ll = -minimum(result2)
            best_θ = minimizer(result2)
        end
        
        if best_ll > ll
            β_hat = best_θ[1:p]
            σ_hat = max(0.01, best_θ[end])
            ll = best_ll
            println("      ✅ MLE convergiu! LL=$(round(ll, digits=2))")
        else
            println("      ⚠️  MLE não melhorou, usando inicialização inteligente")
        end
        
    catch e
        println("      ❌ Erro na otimização: $e")
        println("      🔄 Usando inicialização baseada em momentos")
    end
    
    aic = -2 * ll + 2 * (p + 1)
    
    # Store expanded covariate names instead of original ones
    expanded_covariates = PrepaymentModels.get_expanded_covariate_names(covariates)
    
    return LogNormalPrepaymentModel(
        β_hat, σ_hat, expanded_covariates, ll, aic, n, fitted_transformer
    )
end


function _weibull_loglikelihood(θ::Vector{Float64}, data::DataFrame, X::Matrix{Float64})::Float64
    p = size(X, 2)
    β = θ[1:p]
    σ = θ[end]
    
    @assert σ > 0 "Scale parameter must be positive"
    
    shape = 1.0 / σ
    linear_pred = X * β
    
    ll = 0.0
    
    for i in 1:nrow(data)
        t = data.time[i]
        δ = data.event[i]
        μ = linear_pred[i]
        
        # Weibull survival and hazard
        scale_i = exp(μ)
        
        if δ  # Event observed
            ll += log(shape) - log(scale_i) + (shape - 1) * log(t / scale_i) - (t / scale_i)^shape
        else  # Censored
            ll += -(t / scale_i)^shape
        end
    end
    
    return ll
end

function _lognormal_loglikelihood(θ::Vector{Float64}, data::DataFrame, X::Matrix{Float64})::Float64
    p = size(X, 2)
    β = θ[1:p]
    σ = θ[end]
    
    @assert σ > 0 "Scale parameter must be positive"
    
    linear_pred = X * β
    ll = 0.0
    log_2pi = log(2π)
    
    for i in 1:nrow(data)
        t = data.time[i]
        δ = data.event[i]
        μ = linear_pred[i]
        
        # Avoid extreme values
        if t <= 0 || !isfinite(μ)
            return -Inf
        end
        
        log_t = log(t)
        z = (log_t - μ) / σ
        
        if δ  # Event observed - Log-normal PDF
            # log f(t) = -0.5*z^2 - log(t) - log(σ) - 0.5*log(2π)
            ll += -0.5 * z^2 - log_t - log(σ) - 0.5 * log_2pi
        else  # Censored - Log-normal survival function
            # S(t) = 1 - Φ(z) = 0.5 * erfc(z / sqrt(2))
            # Use log-scale for numerical stability
            if z > 5.0  # For large z, S(t) ≈ 0
                ll += -20.0  # Approximate log(very small number)
            elseif z < -5.0  # For small z, S(t) ≈ 1
                ll += -1e-10  # log(1) ≈ 0
            else
                survival_prob = 0.5 * erfc(z / sqrt(2))
                if survival_prob <= 1e-15
                    ll += -35.0  # log(very small number)
                else
                    ll += log(survival_prob)
                end
            end
        end
        
        # Check for numerical issues
        if !isfinite(ll)
            return -Inf
        end
    end
    
    return ll
end

function survival_probability(model::WeibullPrepaymentModel,
                            covariate_values::Dict{Symbol, Float64},
                            time::Float64)::Float64
    
    # Build linear predictor with intercept + covariates
    linear_pred = model.scale_coefficients[1]  # intercept
    
    for (i, var) in enumerate(model.covariate_names)
        if haskey(covariate_values, var)
            linear_pred += model.scale_coefficients[i+1] * covariate_values[var]
        else
            @warn "Missing covariate $var in prediction"
        end
    end
    
    scale = exp(linear_pred)
    shape = model.shape
    
    # Weibull survival function
    return exp(-(time / scale)^shape)
end

function survival_probability(model::LogNormalPrepaymentModel,
                            covariate_values::Dict{Symbol, Float64},
                            time::Float64)::Float64
    
    linear_pred = model.location_coefficients[1]  # Intercept
    for (var, coef) in zip(model.covariate_names, model.location_coefficients[2:end])
        linear_pred += coef * covariate_values[var]
    end
    
    # Simplified survival function without using Distributions
    z = (log(time) - linear_pred) / model.scale
    return 0.5 * (1 - erf(z / sqrt(2)))
end

function hazard_function(model::WeibullPrepaymentModel,
                        covariate_values::Dict{Symbol, Float64},
                        time::Float64)::Float64
    
    linear_pred = 0.0
    for (var, coef) in zip(model.covariate_names, model.scale_coefficients)
        linear_pred += coef * covariate_values[var]
    end
    
    scale = exp(linear_pred)
    shape = model.shape
    
    # Weibull hazard function
    return (shape / scale) * (time / scale)^(shape - 1)
end

function median_survival_time(model::WeibullPrepaymentModel,
                             covariate_values::Dict{Symbol, Float64})::Float64
    
    linear_pred = 0.0
    for (var, coef) in zip(model.covariate_names, model.scale_coefficients)
        linear_pred += coef * covariate_values[var]
    end
    
    scale = exp(linear_pred)
    shape = model.shape
    
    # Weibull median: scale * (log(2))^(1/shape)
    return scale * log(2.0)^(1.0 / shape)
end

function median_survival_time(model::LogNormalPrepaymentModel,
                             covariate_values::Dict{Symbol, Float64})::Float64
    
    linear_pred = model.location_coefficients[1]
    for (var, coef) in zip(model.covariate_names, model.location_coefficients[2:end])
        linear_pred += coef * covariate_values[var]
    end
    
    # Log-normal median: exp(μ)
    return exp(linear_pred)
end

function _build_design_matrix(data::DataFrame, covariates::Vector{Symbol})::Matrix{Float64}
    # Use unified expansion function
    expanded_covariates = PrepaymentModels.get_expanded_covariate_names(covariates)
    
    n = nrow(data)
    p = length(expanded_covariates) + 1  # +1 for intercept
    
    X = ones(n, p)  # Initialize with intercept
    
    for (i, var) in enumerate(expanded_covariates)
        var_str = string(var)
        if var_str in names(data)
            X[:, i+1] = data[!, var]
        else
            @warn "Covariate $var not found in DataFrame (available: $(names(data))), using zeros"
            X[:, i+1] .= 0.0
        end
    end
    
    return X
end

function _fit_optimized_bernoulli_beta(data::LoanData, covariates::Vector{Symbol}, regularization::Float64, fitted_transformer::FeatureTransformer)::OptimizedBernoulliBetaModel
    """
    Modelo Bernoulli-Beta com MLE real para parâmetros otimizados
    """
    
    survival_df = _prepare_survival_data_parametric(data, fitted_transformer)
    n = nrow(survival_df)
    X = _build_design_matrix(survival_df, covariates)
    p = size(X, 2)
    
    events = survival_df.event
    n_events = sum(events)
    
    println("   🔧 Ajustando Bernoulli-Beta com MLE REAL:")
    println("      📊 $(n_events) eventos de $(n) observações")
    println("      🎯 Regularização: $(regularization)")
    
    # Smart initialization using data
    event_rate = n_events / n
    
    # Initialize Bernoulli coefficients using logistic regression
    β_bernoulli = zeros(p)
    β_bernoulli[1] = log(event_rate / (1 - event_rate))
    if p > 1
        # Use simple correlation-based initialization
        for j in 2:p
            corr_with_events = cor(X[:, j], Float64.(events))
            β_bernoulli[j] = 0.5 * corr_with_events  # Scale correlation
        end
    end
    
    # Initialize Beta parameters from timing data
    γ_α = zeros(p)
    γ_β = zeros(p)
    
    # Get timing data for events only
    event_times = survival_df.time[events]
    event_contracts = Float64[]
    
    for i in 1:n
        if events[i]
            loan_idx = findfirst(data.loan_id .== survival_df.loan_id[i])
            contract_length = !isnothing(loan_idx) ? Float64(data.loan_term[loan_idx]) : 36.0
            push!(event_contracts, contract_length)
        end
    end
    
    if length(event_times) > 0
        relative_times = event_times ./ event_contracts
        # Method of moments for Beta distribution
        u_mean = mean(relative_times)
        u_var = var(relative_times)
        
        if u_var > 0 && u_mean > 0 && u_mean < 1
            # Method of moments: α = μ²(1-μ)/σ² - μ, β = α(1-μ)/μ
            common_term = u_mean * (1 - u_mean) / u_var - 1
            α_init = u_mean * common_term
            β_init = (1 - u_mean) * common_term
            
            γ_α[1] = log(max(0.5, α_init))
            γ_β[1] = log(max(0.5, β_init))
        else
            γ_α[1] = log(2.0)
            γ_β[1] = log(1.5)
        end
    else
        γ_α[1] = log(2.0)  
        γ_β[1] = log(1.5)
    end
    
    # Small covariate effects for Beta parameters
    if p > 1
        γ_α[2:end] .= 0.01 * randn(p-1)
        γ_β[2:end] .= 0.01 * randn(p-1)
    end
    
    # Combine all parameters
    θ₀ = vcat(β_bernoulli, γ_α, γ_β)  # [β..., γ_α..., γ_β...]
    
    println("      📊 Inicialização inteligente:")
    println("         Bernoulli intercept: $(round(β_bernoulli[1], digits=3))")
    println("         Beta α intercept: $(round(exp(γ_α[1]), digits=3))")
    println("         Beta β intercept: $(round(exp(γ_β[1]), digits=3))")
    
    # MLE Objective function
    function bb_objective(θ)
        try
            n_params_each = p
            β = θ[1:n_params_each]
            γ_α_curr = θ[(n_params_each+1):(2*n_params_each)]
            γ_β_curr = θ[(2*n_params_each+1):(3*n_params_each)]
            
            # Check parameter bounds
            if any(abs.(β) .> 10.0) || any(abs.(γ_α_curr) .> 5.0) || any(abs.(γ_β_curr) .> 5.0)
                return 1e6
            end
            
            ll = _bernoulli_beta_loglikelihood(β, γ_α_curr, γ_β_curr, data, survival_df, X)
            
            if !isfinite(ll)
                return 1e6
            end
            
            # Add regularization
            reg_penalty = regularization * (sum(β[2:end].^2) + sum(γ_α_curr[2:end].^2) + sum(γ_β_curr[2:end].^2))
            
            return -(ll - reg_penalty)  # Negative for minimization
            
        catch e
            return 1e6
        end
    end
    
    # Initial likelihood
    ll_init = _bernoulli_beta_loglikelihood(β_bernoulli, γ_α, γ_β, data, survival_df, X)
    
    # MLE Optimization
    β_hat = β_bernoulli
    γ_α_hat = γ_α  
    γ_β_hat = γ_β
    ll_final = ll_init
    
    try
        println("      🔧 Executando MLE...")
        
        # Strategy 1: BFGS
        result1 = optimize(bb_objective, θ₀, BFGS(), Optim.Options(iterations=100, g_tol=1e-6))
        
        if converged(result1) && -minimum(result1) > ll_final
            θ_opt = minimizer(result1)
            β_hat = θ_opt[1:p]
            γ_α_hat = θ_opt[(p+1):(2*p)]
            γ_β_hat = θ_opt[(2*p+1):(3*p)]
            ll_final = -minimum(result1)
            println("      ✅ BFGS MLE convergiu! LL=$(round(ll_final, digits=2))")
        else
            # Strategy 2: Nelder-Mead backup
            result2 = optimize(bb_objective, θ₀, NelderMead(), Optim.Options(iterations=200))
            if converged(result2) && -minimum(result2) > ll_final
                θ_opt = minimizer(result2)
                β_hat = θ_opt[1:p]
                γ_α_hat = θ_opt[(p+1):(2*p)]
                γ_β_hat = θ_opt[(2*p+1):(3*p)]
                ll_final = -minimum(result2)
                println("      ✅ Nelder-Mead MLE convergiu! LL=$(round(ll_final, digits=2))")
            else
                println("      ⚠️  MLE não convergiu, usando inicialização inteligente")
            end
        end
        
    catch e
        println("      ❌ Erro na otimização MLE: $e")
        println("      🔄 Usando inicialização baseada em dados")
    end
    
    # Add regularization to final likelihood
    reg_penalty = regularization * (sum(β_hat[2:end].^2) + sum(γ_α_hat[2:end].^2) + sum(γ_β_hat[2:end].^2))
    ll_regularized = ll_final - reg_penalty
    
    # Information criteria
    n_params = 3 * p
    aic = -2 * ll_regularized + 2 * n_params
    bic = -2 * ll_regularized + log(n) * n_params
    
    println("      📊 Parâmetros finais estimados via MLE:")
    println("         Bernoulli intercept: $(round(β_hat[1], digits=3))")
    println("         Beta α intercept: $(round(exp(γ_α_hat[1]), digits=3))")
    println("         Beta β intercept: $(round(exp(γ_β_hat[1]), digits=3))")
    
    # Store expanded covariate names instead of original ones
    expanded_covariates = PrepaymentModels.get_expanded_covariate_names(covariates)
    
    return OptimizedBernoulliBetaModel(
        β_hat, γ_α_hat, γ_β_hat, 
        expanded_covariates, ll_regularized, aic, bic, n, n_events,
        regularization, fitted_transformer
    )
end

function _bernoulli_beta_loglikelihood(β::Vector{Float64}, γ_α::Vector{Float64}, γ_β::Vector{Float64}, 
                                      data::LoanData, survival_df::DataFrame, X::Matrix{Float64})::Float64
    """
    Log-likelihood para modelo Bernoulli-Beta
    """
    n = nrow(survival_df)
    ll = 0.0
    
    for i in 1:n
        x_i = X[i, :]
        t_i = survival_df.time[i]
        event_i = survival_df.event[i]
        
        # Bernoulli probability (logistic regression)
        logit_p = dot(x_i, β)
        if logit_p > 50  # Prevent overflow
            p_prepay = 1.0 - 1e-15
        elseif logit_p < -50
            p_prepay = 1e-15
        else
            p_prepay = 1.0 / (1.0 + exp(-logit_p))
        end
        
        if event_i
            # Find contract length for this loan
            loan_idx = findfirst(data.loan_id .== survival_df.loan_id[i])
            contract_length = !isnothing(loan_idx) ? Float64(data.loan_term[loan_idx]) : 36.0
            
            if t_i <= contract_length && t_i > 0
                # Event within contract - use both Bernoulli and Beta components
                
                # Beta parameters (ensure positive)
                log_α = dot(x_i, γ_α)
                log_β_param = dot(x_i, γ_β)
                
                α_i = exp(min(5.0, max(-5.0, log_α)))  # Bounded exp
                β_i = exp(min(5.0, max(-5.0, log_β_param)))
                
                # Relative time within contract
                u = t_i / contract_length
                u = max(0.001, min(0.999, u))  # Keep in (0,1)
                
                # Beta log-density
                if α_i > 0.1 && β_i > 0.1 && α_i < 100 && β_i < 100
                    log_beta_density = (α_i - 1) * log(u) + (β_i - 1) * log(1 - u) - 
                                      (loggamma(α_i) + loggamma(β_i) - loggamma(α_i + β_i))
                    
                    # Combined likelihood: P(prepay) * Beta_density(timing)
                    ll += log(max(1e-15, p_prepay)) + log_beta_density
                else
                    # Fallback if Beta parameters are extreme
                    ll += log(max(1e-15, p_prepay)) - 5.0  # Penalty
                end
            else
                # Event outside contract - shouldn't happen in prepayment context
                ll += log(max(1e-15, 1 - p_prepay))
            end
        else
            # --- Início da Lógica Corrigida para Censura ---

            # 1. Obter a duração do contrato para a observação `i`.
            loan_idx = findfirst(data.loan_id .== survival_df.loan_id[i])
            contract_length = !isnothing(loan_idx) ? Float64(data.loan_term[loan_idx]) : 36.0

            # 2. Verificar se a censura ocorre após o fim do contrato.
            #    Nesse caso, a probabilidade de pré-pagar depois é zero, e a lógica antiga se aplica.
            if survival_df.time[i] >= contract_length
                ll += log(max(1e-15, 1 - p_prepay))
            else
                # 3. Se a censura ocorre DURANTE o contrato, calcular a contribuição completa.

                # 3a. Calcular os parâmetros da distribuição Beta (α_i, β_i) para a observação `i`.
                log_α = dot(x_i, γ_α)
                log_β_param = dot(x_i, γ_β)
                α_i = exp(min(5.0, max(-5.0, log_α)))
                β_i = exp(min(5.0, max(-5.0, log_β_param)))
                
                # 3b. Calcular o tempo relativo de censura `u_i`.
                u_i = survival_df.time[i] / contract_length
                u_i = max(0.001, min(0.999, u_i)) # Manter no intervalo (0, 1)

                # 3c. Calcular a função de sobrevivência da Beta no ponto u_i.
                #     P(T > u_i) = 1 - CDF_Beta(u_i; α_i, β_i)
                survival_beta = 1.0 - _beta_cdf(u_i, α_i, β_i)

                # 3d. Calcular a verossimilhança combinada para a observação censurada.
                likelihood_censored = (1 - p_prepay) + p_prepay * survival_beta

                # 3e. Adicionar a contribuição à log-verossimilhança total.
                ll += log(max(1e-15, likelihood_censored))
            end
            # --- Fim da Lógica Corrigida para Censura ---
        end
        
        # Check for numerical issues
        if !isfinite(ll)
            return -Inf
        end
    end
    
    return ll
end


# === FUNÇÕES DE PREDIÇÃO ===

function predict_prepayment(model::WeibullPrepaymentModel,
                           data::LoanData,
                           prediction_horizon::Int=36)::Vector{Float64}
    
    n_loans = length(data.loan_id)
    predictions = Vector{Float64}(undef, n_loans)
    
    for i in 1:n_loans
        covariate_dict = transform_single(model.feature_transformer, data, i)
        # Filter to only model covariates
        model_covariates = Dict{Symbol, Float64}()
        for covar_name in model.covariate_names
            if haskey(covariate_dict, covar_name)
                model_covariates[covar_name] = covariate_dict[covar_name]
            else
                model_covariates[covar_name] = 0.0
            end
        end
        covariate_dict = model_covariates
        
        # Calculate survival probability at horizon
        survival_prob = survival_probability(model, covariate_dict, Float64(prediction_horizon))
        
        # Probability of prepayment within horizon
        predictions[i] = 1.0 - survival_prob
    end
    
    return predictions
end

function predict_prepayment(model::LogNormalPrepaymentModel,
                           data::LoanData,
                           prediction_horizon::Int=36)::Vector{Float64}
    
    n_loans = length(data.loan_id)
    predictions = Vector{Float64}(undef, n_loans)
    
    for i in 1:n_loans
        covariate_dict = transform_single(model.feature_transformer, data, i)
        # Filter to only model covariates
        model_covariates = Dict{Symbol, Float64}()
        for covar_name in model.covariate_names
            if haskey(covariate_dict, covar_name)
                model_covariates[covar_name] = covariate_dict[covar_name]
            else
                model_covariates[covar_name] = 0.0
            end
        end
        covariate_dict = model_covariates
        
        # Calculate survival probability at horizon
        survival_prob = survival_probability(model, covariate_dict, Float64(prediction_horizon))
        
        # Probability of prepayment within horizon
        predictions[i] = 1.0 - survival_prob
    end
    
    return predictions
end


function predict_prepayment(model::OptimizedBernoulliBetaModel,
                           data::LoanData,
                           prediction_horizon::Int=36)::Vector{Float64}
    
    n_loans = length(data.loan_id)
    predictions = Vector{Float64}(undef, n_loans)
    
    for i in 1:n_loans
        covariate_dict = transform_single(model.feature_transformer, data, i)
        # Filter to only model covariates
        model_covariates = Dict{Symbol, Float64}()
        for covar_name in model.covariate_names
            if haskey(covariate_dict, covar_name)
                model_covariates[covar_name] = covariate_dict[covar_name]
            else
                model_covariates[covar_name] = 0.0
            end
        end
        covariate_dict = model_covariates
        
        # Build design vector
        x = [1.0]  # Intercept
        for var in model.covariate_names
            push!(x, get(covariate_dict, var, 0.0))
        end
        
        # Bernoulli probability (will prepay at some point)
        logit_p = dot(x, model.bernoulli_coefficients)
        p_prepay = 1.0 / (1.0 + exp(-logit_p))
        
        # Contract duration
        contract_length = Float64(data.loan_term[i])
        
        if prediction_horizon >= contract_length
            # Horizon extends beyond contract end - use full Bernoulli probability
            predictions[i] = p_prepay
        else
            # Horizon is within contract - use Beta component for timing
            
            # Beta parameters
            log_α = dot(x, model.beta_alpha_coefficients)
            log_β_param = dot(x, model.beta_beta_coefficients)
            
            α_i = exp(min(5.0, max(-5.0, log_α)))
            β_i = exp(min(5.0, max(-5.0, log_β_param)))
            
            # Relative horizon within contract
            u_horizon = prediction_horizon / contract_length
            u_horizon = max(0.001, min(0.999, u_horizon))
            
            # P(prepayment by horizon) = P(prepay) × P(timing ≤ horizon | prepay)
            # P(timing ≤ horizon | prepay) = CDF_Beta(u_horizon; α, β)
            beta_cdf = _beta_cdf(u_horizon, α_i, β_i)
            
            predictions[i] = p_prepay * beta_cdf
        end
    end
    
    return predictions
end

function _beta_cdf(x::Float64, α::Float64, β::Float64)::Float64
    """
    Calculate CDF of Beta distribution using the regularized incomplete beta function I_x(α, β).
    Relies on a robust implementation from SpecialFunctions.jl.
    """
    if x <= 0.0
        return 0.0
    elseif x >= 1.0
        return 1.0
    elseif α <= 0.0 || β <= 0.0
        @warn "Beta CDF called with invalid parameters α=$α, β=$β. Returning fallback."
        return 0.5 # Fallback for invalid parameters
    else
        # I_x(α, β) = P(X ≤ x) where X ~ Beta(α, β)
        try
            return beta_inc(α, β, x)[1]  # beta_inc returns (I_x, 1-I_x)
        catch e
            @warn "Error calculating beta_inc: $e. Using fallback."
            # Fallback: simple approximation based on mean
            mean_beta = α / (α + β)
            return x < mean_beta ? 0.3 : 0.7
        end
    end
end


function model_comparison(models::Vector{ParametricSurvivalModel})::DataFrame
    comparison = DataFrame(
        Model = String[],
        LogLikelihood = Float64[],
        AIC = Float64[],
        Parameters = Int[],
        Distribution = String[]
    )
    
    for (i, model) in enumerate(models)
        model_name = "Model_$i"
        
        if isa(model, WeibullPrepaymentModel)
            push!(comparison, (
                model_name,
                model.loglikelihood,
                model.aic,
                length(model.scale_coefficients) + 1,
                "Weibull"
            ))
        elseif isa(model, LogNormalPrepaymentModel)
            push!(comparison, (
                model_name,
                model.loglikelihood,
                model.aic,
                length(model.location_coefficients) + 1,
                "LogNormal"
            ))
        elseif isa(model, OptimizedBernoulliBetaModel)
            push!(comparison, (
                model_name,
                model.loglikelihood,
                model.aic,
                3 * length(model.covariate_names) + 3,  # 3 sets of coefficients
                "BB Otimizado"
            ))
        end
    end
    
    # Sort by AIC (lower is better)
    sort!(comparison, :AIC)
    
    return comparison
end

function _prepare_survival_data_parametric(data::LoanData, fitted_transformer::FeatureTransformer)::DataFrame
    n = length(data.loan_id)
    
    # Calculate time to event (months from origination)
    times = Vector{Float64}(undef, n)
    events = Vector{Bool}(undef, n)
    
    for i in 1:n
        if !ismissing(data.prepayment_date[i])
            # Prepayment occurred
            times[i] = _calculate_time_difference_parametric(data.origination_date[i], data.prepayment_date[i])
            events[i] = true
        elseif !ismissing(data.default_date[i])
            # Default occurred (competing risk - censored for prepayment)
            times[i] = _calculate_time_difference_parametric(data.origination_date[i], data.default_date[i])
            events[i] = false
        else
            # Right censored (loan still active or data cutoff)
            times[i] = _calculate_time_difference_parametric(data.origination_date[i], Date(2024, 12, 31))
            events[i] = false
        end
    end
    
    # Use centralized transformer for feature engineering
    df = PrepaymentModels.transform(fitted_transformer, data)
    
    # Add survival data
    df[!, :loan_id] = data.loan_id
    df[!, :time] = times
    df[!, :event] = events
    
    return df
end

function _calculate_time_difference_parametric(start_date::Date, end_date::Date)::Float64
    """
    Calcula diferença entre datas em unidades de tempo simplificadas.
    Retorna dias divididos por 30.44 (média de dias por mês) para compatibilidade.
    """
    if end_date <= start_date
        return 0.0
    end
    
    # Diferença em dias, convertida para "meses" aproximados 
    days_diff = (end_date - start_date).value
    return Float64(days_diff) / 30.44  # Média de dias por mês (365.25/12)
end

