"""
Versão OTIMIZADA do modelo Bernoulli-Beta para resolver problemas de scaling
"""

using LinearAlgebra: dot
using SpecialFunctions: loggamma, logbeta
using Optim

struct OptimizedBernoulliBetaModel <: ParametricSurvivalModel
    # Componentes do modelo
    bernoulli_coefficients::Vector{Float64}
    beta_alpha_coefficients::Vector{Float64} 
    beta_beta_coefficients::Vector{Float64}
    
    # Metadados
    covariate_names::Vector{Symbol}
    loglikelihood::Float64
    aic::Float64
    bic::Float64  # Adicionar BIC para penalizar complexidade
    n_observations::Int
    n_events::Int
    
    # Parâmetros de regularização
    regularization_strength::Float64
end

function fit_optimized_bernoulli_beta(data::LoanData; 
                                    covariates::Vector{Symbol}=Symbol[],
                                    regularization::Float64=0.01)::OptimizedBernoulliBetaModel
    """
    Versão otimizada do ajuste Bernoulli-Beta com:
    1. Estabilidade numérica melhorada
    2. Regularização L2 
    3. MLE robusto
    4. Controle de overfitting
    """
    
    survival_df = PrepaymentModels._prepare_survival_data(data, covariates)
    n = nrow(survival_df)
    X = _build_design_matrix(survival_df, covariates)
    p = size(X, 2)
    
    events = survival_df.event
    n_events = sum(events)
    
    println("   🔧 Ajustando Bernoulli-Beta OTIMIZADO:")
    println("      📊 $(n_events) eventos de $(n) observações")
    println("      🎯 Regularização: $(regularization)")
    
    # === INICIALIZAÇÃO INTELIGENTE ===
    
    # 1. Inicializar Bernoulli com regressão logística simples
    event_rate = n_events / n
    β_init = zeros(p)
    β_init[1] = log(event_rate / (1 - event_rate))  # Logit da taxa observada
    
    # 2. Inicializar Beta com MOM robusto (apenas dados válidos)
    α_init, β_init_beta = _robust_beta_initialization(survival_df, data, events)
    
    γ_α_init = zeros(p)
    γ_β_init = zeros(p) 
    γ_α_init[1] = log(α_init)
    γ_β_init[1] = log(β_init_beta)
    
    # === OTIMIZAÇÃO COM REGULARIZAÇÃO ===
    
    # Parâmetros iniciais
    θ_init = vcat(β_init, γ_α_init, γ_β_init)
    
    # Função objetivo com regularização L2
    function objective(θ::Vector{Float64})
        β_bernoulli = θ[1:p]
        γ_α = θ[(p+1):(2p)]
        γ_β = θ[(2p+1):(3p)]
        
        # Log-likelihood
        ll = _compute_stable_loglikelihood(survival_df, data, X, β_bernoulli, γ_α, γ_β)
        
        # Regularização L2 (exceto intercepts)
        reg_penalty = regularization * (
            sum(β_bernoulli[2:end].^2) + 
            sum(γ_α[2:end].^2) + 
            sum(γ_β[2:end].^2)
        )
        
        return -(ll - reg_penalty)  # Negativo para minimização
    end
    
    # Otimizar com gradiente
    println("      ⚙️  Otimizando parâmetros...")
    
    # Inicializar variáveis fora do try-catch
    β_final = β_init
    γ_α_final = γ_α_init  
    γ_β_final = γ_β_init
    ll_opt = _compute_stable_loglikelihood(survival_df, data, X, β_final, γ_α_final, γ_β_final)
    
    try
        result = Optim.optimize(objective, θ_init, BFGS(), 
                               Optim.Options(iterations=1000, show_trace=false))
        
        if Optim.converged(result)
            θ_opt = Optim.minimizer(result)
            ll_opt = -Optim.minimum(result)
            
            β_final = θ_opt[1:p]
            γ_α_final = θ_opt[(p+1):(2p)]
            γ_β_final = θ_opt[(2p+1):(3p)]
            
            println("      ✅ Otimização convergiu!")
        else
            println("      ⚠️  Otimização não convergiu, usando inicialização")
            # Variáveis já inicializadas acima
        end
        
    catch e
        println("      ❌ Erro na otimização: $e")
        println("      🔄 Usando inicialização simples")
        # Variáveis já inicializadas acima
    end
    
    # === CRITÉRIOS DE INFORMAÇÃO ===
    
    n_params = 3 * p
    aic = -2 * ll_opt + 2 * n_params
    bic = -2 * ll_opt + log(n) * n_params  # BIC penaliza mais
    
    println("      📈 Log-likelihood: $(round(ll_opt, digits=2))")
    println("      📊 AIC: $(round(aic, digits=2))")
    println("      📋 BIC: $(round(bic, digits=2))")
    
    return OptimizedBernoulliBetaModel(
        β_final, γ_α_final, γ_β_final, 
        covariates, ll_opt, aic, bic, n, n_events,
        regularization
    )
end

function _robust_beta_initialization(survival_df::DataFrame, original_data::LoanData, 
                                   events::Vector{Bool})::Tuple{Float64, Float64}
    """
    Inicialização robusta dos parâmetros Beta usando apenas dados válidos
    """
    
    # Coletar tempos relativos válidos
    valid_relative_times = Float64[]
    
    for i in findall(events)
        loan_idx = findfirst(original_data.loan_id .== survival_df.loan_id[i])
        if !isnothing(loan_idx)
            contract_length = original_data.loan_term[loan_idx]
            time_to_event = survival_df.time[i]
            
            # Só usar se realmente válido
            if time_to_event > 0 && time_to_event <= contract_length
                rel_time = time_to_event / contract_length
                # Clampar para evitar extremos
                rel_time = max(0.01, min(0.99, rel_time))
                push!(valid_relative_times, rel_time)
            end
        end
    end
    
    if length(valid_relative_times) < 5
        println("      ⚠️  Poucos dados válidos para Beta, usando defaults")
        return 1.0, 1.0
    end
    
    # Method of moments robusto
    μ = mean(valid_relative_times)
    σ² = var(valid_relative_times)
    
    println("      📊 Beta init: n_valid=$(length(valid_relative_times)), μ=$(round(μ, digits=3)), σ²=$(round(σ², digits=4))")
    
    # Verificar se MOM é aplicável
    if σ² > 0 && σ² < μ * (1 - μ) * 0.95  # Margem de segurança
        # Method of moments
        common_factor = (μ * (1 - μ) / σ²) - 1
        if common_factor > 0.1  # Evitar valores muito pequenos
            α = μ * common_factor
            β = (1 - μ) * common_factor
            
            # Clampar para valores razoáveis
            α = max(0.5, min(5.0, α))
            β = max(0.5, min(5.0, β))
            
            println("      🧮 MOM: α=$(round(α, digits=3)), β=$(round(β, digits=3))")
            return α, β
        end
    end
    
    # Fallback: baseado na média observada
    α = max(0.5, μ * 2.0)
    β = max(0.5, (1 - μ) * 2.0)
    
    println("      🔄 Fallback: α=$(round(α, digits=3)), β=$(round(β, digits=3))")
    return α, β
end

function _compute_stable_loglikelihood(survival_df::DataFrame, original_data::LoanData,
                                     X::Matrix{Float64}, β_bernoulli::Vector{Float64},
                                     γ_α::Vector{Float64}, γ_β::Vector{Float64})::Float64
    """
    Cálculo numericamente estável do log-likelihood
    """
    
    n = nrow(survival_df)
    ll = 0.0
    
    # Cache para evitar recálculos
    logit_probs = X * β_bernoulli
    log_α_params = X * γ_α  
    log_β_params = X * γ_β
    
    for i in 1:n
        # Bernoulli probability (em log-space)
        logit_p = logit_probs[i]
        log_p_prepay = -log(1.0 + exp(-logit_p))  # log(sigmoid(x))
        log_p_not_prepay = -log(1.0 + exp(logit_p))  # log(1 - sigmoid(x))
        
        if survival_df.event[i]
            # Evento observado - precisa de timing Beta
            
            # Encontrar contract length
            loan_idx = findfirst(original_data.loan_id .== survival_df.loan_id[i])
            if !isnothing(loan_idx)
                contract_length = Float64(original_data.loan_term[loan_idx])
                time_months = survival_df.time[i]
                
                # Verificar validade
                if time_months > 0 && time_months <= contract_length + 0.1  # Small tolerance
                    rel_time = max(0.001, min(0.999, time_months / contract_length))
                    
                    # Beta parameters (em log-space para estabilidade)
                    log_α = log_α_params[i]
                    log_β = log_β_params[i]
                    
                    # Clampar para evitar extremos
                    log_α = max(-2.0, min(2.0, log_α))  # α ∈ [0.135, 7.39]
                    log_β = max(-2.0, min(2.0, log_β))  # β ∈ [0.135, 7.39]
                    
                    α = exp(log_α)
                    β = exp(log_β)
                    
                    # Beta log-density (numericamente estável)
                    log_beta_density = (α - 1) * log(rel_time) + (β - 1) * log(1 - rel_time) - 
                                      logbeta(α, β)
                    
                    # Total likelihood para evento
                    ll += log_p_prepay + log_beta_density
                else
                    # Evento inválido - tratar como censurado
                    ll += log_p_not_prepay
                end
            else
                # Não encontrou loan - usar prior neutro
                ll += log_p_prepay - 1.0  # Penalidade pequena
            end
        else
            # Censurado - não houve pré-pagamento
            ll += log_p_not_prepay
        end
    end
    
    return ll
end

function _build_design_matrix(data::DataFrame, covariates::Vector{Symbol})::Matrix{Float64}
    """
    Construir matriz de design com normalização para estabilidade numérica
    """
    n = nrow(data)
    p = length(covariates) + 1  # +1 para intercept
    
    X = ones(n, p)  # Inicializar com intercept
    
    for (i, var) in enumerate(covariates)
        raw_values = data[!, var]
        
        # Normalizar covariáveis para melhorar estabilidade
        if var == :interest_rate
            # Centrar em torno da média
            X[:, i+1] = (raw_values .- mean(raw_values)) ./ std(raw_values)
        elseif var == :credit_score
            # Normalizar para [0,1] aproximadamente
            X[:, i+1] = (raw_values .- 500.0) ./ 250.0
        else
            # Padronizar outras variáveis
            X[:, i+1] = (raw_values .- mean(raw_values)) ./ std(raw_values)
        end
    end
    
    return X
end