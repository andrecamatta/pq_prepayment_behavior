"""
Implementação de modelos Cox para análise de pré-pagamento
"""

using Survival  # Biblioteca oficial de análise de sobrevivência
using Statistics  # Para mean()
using LinearAlgebra  # Para rank()

struct CoxPrepaymentModel
    formula::String  # Simplified as string instead of FormulaTerm
    covariate_names::Vector{Symbol}  # Store covariate names used in fitting
    coefficients::Vector{Float64}
    baseline_hazard::Vector{Float64}
    times::Vector{Float64}
    loglikelihood::Float64
    n_events::Int
    n_observations::Int
    # Centralized feature transformer
    feature_transformer::FeatureTransformer
end

function fit_cox_model(data::LoanData; 
                      covariates::Vector{Symbol}=Symbol[],
                      stratify_by::Union{Symbol, Nothing}=nothing)::CoxPrepaymentModel
    
    if isempty(covariates)
        # Modelo nulo - apenas intercepto
        @warn "Nenhuma covariável especificada - ajustando modelo nulo"
        return CoxPrepaymentModel(
            "Surv(time, event) ~ 1",
            Symbol[],  # No covariates
            Float64[],
            Float64[],
            Float64[],
            -Inf,
            sum(survival_df.event),
            nrow(survival_df)
        )
    end
    
    # Construir fórmula usando Survival.jl
    covariate_str = join(string.(covariates), " + ")
    formula_str = "Surv(time, event) ~ $covariate_str"
    
    if !isnothing(stratify_by)
        formula_str *= " + strata($stratify_by)"
    end
    
    try
        # Initialize and fit feature transformer
        transformer = FeatureTransformer(covariates)
        fitted_transformer = fit!(transformer, data)
        
        # Transform data using the fitted transformer
        survival_df = _prepare_survival_data(data, fitted_transformer)
        
        # Verificar dados antes de ajustar
        if any(survival_df.time .<= 0)
            error("Tempos de sobrevivência inválidos encontrados")
        end
        
        # Usar biblioteca oficial Survival.jl
        # Criar EventTime objects
        event_times = [Survival.EventTime(survival_df.time[i], survival_df.event[i]) 
                      for i in 1:nrow(survival_df)]
        
        # Criar matriz de covariáveis
        X = _build_design_matrix_cox(survival_df, covariates)
        
        # Verificar matriz de design
        if rank(X) < size(X, 2)
            error("Matriz de covariáveis não tem rank completo")
        end
        
        # Ajustar modelo Cox usando Survival.jl com opções mais robustas
        model = Survival.coxph(X, event_times)
        
        # Verificar convergência dos coeficientes
        if any(abs.(Survival.coef(model)) .> 10.0)
            @warn "Coeficientes muito altos detectados - possível problema de convergência"
            println("Coeficientes: ", Survival.coef(model))
        end
        
        # Extrair baseline hazard usando Breslow
        baseline_hazard, times = _extract_baseline_hazard_survival(model, survival_df, X)
        
        # Store expanded covariate names instead of original ones
        expanded_covariates = PrepaymentModels.get_expanded_covariate_names(covariates)
        
        return CoxPrepaymentModel(
            formula_str,
            expanded_covariates,  # Store the expanded covariate names
            Survival.coef(model),
            baseline_hazard,
            times,
            Survival.loglikelihood(model),
            sum(survival_df.event),
            nrow(survival_df),
            fitted_transformer # Store the fitted transformer
        )
        
    catch e
        error("Falha ao ajustar modelo Cox: $e")
    end
end

function hazard_ratio(model::CoxPrepaymentModel, 
                     covariate_values::Dict{Symbol, Float64})::Float64
    
    linear_predictor = 0.0
    
    # Match covariates to model coefficients using stored covariate names
    for (i, var) in enumerate(model.covariate_names)
        if haskey(covariate_values, var) && i <= length(model.coefficients)
            linear_predictor += model.coefficients[i] * covariate_values[var]
        end
    end
    
    return exp(linear_predictor)
end

function survival_curve(model::CoxPrepaymentModel,
                       covariate_values::Dict{Symbol, Float64};
                       times::Union{Vector{Float64}, Nothing}=nothing)::Vector{Float64}
    
    hr = hazard_ratio(model, covariate_values)
    
    if isnothing(times)
        times = model.times
    end
    
    # Interpolate baseline cumulative hazard
    baseline_cumhaz = cumsum(model.baseline_hazard)
    cumhaz_interp = _interpolate_hazard(model.times, baseline_cumhaz, times)
    
    # Apply hazard ratio
    individual_cumhaz = hr .* cumhaz_interp
    
    # Convert to survival probability
    return exp.(-individual_cumhaz)
end

function cumulative_hazard(model::CoxPrepaymentModel,
                          covariate_values::Dict{Symbol, Float64};
                          times::Union{Vector{Float64}, Nothing}=nothing)::Vector{Float64}
    
    hr = hazard_ratio(model, covariate_values)
    
    if isnothing(times)
        times = model.times
    end
    
    baseline_cumhaz = cumsum(model.baseline_hazard)
    cumhaz_interp = _interpolate_hazard(model.times, baseline_cumhaz, times)
    
    return hr .* cumhaz_interp
end

# Function moved to FeatureTransformer.jl

function predict_prepayment(model::CoxPrepaymentModel,
                           data::LoanData,
                           prediction_horizon::Int=36)::Vector{Float64}
    
    n_loans = length(data.loan_id)
    predictions = Vector{Float64}(undef, n_loans)
    
    # Extract covariates for each loan using centralized transformer
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
        
        # Get survival curve for this loan
        survival_probs = survival_curve(model, model_covariates, 
                                      times=collect(1.0:prediction_horizon))
        
        # Probability of prepayment within horizon
        predictions[i] = 1.0 - survival_probs[end]
    end
    
    return predictions
end

function _prepare_survival_data(data::LoanData, transformer::FeatureTransformer)::DataFrame
    n = length(data.loan_id)
    
    # Calculate time to event (months from origination)
    times = Vector{Float64}(undef, n)
    events = Vector{Bool}(undef, n)
    
    for i in 1:n
        if !ismissing(data.prepayment_date[i])
            # Prepayment occurred
            times[i] = _calculate_time_difference(data.origination_date[i], data.prepayment_date[i])
            events[i] = true
        elseif !ismissing(data.default_date[i])
            # Default occurred (competing risk - censored for prepayment)
            times[i] = _calculate_time_difference(data.origination_date[i], data.default_date[i])
            events[i] = false
        else
            # Right censored (loan still active or data cutoff)
            times[i] = _calculate_time_difference(data.origination_date[i], Date(2024, 12, 31))
            events[i] = false
        end
    end
    
    # Use centralized transformer for feature engineering
    df = transform(transformer, data)
    
    # Add survival data
    df[!, :loan_id] = data.loan_id
    df[!, :time] = times
    df[!, :event] = events
    
    return df
end


# Função removida - usar apenas Survival.jl

function _extract_baseline_hazard_survival(model, data::DataFrame, X::Matrix{Float64})::Tuple{Vector{Float64}, Vector{Float64}}
    # Extrair baseline hazard de modelo Survival.jl usando Breslow estimator
    
    # Ordenar dados por tempo
    perm = sortperm(data.time)
    times_sorted = data.time[perm]
    events_sorted = data.event[perm]
    X_sorted = X[perm, :]
    
    # Calcular linear predictors usando coeficientes do modelo
    linear_pred = X_sorted * Survival.coef(model)
    exp_linear_pred = exp.(linear_pred)
    
    unique_times = sort(unique(times_sorted[events_sorted]))
    baseline_hazard = Vector{Float64}(undef, length(unique_times))
    
    for (i, t) in enumerate(unique_times)
        # Número de eventos no tempo t
        events_at_t = sum((times_sorted .== t) .& events_sorted)
        
        # Risk set R(t) = {j: T_j >= t}
        risk_set_mask = times_sorted .>= t
        
        # Breslow estimator: λ₀(t) = d_i / Σ_{j∈R(t)} exp(β'X_j)
        denominator = sum(exp_linear_pred[risk_set_mask])
        
        if denominator > 0
            baseline_hazard[i] = events_at_t / denominator
        else
            baseline_hazard[i] = 0.0
        end
    end
    
    return baseline_hazard, unique_times
end

function _interpolate_hazard(time_points::Vector{Float64}, 
                           hazard_values::Vector{Float64},
                           query_times::Vector{Float64})::Vector{Float64}
    
    interpolated = Vector{Float64}(undef, length(query_times))
    
    for (i, t) in enumerate(query_times)
        if t <= time_points[1]
            interpolated[i] = hazard_values[1]
        elseif t >= time_points[end]
            interpolated[i] = hazard_values[end]
        else
            # Linear interpolation
            idx = searchsortedfirst(time_points, t) - 1
            t1, t2 = time_points[idx], time_points[idx+1]
            h1, h2 = hazard_values[idx], hazard_values[idx+1]
            
            interpolated[i] = h1 + (h2 - h1) * (t - t1) / (t2 - t1)
        end
    end
    
    return interpolated
end

function _build_design_matrix_cox(df::DataFrame, covariate_names::Vector{Symbol})::Matrix{Float64}
    # Use unified expansion function
    expanded_covariates = PrepaymentModels.get_expanded_covariate_names(covariate_names)
    
    n = nrow(df)
    p = length(expanded_covariates)
    X = Matrix{Float64}(undef, n, p)
    
    for (j, var) in enumerate(expanded_covariates)
        if string(var) in names(df)
            X[:, j] = df[!, var]
        else
            @warn "Covariate $var not found in DataFrame (available: $(names(df)[1:min(10, end)]))"
            X[:, j] .= 0.0
        end
    end
    
    return X
end

function _calculate_time_difference(start_date::Date, end_date::Date)::Float64
    """
    Calcula diferença entre datas em unidades de tempo simplificadas.
    Retorna dias divididos por 30.44 (média de dias por mês) para compatibilidade com código existente.
    """
    if end_date <= start_date
        return 0.0
    end
    
    # Diferença em dias, convertida para "meses" aproximados 
    days_diff = (end_date - start_date).value
    return Float64(days_diff) / 30.44  # Média de dias por mês (365.25/12)
end

# Function removed - now using centralized FeatureTransformer
