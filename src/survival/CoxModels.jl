"""
Implementação de modelos Cox para análise de pré-pagamento
"""

using Survival  # Biblioteca oficial de análise de sobrevivência
using Statistics  # Para mean()

struct CoxPrepaymentModel
    formula::String  # Simplified as string instead of FormulaTerm
    covariate_names::Vector{Symbol}  # Store covariate names used in fitting
    coefficients::Vector{Float64}
    baseline_hazard::Vector{Float64}
    times::Vector{Float64}
    loglikelihood::Float64
    n_events::Int
    n_observations::Int
end

function fit_cox_model(data::LoanData; 
                      covariates::Vector{Symbol}=Symbol[],
                      stratify_by::Union{Symbol, Nothing}=nothing)::CoxPrepaymentModel
    
    survival_df = _prepare_survival_data(data, covariates)
    
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
        # Usar biblioteca oficial Survival.jl
        # Criar EventTime objects
        event_times = [Survival.EventTime(survival_df.time[i], survival_df.event[i]) 
                      for i in 1:nrow(survival_df)]
        
        # Criar matriz de covariáveis
        X = _build_design_matrix_cox(survival_df, covariates)
        
        # Ajustar modelo Cox usando Survival.jl
        model = Survival.coxph(X, event_times)
        
        # Extrair baseline hazard usando Breslow
        baseline_hazard, times = _extract_baseline_hazard_survival(model, survival_df, X)
        
        return CoxPrepaymentModel(
            formula_str,
            covariates,  # Store the covariate names
            Survival.coef(model),
            baseline_hazard,
            times,
            Survival.loglikelihood(model),
            sum(survival_df.event),
            nrow(survival_df)
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

function predict_prepayment(model::CoxPrepaymentModel,
                           data::LoanData,
                           prediction_horizon::Int=36)::Vector{Float64}
    
    n_loans = length(data.loan_id)
    predictions = Vector{Float64}(undef, n_loans)
    
    # Extract covariates for each loan
    for i in 1:n_loans
        covariate_dict = _extract_loan_covariates(data, i, model.covariate_names)
        
        # Get survival curve for this loan
        survival_probs = survival_curve(model, covariate_dict, 
                                      times=collect(1.0:prediction_horizon))
        
        # Probability of prepayment within horizon
        predictions[i] = 1.0 - survival_probs[end]
    end
    
    return predictions
end

function _prepare_survival_data(data::LoanData, covariates::Vector{Symbol})::DataFrame
    n = length(data.loan_id)
    
    # Calculate time to event (months from origination)
    times = Vector{Float64}(undef, n)
    events = Vector{Bool}(undef, n)
    
    for i in 1:n
        if !ismissing(data.prepayment_date[i])
            # Prepayment occurred - calcular meses reais
            times[i] = _calculate_months_between(data.origination_date[i], data.prepayment_date[i])
            events[i] = true
        elseif !ismissing(data.default_date[i])
            # Default occurred (competing risk - censored for prepayment)
            times[i] = _calculate_months_between(data.origination_date[i], data.default_date[i])
            events[i] = false
        else
            # Right censored (loan still active or data cutoff)
            times[i] = _calculate_months_between(data.origination_date[i], Date(2024, 12, 31))
            events[i] = false
        end
    end
    
    # Calculate DTI (Debt-to-Income ratio) for Brazilian data
    monthly_payments = [(data.loan_amount[i] * (data.interest_rate[i]/100/12)) / 
                       (1 - (1 + data.interest_rate[i]/100/12)^(-data.loan_term[i]))
                       for i in 1:n]
    dti_ratios = [(monthly_payments[i] * 12) / data.borrower_income[i] for i in 1:n]
    
    # Build DataFrame with survival data (Brazilian covariates)
    df = DataFrame(
        loan_id = data.loan_id,
        time = times,
        event = events,
        interest_rate = data.interest_rate,
        loan_amount_log = log.(data.loan_amount),
        loan_term = Float64.(data.loan_term),
        credit_score = Float64.(data.credit_score),
        borrower_income_log = log.(data.borrower_income),
        dti_ratio = dti_ratios
    )
    
    # Add loan type as dummy variables
    for loan_type in unique(data.loan_type)
        safe_name = Symbol(replace(loan_type, " " => "_"))
        df[!, safe_name] = Float64.(data.loan_type .== loan_type)
    end
    
    # Add collateral type as dummy
    df[!, :has_collateral] = Float64.(data.collateral_type .== "Com Garantia")
    
    # Add requested covariates
    for var in covariates
        if var in names(data.covariates)
            df[!, var] = data.covariates[!, var]
        end
    end
    
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
    n = nrow(df)
    p = length(covariate_names)
    X = Matrix{Float64}(undef, n, p)
    
    for (j, var) in enumerate(covariate_names)
        if var == :credit_score
            # Normalizar credit score para melhor convergência
            X[:, j] = (df[!, var] .- 700.0) ./ 100.0
        elseif var == :interest_rate  
            # Centrar taxa de juros
            X[:, j] = df[!, var] .- mean(df[!, var])
        else
            X[:, j] = df[!, var]
        end
    end
    
    return X
end

function _calculate_months_between(start_date::Date, end_date::Date)::Float64
    """
    Calcula diferença em meses entre duas datas de forma precisa
    Não assume que todos os meses têm 30 dias
    """
    
    if end_date <= start_date
        return 0.0
    end
    
    # Calcular diferença em anos e meses
    years_diff = Dates.year(end_date) - Dates.year(start_date)
    months_diff = Dates.month(end_date) - Dates.month(start_date)
    
    # Total de meses inteiros
    total_months = years_diff * 12 + months_diff
    
    # Ajustar para dias dentro do mês
    start_day = Dates.day(start_date)
    end_day = Dates.day(end_date)
    
    if end_day >= start_day
        # Mesmo dia ou mais tarde no mês
        days_fraction = (end_day - start_day) / Dates.daysinmonth(end_date)
        return Float64(total_months) + days_fraction
    else
        # Dia anterior no mês - subtrair um mês e calcular fração
        total_months -= 1
        
        # Calcular dias do mês anterior
        prev_month_date = end_date - Month(1)
        days_in_prev_month = Dates.daysinmonth(prev_month_date)
        days_from_start = days_in_prev_month - start_day + end_day
        days_fraction = days_from_start / days_in_prev_month
        
        return Float64(total_months) + days_fraction
    end
end

function _extract_loan_covariates(data::LoanData, loan_idx::Int, 
                                 covariate_names::Vector{Symbol})::Dict{Symbol, Float64}
    
    # Extract covariates for a specific loan (Brazilian data)
    all_covariates = Dict{Symbol, Float64}()
    
    # Basic loan characteristics
    all_covariates[:interest_rate] = data.interest_rate[loan_idx]
    all_covariates[:loan_amount_log] = log(data.loan_amount[loan_idx])
    all_covariates[:loan_term] = Float64(data.loan_term[loan_idx])
    all_covariates[:credit_score] = Float64(data.credit_score[loan_idx])
    all_covariates[:borrower_income_log] = log(data.borrower_income[loan_idx])
    
    # Calculate DTI
    monthly_payment = (data.loan_amount[loan_idx] * (data.interest_rate[loan_idx]/100/12)) / 
                     (1 - (1 + data.interest_rate[loan_idx]/100/12)^(-data.loan_term[loan_idx]))
    all_covariates[:dti_ratio] = (monthly_payment * 12) / data.borrower_income[loan_idx]
    
    # Loan type dummies
    for loan_type in ["Crédito Pessoal", "Cartão de Crédito", "Cheque Especial", "CDC Veículo"]
        safe_name = Symbol(replace(loan_type, " " => "_"))
        all_covariates[safe_name] = Float64(data.loan_type[loan_idx] == loan_type)
    end
    
    # Collateral
    all_covariates[:has_collateral] = Float64(data.collateral_type[loan_idx] == "Com Garantia")
    
    # Return only the covariates that were used in the model
    model_covariates = Dict{Symbol, Float64}()
    for covar_name in covariate_names
        if haskey(all_covariates, covar_name)
            model_covariates[covar_name] = all_covariates[covar_name]
        else
            @warn "Covariate $covar_name not found in loan data for loan $loan_idx"
            model_covariates[covar_name] = 0.0  # Default value
        end
    end
    
    return model_covariates
end