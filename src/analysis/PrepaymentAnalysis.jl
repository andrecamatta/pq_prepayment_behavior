"""
Funções específicas para análise de pré-pagamento de empréstimos
Inclui análises econômicas, seasoning effects, e validação temporal
"""

struct PrepaymentAnalysisResult
    model::Union{CoxPrepaymentModel, ParametricSurvivalModel}
    seasoning_effect::DataFrame
    economic_factors::DataFrame
    validation_metrics::Dict{String, Float64}
    risk_segments::DataFrame
end

function analyze_prepayment_behavior(data::LoanData;
                                   model_type::Symbol=:cox,
                                   include_economic_vars::Bool=true,
                                   validation_split::Float64=0.8)::PrepaymentAnalysisResult
    
    # Split data for validation
    n_total = length(data.loan_id)
    n_train = Int(floor(n_total * validation_split))
    train_indices = sample(1:n_total, n_train, replace=false)
    test_indices = setdiff(1:n_total, train_indices)
    
    train_data = _subset_loan_data(data, train_indices)
    test_data = _subset_loan_data(data, test_indices)
    
    # Define covariates
    covariates = [:interest_rate, :ltv_ratio, :credit_score]
    if include_economic_vars
        push!(covariates, :unemployment_rate, :hpi_change)
    end
    
    # Fit model
    if model_type == :cox
        model = fit_cox_model(train_data, covariates=covariates)
    else
        model = fit_parametric_model(train_data, distribution=model_type, covariates=covariates)
    end
    
    # Analyze seasoning effect
    seasoning_effect = _analyze_seasoning_effect(data)
    
    # Analyze economic factors
    economic_factors = _analyze_economic_factors(data)
    
    # Validate model
    validation_metrics = _validate_model_performance(model, test_data)
    
    # Risk segmentation
    risk_segments = _create_risk_segments(model, data)
    
    return PrepaymentAnalysisResult(
        model, seasoning_effect, economic_factors, 
        validation_metrics, risk_segments
    )
end

function _analyze_seasoning_effect(data::LoanData)::DataFrame
    # Analyze how prepayment hazard changes with loan age
    
    max_seasoning = 120  # 10 years in months
    seasoning_bins = collect(0:6:max_seasoning)
    n_bins = length(seasoning_bins) - 1
    
    seasoning_df = DataFrame(
        seasoning_months = Int[],
        n_loans = Int[],
        n_prepayments = Int[],
        prepayment_rate = Float64[],
        cumulative_prepayment_rate = Float64[]
    )
    
    for i in 1:n_bins
        lower_bound = seasoning_bins[i]
        upper_bound = seasoning_bins[i+1]
        
        # Find loans in this seasoning range
        in_range = Int[]
        for j in 1:length(data.loan_id)
            if !ismissing(data.prepayment_date[j])
                seasoning = Dates.value(data.prepayment_date[j] - data.origination_date[j]) ÷ 30
                if lower_bound <= seasoning < upper_bound
                    push!(in_range, j)
                end
            end
        end
        
        n_loans_at_risk = _count_loans_at_risk(data, lower_bound)
        n_prepayments = length(in_range)
        prepayment_rate = n_loans_at_risk > 0 ? n_prepayments / n_loans_at_risk : 0.0
        
        push!(seasoning_df, (
            lower_bound,
            n_loans_at_risk,
            n_prepayments,
            prepayment_rate,
            0.0  # Will compute cumulative later
        ))
    end
    
    # Compute cumulative prepayment rates
    seasoning_df.cumulative_prepayment_rate = cumsum(seasoning_df.prepayment_rate)
    
    return seasoning_df
end

function _analyze_economic_factors(data::LoanData)::DataFrame
    # Analyze impact of economic conditions on prepayment
    # This would typically join with external economic data
    
    # Group by origination year and quarter
    economic_df = DataFrame(
        period = String[],
        avg_interest_rate = Float64[],
        prepayment_rate = Float64[],
        n_loans = Int[],
        avg_credit_score = Float64[],
        avg_ltv = Float64[]
    )
    
    # Group loans by origination period
    periods = _create_time_periods(data)
    
    for period in unique(periods)
        period_mask = periods .== period
        period_loans = findall(period_mask)
        
        if length(period_loans) > 10  # Minimum sample size
            avg_rate = mean(data.interest_rate[period_loans])
            prepay_rate = sum(!ismissing.(data.prepayment_date[period_loans])) / length(period_loans)
            avg_credit = mean(data.credit_score[period_loans])
            avg_ltv = mean(data.ltv_ratio[period_loans])
            
            push!(economic_df, (
                period,
                avg_rate,
                prepay_rate,
                length(period_loans),
                avg_credit,
                avg_ltv
            ))
        end
    end
    
    return economic_df
end

function _validate_model_performance(model::Union{CoxPrepaymentModel, ParametricSurvivalModel},
                                   test_data::LoanData)::Dict{String, Float64}
    
    predictions = predict_prepayment(model, test_data)
    
    # Actual outcomes
    actual = [!ismissing(date) for date in test_data.prepayment_date]
    
    # Concordance index (C-statistic)
    c_index = _compute_concordance_index(predictions, actual)
    
    # Brier score at different time points
    brier_12m = _compute_brier_score(predictions, actual, 12)
    brier_24m = _compute_brier_score(predictions, actual, 24)
    brier_36m = _compute_brier_score(predictions, actual, 36)
    
    # Calibration slope
    calibration_slope = _compute_calibration_slope(predictions, actual)
    
    return Dict(
        "concordance_index" => c_index,
        "brier_score_12m" => brier_12m,
        "brier_score_24m" => brier_24m,
        "brier_score_36m" => brier_36m,
        "calibration_slope" => calibration_slope
    )
end

function _create_risk_segments(model::Union{CoxPrepaymentModel, ParametricSurvivalModel},
                              data::LoanData)::DataFrame
    
    predictions = predict_prepayment(model, data)
    
    # Create quintiles based on predicted risk
    risk_quintiles = _create_quintiles(predictions)
    
    segments_df = DataFrame(
        risk_quintile = Int[],
        n_loans = Int[],
        avg_predicted_risk = Float64[],
        actual_prepayment_rate = Float64[],
        avg_credit_score = Float64[],
        avg_ltv = Float64[],
        avg_interest_rate = Float64[]
    )
    
    for q in 1:5
        quintile_mask = risk_quintiles .== q
        quintile_indices = findall(quintile_mask)
        
        if length(quintile_indices) > 0
            avg_pred_risk = mean(predictions[quintile_indices])
            actual_rate = sum(!ismissing.(data.prepayment_date[quintile_indices])) / length(quintile_indices)
            avg_credit = mean(data.credit_score[quintile_indices])
            avg_ltv = mean(data.ltv_ratio[quintile_indices])
            avg_rate = mean(data.interest_rate[quintile_indices])
            
            push!(segments_df, (
                q,
                length(quintile_indices),
                avg_pred_risk,
                actual_rate,
                avg_credit,
                avg_ltv,
                avg_rate
            ))
        end
    end
    
    return segments_df
end

function _subset_loan_data(data::LoanData, indices::Vector{Int})::LoanData
    return LoanData(
        data.loan_id[indices],
        data.origination_date[indices],
        data.maturity_date[indices],
        data.interest_rate[indices],
        data.loan_amount[indices],
        data.ltv_ratio[indices],
        data.credit_score[indices],
        data.property_type[indices],
        data.prepayment_date[indices],
        data.default_date[indices],
        data.covariates[indices, :]
    )
end

function _count_loans_at_risk(data::LoanData, seasoning_months::Int)::Int
    count = 0
    for i in 1:length(data.loan_id)
        loan_age_at_cutoff = Dates.value(Date(2024, 12, 31) - data.origination_date[i]) ÷ 30
        
        # Loan was active at the seasoning point
        if loan_age_at_cutoff >= seasoning_months
            # Check if loan terminated before this seasoning point
            if !ismissing(data.prepayment_date[i])
                prepay_seasoning = Dates.value(data.prepayment_date[i] - data.origination_date[i]) ÷ 30
                if prepay_seasoning >= seasoning_months
                    count += 1
                end
            elseif !ismissing(data.default_date[i])
                default_seasoning = Dates.value(data.default_date[i] - data.origination_date[i]) ÷ 30
                if default_seasoning >= seasoning_months
                    count += 1
                end
            else
                count += 1  # Still active
            end
        end
    end
    
    return count
end

function _create_time_periods(data::LoanData)::Vector{String}
    periods = Vector{String}(undef, length(data.loan_id))
    
    for i in 1:length(data.loan_id)
        year = Dates.year(data.origination_date[i])
        quarter = (Dates.month(data.origination_date[i]) - 1) ÷ 3 + 1
        periods[i] = "$(year)Q$(quarter)"
    end
    
    return periods
end

function _compute_concordance_index(predictions::Vector{Float64}, 
                                  actual::Vector{Bool})::Float64
    # Simplified C-index calculation
    n = length(predictions)
    concordant_pairs = 0
    total_pairs = 0
    
    for i in 1:n
        for j in (i+1):n
            if actual[i] != actual[j]  # Comparable pair
                total_pairs += 1
                if (actual[i] && predictions[i] > predictions[j]) ||
                   (actual[j] && predictions[j] > predictions[i])
                    concordant_pairs += 1
                end
            end
        end
    end
    
    return total_pairs > 0 ? concordant_pairs / total_pairs : 0.5
end

function _compute_brier_score(predictions::Vector{Float64}, 
                             actual::Vector{Bool}, 
                             time_horizon::Int)::Float64
    # Brier score for binary outcomes
    n = length(predictions)
    score = 0.0
    
    for i in 1:n
        score += (predictions[i] - (actual[i] ? 1.0 : 0.0))^2
    end
    
    return score / n
end

function _compute_calibration_slope(predictions::Vector{Float64}, 
                                   actual::Vector{Bool})::Float64
    # Simplified calibration slope calculation
    # In production would use GLM for logistic regression
    
    # Simple correlation as proxy for calibration slope
    return cor(predictions, Float64.(actual))
end

function _create_quintiles(values::Vector{Float64})::Vector{Int}
    n = length(values)
    sorted_indices = sortperm(values)
    quintiles = Vector{Int}(undef, n)
    
    quintile_size = n ÷ 5
    
    for i in 1:n
        rank = findfirst(==(i), sorted_indices)
        quintile = min(5, (rank - 1) ÷ quintile_size + 1)
        quintiles[i] = quintile
    end
    
    return quintiles
end