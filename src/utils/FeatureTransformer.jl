"""
Centralized feature transformation for prepayment models
"""

using Statistics
using DataFrames

struct FeatureTransformer
    # Normalization parameters
    normalization_params::Dict{Symbol, Tuple{Float64, Float64}}  # (mean, std)
    
    # Credit score parameters
    credit_score_linear_params::Union{Tuple{Float64, Float64}, Nothing}  # (mean, std)
    credit_score_quantiles::Union{Vector{Float64}, Nothing}
    
    # Categorical parameters
    loan_type_categories::Vector{String}
    collateral_categories::Vector{String}
    
    # Covariates to transform
    covariates::Vector{Symbol}
end

function FeatureTransformer(covariates::Vector{Symbol})
    return FeatureTransformer(
        Dict{Symbol, Tuple{Float64, Float64}}(),
        nothing,
        nothing,
        String[],
        String[],
        covariates
    )
end

function fit!(transformer::FeatureTransformer, data::LoanData)::FeatureTransformer
    normalization_params = Dict{Symbol, Tuple{Float64, Float64}}()
    
    # Learn normalization parameters for continuous variables
    if :interest_rate in transformer.covariates
        normalization_params[:interest_rate] = (mean(data.interest_rate), std(data.interest_rate))
    end
    
    # Learn credit score parameters
    credit_score_linear_params = nothing
    credit_score_quantiles = nothing
    if :credit_score in transformer.covariates
        credit_score_linear_params = (mean(data.credit_score), std(data.credit_score))
        n_bins = 5
        credit_score_quantiles = [quantile(data.credit_score, q) for q in range(0, 1, length=n_bins+1)]
    end
    
    # Learn categorical parameters
    loan_type_categories = unique(data.loan_type)
    collateral_categories = unique(data.collateral_type)
    
    return FeatureTransformer(
        normalization_params,
        credit_score_linear_params,
        credit_score_quantiles,
        loan_type_categories,
        collateral_categories,
        transformer.covariates
    )
end

function transform(transformer::FeatureTransformer, data::LoanData)::DataFrame
    n = length(data.loan_id)
    
    # Base features
    df = DataFrame(
        loan_amount_log = log.(data.loan_amount),
        loan_term = Float64.(data.loan_term),
        borrower_income_log = log.(data.borrower_income),
        dti_ratio = [_calculate_dti_ratio(data.loan_amount[i], data.interest_rate[i], 
                                         data.loan_term[i], data.borrower_income[i]) for i in 1:n]
    )
    
    # Apply normalization
    if :interest_rate in transformer.covariates && haskey(transformer.normalization_params, :interest_rate)
        mean_val, std_val = transformer.normalization_params[:interest_rate]
        df.interest_rate = (data.interest_rate .- mean_val) ./ std_val
    end
    
    # Apply credit score expansion
    if :credit_score in transformer.covariates && !isnothing(transformer.credit_score_quantiles)
        credit_score_features = PrepaymentModels.expand_credit_score_features(
            data.credit_score,
            transformer.credit_score_linear_params,
            transformer.credit_score_quantiles
        )
        
        for (feature_name, values) in credit_score_features
            df[!, feature_name] = values
        end
    end
    
    # Apply categorical transformations using learned categories
    for loan_type in transformer.loan_type_categories
        safe_name = Symbol(replace(loan_type, " " => "_"))
        df[!, safe_name] = Float64.(data.loan_type .== loan_type)
    end
    
    # Apply collateral transformations using learned categories
    for collateral_type in transformer.collateral_categories
        safe_name = Symbol(replace(collateral_type, " " => "_"))
        df[!, safe_name] = Float64.(data.collateral_type .== collateral_type)
    end
    
    # Maintain backward compatibility: create has_collateral based on "Com Garantia"
    if "Com Garantia" in transformer.collateral_categories
        df[!, :has_collateral] = Float64.(data.collateral_type .== "Com Garantia")
    else
        # Fallback if categories changed
        df[!, :has_collateral] = zeros(Float64, length(data.loan_id))
    end
    
    return df
end

function transform_single(transformer::FeatureTransformer, data::LoanData, loan_idx::Int)::Dict{Symbol, Float64}
    """Transform features for a single loan observation"""
    features = Dict{Symbol, Float64}()
    
    # Base features
    features[:loan_amount_log] = log(data.loan_amount[loan_idx])
    features[:loan_term] = Float64(data.loan_term[loan_idx])
    features[:borrower_income_log] = log(data.borrower_income[loan_idx])
    features[:dti_ratio] = _calculate_dti_ratio(
        data.loan_amount[loan_idx], 
        data.interest_rate[loan_idx], 
        data.loan_term[loan_idx], 
        data.borrower_income[loan_idx]
    )
    
    # Apply normalization
    if :interest_rate in transformer.covariates && haskey(transformer.normalization_params, :interest_rate)
        mean_val, std_val = transformer.normalization_params[:interest_rate]
        features[:interest_rate] = (data.interest_rate[loan_idx] - mean_val) / std_val
    end
    
    # Apply credit score expansion
    if :credit_score in transformer.covariates && !isnothing(transformer.credit_score_quantiles)
        score = data.credit_score[loan_idx]
        transformation_params = Dict{Symbol, Any}(
            :credit_score_linear => transformer.credit_score_linear_params,
            :credit_score_quantiles => transformer.credit_score_quantiles
        )
        credit_score_features = PrepaymentModels.extract_credit_score_features_for_prediction(
            score, 
            transformation_params
        )
        merge!(features, credit_score_features)
    end
    
    # Apply categorical transformations using learned categories
    for loan_type in transformer.loan_type_categories
        safe_name = Symbol(replace(loan_type, " " => "_"))
        features[safe_name] = Float64(data.loan_type[loan_idx] == loan_type)
    end
    
    # Apply collateral transformations using learned categories  
    for collateral_type in transformer.collateral_categories
        safe_name = Symbol(replace(collateral_type, " " => "_"))
        features[safe_name] = Float64(data.collateral_type[loan_idx] == collateral_type)
    end
    
    # Maintain backward compatibility: create has_collateral based on "Com Garantia"
    if "Com Garantia" in transformer.collateral_categories
        features[:has_collateral] = Float64(data.collateral_type[loan_idx] == "Com Garantia")
    else
        # Fallback if categories changed
        features[:has_collateral] = 0.0
    end
    
    return features
end

function _calculate_dti_ratio(loan_amount::Real, interest_rate::Real, loan_term::Real, borrower_income::Real)::Float64
    if interest_rate == 0 || loan_term == 0
        return loan_amount / borrower_income
    end
    monthly_rate = interest_rate / 100 / 12
    monthly_payment = (loan_amount * monthly_rate) / (1 - (1 + monthly_rate)^(-loan_term))
    return (monthly_payment * 12) / borrower_income
end