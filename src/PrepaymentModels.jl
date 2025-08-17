module PrepaymentModels

using CSV
using DataFrames
using Dates
using Distributions
using GLM
using Random
using StatsBase
using StatsModels
using Survival
using Tables

include("data/DataLoader.jl")
include("utils/FeatureTransformer.jl")
include("survival/CoxModels.jl")
include("survival/ParametricModels.jl")
include("analysis/PrepaymentAnalysis.jl")
include("utils/ValidationUtils.jl")

# === UNIFIED CREDIT SCORE EXPANSION ===

function expand_credit_score_features(
    credit_scores::Vector{Int}, 
    normalization_params::Tuple{Float64, Float64}, 
    quantiles::Vector{Float64}
)::Dict{Symbol, Vector{Float64}}
    """
    Unified credit score expansion function.
    Uses pre-calculated parameters to ensure consistency between training and prediction.
    """
    n = length(credit_scores)
    n_bins = length(quantiles) - 1
    features = Dict{Symbol, Vector{Float64}}()

    # Create dummy variables for each bin (excluding first as reference)
    for bin_idx in 2:n_bins
        feature_name = Symbol("credit_score_q$(bin_idx)")
        bin_indicator = zeros(Float64, n)
        
        lower_bound = quantiles[bin_idx]
        upper_bound = quantiles[bin_idx + 1]

        for (i, score) in enumerate(credit_scores)
            if score > lower_bound && score <= upper_bound
                bin_indicator[i] = 1.0
            end
        end
        features[feature_name] = bin_indicator
    end
    
    # Also include a normalized linear term for fine-grained control
    mean_val, std_val = normalization_params
    features[:credit_score_linear] = (Float64.(credit_scores) .- mean_val) ./ std_val
    
    return features
end

function get_expanded_covariate_names(original_covariates::Vector{Symbol})::Vector{Symbol}
    """
    Get the expanded list of covariate names after credit score expansion
    """
    expanded_covariates = Symbol[]
    
    for var in original_covariates
        if var == :credit_score
            # Add all credit score features instead of the original variable
            credit_score_features = [:credit_score_linear, :credit_score_q2, :credit_score_q3, :credit_score_q4, :credit_score_q5]
            append!(expanded_covariates, credit_score_features)
        else
            push!(expanded_covariates, var)
        end
    end
    
    return expanded_covariates
end

function extract_credit_score_features_for_prediction(
    score::Int, 
    transformation_params::Dict{Symbol, Any}
)::Dict{Symbol, Float64}
    """
    Extract credit score features for a single loan during prediction.
    Uses parameters stored in the fitted model object.
    """
    features = Dict{Symbol, Float64}()
    
    # Normalized linear term using stored parameters
    if haskey(transformation_params, :credit_score_linear)
        mean_val, std_val = transformation_params[:credit_score_linear]
        features[:credit_score_linear] = (Float64(score) - mean_val) / std_val
    end
    
    # Binned features using stored quantiles
    if haskey(transformation_params, :credit_score_quantiles)
        quantiles = transformation_params[:credit_score_quantiles]
        n_bins = length(quantiles) - 1
        
        for bin_idx in 2:n_bins
            feature_name = Symbol("credit_score_q$(bin_idx)")
            lower_bound = quantiles[bin_idx]
            upper_bound = quantiles[bin_idx + 1]
            features[feature_name] = Float64(score > lower_bound && score <= upper_bound)
        end
    end
    
    return features
end

export 
    # Data loading
    LoanData,
    load_official_bank_data,
    preprocess_loan_data,
    summarize_loan_data,
    
    # Feature transformation
    FeatureTransformer,
    fit!,
    transform,
    transform_single,
    
    # Survival models
    CoxPrepaymentModel,
    ParametricPrepaymentModel,
    WeibullPrepaymentModel,
    LogNormalPrepaymentModel,
    fit_cox_model,
    fit_parametric_model,
    
    # Analysis
    survival_curve,
    hazard_ratio,
    cumulative_hazard,
    predict_prepayment,
    survival_probability,
    analyze_prepayment_behavior,
    
    # Validation
    validate_model,
    compute_concordance,
    likelihood_ratio_test,
    model_comparison

end
