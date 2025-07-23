module PrepaymentModels

using CSV
using DataFrames
using Dates
using Distributions
using GLM
using Random
using StatsBase
using StatsModels
# using Survival  # Não disponível no registry
using Tables

include("data/DataLoader.jl")
include("survival/CoxModels.jl")
include("survival/ParametricModels.jl")
include("analysis/PrepaymentAnalysis.jl")
include("utils/ValidationUtils.jl")

export 
    # Data loading
    load_official_bank_data,
    preprocess_loan_data,
    summarize_loan_data,
    
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