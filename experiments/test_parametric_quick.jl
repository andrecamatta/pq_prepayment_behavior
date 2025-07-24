#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics, Random
Random.seed!(42)

include("src/PrepaymentModels.jl")
using .PrepaymentModels

println("🧪 TESTE RÁPIDO DOS MODELOS PARAMÉTRICOS")
println(repeat("=", 50))

# Small dataset for testing
filepath = "data/official_based_data/brazilian_loans_2025-07-23_20-16.csv"
raw_data = CSV.read(filepath, DataFrame)
sample_df = raw_data[1:100, :]

function create_loan_data(df::DataFrame)
    return PrepaymentModels.LoanData(
        String.(df.loan_id),
        df.origination_date,
        df.maturity_date,
        df.interest_rate,
        df.loan_amount,
        df.loan_term,
        df.credit_score,
        df.borrower_income,
        df.loan_type,
        df.collateral_type,
        df.prepayment_date,
        fill(missing, nrow(df)),
        DataFrame()
    )
end

loan_data = create_loan_data(sample_df)
covariates = [:interest_rate, :loan_amount_log, :loan_term, :dti_ratio, :borrower_income_log, :has_collateral]

println("📋 Testando com $(length(covariates)) covariáveis")

# Test each model quickly
models_to_test = [
    ("Weibull MLE", :weibull),
    ("Log-Normal", :lognormal), 
    ("Bernoulli-Beta", :bernoulli_beta_optimized)
]

for (name, model_type) in models_to_test
    print("🔧 $name... ")
    
    try
        start_time = time()
        model = PrepaymentModels.fit_parametric_model(loan_data, distribution=model_type, covariates=covariates)
        train_time = time() - start_time
        
        # Quick prediction test
        predictions = PrepaymentModels.predict_prepayment(model, loan_data, 24)
        pred_range = [minimum(predictions), maximum(predictions)]
        pred_unique = length(unique(predictions))
        
        println("✅ ($(round(train_time, digits=2))s) | Predições: $(round(pred_range[1], digits=3))-$(round(pred_range[2], digits=3)) | $(pred_unique) valores únicos")
        
    catch e
        println("❌ Erro: $e")
    end
end

println("\n✅ TESTE CONCLUÍDO")