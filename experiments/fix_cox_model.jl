#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics, Random
import StatsBase
Random.seed!(42)

include("src/PrepaymentModels.jl")
using .PrepaymentModels

println("🔧 IMPLEMENTANDO CORREÇÕES DO MODELO COX")
println(repeat("=", 60))

# Load data
filepath = "data/official_based_data/brazilian_loans_2025-07-23_20-16.csv"
raw_data = CSV.read(filepath, DataFrame)

sample_size = 2000
sample_indices = StatsBase.sample(1:nrow(raw_data), sample_size, replace=false)
sample_df = raw_data[sample_indices, :]

train_data = sample_df[1:1400, :]
test_data = sample_df[1401:end, :]

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

loan_train = create_loan_data(train_data)
loan_test = create_loan_data(test_data)

# Define metric functions  
function concordance_index(predictions::Vector{Float64}, events::AbstractVector{Bool})
    n = length(predictions)
    concordant = 0
    total_pairs = 0
    
    if length(unique(predictions)) == 1
        @warn "Todas as predições são iguais - C-index indefinido, retornando 0.5"
        return 0.5
    end
    
    for i in 1:n
        for j in (i+1):n
            if events[i] != events[j]
                total_pairs += 1
                if (events[i] && predictions[i] > predictions[j]) ||
                   (events[j] && predictions[j] > predictions[i])
                    concordant += 1
                end
            end
        end
    end
    
    return total_pairs > 0 ? concordant / total_pairs : 0.5
end

println("🧪 TESTANDO DIFERENTES ESTRATÉGIAS:\n")

# Strategy 1: Credit Score Binned
println("1️⃣ CREDIT SCORE BINNED:")
try
    # Create modified data with binned credit score
    modified_train = copy(train_data)
    modified_test = copy(test_data)
    
    # Create bins based on quantiles
    all_credit = vcat(train_data.credit_score, test_data.credit_score)
    credit_quantiles = quantile(all_credit, [0.25, 0.5, 0.75])
    
    # Add binary indicators for credit score quartiles
    modified_train.credit_low = Float64.(train_data.credit_score .<= credit_quantiles[1])
    modified_train.credit_high = Float64.(train_data.credit_score .>= credit_quantiles[3])
    
    modified_test.credit_low = Float64.(test_data.credit_score .<= credit_quantiles[1])  
    modified_test.credit_high = Float64.(test_data.credit_score .>= credit_quantiles[3])
    
    # We need to modify the LoanData creation to include these new columns
    # For now, let's test with existing covariates excluding credit_score
    covariates_no_credit = [:interest_rate, :loan_amount_log, :loan_term, :dti_ratio, :borrower_income_log, :has_collateral]
    
    cox_no_credit = PrepaymentModels.fit_cox_model(loan_train, covariates=covariates_no_credit)
    pred_no_credit = PrepaymentModels.predict_prepayment(cox_no_credit, loan_test, 24)
    
    events = .!ismissing.(loan_test.prepayment_date)
    c_idx_no_credit = concordance_index(pred_no_credit, events)
    
    println("   ✅ Sem credit_score:")
    println("   📊 Coeficientes: $(round.(cox_no_credit.coefficients, digits=4))")
    println("   📊 C-Index: $(round(c_idx_no_credit, digits=4))")
    println("   📊 Predições únicas: $(length(unique(pred_no_credit)))")
    
catch e
    println("   ❌ Erro: $e")
end

# Strategy 2: Robust Normalization
println("\n2️⃣ NORMALIZACAO ROBUSTA:")
println("   (Esta requer modificação do código interno, testando conceito)")

# Strategy 3: Different Covariate Combinations  
println("\n3️⃣ COMBINAÇÕES DE COVARIÁVEIS:")

covariate_sets = [
    ([:interest_rate], "Interest only"),
    ([:interest_rate, :loan_amount_log], "Interest + Amount"),
    ([:interest_rate, :loan_amount_log, :loan_term], "Interest + Amount + Term"),
    ([:interest_rate, :loan_amount_log, :loan_term, :dti_ratio], "4 variables"),
    ([:interest_rate, :loan_amount_log, :loan_term, :dti_ratio, :borrower_income_log], "5 variables"),
]

for (covs, desc) in covariate_sets
    try
        cox_model = PrepaymentModels.fit_cox_model(loan_train, covariates=covs)
        predictions = PrepaymentModels.predict_prepayment(cox_model, loan_test, 24)
        events = .!ismissing.(loan_test.prepayment_date)
        c_idx = concordance_index(predictions, events)
        
        max_coef = maximum(abs.(cox_model.coefficients))
        pred_variety = length(unique(predictions))
        
        println("   $(desc):")
        println("     Max |coef|: $(round(max_coef, digits=4))")
        println("     C-Index: $(round(c_idx, digits=4))")  
        println("     Pred. únicas: $pred_variety")
        
    catch e
        println("   $(desc): ❌ Error: $e")
    end
end

# Strategy 4: Check the issue with full covariates but investigate further
println("\n4️⃣ INVESTIGAÇÃO DETALHADA COM TODAS COVARIÁVEIS:")
try 
    full_covariates = [:interest_rate, :credit_score, :loan_amount_log, :loan_term, :dti_ratio, :borrower_income_log, :has_collateral]
    
    cox_full = PrepaymentModels.fit_cox_model(loan_train, covariates=full_covariates)
    pred_full = PrepaymentModels.predict_prepayment(cox_full, loan_test, 24)
    
    println("   📊 Coeficientes completos:")
    for (i, var) in enumerate(full_covariates)
        coef = cox_full.coefficients[i]
        hr = exp(coef)
        println("     $var: $(round(coef, digits=4)) (HR: $(round(hr, digits=4)))")
    end
    
    events = .!ismissing.(loan_test.prepayment_date)
    c_idx_full = concordance_index(pred_full, events)
    pred_range = (minimum(pred_full), maximum(pred_full))
    
    println("   📊 C-Index: $(round(c_idx_full, digits=4))")
    println("   📊 Pred range: $(round(pred_range[1], digits=6)) - $(round(pred_range[2], digits=6))")
    println("   📊 Pred únicas: $(length(unique(pred_full)))")
    
catch e
    println("   ❌ Erro: $e")
end

println("\n" * repeat("=", 60))