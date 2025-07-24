#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics, Random
import StatsBase
Random.seed!(42)

include("src/PrepaymentModels.jl")
using .PrepaymentModels

println("🎯 TESTE DO COX CORRIGIDO (SEM CREDIT_SCORE)")
println(repeat("=", 50))

# Load data
filepath = "data/official_based_data/brazilian_loans_2025-07-23_20-16.csv"
raw_data = CSV.read(filepath, DataFrame)

sample_size = 5000
sample_indices = StatsBase.sample(1:nrow(raw_data), sample_size, replace=false)
sample_df = raw_data[sample_indices, :]

n_train = 3500
train_data = sample_df[1:n_train, :]
test_data = sample_df[n_train+1:end, :]

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

function brier_score(predictions::Vector{Float64}, events::AbstractVector{Bool})
    observed = Float64.(events)
    return mean((predictions .- observed).^2)
end

# Test both versions
println("🔄 COMPARANDO ANTES E DEPOIS:\n")

# Version WITH credit_score (problematic)
println("❌ COM credit_score:")
try
    covariates_with = [:interest_rate, :credit_score, :loan_amount_log, :loan_term, :dti_ratio, :borrower_income_log, :has_collateral]
    cox_with = PrepaymentModels.fit_cox_model(loan_train, covariates=covariates_with)
    pred_with = PrepaymentModels.predict_prepayment(cox_with, loan_test, 24)
    events = .!ismissing.(loan_test.prepayment_date)
    
    c_idx_with = concordance_index(pred_with, events)
    brier_with = brier_score(pred_with, events)
    
    max_coef = maximum(abs.(cox_with.coefficients))
    pred_variety = length(unique(pred_with))
    
    println("   📊 Max |coef|: $(round(max_coef, digits=4))")
    println("   📊 C-Index: $(round(c_idx_with, digits=4))")
    println("   📊 Brier Score: $(round(brier_with, digits=4))")
    println("   📊 Predições únicas: $pred_variety")
    println("   📊 Range: $(round(minimum(pred_with), digits=6)) - $(round(maximum(pred_with), digits=6))")
    
    println("   🔍 Coeficientes detalhados:")
    coef_names = [:interest_rate, :credit_score, :loan_amount_log, :loan_term, :dti_ratio, :borrower_income_log, :has_collateral]
    for (i, name) in enumerate(coef_names)
        coef = cox_with.coefficients[i]
        hr = exp(coef)
        println("     $name: $(round(coef, digits=4)) (HR: $(round(hr, digits=4)))")
    end
    
catch e
    println("   ❌ Erro: $e")
end

println("\n✅ SEM credit_score:")
try
    covariates_without = [:interest_rate, :loan_amount_log, :loan_term, :dti_ratio, :borrower_income_log, :has_collateral]
    cox_without = PrepaymentModels.fit_cox_model(loan_train, covariates=covariates_without)
    pred_without = PrepaymentModels.predict_prepayment(cox_without, loan_test, 24)
    events = .!ismissing.(loan_test.prepayment_date)
    
    c_idx_without = concordance_index(pred_without, events)
    brier_without = brier_score(pred_without, events)
    
    max_coef = maximum(abs.(cox_without.coefficients))
    pred_variety = length(unique(pred_without))
    
    println("   📊 Max |coef|: $(round(max_coef, digits=4))")
    println("   📊 C-Index: $(round(c_idx_without, digits=4))")
    println("   📊 Brier Score: $(round(brier_without, digits=4))")
    println("   📊 Predições únicas: $pred_variety")
    println("   📊 Range: $(round(minimum(pred_without), digits=6)) - $(round(maximum(pred_without), digits=6))")
    
    println("   🔍 Coeficientes detalhados:")
    coef_names = [:interest_rate, :loan_amount_log, :loan_term, :dti_ratio, :borrower_income_log, :has_collateral]
    for (i, name) in enumerate(coef_names)
        coef = cox_without.coefficients[i]
        hr = exp(coef)
        println("     $name: $(round(coef, digits=4)) (HR: $(round(hr, digits=4)))")
    end
    
catch e
    println("   ❌ Erro: $e")
end

println("\n" * repeat("=", 50))