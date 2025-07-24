#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics, Random
import StatsBase
Random.seed!(42)

include("src/PrepaymentModels.jl")
using .PrepaymentModels

println("🔍 DIAGNÓSTICO DETALHADO DO MODELO COX")
println(repeat("=", 50))

# Load data
filepath = "data/official_based_data/brazilian_loans_2025-07-23_20-16.csv"
raw_data = CSV.read(filepath, DataFrame)

# Small sample for debugging
sample_size = 500
sample_indices = StatsBase.sample(1:nrow(raw_data), sample_size, replace=false)
sample_df = raw_data[sample_indices, :]

train_data = sample_df[1:350, :]

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

# Analyze the raw data distribution
println("📊 DISTRIBUIÇÕES DOS DADOS ORIGINAIS:")
println("Credit Score:")
println("  Min: $(minimum(train_data.credit_score))")
println("  Max: $(maximum(train_data.credit_score))")  
println("  Mean: $(round(mean(train_data.credit_score), digits=2))")
println("  Std: $(round(std(train_data.credit_score), digits=2))")

println("\nInterest Rate:")
println("  Min: $(minimum(train_data.interest_rate))")
println("  Max: $(maximum(train_data.interest_rate))")
println("  Mean: $(round(mean(train_data.interest_rate), digits=2))")
println("  Std: $(round(std(train_data.interest_rate), digits=2))")

# Check the survival data preparation
covariates = [:interest_rate, :credit_score]
survival_df = PrepaymentModels._prepare_survival_data(loan_train, covariates)

println("\n📊 DADOS APÓS NORMALIZAÇÃO:")
println("Credit Score normalizado:")
println("  Min: $(round(minimum(survival_df.credit_score), digits=4))")
println("  Max: $(round(maximum(survival_df.credit_score), digits=4))")
println("  Mean: $(round(mean(survival_df.credit_score), digits=4))")
println("  Std: $(round(std(survival_df.credit_score), digits=4))")

println("\nInterest Rate normalizado:")
println("  Min: $(round(minimum(survival_df.interest_rate), digits=4))")
println("  Max: $(round(maximum(survival_df.interest_rate), digits=4))")
println("  Mean: $(round(mean(survival_df.interest_rate), digits=4))")
println("  Std: $(round(std(survival_df.interest_rate), digits=4))")

# Check event rates by credit score quantiles
println("\n📊 ANÁLISE POR QUANTIS DE CREDIT SCORE:")
credit_quantiles = quantile(train_data.credit_score, [0.25, 0.5, 0.75])
println("Quantis: $(credit_quantiles)")

# Low credit score group
low_credit = train_data[train_data.credit_score .<= credit_quantiles[1], :]
high_credit = train_data[train_data.credit_score .>= credit_quantiles[3], :]

low_prepay_rate = mean(.!ismissing.(low_credit.prepayment_date))
high_prepay_rate = mean(.!ismissing.(high_credit.prepayment_date))

println("Taxa pré-pagamento - Credit baixo: $(round(low_prepay_rate*100, digits=1))%")
println("Taxa pré-pagamento - Credit alto: $(round(high_prepay_rate*100, digits=1))%")

println("\n🔧 TESTANDO MODELO COX:")
try
    cox_model = PrepaymentModels.fit_cox_model(loan_train, covariates=covariates)
    println("✅ Modelo ajustado")
    println("Coeficientes: $(cox_model.coefficients)")
    println("Hazard ratios: $(exp.(cox_model.coefficients))")
    
    # Test with a smaller subset of covariates
    println("\n🔧 TESTANDO APENAS COM INTEREST_RATE:")
    cox_simple = PrepaymentModels.fit_cox_model(loan_train, covariates=[:interest_rate])
    println("✅ Modelo simples ajustado")
    println("Coeficiente interest_rate: $(cox_simple.coefficients)")
    println("Hazard ratio: $(exp.(cox_simple.coefficients))")
    
catch e
    println("❌ Erro: $e")
end