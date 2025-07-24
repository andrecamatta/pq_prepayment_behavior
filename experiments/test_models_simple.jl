#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics, Random, StatsBase
Random.seed!(42)

include("src/PrepaymentModels.jl")
using .PrepaymentModels

println("🧪 TESTE SIMPLES DOS MODELOS")
println(repeat("=", 40))

# Carrega dados pequenos
filepath = "data/official_based_data/brazilian_loans_2025-07-23_20-16.csv"
raw_data = CSV.read(filepath, DataFrame)

# Amostra pequena
sample_size = 200
sample_indices = sample(1:nrow(raw_data), sample_size, replace=false)
sample_df = raw_data[sample_indices, :]

# Split
n_train = 140
train_data = sample_df[1:n_train, :]
test_data = sample_df[n_train+1:end, :]

println("📊 Treino: $n_train | Teste: $(nrow(test_data)) empréstimos")

# Criar LoanData
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

covariates = [:interest_rate, :credit_score, :has_collateral]

# === TESTE DOS MODELOS ===
models_results = []

# 1. COX
println("\n🔍 Testando Cox...")
try
    cox_model = PrepaymentModels.fit_cox_model(loan_train, covariates=covariates)
    cox_pred = PrepaymentModels.predict_prepayment(cox_model, loan_test, 24)
    
    println("   ✅ Cox OK")
    println("   📊 Min: $(round(minimum(cox_pred), digits=6))")
    println("   📊 Max: $(round(maximum(cox_pred), digits=6))")
    println("   📊 Mean: $(round(mean(cox_pred), digits=6))")
    
    push!(models_results, ("Cox", cox_pred))
catch e
    println("   ❌ Cox falhou: $e")
end

# 2. WEIBULL
println("\n🔍 Testando Weibull...")
try
    weibull_model = PrepaymentModels.fit_parametric_model(loan_train, distribution=:weibull, covariates=covariates)
    weibull_pred = PrepaymentModels.predict_prepayment(weibull_model, loan_test, 24)
    
    println("   ✅ Weibull OK")
    println("   📊 Min: $(round(minimum(weibull_pred), digits=6))")
    println("   📊 Max: $(round(maximum(weibull_pred), digits=6))")
    println("   📊 Mean: $(round(mean(weibull_pred), digits=6))")
    
    push!(models_results, ("Weibull", weibull_pred))
catch e
    println("   ❌ Weibull falhou: $e")
end

# 3. LOG-NORMAL
println("\n🔍 Testando Log-Normal...")
try
    lognormal_model = PrepaymentModels.fit_parametric_model(loan_train, distribution=:lognormal, covariates=covariates)
    lognormal_pred = PrepaymentModels.predict_prepayment(lognormal_model, loan_test, 24)
    
    println("   ✅ Log-Normal OK")
    println("   📊 Min: $(round(minimum(lognormal_pred), digits=6))")
    println("   📊 Max: $(round(maximum(lognormal_pred), digits=6))")
    println("   📊 Mean: $(round(mean(lognormal_pred), digits=6))")
    
    push!(models_results, ("Log-Normal", lognormal_pred))
catch e
    println("   ❌ Log-Normal falhou: $e")
end

# 4. BERNOULLI-BETA
println("\n🔍 Testando Bernoulli-Beta...")
try
    bb_model = PrepaymentModels.fit_parametric_model(loan_train, distribution=:bernoulli_beta_optimized, covariates=covariates)
    bb_pred = PrepaymentModels.predict_prepayment(bb_model, loan_test, 24)
    
    println("   ✅ Bernoulli-Beta OK")
    println("   📊 Min: $(round(minimum(bb_pred), digits=6))")
    println("   📊 Max: $(round(maximum(bb_pred), digits=6))")
    println("   📊 Mean: $(round(mean(bb_pred), digits=6))")
    
    push!(models_results, ("Bernoulli-Beta", bb_pred))
catch e
    println("   ❌ Bernoulli-Beta falhou: $e")
end

# === RESUMO ===
println("\n🏆 RESUMO:")
println("   Modelos testados: $(length(models_results))")
for (name, pred) in models_results
    println("   $name: $(length(pred)) predições")
end

println("\n" * repeat("=", 40))