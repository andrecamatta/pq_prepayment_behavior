#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics
include("src/PrepaymentModels.jl")
using .PrepaymentModels

# Load sample data
filepath = "data/official_based_data/brazilian_loans_2025-07-23_20-16.csv"
raw_data = CSV.read(filepath, DataFrame)
sample_df = raw_data[1:200, :]

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
current_covariates = [:interest_rate, :loan_amount_log, :loan_term, :dti_ratio, :borrower_income_log, :has_collateral]

# Create survival data
survival_df = PrepaymentModels._prepare_survival_data(loan_data, current_covariates)

println("🔍 ANÁLISE COMPLETA DO DATAFRAME DE SOBREVIVÊNCIA")
println(repeat("=", 60))

println("📋 TODAS AS COLUNAS GERADAS:")
all_cols = names(survival_df)
for (i, col) in enumerate(all_cols)
    println("  $i. $col")
end

println("\n📊 TIPOS DE EMPRÉSTIMO NO DATASET:")
loan_types = unique(sample_df.loan_type)
for (i, lt) in enumerate(loan_types)
    count = sum(sample_df.loan_type .== lt)
    percentage = round(count / nrow(sample_df) * 100, digits=1)
    println("  $i. '$lt': $count observações ($percentage%)")
end

println("\n🔧 VERIFICANDO DUMMIES GERADAS:")
for loan_type in loan_types
    safe_name = Symbol(replace(loan_type, " " => "_"))
    if safe_name in names(survival_df)
        count_dummy = sum(survival_df[!, safe_name] .== 1.0)
        println("  ✅ $safe_name: $count_dummy = 1.0")
    else
        println("  ❌ $safe_name: NÃO ENCONTRADA")
    end
end

println("\n📏 DIMENSÕES:")
println("  • Linhas: $(nrow(survival_df))")
println("  • Colunas: $(ncol(survival_df))")
println("  • Covariáveis: $(ncol(survival_df) - 3) (excluindo loan_id, time, event)")

println("\n📊 AMOSTRA DOS DADOS:")
println(first(survival_df, 3))