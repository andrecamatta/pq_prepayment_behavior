#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics, Random
import StatsBase
Random.seed!(42)

include("src/PrepaymentModels.jl")
using .PrepaymentModels

println("📊 COVARIÁVEIS UTILIZADAS NOS MODELOS DE SOBREVIVÊNCIA")
println(repeat("=", 70))

# Load sample data
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

# Current covariates in use
println("🎯 COVARIÁVEIS ATUALMENTE EM USO:")
current_covariates = [
    :interest_rate,
    :loan_amount_log,
    :loan_term,
    :dti_ratio,
    :borrower_income_log,
    :has_collateral
]

for (i, var) in enumerate(current_covariates)
    println("  $i. $var")
end
println("  TOTAL: $(length(current_covariates)) covariáveis")

# Create survival data to see what gets generated
survival_df = PrepaymentModels._prepare_survival_data(loan_data, current_covariates)

println("\n📋 DETALHAMENTO DAS COVARIÁVEIS:")
println(repeat("-", 70))

# Basic covariates with transformations
println("🏦 VARIÁVEIS DE EMPRÉSTIMO:")
println("  ✅ interest_rate")
println("     - Descrição: Taxa de juros anual (%)")
println("     - Transformação: Normalizada (z-score)")
println("     - Range original: $(round(minimum(sample_df.interest_rate), digits=2)) - $(round(maximum(sample_df.interest_rate), digits=2))%")
println("     - Range transformado: $(round(minimum(survival_df.interest_rate), digits=4)) - $(round(maximum(survival_df.interest_rate), digits=4))")

println("\n  ✅ loan_amount_log")
println("     - Descrição: Valor do empréstimo")
println("     - Transformação: Logaritmo natural")
println("     - Range original: R\$ $(Int(minimum(sample_df.loan_amount))) - R\$ $(Int(maximum(sample_df.loan_amount)))")
println("     - Range transformado: $(round(minimum(survival_df.loan_amount_log), digits=4)) - $(round(maximum(survival_df.loan_amount_log), digits=4))")

println("\n  ✅ loan_term")
println("     - Descrição: Prazo do empréstimo (meses)")
println("     - Transformação: Nenhuma (Float64)")
println("     - Range: $(Int(minimum(sample_df.loan_term))) - $(Int(maximum(sample_df.loan_term))) meses")

println("\n👤 VARIÁVEIS DO MUTUÁRIO:")
println("  ✅ borrower_income_log")
println("     - Descrição: Renda mensal do mutuário")
println("     - Transformação: Logaritmo natural")
println("     - Range original: R\$ $(Int(minimum(sample_df.borrower_income))) - R\$ $(Int(maximum(sample_df.borrower_income)))")
println("     - Range transformado: $(round(minimum(survival_df.borrower_income_log), digits=4)) - $(round(maximum(survival_df.borrower_income_log), digits=4))")

println("\n  ✅ dti_ratio")
println("     - Descrição: Debt-to-Income ratio (Prestação/Renda anual)")
println("     - Transformação: Calculado (payment * 12 / income)")
println("     - Range: $(round(minimum(survival_df.dti_ratio), digits=4)) - $(round(maximum(survival_df.dti_ratio), digits=4))")

println("\n🛡️  VARIÁVEIS DE GARANTIA:")
println("  ✅ has_collateral")
println("     - Descrição: Indica se o empréstimo tem garantia")
println("     - Transformação: Dummy (1 = Com Garantia, 0 = Sem Garantia)")
println("     - Distribuição: $(sum(survival_df.has_collateral)) com garantia / $(nrow(survival_df)) total ($(round(mean(survival_df.has_collateral)*100, digits=1))%)")

# Check loan type dummies that are automatically added
println("\n🏷️  VARIÁVEIS DUMMY DE TIPO DE EMPRÉSTIMO (adicionadas automaticamente):")
loan_types = unique(sample_df.loan_type)
for loan_type in loan_types
    safe_name = Symbol(replace(loan_type, " " => "_"))
    if safe_name in names(survival_df)
        count_type = sum(survival_df[!, safe_name])
        percentage = round(count_type / nrow(survival_df) * 100, digits=1)
        println("  ✅ $safe_name")
        println("     - Descrição: Dummy para '$loan_type'")
        println("     - Distribuição: $count_type / $(nrow(survival_df)) ($percentage%)")
    end
end

println("\n❌ VARIÁVEL REMOVIDA:")
println("  ❌ credit_score")
println("     - Descrição: Score de crédito (300-1000)")
println("     - Motivo da remoção: Coeficientes extremos (>10) causando predições uniformes")
println("     - Transformação original: Normalizada (z-score)")
println("     - Range original: $(Int(minimum(sample_df.credit_score))) - $(Int(maximum(sample_df.credit_score)))")
println("     - Status: TEMPORARIAMENTE REMOVIDO - precisa normalização robusta")

println("\n📊 RESUMO FINAL:")
println("  • Covariáveis numéricas: 5 (interest_rate, loan_amount_log, loan_term, dti_ratio, borrower_income_log)")
println("  • Covariáveis binárias: 1 + $(length(loan_types)) (has_collateral + loan_type_dummies)")
println("  • Total efetivo: $(ncol(survival_df) - 3) (excluindo loan_id, time, event)")
println("  • Transformações: 3 log, 1 normalizada, 1 calculada, $(1 + length(loan_types)) dummies")

println("\n" * repeat("=", 70))