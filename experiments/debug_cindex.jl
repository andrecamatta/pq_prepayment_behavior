#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics, Random
import StatsBase
Random.seed!(42)

include("src/PrepaymentModels.jl")
using .PrepaymentModels

function concordance_index(predictions::Vector{Float64}, events::AbstractVector{Bool})
    """C-Index: Concordância entre predições e eventos"""
    n = length(predictions)
    concordant = 0
    total_pairs = 0
    
    # Verificar se há variação nas predições
    if length(unique(predictions)) == 1
        @warn "Todas as predições são iguais - C-index indefinido, retornando 0.5"
        return 0.5
    end
    
    for i in 1:n
        for j in (i+1):n
            if events[i] != events[j]  # Um teve evento, outro não
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

println("🧪 TESTE ISOLADO CONCORDANCE INDEX")

# Test data
predictions = [1.0, 1.0, 1.0, 1.0, 1.0]
events = [true, false, true, false, true]

println("Testando concordance_index com dados simples...")
try
    result = concordance_index(predictions, events)
    println("✅ Sucesso: $result")
catch e
    println("❌ Erro: $e")
end

# Test Cox model
println("\n🧪 TESTE COX MODEL")
filepath = "data/official_based_data/brazilian_loans_2025-07-23_20-16.csv"
raw_data = CSV.read(filepath, DataFrame)

# Small sample
sample_size = 50
sample_indices = StatsBase.sample(1:nrow(raw_data), sample_size, replace=false)
sample_df = raw_data[sample_indices, :]

# Create LoanData
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

train_data = sample_df[1:35, :]
test_data = sample_df[36:end, :]

loan_train = create_loan_data(train_data)
loan_test = create_loan_data(test_data)

covariates = [:interest_rate, :credit_score]

println("Treinando modelo Cox...")
try
    cox_model = PrepaymentModels.fit_cox_model(loan_train, covariates=covariates)
    println("✅ Cox treinado")
    
    println("Fazendo predições...")
    cox_pred = PrepaymentModels.predict_prepayment(cox_model, loan_test, 24)
    println("✅ Predições feitas: $(length(cox_pred))")
    println("Predições: $(round.(cox_pred[1:min(5, end)], digits=6))")
    
    # Eventos reais no teste
    events = .!ismissing.(loan_test.prepayment_date)
    println("Eventos: $(events[1:min(5, end)])")
    
    println("Calculando concordance_index...")
    println("Tipo de cox_pred: $(typeof(cox_pred))")
    println("Tipo de events: $(typeof(events))")
    
    c_idx = concordance_index(cox_pred, events)
    println("✅ C-Index: $c_idx")
    
catch e
    println("❌ Erro: $e")
    println("Tipo do erro: $(typeof(e))")
    if isa(e, MethodError)
        println("Método chamado: $(e.f)")
        println("Argumentos: $(e.args)")
    end
end