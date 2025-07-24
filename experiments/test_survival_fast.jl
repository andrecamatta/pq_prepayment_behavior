#!/usr/bin/env julia

"""
Comparação rápida das métricas principais com dataset reduzido
"""

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics, Random
import StatsBase
Random.seed!(42)

include("src/PrepaymentModels.jl")
using .PrepaymentModels

println("📊 COMPARAÇÃO RÁPIDA DAS MÉTRICAS DE SURVIVAL")
println(repeat("=", 60))

# Smaller dataset for faster testing
filepath = "data/official_based_data/brazilian_loans_2025-07-23_20-16.csv"
raw_data = CSV.read(filepath, DataFrame)

# Use smaller sample
sample_size = 1000
sample_indices = StatsBase.sample(1:nrow(raw_data), sample_size, replace=false)
sample_df = raw_data[sample_indices, :]

# Split 70/30  
n_train = 700
n_test = 300
train_indices = sample_indices[1:n_train]
test_indices = sample_indices[n_train+1:end]

train_data = raw_data[train_indices, :]
test_data = raw_data[test_indices, :]

println("📊 Treino: $n_train | Teste: $n_test empréstimos")

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

covariates = [
    :interest_rate,
    :loan_amount_log,
    :loan_term,
    :dti_ratio,
    :borrower_income_log,
    :has_collateral
]

println("📋 Usando ", length(covariates), " covariáveis principais")

# === FUNÇÕES DAS MÉTRICAS ===

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

function calibration_error(predictions::Vector{Float64}, events::AbstractVector{Bool}, n_bins::Int=10)
    bin_edges = range(0.0, 1.0, length=n_bins+1)
    calibration_error_total = 0.0
    total_count = 0
    
    for i in 1:n_bins
        bin_mask = (predictions .>= bin_edges[i]) .& (predictions .< bin_edges[i+1])
        
        if sum(bin_mask) > 0
            bin_predictions = predictions[bin_mask]
            bin_events = events[bin_mask]
            
            mean_pred = mean(bin_predictions)
            mean_obs = mean(bin_events)
            bin_size = sum(bin_mask)
            
            calibration_error_total += bin_size * abs(mean_pred - mean_obs)
            total_count += bin_size
        end
    end
    
    return total_count > 0 ? calibration_error_total / total_count : 0.0
end

# === TESTAR MODELOS ===

models = [
    ("Cox", :cox),
    ("Weibull MLE", :weibull),
    ("Log-Normal", :lognormal),
    ("Bernoulli-Beta", :bernoulli_beta_optimized)
]

results = DataFrame(
    Modelo = String[],
    C_Index = Float64[],
    Brier_Score = Float64[],
    Calibration_Error = Float64[],
    Status = String[]
)

println("\n🏋️  TREINANDO E AVALIANDO MODELOS...")
println(repeat("-", 60))

for (name, model_type) in models
    print("$name... ")
    
    try
        start_time = time()
        if model_type == :cox
            model = PrepaymentModels.fit_cox_model(loan_train, covariates=covariates)
        else
            model = PrepaymentModels.fit_parametric_model(loan_train, distribution=model_type, covariates=covariates)
        end
        train_time = time() - start_time
        
        # Predições
        start_pred = time()
        predictions = PrepaymentModels.predict_prepayment(model, loan_test, 24)
        pred_time = time() - start_pred
        
        # Eventos reais
        events = .!ismissing.(loan_test.prepayment_date)
        
        # Métricas
        c_idx = concordance_index(predictions, events)
        brier = brier_score(predictions, events)
        calib_error = calibration_error(predictions, events)
        
        push!(results, (name, c_idx, brier, calib_error, "✅"))
        
        println("✅ ($(round(train_time, digits=2))s treino, $(round(pred_time, digits=3))s predição)")
        
    catch e
        println("❌ Erro: $e")
        push!(results, (name, NaN, NaN, NaN, "❌ Falhou"))
    end
end

# === RESULTADOS ===

println("\n📊 RESULTADOS DAS 3 MÉTRICAS PRINCIPAIS")
println(repeat("=", 70))
println("Modelo           C-Index    Brier Score   Calibration   Status")
println(repeat("-", 70))

for row in eachrow(results)
    if row.Status == "✅"
        c_str = rpad(round(row.C_Index, digits=4), 10)
        b_str = rpad(round(row.Brier_Score, digits=4), 13)
        cal_str = rpad(round(row.Calibration_Error, digits=4), 13)
        println("$(rpad(row.Modelo, 16)) $c_str $b_str $cal_str $(row.Status)")
    else
        println("$(rpad(row.Modelo, 16)) $(rpad("N/A", 10)) $(rpad("N/A", 13)) $(rpad("N/A", 13)) $(row.Status)")
    end
end

# Rankings
successful_results = filter(row -> row.Status == "✅", results)

if nrow(successful_results) > 0
    println("\n🏆 RANKINGS POR MÉTRICA")
    println(repeat("=", 40))
    
    # C-Index (maior é melhor)
    c_ranking = sort(successful_results, :C_Index, rev=true)
    println("\n1️⃣ MELHOR C-INDEX:")
    for (i, row) in enumerate(eachrow(c_ranking))
        medal = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
        println("   $medal $(rpad(row.Modelo, 18)) $(round(row.C_Index, digits=4))")
    end
    
    # Brier Score (menor é melhor)
    brier_ranking = sort(successful_results, :Brier_Score)
    println("\n2️⃣ MELHOR BRIER SCORE:")
    for (i, row) in enumerate(eachrow(brier_ranking))
        medal = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
        println("   $medal $(rpad(row.Modelo, 18)) $(round(row.Brier_Score, digits=4))")
    end
    
    # Calibration Error (menor é melhor)
    calib_ranking = sort(successful_results, :Calibration_Error)
    println("\n3️⃣ MELHOR CALIBRAÇÃO:")
    for (i, row) in enumerate(eachrow(calib_ranking))
        medal = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
        println("   $medal $(rpad(row.Modelo, 18)) $(round(row.Calibration_Error, digits=4))")
    end
    
    best_model = c_ranking[1, :]
    println("\n🎯 MELHOR MODELO (C-Index): $(best_model.Modelo)")
    println("   📊 C-Index: $(round(best_model.C_Index, digits=4))")
    println("   📊 Brier Score: $(round(best_model.Brier_Score, digits=4))")
    println("   📊 Calibration Error: $(round(best_model.Calibration_Error, digits=4))")
end

println("\n" * repeat("=", 60))