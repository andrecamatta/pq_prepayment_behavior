#!/usr/bin/env julia

"""
Comparação completa das métricas com dataset de 5000 empréstimos
"""

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics, Random
import StatsBase
Random.seed!(42)

include("src/PrepaymentModels.jl")
using .PrepaymentModels

println("📊 COMPARAÇÃO COMPLETA COM 5000 EMPRÉSTIMOS")
println(repeat("=", 60))

# Dataset completo
filepath = "data/official_based_data/brazilian_loans_2025-07-23_20-16.csv"
raw_data = CSV.read(filepath, DataFrame)

# Use 5000 loans as requested
sample_size = 5000
sample_indices = StatsBase.sample(1:nrow(raw_data), sample_size, replace=false)
sample_df = raw_data[sample_indices, :]

# Split 70/30  
n_train = 3500
n_test = 1500
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

# Covariáveis expandidas (sem credit_score devido a problemas de convergência)
covariates = [
    :interest_rate,
    :loan_amount_log,
    :loan_term,
    :dti_ratio,
    :borrower_income_log,
    :has_collateral
]

println("📋 Usando ", length(covariates), " covariáveis principais: ", covariates)

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
    ("Log-Normal MLE", :lognormal),
    ("Bernoulli-Beta MLE", :bernoulli_beta_optimized)
]

results = DataFrame(
    Modelo = String[],
    C_Index = Float64[],
    Brier_Score = Float64[],
    Calibration_Error = Float64[],
    Train_Time = Float64[],
    Pred_Time = Float64[],
    Status = String[]
)

println("\n🏋️  TREINANDO E AVALIANDO MODELOS...")
println(repeat("-", 80))

for (name, model_type) in models
    print("$name... ")
    
    try
        # Treinar modelo
        start_time = time()
        if model_type == :cox
            model = PrepaymentModels.fit_cox_model(loan_train, covariates=covariates)
        else
            model = PrepaymentModels.fit_parametric_model(loan_train, distribution=model_type, covariates=covariates)
        end
        train_time = time() - start_time
        
        # Fazer predições no conjunto de teste
        start_pred = time()
        predictions = PrepaymentModels.predict_prepayment(model, loan_test, 24)
        pred_time = time() - start_pred
        
        # Eventos reais no teste
        events = .!ismissing.(loan_test.prepayment_date)
        
        # Verificar diversidade das predições
        unique_preds = length(unique(predictions))
        pred_range = [minimum(predictions), maximum(predictions)]
        
        # Calcular métricas
        c_idx = concordance_index(predictions, events)
        brier = brier_score(predictions, events)
        calib_error = calibration_error(predictions, events)
        
        push!(results, (name, c_idx, brier, calib_error, train_time, pred_time, "✅"))
        
        println("✅")
        println("      Tempo treino: $(round(train_time, digits=2))s | Predição: $(round(pred_time, digits=3))s")
        println("      Predições: $(round(pred_range[1], digits=3))-$(round(pred_range[2], digits=3)) | $(unique_preds) valores únicos")
        
    catch e
        println("❌ Erro: $e")
        push!(results, (name, NaN, NaN, NaN, NaN, NaN, "❌ Falhou"))
    end
end

# === RESULTADOS ===

println("\n📊 RESULTADOS COMPLETOS - DATASET 5000 EMPRÉSTIMOS")
println(repeat("=", 90))
println("Modelo              C-Index    Brier Score   Calibration   Treino(s)  Pred(s)   Status")
println(repeat("-", 90))

for row in eachrow(results)
    if row.Status == "✅"
        c_str = rpad(round(row.C_Index, digits=4), 10)
        b_str = rpad(round(row.Brier_Score, digits=4), 13)
        cal_str = rpad(round(row.Calibration_Error, digits=4), 13)
        train_str = rpad(round(row.Train_Time, digits=1), 9)
        pred_str = rpad(round(row.Pred_Time, digits=3), 8)
        println("$(rpad(row.Modelo, 19)) $c_str $b_str $cal_str $train_str $pred_str $(row.Status)")
    else
        println("$(rpad(row.Modelo, 19)) $(rpad("N/A", 10)) $(rpad("N/A", 13)) $(rpad("N/A", 13)) $(rpad("N/A", 9)) $(rpad("N/A", 8)) $(row.Status)")
    end
end

# === ANÁLISE DETALHADA ===

successful_results = filter(row -> row.Status == "✅", results)

if nrow(successful_results) > 0
    println("\n🏆 RANKINGS DETALHADOS POR MÉTRICA")
    println(repeat("=", 50))
    
    # 1. C-Index (maior é melhor)
    c_ranking = sort(successful_results, :C_Index, rev=true)
    println("\n1️⃣ RANKING C-INDEX (Discriminação):")
    for (i, row) in enumerate(eachrow(c_ranking))
        medal = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
        println("   $medal $(rpad(row.Modelo, 20)) $(round(row.C_Index, digits=4))")
    end
    
    # 2. Brier Score (menor é melhor)
    brier_ranking = sort(successful_results, :Brier_Score)
    println("\n2️⃣ RANKING BRIER SCORE (Performance Geral):")
    for (i, row) in enumerate(eachrow(brier_ranking))
        medal = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
        println("   $medal $(rpad(row.Modelo, 20)) $(round(row.Brier_Score, digits=4))")
    end
    
    # 3. Calibration Error (menor é melhor)
    calib_ranking = sort(successful_results, :Calibration_Error)
    println("\n3️⃣ RANKING CALIBRAÇÃO (Precisão das Probabilidades):")
    for (i, row) in enumerate(eachrow(calib_ranking))
        medal = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
        println("   $medal $(rpad(row.Modelo, 20)) $(round(row.Calibration_Error, digits=4))")
    end
    
    # Performance combinada
    println("\n💯 SCORING COMBINADO (Brier 50% + Calibração 30% + C-Index 20%):")
    
    scoring_results = copy(successful_results)
    
    # Normalizar métricas
    max_c = maximum(scoring_results.C_Index)
    min_brier = minimum(scoring_results.Brier_Score)
    max_brier = maximum(scoring_results.Brier_Score)
    min_calib = minimum(scoring_results.Calibration_Error)
    max_calib = maximum(scoring_results.Calibration_Error)
    
    scores = Float64[]
    
    for row in eachrow(scoring_results)
        c_norm = row.C_Index / max_c
        brier_norm = 1.0 - (row.Brier_Score - min_brier) / (max_brier - min_brier + 1e-10)
        calib_norm = 1.0 - (row.Calibration_Error - min_calib) / (max_calib - min_calib + 1e-10)
        
        score = 0.5 * brier_norm + 0.3 * calib_norm + 0.2 * c_norm
        push!(scores, score)
    end
    
    scoring_results.Score = scores
    final_ranking = sort(scoring_results, :Score, rev=true)
    
    for (i, row) in enumerate(eachrow(final_ranking))
        medal = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
        println("   $medal $(rpad(row.Modelo, 20)) Score: $(round(row.Score, digits=3))")
    end
    
    # Modelo recomendado
    best_model = final_ranking[1, :]
    println("\n🎯 MODELO CAMPEÃO GERAL: $(best_model.Modelo)")
    println("   📊 C-Index: $(round(best_model.C_Index, digits=4))")
    println("   📊 Brier Score: $(round(best_model.Brier_Score, digits=4))")
    println("   📊 Calibration Error: $(round(best_model.Calibration_Error, digits=4))")
    println("   ⏱️  Tempo treino: $(round(best_model.Train_Time, digits=1))s")
    println("   🏆 Score Final: $(round(best_model.Score, digits=3))")
    
    # Taxa de eventos no teste
    event_rate = mean(.!ismissing.(loan_test.prepayment_date))
    println("\n📈 Taxa de pré-pagamento no teste: $(round(100*event_rate, digits=1))%")
    
end

println("\n" * repeat("=", 60))