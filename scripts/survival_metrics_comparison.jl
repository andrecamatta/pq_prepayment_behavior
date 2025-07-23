#!/usr/bin/env julia

"""
Comparação focada nas 3 métricas principais de survival analysis:
1. C-Index (Concordance Index)
2. Brier Score  
3. Calibration Error
"""

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics, Random, StatsBase
Random.seed!(42)

include("../src/PrepaymentModels.jl")
using .PrepaymentModels

println("📊 COMPARAÇÃO DAS 3 MÉTRICAS PRINCIPAIS DE SURVIVAL")
println(repeat("=", 60))

# Dataset pequeno para teste rápido
filepath = "data/official_based_data/brazilian_loans_2025-07-23_10-18.csv"
raw_data = CSV.read(filepath, DataFrame)

# Amostra de 5000 empréstimos
sample_size = 5000
sample_indices = sample(1:nrow(raw_data), sample_size, replace=false)
sample_df = raw_data[sample_indices, :]

# Split 70/30
n_train = 3500
n_test = 1500
train_indices = sample_indices[1:n_train]
test_indices = sample_indices[n_train+1:end]

train_data = raw_data[train_indices, :]
test_data = raw_data[test_indices, :]

println("📊 Treino: $n_train | Teste: $n_test empréstimos")

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

covariates = [:interest_rate, :credit_score]

# === FUNÇÕES DAS MÉTRICAS ===

function c_index(predictions::Vector{Float64}, events::Vector{Bool})
    """C-Index: Concordância entre predições e eventos"""
    n = length(predictions)
    concordant = 0
    total_pairs = 0
    
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

function brier_score(predictions::Vector{Float64}, events::Vector{Bool})
    """Brier Score: Erro quadrático médio das probabilidades"""
    observed = Float64.(events)
    return mean((predictions .- observed).^2)
end

function calibration_error(predictions::Vector{Float64}, events::Vector{Bool}, n_bins::Int=10)
    """Erro de Calibração: Diferença entre predições e observações por bin"""
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

# === TREINAR E AVALIAR MODELOS ===

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
        if model_type == :cox
            predictions = PrepaymentModels.predict_prepayment(model, loan_test, 24)
        else
            predictions = PrepaymentModels.predict_prepayment(model, loan_test, 24)
        end
        pred_time = time() - start_pred
        
        # Eventos reais no teste
        events = .!ismissing.(loan_test.prepayment_date)
        
        # Calcular métricas
        c_idx = c_index(predictions, events)
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

# === RANKINGS ===

successful_results = filter(row -> row.Status == "✅", results)

if nrow(successful_results) > 0
    println("\n🏆 RANKINGS POR MÉTRICA")
    println(repeat("=", 40))
    
    # 1. C-Index (maior é melhor)
    c_ranking = sort(successful_results, :C_Index, rev=true)
    println("\n1️⃣ MELHOR C-INDEX (Maior = Melhor):")
    for (i, row) in enumerate(eachrow(c_ranking))
        medal = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
        println("   $medal $(rpad(row.Modelo, 18)) $(round(row.C_Index, digits=4))")
    end
    
    # 2. Brier Score (menor é melhor)
    brier_ranking = sort(successful_results, :Brier_Score)
    println("\n2️⃣ MELHOR BRIER SCORE (Menor = Melhor):")
    for (i, row) in enumerate(eachrow(brier_ranking))
        medal = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
        println("   $medal $(rpad(row.Modelo, 18)) $(round(row.Brier_Score, digits=4))")
    end
    
    # 3. Calibration Error (menor é melhor)
    calib_ranking = sort(successful_results, :Calibration_Error)
    println("\n3️⃣ MELHOR CALIBRAÇÃO (Menor = Melhor):")
    for (i, row) in enumerate(eachrow(calib_ranking))
        medal = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
        println("   $medal $(rpad(row.Modelo, 18)) $(round(row.Calibration_Error, digits=4))")
    end
    
    # === MODELO RECOMENDADO PARA PRECIFICAÇÃO ===
    println("\n💰 RECOMENDAÇÃO PARA PRECIFICAÇÃO")
    println(repeat("=", 50))
    
    # Scoring ponderado: Brier (50%) + Calibration (30%) + C-Index (20%)
    println("Critério: Brier Score (50%) + Calibração (30%) + C-Index (20%)")
    
    scoring_results = copy(successful_results)
    
    # Normalizar métricas (0-1, onde 1 é melhor)
    max_c = maximum(scoring_results.C_Index)
    min_brier = minimum(scoring_results.Brier_Score)
    max_brier = maximum(scoring_results.Brier_Score)
    min_calib = minimum(scoring_results.Calibration_Error)
    max_calib = maximum(scoring_results.Calibration_Error)
    
    scoring_results.Score = Float64[]
    
    for row in eachrow(scoring_results)
        c_norm = row.C_Index / max_c
        brier_norm = 1.0 - (row.Brier_Score - min_brier) / (max_brier - min_brier + 1e-10)
        calib_norm = 1.0 - (row.Calibration_Error - min_calib) / (max_calib - min_calib + 1e-10)
        
        score = 0.5 * brier_norm + 0.3 * calib_norm + 0.2 * c_norm
        push!(scoring_results.Score, score)
    end
    
    final_ranking = sort(scoring_results, :Score, rev=true)
    
    println("\nRanking Final:")
    for (i, row) in enumerate(eachrow(final_ranking))
        medal = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
        println("   $medal $(rpad(row.Modelo, 18)) Score: $(round(row.Score, digits=3))")
    end
    
    best_model = final_ranking[1, :]
    println("\n🎯 MODELO RECOMENDADO: $(best_model.Modelo)")
    println("   📊 C-Index: $(round(best_model.C_Index, digits=4))")
    println("   📊 Brier Score: $(round(best_model.Brier_Score, digits=4))")
    println("   📊 Calibration Error: $(round(best_model.Calibration_Error, digits=4))")
    println("   🏆 Score Final: $(round(best_model.Score, digits=3))")
end

println("\n" * repeat("=", 60))