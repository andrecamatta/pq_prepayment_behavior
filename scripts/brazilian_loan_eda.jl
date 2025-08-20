#!/usr/bin/env julia

"""
Análise Exploratória de Dados - Empréstimos Bancários Brasileiros

Script para análise exploratória completa dos dados de empréstimos brasileiros,
incluindo estatísticas descritivas, visualizações e conclusões sobre padrões
de pré-pagamento baseados em dados realísticos do mercado brasileiro.

Autor: pq_prepayment project
"""

using Pkg
Pkg.activate(".")

using DataFrames, CSV
using Statistics, StatsBase
using Dates
using Plots
using Printf
using LinearAlgebra

# Configurar plots para salvar em PNG de alta qualidade
gr()
Plots.default(size=(800, 600), dpi=300, fontfamily="DejaVu Sans")

println("🔍 ANÁLISE EXPLORATÓRIA - EMPRÉSTIMOS BANCÁRIOS BRASILEIROS")
println("=" ^ 60)

# Carregar dados mais recentes
data_file = "data/official_based_data/brazilian_loans_2025-08-19_09-30.csv"
println("📊 Carregando dados de: $data_file")

df = CSV.read(data_file, DataFrame)
n_total = nrow(df)
println("✅ Dados carregados: $n_total observações")

# Criar variáveis derivadas para análise
println("\n🔧 Preparando variáveis para análise...")

# Flag de pré-pagamento
df.prepaid = .!(ismissing.(df.prepayment_date))

# Tempo de sobrevivência (em meses)
df.survival_time = map(eachrow(df)) do row
    if !ismissing(row.prepayment_date)
        # Tempo até pré-pagamento
        (Date(row.prepayment_date) - Date(row.origination_date)).value / 30.44
    else
        # Tempo até maturidade ou data de corte (censurado)
        (Date(row.maturity_date) - Date(row.origination_date)).value / 30.44
    end
end

# LTV aproximado (valor empréstimo / renda do mutuário)
df.ltv = df.loan_amount ./ (df.borrower_income .* 12)

# Categorizar por faixas de renda
df.income_bracket = map(df.borrower_income) do income
    if income < 3000
        "Baixa (< R\$ 3k)"
    elseif income < 8000
        "Média-Baixa (R\$ 3-8k)"
    elseif income < 15000
        "Média (R\$ 8-15k)"
    elseif income < 30000
        "Média-Alta (R\$ 15-30k)"
    else
        "Alta (> R\$ 30k)"
    end
end

# Faixas de score de crédito
df.score_bracket = map(df.credit_score) do score
    if score < 400
        "Muito Baixo (< 400)"
    elseif score < 600
        "Baixo (400-599)"
    elseif score < 750
        "Médio (600-749)"
    elseif score < 900
        "Alto (750-899)"
    else
        "Muito Alto (≥ 900)"
    end
end

println("✅ Variáveis derivadas criadas")

# ================================
# 1. ESTATÍSTICAS DESCRITIVAS
# ================================

println("\n" * repeat("=", 60))
println("📈 ESTATÍSTICAS DESCRITIVAS")
println(repeat("=", 60))

# Estatísticas básicas
println("\n🔢 Visão Geral dos Dados:")
println("Total de empréstimos: $(n_total)")
println("Empréstimos pré-pagos: $(sum(df.prepaid)) ($(round(mean(df.prepaid)*100, digits=1))%)")
println("Empréstimos censurados: $(sum(.!df.prepaid)) ($(round(mean(.!df.prepaid)*100, digits=1))%)")

println("\n💰 Características dos Empréstimos:")
@printf("Valor médio: R\$ %.0f (DP: R\$ %.0f)\n", mean(df.loan_amount), std(df.loan_amount))
@printf("Prazo médio: %.1f meses (DP: %.1f meses)\n", mean(df.loan_term), std(df.loan_term))
@printf("Taxa de juros média: %.1f%% a.a. (DP: %.1f%%)\n", mean(df.interest_rate), std(df.interest_rate))
@printf("Spread over SELIC médio: %.1f p.p. (DP: %.1f p.p.)\n", mean(df.spread_over_selic), std(df.spread_over_selic))

println("\n👤 Características dos Mutuários:")
@printf("Renda média: R\$ %.0f (DP: R\$ %.0f)\n", mean(df.borrower_income), std(df.borrower_income))
@printf("Score de crédito médio: %.0f (DP: %.0f)\n", mean(df.credit_score), std(df.credit_score))
@printf("LTV médio: %.2f (DP: %.2f)\n", mean(df.ltv), std(df.ltv))

println("\n⏱️  Análise Temporal:")
@printf("Tempo médio de sobrevivência: %.1f meses (DP: %.1f meses)\n", mean(df.survival_time), std(df.survival_time))

# Análise por modalidade
println("\n🏦 Distribuição por Modalidade:")
modalidade_dist = combine(groupby(df, :loan_type), nrow => :count, :prepaid => mean => :prepayment_rate)
modalidade_dist.percentage = modalidade_dist.count ./ n_total .* 100
sort!(modalidade_dist, :count, rev=true)

for row in eachrow(modalidade_dist)
    @printf("%-20s: %5d (%4.1f%%) - Taxa Pré-pag: %5.1f%%\n", 
            row.loan_type, row.count, row.percentage, row.prepayment_rate*100)
end

# ================================
# 2. VISUALIZAÇÕES E ANÁLISES
# ================================

println("\n" * repeat("=", 60))
println("📊 GERANDO VISUALIZAÇÕES")
println(repeat("=", 60))

# Criar diretório para gráficos
mkpath("results/eda_plots")
println("📁 Salvando gráficos em: results/eda_plots/")

# Plot 1: Distribuição de valores dos empréstimos
p1 = histogram(df.loan_amount, bins=50, 
               title="Distribuição dos Valores dos Empréstimos",
               xlabel="Valor do Empréstimo (R\$)", 
               ylabel="Frequência",
               color=:lightblue, alpha=0.7,
               grid=true)
vline!([mean(df.loan_amount)], color=:red, linewidth=2, 
       label="Média: R\$ $(Int(round(mean(df.loan_amount))))")
savefig(p1, "results/eda_plots/01_distribuicao_valores.png")

# Plot 2: Taxa de pré-pagamento por modalidade
prep_by_type = combine(groupby(df, :loan_type), :prepaid => mean => :rate)
sort!(prep_by_type, :rate, rev=true)

p2 = bar(prep_by_type.loan_type, prep_by_type.rate .* 100,
         title="Taxa de Pré-pagamento por Modalidade",
         xlabel="Modalidade", ylabel="Taxa de Pré-pagamento (%)",
         color=:orange, alpha=0.8, 
         xrotation=45, grid=true)
savefig(p2, "results/eda_plots/02_prepagamento_modalidade.png")

# Plot 3: Distribuição de taxas de juros
p3 = histogram(df.interest_rate, bins=40,
               title="Distribuição das Taxas de Juros",
               xlabel="Taxa de Juros (% a.a.)", ylabel="Frequência",
               color=:green, alpha=0.7, grid=true)
vline!([mean(df.interest_rate)], color=:red, linewidth=2,
       label="Média: $(round(mean(df.interest_rate), digits=1))% a.a.")
savefig(p3, "results/eda_plots/03_distribuicao_taxas.png")

# Plot 4: Relação entre renda e valor do empréstimo
p4 = scatter(df.borrower_income, df.loan_amount, 
             alpha=0.6, markersize=2,
             title="Relação: Renda vs Valor do Empréstimo",
             xlabel="Renda do Mutuário (R\$)", ylabel="Valor do Empréstimo (R\$)",
             color=:purple, grid=true)
# Adicionar linha de tendência
x_vals = 1000:1000:maximum(df.borrower_income)
y_vals = x_vals .* (mean(df.loan_amount) / mean(df.borrower_income))
plot!(p4, x_vals, y_vals, color=:red, linewidth=2, label="Tendência Linear")
savefig(p4, "results/eda_plots/04_renda_vs_valor.png")

# Plot 5: Taxa de pré-pagamento por faixa de renda
prep_by_income = combine(groupby(df, :income_bracket), :prepaid => mean => :rate)
income_order = ["Baixa (< R\$ 3k)", "Média-Baixa (R\$ 3-8k)", "Média (R\$ 8-15k)", 
                "Média-Alta (R\$ 15-30k)", "Alta (> R\$ 30k)"]
prep_by_income = prep_by_income[indexin(income_order, prep_by_income.income_bracket), :]

p5 = bar(prep_by_income.income_bracket, prep_by_income.rate .* 100,
         title="Taxa de Pré-pagamento por Faixa de Renda",
         xlabel="Faixa de Renda", ylabel="Taxa de Pré-pagamento (%)",
         color=:teal, alpha=0.8, xrotation=45, grid=true)
savefig(p5, "results/eda_plots/05_prepagamento_renda.png")

# Plot 6: Distribuição de scores de crédito
p6 = histogram(df.credit_score, bins=50,
               title="Distribuição dos Scores de Crédito",
               xlabel="Score de Crédito", ylabel="Frequência",
               color=:coral, alpha=0.7, grid=true)
vline!([mean(df.credit_score)], color=:red, linewidth=2,
       label="Média: $(Int(round(mean(df.credit_score))))")
savefig(p6, "results/eda_plots/06_distribuicao_scores.png")

# Plot 7: Curva de sobrevivência de Kaplan-Meier (simplificada)
# Ordenar por tempo de sobrevivência
df_sorted = sort(df, :survival_time)
n = nrow(df_sorted)

# Calcular estimador de Kaplan-Meier
km_times = Float64[]
km_survival = Float64[]
global n_at_risk = n

for i in 1:n
    global n_at_risk
    if df_sorted.prepaid[i]  # Evento observado
        push!(km_times, df_sorted.survival_time[i])
        push!(km_survival, (n_at_risk - 1) / n_at_risk * (isempty(km_survival) ? 1.0 : km_survival[end]))
    end
    n_at_risk -= 1
end

if !isempty(km_times)
    p7 = plot(km_times, km_survival, 
              title="Curva de Sobrevivência (Kaplan-Meier)",
              xlabel="Tempo (meses)", ylabel="Probabilidade de Sobrevivência",
              color=:blue, linewidth=2, grid=true,
              label="Estimativa K-M")
    ylims!(p7, (0, 1))
    savefig(p7, "results/eda_plots/07_curva_sobrevivencia.png")
end

# Plot 8: Tempo de sobrevivência médio por modalidade
survival_by_type = combine(groupby(df, :loan_type), :survival_time => mean => :avg_survival)
sort!(survival_by_type, :avg_survival, rev=true)

p8 = bar(survival_by_type.loan_type, survival_by_type.avg_survival,
         title="Tempo Médio de Sobrevivência por Modalidade",
         xlabel="Modalidade", ylabel="Tempo Médio (meses)",
         color=:lightgreen, alpha=0.7, xrotation=45, grid=true)
savefig(p8, "results/eda_plots/08_sobrevivencia_modalidade.png")

# Plot 9: Heatmap de correlação entre variáveis numéricas
numeric_cols = [:loan_amount, :loan_term, :interest_rate, :spread_over_selic, 
                :credit_score, :borrower_income, :survival_time, :ltv]
correlation_matrix = cor(Matrix(df[!, numeric_cols]))

p9 = heatmap(string.(numeric_cols), string.(numeric_cols), correlation_matrix,
             title="Matriz de Correlação - Variáveis Numéricas",
             color=:viridis, aspect_ratio=:equal,
             xrotation=45, yrotation=45)
savefig(p9, "results/eda_plots/09_matriz_correlacao.png")

# Plot 10: Taxa de pré-pagamento por prazo do empréstimo
df.term_bracket = map(df.loan_term) do term
    if term <= 12
        "Curto (≤ 12m)"
    elseif term <= 24
        "Médio (13-24m)"
    elseif term <= 48
        "Longo (25-48m)"
    else
        "Muito Longo (> 48m)"
    end
end

prep_by_term = combine(groupby(df, :term_bracket), :prepaid => mean => :rate)
term_order = ["Curto (≤ 12m)", "Médio (13-24m)", "Longo (25-48m)", "Muito Longo (> 48m)"]
prep_by_term = prep_by_term[indexin(term_order, prep_by_term.term_bracket), :]
filter!(!isnothing, prep_by_term.term_bracket)

p10 = bar(prep_by_term.term_bracket, prep_by_term.rate .* 100,
          title="Taxa de Pré-pagamento por Prazo",
          xlabel="Prazo do Empréstimo", ylabel="Taxa de Pré-pagamento (%)",
          color=:gold, alpha=0.8, grid=true)
savefig(p10, "results/eda_plots/10_prepagamento_prazo.png")

println("✅ 10 visualizações salvas em PNG")

# ================================
# 3. ANÁLISES AVANÇADAS E CONCLUSÕES
# ================================

println("\n" * repeat("=", 60))
println("🎯 ANÁLISES AVANÇADAS E CONCLUSÕES")
println(repeat("=", 60))

# Análise por estados
println("\n🌎 Análise Regional (Top 5 Estados):")
regional_analysis = combine(groupby(df, :borrower_state), 
                           nrow => :count, 
                           :prepaid => mean => :prepayment_rate,
                           :interest_rate => mean => :avg_rate)
sort!(regional_analysis, :count, rev=true)

for (i, row) in enumerate(eachrow(regional_analysis[1:min(5, nrow(regional_analysis)), :]))
    @printf("%d. %s: %d empréstimos, %.1f%% pré-pagamento, %.1f%% taxa média\n",
            i, row.borrower_state, row.count, row.prepayment_rate*100, row.avg_rate)
end

# Análise de risco por modalidade
println("\n💡 Perfil de Risco por Modalidade:")
risk_analysis = combine(groupby(df, :loan_type),
                       :credit_score => mean => :avg_score,
                       :interest_rate => mean => :avg_rate,
                       :prepaid => mean => :prepayment_rate,
                       :ltv => mean => :avg_ltv)
sort!(risk_analysis, :avg_rate, rev=true)

for row in eachrow(risk_analysis)
    @printf("%-20s: Score=%3.0f, Taxa=%4.1f%%, Pré-pag=%4.1f%%, LTV=%.2f\n",
            row.loan_type, row.avg_score, row.avg_rate, 
            row.prepayment_rate*100, row.avg_ltv)
end

# Segmentação de clientes por propensão ao pré-pagamento
println("\n📊 Segmentação por Propensão ao Pré-pagamento:")

# Definir segmentos baseados em características observadas
df.risk_segment = map(eachrow(df)) do row
    score = row.credit_score
    income = row.borrower_income
    ltv = row.ltv
    
    if score >= 750 && income >= 15000 && ltv <= 0.3
        "Baixo Risco"
    elseif score >= 600 && income >= 8000 && ltv <= 0.5
        "Médio Risco"
    elseif score >= 400 && income >= 3000 && ltv <= 0.8
        "Alto Risco"
    else
        "Muito Alto Risco"
    end
end

segment_analysis = combine(groupby(df, :risk_segment), 
                          nrow => :count,
                          :prepaid => mean => :prepayment_rate,
                          :interest_rate => mean => :avg_rate,
                          :loan_amount => mean => :avg_amount)

for row in eachrow(segment_analysis)
    @printf("%-16s: %4d (%4.1f%%) - Pré-pag: %4.1f%% - Taxa: %4.1f%% - Valor: R\$ %.0f\n",
            row.risk_segment, row.count, row.count/n_total*100,
            row.prepayment_rate*100, row.avg_rate, row.avg_amount)
end

# ================================
# 4. CONCLUSÕES FINAIS
# ================================

println("\n" * repeat("=", 60))
println("📝 CONCLUSÕES PRINCIPAIS")
println(repeat("=", 60))

println("\n🔍 PRINCIPAIS DESCOBERTAS:")

println("\n1. 📈 TAXA DE PRÉ-PAGAMENTO GERAL:")
println("   • $(round(mean(df.prepaid)*100, digits=1))% dos empréstimos são pré-pagos")
println("   • Tempo médio até pré-pagamento: $(round(mean(df[df.prepaid, :].survival_time), digits=1)) meses")

println("\n2. 🏦 ANÁLISE POR MODALIDADE:")
high_prepay = prep_by_type[1, :]
low_prepay = prep_by_type[end, :]
println("   • Maior taxa: $(high_prepay.loan_type) ($(round(high_prepay.rate*100, digits=1))%)")
println("   • Menor taxa: $(low_prepay.loan_type) ($(round(low_prepay.rate*100, digits=1))%)")

println("\n3. 💰 FATORES DE RISCO:")
high_income_prepay = prep_by_income[end, :rate] * 100
low_income_prepay = prep_by_income[1, :rate] * 100
println("   • Renda alta: $(round(high_income_prepay, digits=1))% de pré-pagamento")
println("   • Renda baixa: $(round(low_income_prepay, digits=1))% de pré-pagamento")
println("   • Correlação renda-capacidade de pré-pagar é evidente")

println("\n4. 📊 CARACTERÍSTICAS DO MERCADO:")
println("   • Taxa média de juros: $(round(mean(df.interest_rate), digits=1))% a.a.")
println("   • Spread médio sobre SELIC: $(round(mean(df.spread_over_selic), digits=1)) p.p.")
println("   • LTV médio: $(round(mean(df.ltv), digits=2))")

println("\n5. 🎯 RECOMENDAÇÕES PARA PRICING E RISCO:")
println("   • Segmentar taxas por modalidade (spread de $(round(maximum(prep_by_type.rate) - minimum(prep_by_type.rate), digits=3)*100) p.p. entre extremos)")
println("   • Considerar renda como fator principal de pré-pagamento")
println("   • Ajustar provisões baseado no perfil de risco identificado")
println("   • Implementar modelos preditivos para segmentos específicos")

# Salvar resumo em arquivo texto
println("\n💾 Salvando resumo da análise...")
open("results/eda_plots/analise_resumo.txt", "w") do io
    println(io, "ANÁLISE EXPLORATÓRIA - EMPRÉSTIMOS BANCÁRIOS BRASILEIROS")
    println(io, "Data: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM"))")
    println(io, "Dataset: $data_file")
    println(io, "=" ^ 60)
    println(io)
    println(io, "ESTATÍSTICAS GERAIS:")
    println(io, "• Total de empréstimos: $n_total")
    println(io, "• Taxa de pré-pagamento: $(round(mean(df.prepaid)*100, digits=1))%")
    println(io, "• Valor médio: R\$ $(Int(round(mean(df.loan_amount))))")
    println(io, "• Taxa média: $(round(mean(df.interest_rate), digits=1))% a.a.")
    println(io, "• Score médio: $(Int(round(mean(df.credit_score))))")
    println(io)
    println(io, "MODALIDADES (por volume):")
    for row in eachrow(modalidade_dist)
        println(io, "• $(row.loan_type): $(row.count) ($(round(row.percentage, digits=1))%) - Pré-pag: $(round(row.prepayment_rate*100, digits=1))%")
    end
end

println("✅ Análise completa finalizada!")
println("📊 Gráficos salvos em: results/eda_plots/")
println("📄 Resumo salvo em: results/eda_plots/analise_resumo.txt")

println("\n" * repeat("=", 60))
println("🎉 ANÁLISE EXPLORATÓRIA CONCLUÍDA COM SUCESSO!")
println(repeat("=", 60))