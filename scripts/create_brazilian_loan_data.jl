#!/usr/bin/env julia

"""
Criação de dataset realístico de empréstimos bancários brasileiros em taxa fixa
Baseado em dados públicos do BCB, IBGE, Serasa e regulamentações brasileiras
"""

using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Dates
using Statistics
using Random
using Distributions

println("🇧🇷 CRIANDO DATASET DE EMPRÉSTIMOS BANCÁRIOS BRASILEIROS")
println(repeat("=", 60))

Random.seed!(42)  # Reprodutibilidade

# Configurações
n_loans = 50000
data_dir = "data/official_based_data"
mkpath(data_dir)

println("📊 Fontes de dados utilizadas (Brasil):")
println("  • Taxas: BCB - Focus - Relatório de Mercado")
println("  • Rendas: IBGE PNAD Contínua")
println("  • Scores: Distribuições Serasa + SPC")
println("  • Geografia: IBGE - População por Estado")
println("  • Pré-pagamento: Literatura brasileira + CDC Art. 52")

# MODELAGEM DE CICLOS ECONÔMICOS BRASILEIROS (2019-2024)
println("  ✓ Implementando ciclos econômicos brasileiros...")

# Taxas de juros REAIS baseadas no BCB Focus e Selic
bcb_rates = Dict(
    Date(2019,1,1) => 6.50,   # Jan 2019 - Selic 6.50%
    Date(2019,7,1) => 6.00,   # Jul 2019  
    Date(2020,1,1) => 4.50,   # Jan 2020
    Date(2020,4,1) => 3.00,   # Apr 2020 - COVID - queda abrupta
    Date(2020,7,1) => 2.25,   # Jul 2020 - Mínimo histórico
    Date(2021,1,1) => 2.00,   # Jan 2021 - Piso histórico
    Date(2021,7,1) => 4.25,   # Jul 2021 - Início do ciclo de alta
    Date(2022,1,1) => 10.75,  # Jan 2022 - Alta agressiva
    Date(2022,7,1) => 13.25,  # Jul 2022 - Pico do ciclo
    Date(2022,12,1) => 13.75, # Dec 2022 - Teto do ciclo
    Date(2023,6,1) => 13.75,  # Jun 2023 - Manutenção
    Date(2024,1,1) => 11.25   # Jan 2024 - Início dos cortes
)

# DADOS MACROECONÔMICOS BRASILEIROS (2019-2024)
# Baseado em BCB, IBGE e dados oficiais
macro_conditions = Dict(
    # PIB Growth (% anual)
    :gdp_growth => Dict(
        Date(2019,1,1) => 1.1,   # 2019: crescimento fraco
        Date(2020,1,1) => -3.9,  # 2020: recessão COVID severa
        Date(2021,1,1) => 4.6,   # 2021: recuperação forte
        Date(2022,1,1) => 2.9,   # 2022: desaceleração
        Date(2023,1,1) => 2.9,   # 2023: crescimento moderado
        Date(2024,1,1) => 2.1    # 2024: projeção conservadora
    ),
    
    # Desemprego (% da força de trabalho - PNAD)
    :unemployment => Dict(
        Date(2019,1,1) => 11.9,  # 2019: alto desemprego
        Date(2020,1,1) => 13.5,  # 2020: pico COVID
        Date(2020,7,1) => 14.7,  # Jul 2020: máximo histórico
        Date(2021,1,1) => 14.2,  # 2021: ainda elevado
        Date(2022,1,1) => 11.2,  # 2022: melhora
        Date(2023,1,1) => 8.5,   # 2023: normalização
        Date(2024,1,1) => 7.8    # 2024: nível mais baixo
    ),
    
    # Aperto de Crédito (Índice 0-100, baseado em surveys BCB)
    :credit_tightening => Dict(
        Date(2019,1,1) => 45,    # 2019: neutro
        Date(2020,3,1) => 85,    # Mar 2020: aperto severo COVID
        Date(2020,7,1) => 70,    # Jul 2020: melhora gradual
        Date(2021,1,1) => 35,    # 2021: afrouxamento
        Date(2021,7,1) => 55,    # Jul 2021: endurecimento
        Date(2022,1,1) => 75,    # 2022: aperto monetário
        Date(2023,1,1) => 60,    # 2023: ainda restritivo
        Date(2024,1,1) => 40     # 2024: normalização gradual
    ),
    
    # Inflação (IPCA % anual)
    :inflation => Dict(
        Date(2019,1,1) => 4.3,   # 2019: controlada
        Date(2020,1,1) => 4.5,   # 2020: início da pressão
        Date(2021,1,1) => 10.1,  # 2021: disparada pós-COVID
        Date(2022,1,1) => 11.9,  # 2022: pico inflacionário
        Date(2023,1,1) => 4.6,   # 2023: convergência para meta
        Date(2024,1,1) => 3.8    # 2024: dentro da meta
    )
)

function get_macro_indicator(date::Date, indicator::Symbol)
    conditions = macro_conditions[indicator]
    closest_value = 0.0
    min_diff = typemax(Int)
    
    for (ref_date, value) in conditions
        diff = abs((date - ref_date).value)
        if diff < min_diff
            min_diff = diff
            closest_value = value
        end
    end
    
    return closest_value
end

function get_selic_rate(date::Date)::Float64
    """
    Interpola a taxa Selic para uma data específica usando os dados do BCB
    """
    closest_rate = 8.0  # Default fallback
    min_diff = typemax(Int)
    
    for (ref_date, rate) in bcb_rates
        diff = abs((date - ref_date).value)
        if diff < min_diff
            min_diff = diff
            closest_rate = rate
        end
    end
    
    return closest_rate
end

println("\n⏳ Gerando $n_loans empréstimos...")

# 1. DATAS DE ORIGINAÇÃO
start_date = Date(2019, 1, 1)
end_date = Date(2024, 1, 1)
date_range = (end_date - start_date).value

orig_dates = Date[]
for _ in 1:n_loans
    # Concentração em 2020-2021 devido a juros baixos
    if rand() < 0.35  # 35% durante período de juros baixos (2020-2021)
        low_rate_start = Date(2020, 3, 1)
        low_rate_end = Date(2021, 12, 31)
        low_rate_range = (low_rate_end - low_rate_start).value
        date = low_rate_start + Day(rand(1:low_rate_range))
    else  # 65% distribuído normalmente
        date = start_date + Day(rand(1:date_range))
    end
    push!(orig_dates, date)
end

println("  ✓ Datas de originação: $(minimum(orig_dates)) a $(maximum(orig_dates))")

# 2. TAXAS DE JUROS baseadas no BCB Focus (calculadas após correlações)
rates = Float64[]

# 3. VALORES DOS EMPRÉSTIMOS (realidade brasileira) - CORRIGIDO
amounts = Float64[]
for _ in 1:n_loans
    # Empréstimos brasileiros: distribuição log-normal ajustada para rendas menores
    # Mediana ~R$ 8k baseado em proporção realística da renda (1-2x renda mensal)
    log_amount = log(8000) + 1.1 * randn()
    amount = exp(log_amount)
    push!(amounts, max(500, min(80000, amount)))  # R$500-R$80k range mais realístico
end

println("  ✓ Valores: R\$$(round(Int,minimum(amounts)/1000))k a R\$$(round(Int,maximum(amounts)/1000))k")

# 4. PRAZOS DOS EMPRÉSTIMOS (padrão brasileiro)
loan_terms = Int[]
for _ in 1:n_loans
    r = rand()
    if r < 0.20      # 20% - 12 meses
        term = 12
    elseif r < 0.40  # 20% - 18 meses
        term = 18
    elseif r < 0.60  # 20% - 24 meses
        term = 24
    elseif r < 0.80  # 20% - 36 meses
        term = 36
    else             # 20% - 48-60 meses
        term = rand([48, 60])
    end
    push!(loan_terms, term)
end

println("  ✓ Prazos: $(minimum(loan_terms)) a $(maximum(loan_terms)) meses")

# 5. GERAÇÃO CORRELACIONADA DE VARIÁVEIS SOCIOECONÔMICAS
using LinearAlgebra

println("  ✓ Implementando matriz de correlação realista entre variáveis...")

# Matriz de correlação baseada em evidências empíricas brasileiras
correlation_matrix = [
    1.0  0.65 -0.45  0.30;  # income vs [income, score, rate, dti]
    0.65 1.0  -0.55  0.20;  # score vs [income, score, rate, dti]  
   -0.45 -0.55 1.0  -0.25;  # rate vs [income, score, rate, dti]
    0.30  0.20 -0.25 1.0    # dti vs [income, score, rate, dti]
]

# Verificar se é positiva definida
@assert isposdef(correlation_matrix) "Matriz de correlação deve ser positiva definida"

# Decomposição de Cholesky para geração multivariada
L = cholesky(correlation_matrix).L

# Médias e desvios das variáveis (transformadas para Normal)
# CORRIGIDO: Rendas realísticas brasileiras baseadas em IBGE PNAD 2023
mu_log_income = log(6000)    # Mediana ~R$ 6k (realística para Brasil)
sigma_log_income = 1.2       # Maior dispersão para capturar desigualdade brasileira
mu_score_norm = 0.0  # Score normalizado [-2, 2]
sigma_score_norm = 1.0
mu_rate_norm = 0.0   # Taxa normalizada
sigma_rate_norm = 1.0
mu_dti_norm = 0.0    # DTI normalizado
sigma_dti_norm = 1.0

borrower_incomes = Float64[]
credit_scores = Int[]
rate_adjustments = Float64[]  # Ajustes nas taxas baseados em correlação
dti_factors = Float64[]       # Fatores DTI correlacionados

println("  ✓ Gerando $(n_loans) observações correlacionadas...")

for i in 1:n_loans
    # Gerar 4 variáveis normais independentes
    z = randn(4)
    
    # Aplicar correlação via Cholesky
    correlated_z = L * z
    
    # 1. RENDA (log-normal correlacionada) - CORRIGIDA para realidade brasileira
    log_income = mu_log_income + sigma_log_income * correlated_z[1]
    income = exp(log_income)
    # Limites realísticos: salário mínimo (~R$ 1.500) até classe alta (~R$ 50k)
    income = max(1500, min(50000, income))
    push!(borrower_incomes, income)
    
    # 2. SCORE DE CRÉDITO (correlacionado com renda)
    score_norm = mu_score_norm + sigma_score_norm * correlated_z[2]
    # Mapear de Normal para distribuição Serasa brasileira
    score_percentile = cdf(Normal(), score_norm)
    
    if score_percentile < 0.28
        score = round(Int, 100 + 300 * (score_percentile / 0.28))
    elseif score_percentile < 0.50
        score = round(Int, 401 + 199 * ((score_percentile - 0.28) / 0.22))
    elseif score_percentile < 0.73
        score = round(Int, 601 + 199 * ((score_percentile - 0.50) / 0.23))
    else
        score = round(Int, 801 + 199 * ((score_percentile - 0.73) / 0.27))
    end
    push!(credit_scores, clamp(score, 0, 1000))
    
    # 3. AJUSTE DE TAXA (correlacionado negativamente com renda e score)
    rate_adjustment = correlated_z[3] * 5.0  # ±5% de ajuste máximo
    push!(rate_adjustments, rate_adjustment)
    
    # 4. FATOR DTI (correlacionado com outras variáveis)
    dti_factor = 1.0 + 0.3 * correlated_z[4]  # Multiplicador para DTI
    push!(dti_factors, max(0.5, min(2.0, dti_factor)))
end

println("  ✓ Rendas correlacionadas: R\$$(round(Int,minimum(borrower_incomes)/1000))k a R\$$(round(Int,maximum(borrower_incomes)/1000))k")
println("  ✓ Scores correlacionados: $(minimum(credit_scores)) a $(maximum(credit_scores))")

# Verificar correlações empíricas geradas
log_incomes = log.(borrower_incomes)
norm_scores = (credit_scores .- 500) ./ 200  # Normalizar scores
println("  ✓ Correlação Renda-Score: $(round(cor(log_incomes, norm_scores), digits=3))")

# CALCULAR TAXAS COM CORRELAÇÕES
for (i, date) in enumerate(orig_dates)
    # Encontrar Selic mais próxima
    closest_rate = 8.0  # Default
    min_diff = typemax(Int)
    
    for (ref_date, rate) in bcb_rates
        diff = abs((date - ref_date).value)
        if diff < min_diff
            min_diff = diff
            closest_rate = rate
        end
    end
    
    # Spreads bancários brasileiros sobre Selic:
    # Pessoa Física: +15-45% a.a. típico
    base_spread = 18.0 + 25.0 * rand()^1.3  # Spread de 18-43% típico no Brasil
    
    # APLICAR AJUSTE CORRELACIONADO (renda alta = taxa menor)
    correlated_rate = closest_rate + base_spread + rate_adjustments[i]
    push!(rates, max(5.0, min(60.0, correlated_rate)))  # Limites realísticos
end

println("  ✓ Taxas correlacionadas: $(round(minimum(rates), digits=2))% a $(round(maximum(rates), digits=2))%")

# CALCULAR SPREAD SOBRE SELIC para capturar sensibilidade aos juros
println("  ✓ Calculando spread sobre Selic para sensibilidade aos juros...")
spread_over_selic = Float64[]

for (i, date) in enumerate(orig_dates)
    selic_at_origination = get_selic_rate(date)
    spread = rates[i] - selic_at_origination
    push!(spread_over_selic, spread)
end

println("  ✓ Spread sobre Selic: $(round(minimum(spread_over_selic), digits=2))% a $(round(maximum(spread_over_selic), digits=2))%")

# 6. TIPOS DE EMPRÉSTIMO (mercado brasileiro)
loan_types = String[]
for _ in 1:n_loans
    r = rand()
    if r < 0.45
        push!(loan_types, "Crédito Pessoal")
    elseif r < 0.70
        push!(loan_types, "Cartão de Crédito")
    elseif r < 0.85
        push!(loan_types, "Cheque Especial")
    else
        push!(loan_types, "CDC Veículo")
    end
end

# 8. GARANTIA
collateral_types = String[]
for i in 1:n_loans
    if loan_types[i] == "CDC Veículo"
        push!(collateral_types, "Com Garantia")
    elseif loan_types[i] == "Crédito Pessoal" && rand() < 0.15
        push!(collateral_types, "Com Garantia")
    else
        push!(collateral_types, "Sem Garantia")
    end
end

# 9. ESTADOS BRASILEIROS (distribuição IBGE)
state_weights = Dict(
    "SP" => 0.220, "RJ" => 0.081, "MG" => 0.100, "BA" => 0.070,
    "PR" => 0.054, "RS" => 0.054, "PE" => 0.046, "CE" => 0.043,
    "PA" => 0.041, "SC" => 0.036, "GO" => 0.034, "MA" => 0.033,
    "PB" => 0.019, "ES" => 0.019, "PI" => 0.016, "AL" => 0.016,
    "RN" => 0.017, "MT" => 0.016, "MS" => 0.013, "DF" => 0.014,
    "SE" => 0.011, "RO" => 0.009, "TO" => 0.007, "AC" => 0.004,
    "AM" => 0.020, "RR" => 0.003, "AP" => 0.004
)

borrower_states = String[]
state_list = collect(keys(state_weights))
weights = collect(values(state_weights))
for _ in 1:n_loans
    idx = findfirst(cumsum(weights) .>= rand())
    push!(borrower_states, state_list[idx])
end

# 10. FINALIDADES (padrão brasileiro)
loan_purposes = String[]
for _ in 1:n_loans
    r = rand()
    if r < 0.25
        push!(loan_purposes, "Pagamento de Dívidas")
    elseif r < 0.45
        push!(loan_purposes, "Compras Diversas")
    elseif r < 0.60
        push!(loan_purposes, "Emergência")
    elseif r < 0.75
        push!(loan_purposes, "Capital de Giro")
    else
        push!(loan_purposes, "Outros")
    end
end

println("  ✓ Características dos empréstimos definidas")

# 11. MODELAGEM DE PRÉ-PAGAMENTOS COM FATORES COMPORTAMENTAIS
println("\n🔄 Modelando comportamento de pré-pagamento brasileiro com fatores comportamentais...")

current_date = Date(2024, 7, 20)
current_avg_rate = 11.25  # Selic atual (2024)

prepay_dates = Vector{Union{Date, Missing}}(undef, n_loans)

for i in 1:n_loans
    # Calcular tempo até vencimento vs tempo até data atual
    months_to_maturity = loan_terms[i]  # Prazo do empréstimo
    months_since_orig = max(1, (current_date - orig_dates[i]).value ÷ 30)
    
    # Usar o menor: tempo até vencimento ou tempo até hoje
    months_available = min(months_to_maturity, months_since_orig)
    
    if months_available < 3  # Mínimo de seasoning
        prepay_dates[i] = missing
        continue
    end
    
    # COMPONENTES COMPORTAMENTAIS BRASILEIROS:
    
    # 1. INCENTIVO DE TAXA (mais sensível que mercados desenvolvidos)
    rate_diff = rates[i] - current_avg_rate
    rate_incentive = max(0, rate_diff / 8.0)  # Brasileiros mais sensíveis a juros
    
    # 2. SEASONING RAMP com SUNK COST FALLACY
    # Brasileiros têm dificuldade de abandonar empréstimos já pagos parcialmente
    if months_available <= 3
        seasoning_mult = 0.3
    elseif months_available <= 12
        seasoning_mult = months_available / 12.0
        # Sunk cost: quanto mais pagou, menor propensão a quitar antecipadamente
        sunk_cost_penalty = 1.0 - 0.3 * (months_available / 12.0)^2
        seasoning_mult *= sunk_cost_penalty
    else
        seasoning_mult = 1.0 + 0.2 * min(1.0, (months_available - 12) / 12.0)
        # Sunk cost mais forte após 12 meses
        sunk_cost_penalty = 1.0 - 0.5 * min(1.0, (months_available - 12) / 24.0)
        seasoning_mult *= sunk_cost_penalty
    end
    
    # 3. DESCONTO HIPERBÓLICO (preferência pelo presente)
    # Brasileiros descontam o futuro mais agressivamente
    time_to_maturity = months_to_maturity - months_available
    if time_to_maturity > 0
        # Desconto hiperbólico: β = 0.7 (literatura brasileira)
        hyperbolic_discount = 0.7 / (1 + 0.1 * time_to_maturity)
        # Quanto maior o desconto, maior urgência para quitar
        urgency_mult = 1.0 + (1.0 - hyperbolic_discount) * 0.8
    else
        urgency_mult = 1.5  # Muito próximo do vencimento
    end
    
    # 4. CREDIT QUALITY com VIÉS DE CONFIANÇA
    credit_mult = (credit_scores[i] - 200) / 600.0
    credit_mult = max(0.4, min(1.8, credit_mult))
    # Scores altos geram overconfidence = mais pré-pagamento
    if credit_scores[i] > 700
        overconfidence_mult = 1.0 + 0.3 * ((credit_scores[i] - 700) / 300.0)
        credit_mult *= overconfidence_mult
    end
    
    # 5. TIPO DE EMPRÉSTIMO (efeito brasileiro)
    type_mult = if loan_types[i] == "Cartão de Crédito" 
        1.25 # Volátil, mas com alta rolagem de dívida
    elseif loan_types[i] == "Cheque Especial"
        1.15 # Levemente mais propenso a quitação
    elseif loan_types[i] == "CDC Veículo"
        0.6  # Mais estável - garantia
    else  # Crédito Pessoal
        1.0  # Baseline
    end
    
    # 6. DEBT-TO-INCOME com STRESS COMPORTAMENTAL
    monthly_payment = (amounts[i] * (rates[i]/100/12)) / 
                     (1 - (1 + rates[i]/100/12)^(-loan_terms[i]))
    dti = (monthly_payment * 12) / borrower_incomes[i] * dti_factors[i]  # Usar fator correlacionado
    
    # DTI alto causa stress = comportamento irracional
    if dti > 0.4
        stress_mult = 1.0 + 2.0 * (dti - 0.4)^2  # Stress exponencial
    else
        stress_mult = 1.0
    end
    dti_mult = stress_mult
    
    # 7. GARANTIA (efeito brasileiro)
    collateral_mult = collateral_types[i] == "Com Garantia" ? 0.6 : 1.0
    
    # 8. INFLUÊNCIA SOCIAL REGIONAL (Brasil)
    # Regiões com maior dinamismo = maior influência social para quitação
    base_region_mult = if borrower_states[i] in ["SP", "RJ", "MG"]
        1.1  # Sudeste mais dinâmico
    elseif borrower_states[i] in ["RS", "PR", "SC"]
        1.05  # Sul
    elseif borrower_states[i] in ["GO", "DF", "MT", "MS"]
        0.95  # Centro-Oeste
    elseif borrower_states[i] in ["BA", "PE", "CE"]
        0.90  # Nordeste
    else
        0.85  # Norte
    end
    
    # EFEITO DE REDE: Quanto maior a renda regional, maior pressão social
    regional_income_effect = 1.0 + 0.2 * min(1.0, (borrower_incomes[i] - 30000) / 50000)
    region_mult = base_region_mult * regional_income_effect
    
    # 9. CICLO ECONÔMICO COMPORTAMENTAL
    orig_date = orig_dates[i]
    unemployment_rate = get_macro_indicator(orig_date, :unemployment)
    gdp_growth = get_macro_indicator(orig_date, :gdp_growth)
    credit_tightening = get_macro_indicator(orig_date, :credit_tightening)
    
    # Desemprego alto = medo de perder emprego = quitar antecipadamente
    unemployment_mult = 1.0 + max(0, (unemployment_rate - 8.0) / 10.0)
    
    # Crescimento baixo = pessimismo = quitar antecipadamente  
    gdp_mult = 1.0 + max(0, (2.0 - gdp_growth) / 5.0)
    
    # Aperto de crédito = dificuldade refinanciamento = manter empréstimo
    credit_mult_macro = 1.0 - max(0, (credit_tightening - 50) / 100.0)
    
    macro_mult = unemployment_mult * gdp_mult * credit_mult_macro
    
    # 10. DIREITO CDC Art. 52 (sem penalidade = mais pré-pagamento)
    cdc_mult = 1.08 # CDC facilita pré-pagamento (efeito limitado)
    
    # APR ANUAL brasileiro com fatores comportamentais
    base_apr = 0.02  # 2% base
    total_apr = base_apr * (1 + rate_incentive) * seasoning_mult * urgency_mult *
                credit_mult * type_mult * dti_mult * collateral_mult * 
                region_mult * macro_mult * cdc_mult
    
    # Limites realísticos brasileiros
    total_apr = max(0.03, min(0.60, total_apr))
    
    # Converter para SMM
    smm = 1 - (1 - total_apr)^(1/12)
    
    # Simular mês de pré-pagamento - LIMITADO PELO PRAZO DO EMPRÉSTIMO
    prepaid = false
    for month in 1:months_available
        # Efeito sazonal brasileiro com viés comportamental
        current_month = (Dates.month(orig_dates[i]) + month - 1) % 12 + 1
        seasonal_mult = if current_month in [11, 12, 1]  # Nov-Jan (13º salário)
            1.25 # Moderado - dinheiro "extra" calibrado
        elseif current_month in [6, 7]  # Jun-Jul (férias, 2º férias)
            1.15  # Leve impulso de férias
        elseif current_month in [3, 4]  # Mar-Abr (IR, planejamento)
            1.1  # Leve planejamento financeiro
        else
            1.0
        end
        
        # MOMENTUM COMPORTAMENTAL: Primeiros meses têm "luna de mel"
        if month <= 6
            momentum_mult = 0.7  # Honeymoon period
        else
            momentum_mult = 1.0 + 0.1 * min(1.0, (month - 6) / 12.0)  # Momentum crescente
        end
        
        if rand() < smm * seasonal_mult * momentum_mult
            prepay_dates[i] = orig_dates[i] + Month(month)
            prepaid = true
            break
        end
    end
    
    if !prepaid
        prepay_dates[i] = missing
    end
end

println("  ✓ Pré-pagamentos modelados com fatores comportamentais brasileiros")
println("    - Sunk cost fallacy implementado")
println("    - Desconto hiperbólico (β=0.7)")
println("    - Influência social regional")
println("    - Stress comportamental por DTI")
println("    - Viés de overconfidence")
println("    - Efeitos de ciclos econômicos")

# 12. CONSTRUIR DATASET FINAL
println("\n📋 Construindo dataset brasileiro final...")

loan_data = DataFrame(
    loan_id = ["BR$(lpad(i, 8, '0'))" for i in 1:n_loans],
    origination_date = orig_dates,
    maturity_date = orig_dates .+ Month.(loan_terms),
    interest_rate = round.(rates, digits=2),
    spread_over_selic = round.(spread_over_selic, digits=2),  # Nova variável para sensibilidade aos juros
    loan_amount = round.(amounts, digits=0),
    loan_term = loan_terms,
    credit_score = credit_scores,
    borrower_income = round.(borrower_incomes, digits=0),
    loan_type = loan_types,
    collateral_type = collateral_types,
    prepayment_date = prepay_dates,
    
    # Covariates brasileiras
    borrower_state = borrower_states,
    loan_purpose = loan_purposes,
    outstanding_balance = round.(amounts .* rand(0.70:0.01:1.0, n_loans), digits=0),
    payment_amount = round.([
        (amounts[i] * (rates[i]/100/12)) / 
        (1 - (1 + rates[i]/100/12)^(-loan_terms[i]))
        for i in 1:n_loans
    ], digits=2)
)

# Salvar dataset
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
output_file = joinpath(data_dir, "brazilian_loans_$(timestamp).csv")
CSV.write(output_file, loan_data)

println("✅ Dataset brasileiro salvo: $(basename(output_file))")

# ANÁLISE FINAL
prepaid_count = sum(.!ismissing.(loan_data.prepayment_date))
prepay_rate = prepaid_count / n_loans

println("\n📊 ESTATÍSTICAS FINAIS DO DATASET BRASILEIRO:")
println("  🇧🇷 Total empréstimos: $(nrow(loan_data))")
println("  📅 Período: $(minimum(orig_dates)) a $(maximum(orig_dates))")
println("  💰 Valor médio: R\$$(round(Int, mean(loan_data.loan_amount)/1000))k")
println("  📈 Taxa média: $(round(mean(loan_data.interest_rate), digits=1))%")
println("  📊 Spread sobre Selic médio: $(round(mean(loan_data.spread_over_selic), digits=1))%")
println("  ⏳ Prazo médio: $(round(Int, mean(loan_data.loan_term))) meses")
println("  💼 Renda média: R\$$(round(Int, mean(loan_data.borrower_income)/1000))k")
println("  🎯 Score médio: $(round(Int, mean(loan_data.credit_score)))")
println("  🔄 Taxa pré-pagamento: $(round(100*prepay_rate, digits=1))%")

# Análise por tipo de empréstimo
println("\n📈 Análise por tipo de empréstimo:")
type_stats = combine(groupby(loan_data, :loan_type)) do df
    prep_mask = .!ismissing.(df.prepayment_date)
    (
        count = nrow(df),
        avg_rate = round(mean(df.interest_rate), digits=1),
        avg_amount = round(Int, mean(df.loan_amount)/1000),
        prepay_pct = round(100 * mean(prep_mask), digits=1)
    )
end

show(type_stats, allrows=true)

# Análise regional
println("\n\n🗺️ Análise por região (Top 5):")
regional_stats = combine(groupby(loan_data, :borrower_state)) do df
    prep_mask = .!ismissing.(df.prepayment_date)
    (
        count = nrow(df),
        avg_income = round(Int, mean(df.borrower_income)/1000),
        prepay_pct = round(100 * mean(prep_mask), digits=1)
    )
end

# Ordenar por count e mostrar top 5
regional_stats_sorted = sort(regional_stats, :count, rev=true)[1:5, :]
show(regional_stats_sorted, allrows=true)

println("\n")
println("🎉 DATASET BRASILEIRO COMPLETO CRIADO!")
println("\n📚 METODOLOGIA BASEADA EM DADOS BRASILEIROS:")
println("  • Taxas: BCB Focus - relatórios Selic e spread bancário")
println("  • Rendas: IBGE PNAD Contínua - distribuição real brasileira")  
println("  • Scores: Serasa (0-1000) - distribuição atualizada")
println("  • Geografia: IBGE - proporção populacional por estado")
println("  • Pré-pagamento: CDC Art. 52 + sazonalidade brasileira (13º, férias)")
println("  • Tipos: Mercado brasileiro (crédito pessoal, cartão, CDC)")

println("\n🇧🇷 CARACTERÍSTICAS ÚNICAS BRASILEIRAS:")
println("  • Sem penalidade de pré-pagamento (CDC)")
println("  • Sazonalidade (13º salário, férias)")
println("  • Scores Serasa (0-1000)")
println("  • Spreads bancários altos")
println("  • Efeitos regionais")

println("\n▶️  PRÓXIMO PASSO:")
println("   julia --project=. scripts/example_analysis.jl")
println("   (Use os novos dados brasileiros realísticos)")

println("\n" * repeat("=", 60))