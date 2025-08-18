"""
Módulo para carregamento de dados brasileiros baseados em estatísticas oficiais
Carrega datasets criados a partir de fontes públicas do BCB, IBGE e Serasa
"""

using CSV
using DataFrames
using Dates

struct LoanData
    loan_id::Vector{String}
    origination_date::Vector{Date}
    maturity_date::Vector{Date}
    interest_rate::Vector{Float64}
    spread_over_selic::Vector{Float64}  # Spread sobre taxa Selic para capturar sensibilidade aos juros
    loan_amount::Vector{Float64}
    loan_term::Vector{Int}
    credit_score::Vector{Int}
    borrower_income::Vector{Float64}
    loan_type::Vector{String}
    collateral_type::Vector{String}
    prepayment_date::Vector{Union{Date, Missing}}
    default_date::Vector{Union{Date, Missing}}
    covariates::DataFrame
end

"""
Carrega dados brasileiros baseados em estatísticas oficiais do BCB, IBGE e Serasa
"""
function load_official_bank_data(data_dir::String="data/official_based_data")::LoanData
    @assert isdir(data_dir) "Diretório não encontrado: $data_dir"
    
    # Encontrar arquivo de empréstimos bancários brasileiros mais recente
    csv_files = filter(f -> endswith(f, ".csv") && contains(f, "brazilian_loans"), readdir(data_dir, join=true))
    @assert !isempty(csv_files) "Nenhum arquivo de empréstimos bancários brasileiros encontrado em $data_dir"
    
    # Usar o arquivo mais recente
    latest_file = csv_files[end]  # Assume que estão ordenados por nome/data
    
    println("📊 Carregando dados oficiais: $(basename(latest_file))")
    
    df = CSV.read(latest_file, DataFrame)
    
    println("   ✅ $(nrow(df)) empréstimos carregados")
    
    return dataframe_to_loandata(df)
end

"""
Converte DataFrame para estrutura LoanData
"""
function dataframe_to_loandata(df::DataFrame)::LoanData
    # Campos obrigatórios
    loan_ids = df.loan_id
    orig_dates = df.origination_date
    maturity_dates = df.maturity_date
    rates = df.interest_rate
    spreads = df.spread_over_selic  # Nova variável para sensibilidade aos juros
    amounts = df.loan_amount
    terms = df.loan_term
    scores = df.credit_score
    incomes = df.borrower_income
    loan_types = df.loan_type
    collateral_types = df.collateral_type
    prepay_dates = df.prepayment_date
    
    # Default dates (não modelados no dataset atual)
    default_dates = fill(missing, nrow(df))
    
    # Construir covariates a partir de outras colunas (incluindo macro)
    covariates = DataFrame()
    
    # Adicionar variáveis macroeconômicas se existirem no CSV
    macro_vars = [:selic_rate_origination, :unemployment_rate_origination, 
                 :gdp_growth_origination, :inflation_rate_origination, 
                 :credit_tightening_origination, :refinancing_incentive,
                 :selic_environment_change, :macro_stress_indicator]
    
    for var in macro_vars
        if var in names(df)
            covariates[!, var] = df[!, var]
        end
    end
    
    # Adicionar covariates disponíveis
    if hasproperty(df, :borrower_state)
        covariates.borrower_state = df.borrower_state
    end
    
    if hasproperty(df, :loan_purpose)
        covariates.loan_purpose = df.loan_purpose
    end
    
    if hasproperty(df, :outstanding_balance)
        covariates.outstanding_balance = df.outstanding_balance
    end
    
    if hasproperty(df, :payment_amount)
        covariates.payment_amount = df.payment_amount
    end
    
    # Se não temos covariates, criar DataFrame vazio com número correto de linhas
    if isempty(names(covariates))
        covariates = DataFrame(dummy = ones(Int, nrow(df)))
    end
    
    return LoanData(
        loan_ids,
        orig_dates,
        maturity_dates,
        rates,
        spreads,  # spread_over_selic
        amounts,
        terms,
        scores,
        incomes,
        loan_types,
        collateral_types,
        prepay_dates,
        default_dates,
        covariates
    )
end

"""
Pré-processamento básico dos dados
"""
function preprocess_loan_data(data::LoanData; 
                             min_seasoning_months::Int=6,
                             max_dti_ratio::Float64=0.5,
                             min_credit_score::Int=300)::LoanData
    
    valid_indices = _filter_valid_loans(data, min_seasoning_months, max_dti_ratio, min_credit_score)
    
    println("   📋 Filtros aplicados:")
    println("      - Seasoning mínimo: $min_seasoning_months meses")
    println("      - DTI máximo: $(max_dti_ratio*100)%")
    println("      - Score mínimo: $min_credit_score")
    println("   ✅ $(length(valid_indices)) empréstimos válidos de $(length(data.loan_id)) total")
    
    return LoanData(
        data.loan_id[valid_indices],
        data.origination_date[valid_indices],
        data.maturity_date[valid_indices],
        data.interest_rate[valid_indices],
        data.spread_over_selic[valid_indices],
        data.loan_amount[valid_indices],
        data.loan_term[valid_indices],
        data.credit_score[valid_indices],
        data.borrower_income[valid_indices],
        data.loan_type[valid_indices],
        data.collateral_type[valid_indices],
        data.prepayment_date[valid_indices],
        data.default_date[valid_indices],
        data.covariates[valid_indices, :]
    )
end

function _filter_valid_loans(data::LoanData, min_seasoning::Int, max_dti::Float64, min_credit::Int)::Vector{Int}
    n = length(data.loan_id)
    valid = trues(n)
    
    current_date = Date(2024, 7, 20)  # Data de referência
    
    for i in 1:n
        # Seasoning mínimo
        months_since_orig = (current_date - data.origination_date[i]).value ÷ 30
        if months_since_orig < min_seasoning
            valid[i] = false
            continue
        end
        
        # Se tem pré-pagamento, verificar seasoning do pré-pagamento
        if !ismissing(data.prepayment_date[i])
            prepay_seasoning = (data.prepayment_date[i] - data.origination_date[i]).value ÷ 30
            if prepay_seasoning < min_seasoning
                valid[i] = false
                continue
            end
        end
        
        # DTI filter (aproximado pela relação loan/income)
        monthly_payment = (data.loan_amount[i] * (data.interest_rate[i]/100/12)) / 
                         (1 - (1 + data.interest_rate[i]/100/12)^(-data.loan_term[i]))
        dti_ratio = (monthly_payment * 12) / data.borrower_income[i]
        
        if dti_ratio > max_dti
            valid[i] = false
            continue
        end
        
        # Credit score filter
        if data.credit_score[i] < min_credit
            valid[i] = false
            continue
        end
    end
    
    return findall(valid)
end

"""
Estatísticas básicas do dataset
"""
function summarize_loan_data(data::LoanData)
    n = length(data.loan_id)
    prepaid_count = sum(.!ismissing.(data.prepayment_date))
    
    println("📊 Resumo do Dataset:")
    println("   🏦 Total de empréstimos: $n")
    println("   📅 Período: $(minimum(data.origination_date)) a $(maximum(data.origination_date))")
    println("   💰 Valor médio: \$$(round(Int, mean(data.loan_amount)/1000))k")
    println("   📊 Faixa de valores: \$$(round(Int, minimum(data.loan_amount)/1000))k - \$$(round(Int, maximum(data.loan_amount)/1000))k")
    println("   📈 Taxa média: $(round(mean(data.interest_rate), digits=2))%")
    println("   📊 Faixa de taxas: $(round(minimum(data.interest_rate), digits=2))% - $(round(maximum(data.interest_rate), digits=2))%")
    println("   ⏳ Prazo médio: $(round(Int, mean(data.loan_term))) meses")
    println("   💼 Renda média: \$$(round(Int, mean(data.borrower_income)/1000))k")
    println("   💳 Score médio: $(round(Int, mean(data.credit_score)))")
    println("   🔄 Pré-pagamentos: $prepaid_count ($(round(100*prepaid_count/n, digits=1))%)")
end