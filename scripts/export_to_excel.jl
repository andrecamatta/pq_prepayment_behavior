#!/usr/bin/env julia

"""
Converte dataset CSV de empréstimos brasileiros para formato Excel (.xlsx)
"""

using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using XLSX
using Dates
using Statistics

println("📊 EXPORTANDO DATASET BRASILEIRO PARA EXCEL")
println(repeat("=", 50))

# Encontrar o arquivo CSV mais recente
data_dir = "data/official_based_data"
csv_files = filter(f -> endswith(f, ".csv") && startswith(f, "brazilian_loans"), readdir(data_dir))

if isempty(csv_files)
    error("❌ Nenhum arquivo CSV de empréstimos brasileiros encontrado em $data_dir")
end

# Pegar o arquivo mais recente
latest_file = sort(csv_files)[end]
csv_path = joinpath(data_dir, latest_file)

println("📄 Carregando: $latest_file")

# Carregar dados
loan_data = CSV.read(csv_path, DataFrame)

println("  ✓ $(nrow(loan_data)) empréstimos carregados")

# Preparar nome do arquivo Excel
excel_filename = replace(latest_file, ".csv" => ".xlsx")
excel_path = joinpath(data_dir, excel_filename)

println("💾 Exportando para: $excel_filename")

# Criar workbook Excel com múltiplas abas
XLSX.openxlsx(excel_path, mode="w") do xf
    
    # Aba 1: Dados completos
    sheet_data = xf[1]
    XLSX.rename!(sheet_data, "Dados_Emprestimos")
    
    # Escrever headers
    headers = names(loan_data)
    for (col, header) in enumerate(headers)
        sheet_data[1, col] = header
    end
    
    # Escrever dados
    for row in 1:nrow(loan_data)
        for (col, value) in enumerate(eachrow(loan_data)[row])
            if ismissing(value)
                sheet_data[row + 1, col] = ""
            elseif value isa Date
                sheet_data[row + 1, col] = value
            else
                sheet_data[row + 1, col] = value
            end
        end
    end
    
    println("  ✓ Aba 'Dados_Emprestimos' criada com $(nrow(loan_data)) linhas")
    
    # Aba 2: Estatísticas resumo
    sheet_stats = XLSX.addsheet!(xf, "Estatisticas_Resumo")
    
    # Estatísticas gerais
    stats_data = [
        ["Métrica", "Valor"],
        ["Total de Empréstimos", nrow(loan_data)],
        ["Valor Médio (R\$)", round(Int, mean(loan_data.loan_amount))],
        ["Taxa Média (%)", round(mean(loan_data.interest_rate), digits=2)],
        ["Prazo Médio (meses)", round(Int, mean(loan_data.loan_term))],
        ["Renda Média (R\$)", round(Int, mean(loan_data.borrower_income))],
        ["Score Médio", round(Int, mean(loan_data.credit_score))],
        ["Taxa de Pré-pagamento (%)", round(100 * mean(.!ismissing.(loan_data.prepayment_date)), digits=1)],
        [""],
        ["Distribuição por Tipo de Empréstimo", ""],
    ]
    
    # Adicionar estatísticas por tipo
    type_stats = combine(groupby(loan_data, :loan_type)) do df
        prep_mask = .!ismissing.(df.prepayment_date)
        (
            count = nrow(df),
            avg_rate = round(mean(df.interest_rate), digits=1),
            prepay_pct = round(100 * mean(prep_mask), digits=1)
        )
    end
    
    for row in eachrow(type_stats)
        push!(stats_data, [row.loan_type, "$(row.count) empréstimos"])
        push!(stats_data, ["  - Taxa média", "$(row.avg_rate)%"])
        push!(stats_data, ["  - Pré-pagamento", "$(row.prepay_pct)%"])
    end
    
    # Escrever estatísticas
    for (row, data) in enumerate(stats_data)
        for (col, value) in enumerate(data)
            sheet_stats[row, col] = value
        end
    end
    
    println("  ✓ Aba 'Estatisticas_Resumo' criada")
    
    # Aba 3: Análise regional
    sheet_regional = XLSX.addsheet!(xf, "Analise_Regional")
    
    regional_stats = combine(groupby(loan_data, :borrower_state)) do df
        prep_mask = .!ismissing.(df.prepayment_date)
        (
            count = nrow(df),
            avg_income = round(Int, mean(df.borrower_income)),
            avg_score = round(Int, mean(df.credit_score)),
            prepay_pct = round(100 * mean(prep_mask), digits=1)
        )
    end
    
    # Ordenar por count
    regional_stats = sort(regional_stats, :count, rev=true)
    
    # Headers regionais
    regional_headers = ["Estado", "Quantidade", "Renda Média (R\$)", "Score Médio", "Pré-pagamento (%)"]
    for (col, header) in enumerate(regional_headers)
        sheet_regional[1, col] = header
    end
    
    # Dados regionais
    for (row, data) in enumerate(eachrow(regional_stats))
        sheet_regional[row + 1, 1] = data.borrower_state
        sheet_regional[row + 1, 2] = data.count
        sheet_regional[row + 1, 3] = data.avg_income
        sheet_regional[row + 1, 4] = data.avg_score  
        sheet_regional[row + 1, 5] = data.prepay_pct
    end
    
    println("  ✓ Aba 'Analise_Regional' criada com $(nrow(regional_stats)) estados")
    
end

println("\n✅ EXPORTAÇÃO COMPLETA!")
println("📁 Arquivo Excel salvo: $excel_path")
println("📋 Abas criadas:")
println("  • Dados_Emprestimos: Dataset completo")
println("  • Estatisticas_Resumo: Métricas principais")
println("  • Analise_Regional: Análise por estado")

println("\n" * repeat("=", 50))