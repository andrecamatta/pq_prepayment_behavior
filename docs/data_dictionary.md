# Dicionário de Dados - Dataset de Empréstimos Bancários Brasileiros

## Visão Geral

Este documento descreve o dataset sintético de empréstimos bancários brasileiros criado para análise de modelos de pré-pagamento. O dataset é baseado em dados reais do Banco Central do Brasil (BCB), IBGE, Serasa e regulamentações brasileiras.

**Arquivo**: `brazilian_loans_YYYY-MM-DD_HH-MM.csv` e `.xlsx`
**Período**: Janeiro 2019 a Janeiro 2024
**Registros**: 50.000 empréstimos simulados
**Fonte**: Gerado por `scripts/create_brazilian_loan_data.jl`

## Metodologia

### Fontes de Dados Brasileiras
- **Taxas de Juros**: BCB Focus - Relatórios Selic e spread bancário
- **Rendas**: IBGE PNAD Contínua - distribuição real brasileira
- **Scores de Crédito**: Serasa (escala 0-1000) - distribuição atualizada
- **Geografia**: IBGE - proporção populacional por estado
- **Pré-pagamento**: CDC Art. 52 + sazonalidade brasileira (13º salário, férias)
- **Tipos de Empréstimo**: Mercado brasileiro (crédito pessoal, cartão, CDC)

### Características Únicas Brasileiras
- ✅ **Sem penalidade de pré-pagamento** (conforme CDC Art. 52)
- ✅ **Sazonalidade específica** (13º salário em dezembro, férias em janeiro/julho)
- ✅ **Scores Serasa** (escala 0-1000, não FICO)
- ✅ **Spreads bancários altos** (característicos do mercado brasileiro)
- ✅ **Efeitos regionais** (baseados em dados IBGE)
- ✅ **Fatores comportamentais**: sunk cost fallacy, desconto hiperbólico, overconfidence

---

## Variáveis do Dataset

### 📋 Identificação e Datas

| Variável | Tipo | Descrição | Exemplo | Observações |
|----------|------|-----------|---------|-------------|
| `loan_id` | String | Identificador único do empréstimo | "BR00000001" | Formato: BR + 8 dígitos |
| `origination_date` | Date | Data de originação do empréstimo | 2021-04-03 | Período: 2019-01-01 a 2024-01-01 |
| `maturity_date` | Date | Data de vencimento original | 2022-10-03 | origination_date + loan_term |

### 💰 Características Financeiras

| Variável | Tipo | Descrição | Range/Valores | Distribuição |
|----------|------|-----------|---------------|--------------|
| `interest_rate` | Float | Taxa de juros anual (%) | 5.0% - 60.0% | Baseada em Selic + spread bancário |
| `loan_amount` | Float | Valor do empréstimo (R$) | R$ 500 - R$ 80.000 | Log-normal, mediana ~R$ 8k |
| `loan_term` | Integer | Prazo em meses | 12, 18, 24, 36, 48, 60 | Distribuição: 20% cada |
| `payment_amount` | Float | Valor da parcela mensal (R$) | Calculado | Fórmula Price (SAC) |
| `outstanding_balance` | Float | Saldo devedor atual (R$) | 70-100% do loan_amount | Simulado para análise |

### 👤 Características do Mutuário

| Variável | Tipo | Descrição | Range/Valores | Fonte |
|----------|------|-----------|---------------|-------|
| `credit_score` | Integer | Score de crédito Serasa | 0 - 1000 | Distribuição Serasa real |
| `borrower_income` | Float | Renda mensal do mutuário (R$) | R$ 1.500 - R$ 50.000 | IBGE PNAD Contínua |
| `borrower_state` | String | Estado do mutuário | SP, RJ, MG, ... | Proporção populacional IBGE |

**Interpretação Score Serasa:**
- **0-400**: Muito baixo (28% da população)
- **401-600**: Baixo (22% da população)  
- **601-800**: Médio (23% da população)
- **801-1000**: Alto (27% da população)

### 🏦 Características do Produto

| Variável | Tipo | Descrição | Valores Possíveis | % Distribuição |
|----------|------|-----------|-------------------|----------------|
| `loan_type` | String | Tipo de empréstimo | "Crédito Pessoal"<br>"Cartão de Crédito"<br>"Cheque Especial"<br>"CDC Veículo" | 45%<br>25%<br>15%<br>15% |
| `collateral_type` | String | Tipo de garantia | "Sem Garantia"<br>"Com Garantia" | ~85%<br>~15% |
| `loan_purpose` | String | Finalidade do empréstimo | "Pagamento de Dívidas"<br>"Compras Diversas"<br>"Emergência"<br>"Capital de Giro"<br>"Outros" | 25%<br>20%<br>15%<br>15%<br>25% |

### 🔄 Pré-pagamento

| Variável | Tipo | Descrição | Valores | Observações |
|----------|------|-----------|---------|-------------|
| `prepayment_date` | Date/Missing | Data do pré-pagamento | Data ou vazio | Missing = não houve pré-pagamento |

**Fatores do Modelo de Pré-pagamento:**
1. **Incentivo de Taxa**: Diferença entre taxa do empréstimo e Selic atual
2. **Seasoning**: Tempo desde originação (ramp-up nos primeiros 12 meses)
3. **Sunk Cost Fallacy**: Relutância em abandonar empréstimos já pagos parcialmente
4. **Desconto Hiperbólico**: Preferência temporal brasileira (β=0.7)
5. **Qualidade de Crédito**: Scores altos têm overconfidence
6. **Tipo de Empréstimo**: Cartão/Cheque Especial mais voláteis
7. **DTI Stress**: Alto debt-to-income causa comportamento irracional
8. **Efeito Regional**: Regiões dinâmicas (SE/S) têm mais pré-pagamento
9. **Ciclo Econômico**: Desemprego e crescimento afetam decisões
10. **CDC Art. 52**: Facilita pré-pagamento (sem penalidade)

---

## 📊 Estatísticas do Dataset

### Estatísticas Gerais
- **Total de Empréstimos**: 50.000
- **Período**: Jan/2019 - Jan/2024
- **Valor Médio**: R$ 14.000
- **Taxa Média**: 35,4% a.a.
- **Prazo Médio**: 29 meses
- **Renda Média**: R$ 11.000/mês
- **Score Médio**: 586 pontos
- **Taxa de Pré-pagamento**: Calibrada para mercado brasileiro

### Distribuição de Rendas (Corrigida - Realística)
- **Renda Mediana**: R$ 6.062
- **P25**: R$ 2.669 (próximo ao salário mínimo x2)
- **P75**: R$ 13.507 (classe média brasileira)
- **P90**: R$ 27.870 (classe média alta)
- **Máximo**: R$ 50.000 (teto realístico)

### Por Tipo de Empréstimo

| Tipo | Quantidade | Taxa Média | Valor Médio | Características |
|------|------------|------------|-------------|-----------------|
| Crédito Pessoal | 22.690 (45%) | 35,4% | R$ 14k | Comportamento baseline |
| Cartão de Crédito | 12.296 (25%) | 35,5% | R$ 14k | Rotativo, alta volatilidade |
| Cheque Especial | 7.574 (15%) | 35,5% | R$ 14k | Rotativo moderado |
| CDC Veículo | 7.440 (15%) | 35,5% | R$ 13k | Garantia real, conservador |

### Por Região (Top 5)

| Estado | Quantidade | Renda Média | Características |
|--------|------------|-------------|-----------------|
| SP | 11.025 | R$ 11k | Região mais dinâmica |
| MG | 5.007 | R$ 11k | Economia diversificada |
| RJ | 4.039 | R$ 11k | Centro financeiro |
| BA | 3.445 | R$ 11k | Economia regional |
| RS | 2.673 | R$ 11k | Agronegócio forte |

### Sazonalidade Brasileira

O modelo incorpora padrões sazonais específicos do Brasil:
- **Novembro-Janeiro**: Efeito 13º salário e bônus de fim de ano
- **Junho-Julho**: Período de férias e planejamento financeiro
- **Março-Abril**: Restituição de IR e reorganização fiscal

---

## 📈 Correlações entre Variáveis

O dataset foi construído com correlações realísticas baseadas na literatura brasileira:

| Variáveis | Correlação | Interpretação |
|-----------|------------|---------------|
| Renda ↔ Score | +0.65 | Maior renda → melhor score |
| Renda ↔ Taxa | -0.45 | Maior renda → menor taxa |
| Score ↔ Taxa | -0.55 | Melhor score → menor taxa |
| Renda ↔ DTI | +0.30 | Maior renda → pode ter maior DTI |

---

## 🔧 Campos Calculados

### Payment Amount (Parcela Mensal)
```julia
monthly_rate = interest_rate / 100 / 12
payment_amount = (loan_amount * monthly_rate) / (1 - (1 + monthly_rate)^(-loan_term))
```

### Debt-to-Income Ratio (DTI)
```julia
dti = (payment_amount * 12) / borrower_income
```

### Single Monthly Mortality (SMM)
```julia
# Conversão de APR anual para SMM mensal
smm = 1 - (1 - annual_prepayment_rate)^(1/12)
```

---

## 🚨 Limitações e Considerações

### Dados Sintéticos
- Dataset é **sintético/simulado**, não contém dados reais de clientes
- Baseado em distribuições e correlações da literatura brasileira
- Adequado para **pesquisa acadêmica** e **desenvolvimento de modelos**

### Período de Análise
- Dados limitados ao período até **janeiro/2024**
- Empréstimos originados entre **2019-2024**
- Pré-pagamentos observados até **julho/2024**

### Pressupostos do Modelo
- **Taxa fixa** durante todo o prazo do empréstimo
- **Sem refinanciamento** (apenas pré-pagamento total)
- **Sem inadimplência** (foco apenas em pré-pagamento)
- **CDC Art. 52** aplicável (sem penalidade)

---

## 📁 Arquivos Relacionados

### Scripts
- `scripts/create_brazilian_loan_data.jl` - Geração do dataset
- `scripts/brazilian_loan_eda.jl` - Análise exploratória completa
- `scripts/export_to_excel.jl` - Exportação para Excel
- `scripts/survival_metrics_comparison.jl` - Comparação de modelos

### Dados
- `data/official_based_data/brazilian_loans_*.csv` - Dataset principal
- `data/official_based_data/brazilian_loans_*.xlsx` - Versão Excel com múltiplas abas

### Documentação
- `docs/data_dictionary.md` - Este documento
- `CLAUDE.md` - Instruções do projeto

---

## 📞 Contato e Suporte

Para dúvidas sobre o dataset ou metodologia, consulte:
- Código fonte em `scripts/create_brazilian_loan_data.jl`
- Documentação técnica nos comentários do código
- Referências das fontes de dados (BCB, IBGE, Serasa)

---

*Dataset gerado automaticamente em: `r Dates.now()`*  
*Versão do dicionário: 1.0*  
*Compatível com Julia 1.x e pacotes listados em Project.toml*