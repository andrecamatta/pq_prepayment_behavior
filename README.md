# Modelagem de Pré-pagamento Bancário Brasileiro

Implementação robusta de modelos de sobrevivência para análise de comportamento de pré-pagamento em empréstimos bancários brasileiros usando Julia, com **sensibilidade aos juros via spread sobre Selic** e **todos os 4 modelos convergindo corretamente**.

## 📋 Visão Geral

Este projeto implementa modelos estatísticos avançados para prever o comportamento de pré-pagamento em empréstimos bancários brasileiros, utilizando técnicas de análise de sobrevivência com **validação out-of-sample rigorosa**. 

### 🏆 **Inovações Técnicas:**
- **Sensibilidade aos Juros**: `spread_over_selic` captura incentivos de refinanciamento
- **Eliminação de Multicolinearidade**: Remoção de `interest_rate` para estabilidade numérica
- **Convergência Robusta**: Múltiplas estratégias de otimização MLE
- **Validação Rigorosa**: Comparação out-of-sample entre todos os modelos

### 📊 **Modelos Implementados (Todos Funcionando):**
- **Modelo Bernoulli-Beta** (MLE + Regularização)
- **Modelo Cox** (Partial Likelihood) 
- **Modelos Paramétricos** (Weibull, Log-Normal) com Regularização L2

## 🚀 Instalação e Configuração

### Pré-requisitos

- Julia 1.9+ instalado
- Git para controle de versão

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/andrecamatta/pq_prepayment_behavior.git
cd pq_prepayment_behavior
```

2. Ative o ambiente Julia e instale as dependências:
```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

3. Execute os testes para verificar a instalação:
```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

## 📊 Como Usar

### 1. Geração de Dados

Para criar um dataset brasileiro baseado em estatísticas oficiais (BCB, IBGE, Serasa):

```bash
julia --project=. scripts/create_brazilian_loan_data.jl
```

Este script gera dados sintéticos realistas baseados em fontes brasileiras oficiais:
- **Taxas de juros**: BCB Focus (Selic e spreads bancários)
- **Spread sobre Selic**: Captura incentivos de refinanciamento
- **Rendas**: IBGE PNAD Contínua (distribuição real brasileira)
- **Scores de crédito**: Serasa (0-1000, distribuição atualizada)
- **Geografia**: IBGE (proporção populacional por estado)
- **Comportamento**: CDC Art. 52 + sazonalidade brasileira (13º, férias)

**Taxa de pré-pagamento calibrada**: O modelo produz taxas realísticas para o mercado brasileiro após extensiva calibração de parâmetros comportamentais.

### 2. Análise Exploratória de Dados (EDA)

Para executar uma análise exploratória completa dos dados brasileiros:

```bash
julia --project=. scripts/brazilian_loan_eda.jl
```

**O que este script faz:**
- 📊 **Estatísticas descritivas** completas do dataset brasileiro
- 📈 **Visualizações** (10 gráficos salvos em `results/eda_plots/`)
- 🏦 **Análise por modalidade** (Crédito Pessoal, Cartão, CDC Veículo, Cheque Especial)
- 🌍 **Análise regional** por estado brasileiro
- 📋 **Segmentação por risco** e perfil de pré-pagamento
- 💾 **Relatório resumido** em `results/eda_plots/analise_resumo.txt`

### 3. Exportação para Excel

Para converter os dados CSV para formato Excel com múltiplas planilhas:

```bash
julia --project=. scripts/export_to_excel.jl
```

### 4. Comparação Robusta de Modelos

Para executar uma análise completa comparando todos os 4 modelos de sobrevivência:

```bash
julia --project=. scripts/survival_metrics_comparison.jl
```

**O que este script faz:**

1. **📊 Carrega os dados brasileiros** gerados no passo anterior
2. **🔄 Divide amostra aleatória** (3000 empréstimos: 70% treino / 30% teste)
3. **🏋️ Treina os 4 modelos** usando apenas dados de treino:
   - Modelo Cox (Proportional Hazards)
   - Modelo Weibull MLE com regularização L2
   - Modelo Log-Normal MLE com regularização L2
   - Modelo Bernoulli-Beta otimizado
4. **📈 Avalia performance** em dados de teste (out-of-sample)
5. **🏆 Ranking dos modelos** por critério ponderado

**Covariáveis utilizadas (7 variáveis):**
- `spread_over_selic`: Sensibilidade aos juros vs refinanciamento
- `credit_score`: Score Serasa com binning por quantis
- `loan_amount_log`: Valor do empréstimo (log-transformado)
- `loan_term`: Prazo em meses
- `dti_ratio`: Debt-to-income ratio
- `borrower_income_log`: Renda do mutuário (log-transformada)
- `has_collateral`: Dummy para presença de garantia

**Pipeline de Validação:**
- 🔄 **Split 70/30**: 2100 treino / 900 teste (sem data leakage)
- 🎯 **Treino**: Modelos ajustados apenas em dados de treino
- 📊 **Avaliação**: Métricas calculadas exclusivamente em dados de teste
- ⚖️ **Regularização L2**: Comparação justa entre modelos MLE (λ=0.01)

**Métricas Out-of-Sample:**
- **C-Index**: Capacidade de discriminação (0.5-1.0, maior = melhor)
- **Brier Score**: Acurácia das probabilidades (0-1, menor = melhor)
- **Calibration Error**: Qualidade da calibração (0-1, menor = melhor)
- **Score Final**: Média ponderada (Brier 50% + Calibração 30% + C-Index 20%)

**Saída esperada:**
```
🏆 RANKINGS POR MÉTRICA
🎯 MODELO RECOMENDADO: [Melhor modelo baseado no critério ponderado]
```


## 🏗️ Estrutura do Projeto

```
pq_prepayment_behavior/
├── src/                          # Código fonte principal
│   ├── PrepaymentModels.jl       # Módulo principal com expansão credit_score
│   ├── survival/                 # Modelos de sobrevivência
│   │   ├── CoxModels.jl         # Modelo Cox (Partial Likelihood)
│   │   ├── ParametricModels.jl  # Modelos paramétricos + Bernoulli-Beta
│   │   └── BernoulliBetaOptimized.jl # [Histórico] Integrado em ParametricModels
│   ├── data/                    # Utilitários de dados
│   │   └── DataLoader.jl        # Carregamento e pré-processamento
│   ├── analysis/                # Análises específicas
│   │   └── PrepaymentAnalysis.jl # Análise comportamental
│   └── utils/                   # Utilitários gerais
│       ├── FeatureTransformer.jl # Transformação centralizada de features
│       └── ValidationUtils.jl   # Validação de modelos
├── scripts/                     # Scripts de execução
│   ├── create_brazilian_loan_data.jl    # Geração de dados brasileiros
│   ├── brazilian_loan_eda.jl           # Análise exploratória brasileira
│   ├── export_to_excel.jl              # Exportação Excel
│   └── survival_metrics_comparison.jl   # Comparação out-of-sample
├── test/                        # Testes unitários
├── docs/                        # Documentação
└── data/                        # Dados gerados (não versionados)
```

## 🔬 Modelos Implementados

### 1. Modelo Cox (Proportional Hazards)
- **Método**: Partial Likelihood (não MLE tradicional)
- **Características**: Semi-paramétrico, baseline hazard livre
- **Uso**: Ideal para identificar fatores de risco

### 2. Modelos Paramétricos com Regularização L2
- **Weibull**: MLE + L2, flexível para diferentes formas de hazard
- **Log-Normal**: MLE + L2, adequado para hazards não-monótonos
- **Regularização**: λ=0.01 aplicada aos coeficientes (exceto intercept)

### 3. Modelo Bernoulli-Beta Otimizado
- **Método**: MLE com regularização L2 e inicialização inteligente
- **Componentes**: Probabilidade (Bernoulli) + Timing (Beta)
- **Vantagens**: Captura comportamento não-linear e timing de pré-pagamento

## 📈 Métricas de Avaliação

O projeto utiliza métricas estatísticas rigorosas para comparação de modelos:

- **C-Index** (0.5-1.0): Capacidade de ordenação dos modelos
- **Brier Score** (0-1): Acurácia das probabilidades preditas
- **Calibration Error** (0-1): Qualidade da calibração

## 🎯 Calibração do Modelo de Pré-pagamento

O projeto implementa um **modelo comportamental brasileiro** com calibração rigorosa:

### Características do Modelo
- **Fatores comportamentais**: Sunk cost fallacy, desconto hiperbólico, stress financeiro
- **Sazonalidade brasileira**: 13º salário, férias, planejamento fiscal
- **Regulamentação**: Efeitos do CDC (sem penalidade de pré-pagamento)
- **Segmentação**: Diferentes comportamentos por modalidade de crédito
- **Sensibilidade**: Resposta a diferenciais de taxa de juros

### Modalidades Implementadas
- **CDC Veículo**: Comportamento mais conservador (garantia real)
- **Crédito Pessoal**: Comportamento baseline
- **Cheque Especial**: Comportamento rotativo moderado
- **Cartão de Crédito**: Maior volatilidade, alta rolagem de dívida

## 🔧 Personalização

### Adicionando Novos Modelos

1. Implemente o modelo em `src/survival/`
2. Adicione testes em `test/`
3. Inclua no script de comparação

### Modificando Dados

- Edite `scripts/create_brazilian_loan_data.jl` para ajustar:
  - Distribuições de variáveis
  - Período de análise
  - Tamanho da amostra
  - Parâmetros comportamentais

## 📚 Dependências Principais

- `DataFrames.jl`: Manipulação de dados
- `Survival.jl`: Análise de sobrevivência (Cox models)
- `Optim.jl`: Otimização MLE para modelos paramétricos
- `StatsBase.jl`: Estatísticas e métricas de avaliação
- `CSV.jl`: Leitura/escrita CSV
- `XLSX.jl`: Exportação Excel
- `Dates.jl`: Manipulação de datas
- `SpecialFunctions.jl`: Funções especiais para distribuições

## 📝 Exemplo de Uso em Código

```julia
using PrepaymentModels

# Carregar dados
loan_data = load_official_bank_data("data/official_based_data")

# Pré-processar
processed_data = preprocess_loan_data(loan_data, 
                                    min_seasoning_months=12,
                                    max_dti_ratio=0.50,
                                    min_credit_score=500)

# Treinar modelo Cox com covariáveis otimizadas
covariates = [:spread_over_selic, :credit_score, :loan_amount_log, 
              :loan_term, :dti_ratio, :borrower_income_log, :has_collateral]
cox_model = fit_cox_model(processed_data, covariates=covariates)

# Fazer predições
predictions = predict_prepayment(cox_model, processed_data, 24)
```

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 📞 Contato

- **Autor**: André Camatta
- **GitHub**: [@andrecamatta](https://github.com/andrecamatta)
- **Repositório**: [pq_prepayment_behavior](https://github.com/andrecamatta/pq_prepayment_behavior)

## 🎯 Aplicações

Este projeto é útil para:

- **Instituições Financeiras**: Gestão de risco de crédito
- **Pesquisadores**: Análise de sobrevivência aplicada
- **Reguladores**: Compreensão do comportamento de mercado
- **Desenvolvedores**: Implementação de modelos estatísticos em Julia

---

*Desenvolvido com Julia 💻 para análise quantitativa de risco bancário*