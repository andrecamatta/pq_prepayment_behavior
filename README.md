# Modelagem de Pré-pagamento Bancário

Implementação de modelos de sobrevivência para análise de comportamento de pré-pagamento em empréstimos bancários usando Julia.

## 📋 Visão Geral

Este projeto implementa modelos estatísticos avançados para prever o comportamento de pré-pagamento em empréstimos bancários, utilizando técnicas de análise de sobrevivência. Os modelos incluem:

- **Modelo Cox** (Proportional Hazards)
- **Modelos Paramétricos** (Weibull, Log-Normal)
- **Modelo Bernoulli-Beta Otimizado**

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

Este script gera dados sintéticos realistas baseados em:
- Taxas históricas do PMMS (Freddie Mac)
- Índices de preços habitacionais (FHFA)
- Distribuições de crédito brasileiras

### 2. Exportação para Excel

Para converter os dados CSV para formato Excel com múltiplas planilhas:

```bash
julia --project=. scripts/export_to_excel.jl
```

### 3. Comparação de Modelos

Para executar uma análise completa comparando todos os modelos usando métricas avançadas:

```bash
julia --project=. scripts/survival_metrics_comparison.jl
```

Este script implementa e compara:
- **C-Index**: Concordância entre predições e eventos
- **Brier Score**: Erro quadrático médio das probabilidades
- **Calibration Error**: Diferença entre predições e observações

## 🏗️ Estrutura do Projeto

```
pq_prepayment_behavior/
├── src/                          # Código fonte principal
│   ├── PrepaymentModels.jl       # Módulo principal
│   ├── survival/                 # Modelos de sobrevivência
│   │   ├── CoxModels.jl         # Modelo Cox
│   │   ├── ParametricModels.jl  # Modelos paramétricos
│   │   └── BernoulliBetaOptimized.jl # Modelo Bernoulli-Beta
│   ├── data/                    # Utilitários de dados
│   │   └── DataLoader.jl        # Carregamento e pré-processamento
│   ├── analysis/                # Análises específicas
│   │   └── PrepaymentAnalysis.jl # Análise comportamental
│   └── utils/                   # Utilitários gerais
│       └── ValidationUtils.jl   # Validação de modelos
├── scripts/                     # Scripts de execução
│   ├── create_brazilian_loan_data.jl    # Geração de dados
│   ├── export_to_excel.jl              # Exportação Excel
│   └── survival_metrics_comparison.jl   # Comparação de modelos
├── test/                        # Testes unitários
├── docs/                        # Documentação
└── data/                        # Dados gerados (não versionados)
```

## 🔬 Modelos Implementados

### 1. Modelo Cox (Proportional Hazards)
- Modelo semi-paramétrico clássico
- Não assume forma específica para o hazard baseline
- Ideal para identificar fatores de risco

### 2. Modelos Paramétricos
- **Weibull**: Flexível para diferentes formas de hazard
- **Log-Normal**: Adequado para hazards não-monótonos

### 3. Modelo Bernoulli-Beta Otimizado
- Implementação otimizada para dados de pré-pagamento
- Incorpora características específicas do comportamento bancário

## 📈 Métricas de Avaliação

O projeto utiliza métricas estatísticas rigorosas para comparação de modelos:

- **C-Index** (0.5-1.0): Capacidade de ordenação dos modelos
- **Brier Score** (0-1): Acurácia das probabilidades preditas
- **Calibration Error** (0-1): Qualidade da calibração

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

## 📚 Dependências Principais

- `DataFrames.jl`: Manipulação de dados
- `Distributions.jl`: Distribuições estatísticas
- `StatsBase.jl`: Estatísticas básicas
- `CSV.jl`: Leitura/escrita CSV
- `XLSX.jl`: Exportação Excel
- `Dates.jl`: Manipulação de datas

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

# Treinar modelo Cox
covariates = [:interest_rate, :credit_score]
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