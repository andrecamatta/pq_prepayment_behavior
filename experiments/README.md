# Experiments Directory

Esta pasta contém arquivos de teste, debug e experimentos gerados durante o desenvolvimento do projeto.

## 🧪 Arquivos de Teste dos Modelos

- `test_parametric_quick.jl` - Teste rápido dos modelos paramétricos (100 empréstimos)
- `test_survival_fast.jl` - Teste com 1000 empréstimos 
- `test_survival_5000.jl` - Teste completo com 5000 empréstimos
- `test_models_simple.jl` - Teste básico inicial dos modelos

## 🔧 Arquivos de Debug

- `debug_cox.jl` - Debug específico do modelo Cox
- `debug_cindex.jl` - Debug da função C-index
- `fix_cox_model.jl` - Correções aplicadas ao modelo Cox
- `test_cox_fixed.jl` - Teste das correções do Cox
- `test_cox_solutions.jl` - Soluções testadas para o Cox

## 📊 Análises de Dados

- `check_survival_data.jl` - Verificação dos dados de sobrevivência
- `list_covariates.jl` - Lista e análise das covariáveis utilizadas

## ℹ️ Status

Todos estes arquivos foram utilizados durante o desenvolvimento e debug dos modelos. Os arquivos principais do projeto estão em:

- `scripts/` - Scripts de produção
- `src/` - Código fonte dos modelos
- `test/` - Testes unitários oficiais

**Estes arquivos podem ser removidos se não precisar reproduzir o processo de desenvolvimento.**