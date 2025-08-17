# Experiments Archive

Este diretório continha scripts experimentais de desenvolvimento que foram removidos durante a limpeza do projeto.

## Scripts Removidos (Histórico)
- `check_survival_data.jl` - Verificação de dados de sobrevivência
- `debug_cindex.jl` - Debug do C-index
- `debug_cox.jl` - Debug do modelo Cox
- `fix_cox_model.jl` - Correções do modelo Cox
- `list_covariates.jl` - Listagem de covariáveis
- `test_cox_fixed.jl` - Teste do Cox corrigido
- `test_cox_solutions.jl` - Soluções para Cox
- `test_models_simple.jl` - Testes de modelos simples
- `test_parametric_quick.jl` - Testes paramétricos rápidos
- `test_survival_5000.jl` - Teste com 5000 observações
- `test_survival_fast.jl` - Testes rápidos de sobrevivência

## Pipeline Principal
Para execução do projeto, use os scripts principais:
1. `julia --project=. scripts/create_brazilian_loan_data.jl`
2. `julia --project=. scripts/survival_metrics_comparison.jl`