> Mesmo modelo. Mesmos parâmetros. A única variável foi a qualidade do dado.
> **Qualidade de dados não é detalhe técnico — é resultado de negócio.**

---

## 🔍 Por que os dados são diferentes?

O dado sujo (Bronze) contém problemas reais identificados na camada de qualidade:

- **144k corridas** com `total_amount` ≤ 0 (estornos/erros) → distorce a média para baixo
- **428k corridas** com `passenger_count` nulo → campo crítico ausente
- **6.181 corridas** com dropoff antes do pickup → fisicamente impossível
- **104 registros** vazados de fora de jan–mai/2023 → contaminam a janela temporal

O dado limpo (Gold) passou por **11 regras de qualidade** mapeadas às **6 dimensões do
DAMA-DMBOK** (completeness, validity, accuracy, consistency, timeliness, uniqueness),
com os reprovados isolados em quarentena — não descartados, auditáveis.

---

## 🛠️ Stack

`Databricks Free Edition` · `PySpark` · `Delta Lake` · `MLflow` · `scikit-learn`
`Random Forest Regressor` · `DAMA-DMBOK` · `Arquitetura Medallion`

---

## ▶️ Como executar

1. Conta no [Databricks Free Edition](https://www.databricks.com/learn/free-edition)
2. Execute primeiro o pipeline de qualidade: [NYCtaxi-case](https://github.com/laaurasaporski/NYCtaxi-case)
   — as tabelas `bronze_yellow_taxi` e `gold_yellow_taxi` precisam existir
3. Importe `05_ml_data_quality_impact.py` no workspace
4. Ajuste o email no `mlflow.set_experiment` para o seu
5. Rode as células em ordem

---

## 🔗 Projeto relacionado

Este experimento é o **complemento** do pipeline principal de Data Quality:
👉 [NYCtaxi-case](https://github.com/laaurasaporski/NYCtaxi-case) — ingestão, qualidade,
Medallion architecture e análises dos dados de táxi de NYC.
