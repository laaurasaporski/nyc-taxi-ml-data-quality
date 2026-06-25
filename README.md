> Mesmo modelo. Mesmos parâmetros. A única variável foi a qualidade do dado.
> **Qualidade de dados não é detalhe técnico — é resultado de negócio.**

---

## 🔬 MLflow — Rastreabilidade do Experimento

Os dois runs foram rastreados com **MLflow**, garantindo que o experimento seja
**reproduzível, auditável e comparável** — não é um gráfico estático, é ciência.

### O que foi logado em cada run

| | dado_sujo_bronze | dado_limpo_gold |
|---|---|---|
| **Parâmetros** | n_estimators=100, max_depth=10, random_state=42 | n_estimators=100, max_depth=10, random_state=42 |
| **total_amount_min** | -203.7 🚨 | 1.0 ✅ |
| **total_amount_max** | 572.85 | 1000.0 |
| **RMSE** | 13.74 | **12.23** |
| **MAE** | 5.941 | **5.6** |
| **R²** | 0.66 | **0.706** |
| **Modelo salvo** | ✅ artifact | ✅ artifact |

### Por que MLflow aqui?

- **Rastreabilidade:** cada run tem ID único, timestamp, parâmetros e métricas persistidos
- **Reprodutibilidade:** qualquer pessoa pode replicar o experimento com os mesmos resultados
- **Comparação:** a UI do MLflow gera comparação visual automática entre os dois runs
- **Auditoria:** prova que os parâmetros do modelo foram idênticos — a diferença é **só o dado**

> 💬 *"O MLflow não é enfeite aqui — é a prova de que foi um experimento controlado.
> Mesmos parâmetros registrados em ambos os runs. A diferença nas métricas não é acaso."*

---

## 🔍 Por que os dados são diferentes?

O dado sujo (Bronze) contém problemas reais identificados na camada de qualidade:

- **144k corridas** com `total_amount` ≤ 0 (estornos/erros) → distorce média para baixo
- **428k corridas** com `passenger_count` nulo → campo crítico ausente
- **6.181 corridas** com dropoff antes do pickup → fisicamente impossível
- **104 registros** vazados de fora de jan–mai/2023 → contaminam a janela temporal

O dado limpo (Gold) passou por **11 regras de qualidade** mapeadas às **6 dimensões do
DAMA-DMBOK** (completeness, validity, accuracy, consistency, timeliness, uniqueness).

---

## 🛠️ Stack

`Databricks Free Edition` · `PySpark` · `Delta Lake` · `MLflow` · `scikit-learn`
`Random Forest Regressor` · `DAMA-DMBOK` · `Arquitetura Medallion`

---

## ▶️ Como executar

1. Conta no [Databricks Free Edition](https://www.databricks.com/learn/free-edition)
2. Execute primeiro o pipeline de qualidade:
   👉 [NYCtaxi-case](https://github.com/laaurasaporski/NYCtaxi-case)
   — as tabelas `bronze_yellow_taxi` e `gold_yellow_taxi` precisam existir
3. Importe `05_ml_data_quality_impact.py` no workspace
4. Ajuste o email no `mlflow.set_experiment` para o seu
5. Rode as células em ordem — o MLflow registra tudo automaticamente

---

## 🔗 Projeto relacionado

Este experimento é o **complemento** do pipeline principal de Data Quality:
👉 [NYCtaxi-case](https://github.com/laaurasaporski/NYCtaxi-case) — ingestão,
qualidade, Medallion architecture e análises dos dados de táxi de NYC.
