# NYC Taxi — Impacto da Qualidade de Dados no Modelo ML

Experimento controlado que **prova empiricamente** o impacto da qualidade de dados
num modelo de Machine Learning — usando os dados de táxi amarelo de NYC (jan–mai/2023).

> **A pergunta central:** um modelo treinado em dado sujo vs dado limpo — qual a diferença real?
> A resposta está nos números abaixo.

---

## 🧪 O Experimento

Treinei o **mesmo modelo** (Random Forest), com os **mesmos parâmetros**, na **mesma divisão
treino/teste** — a única variável foi a **qualidade dos dados de entrada**.

| | Dado Sujo (Bronze) | Dado Limpo (Gold) |
|---|---|---|
| Origem | Dado cru, sem validação | Validado com 11 regras DAMA-DMBOK |
| `total_amount` mínimo | **-$203,70** 🚨 | $1,00 ✅ |
| Registros | 99.895 | 99.318 |
| `max_depth` | 10 | 10 |
| `n_estimators` | 100 | 100 |
| `random_state` | 42 | 42 |

---

## 📊 Resultados

| Métrica | Dado Sujo | Dado Limpo | Melhora |
|---|---|---|---|
| **RMSE** (erro médio/corrida) | $13,74 | **$12,23** | **-$1,51** |
| **MAE** | $5,94 | **$5,60** | **-$0,34** |
| **R²** | 0,660 (66,0%) | **0,706 (70,6%)** | **+4,6pp** |

### Tradução em negócio

Erro por corrida reduzido:    $1,51

Corridas/mês (média):         3.067.111

Erro acumulado evitado/mês:   $4.628.235


> Mesmo modelo. Mesmos parâmetros. A única variável foi a qualidade do dado.
> **Qualidade de dados não é detalhe técnico — é resultado de negócio.**

---

## 🔬 MLflow — Rastreabilidade do Experimento

Os dois runs foram rastreados com **MLflow** dentro do Databricks, garantindo que o
experimento seja **reproduzível, auditável e comparável**.

O que foi logado em cada run:

| | dado_sujo_bronze | dado_limpo_gold |
|---|---|---|
| **n_estimators** | 100 | 100 |
| **max_depth** | 10 | 10 |
| **random_state** | 42 | 42 |
| **total_amount_min** | -203.7 🚨 | 1.0 ✅ |
| **RMSE** | 13.74 | **12.23** |
| **MAE** | 5.941 | **5.6** |
| **R²** | 0.66 | **0.706** |
| **Modelo salvo** | ✅ artifact | ✅ artifact |

> 💬 *"O MLflow prova que os parâmetros foram idênticos nos dois runs.
> A diferença nas métricas não é acaso — é o custo mensurável do dado sujo."*

Os runs, artefatos e modelos ficam persistidos no Databricks (não no GitHub).
Para visualizar: abra o experimento `nyc_taxi_data_quality_impact` no MLflow UI
e selecione os dois runs para comparação automática.

---

## 🔍 Por que os dados são diferentes?

O dado sujo (Bronze) contém problemas reais identificados na camada de qualidade:

- **144k corridas** com `total_amount` ≤ 0 (estornos/erros) → distorce a média para baixo
- **428k corridas** com `passenger_count` nulo → campo crítico ausente
- **6.181 corridas** com dropoff antes do pickup → fisicamente impossível
- **104 registros** vazados de fora de jan–mai/2023 → contaminam a janela temporal

O dado limpo (Gold) passou por **11 regras de qualidade** mapeadas às **6 dimensões
do DAMA-DMBOK** (completeness, validity, accuracy, consistency, timeliness, uniqueness),
com os reprovados isolados em quarentena — não descartados, auditáveis.

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
4. Ajuste o email em `mlflow.set_experiment` para o seu
5. Rode as células em ordem — o MLflow registra tudo automaticamente

---

## 🔗 Projeto relacionado

Este experimento é o **complemento** do pipeline principal de Data Quality:
👉 [NYCtaxi-case](https://github.com/laaurasaporski/NYCtaxi-case) — ingestão,
qualidade, Medallion architecture e análises dos dados de táxi de NYC.
