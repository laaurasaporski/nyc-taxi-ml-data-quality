# Databricks notebook source
# ── Configuração ──────────────────────────────────────────────────────
CATALOG = "workspace"
SCHEMA  = "ifood_case"
SEED    = 42
SAMPLE  = 100_000 

from pyspark.sql import functions as F

# DADO SUJO → Bronze (com negativos, nulos, outliers, datas vazadas)
bronze = (spark.table(f"{CATALOG}.{SCHEMA}.bronze_yellow_taxi")
          .withColumn("trip_duration_min",
                      (F.col("tpep_dropoff_datetime").cast("long") -
                       F.col("tpep_pickup_datetime").cast("long")) / 60)
          .withColumn("pickup_hour",  F.hour("tpep_pickup_datetime"))
          .withColumn("pickup_month", F.month("tpep_pickup_datetime"))
          .select("total_amount", "passenger_count", "trip_duration_min",
                  "pickup_hour", "pickup_month", "VendorID")
          .dropna())

# DADO LIMPO → Gold (validado, deduplicado, dentro da janela jan–mai/2023)
gold = (spark.table(f"{CATALOG}.{SCHEMA}.gold_yellow_taxi")
        .select("total_amount", "passenger_count", "trip_duration_min",
                "pickup_hour", "pickup_month", "VendorID")
        .dropna())

print(f"Dado SUJO  (Bronze): {bronze.count():,} registros")
print(f"Dado LIMPO (Gold):   {gold.count():,} registros")
print(f"\nDiferença: {bronze.count() - gold.count():,} registros removidos pela qualidade")

# COMMAND ----------

import pandas as pd
import numpy as np

# Amostra representativa — 100k de cada, mesma proporção treino/teste
frac_bronze = SAMPLE / bronze.count()
frac_gold   = SAMPLE / gold.count()

pdf_sujo  = bronze.sample(fraction=frac_bronze, seed=SEED).toPandas()
pdf_limpo = gold.sample(fraction=frac_gold,     seed=SEED).toPandas()

pdf_sujo  = pdf_sujo.sample(n=min(SAMPLE, len(pdf_sujo)),   random_state=SEED).reset_index(drop=True)
pdf_limpo = pdf_limpo.sample(n=min(SAMPLE, len(pdf_limpo)), random_state=SEED).reset_index(drop=True)

print(f"Amostra SUJO:  {len(pdf_sujo):,} registros")
print(f"Amostra LIMPO: {len(pdf_limpo):,} registros")


print("\nEstatísticas do dado SUJO (total_amount):")
print(pdf_sujo["total_amount"].describe().round(2))
print("\nEstatísticas do dado LIMPO (total_amount):")
print(pdf_limpo["total_amount"].describe().round(2))



# MAGIC %md




import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATURES = ["passenger_count", "trip_duration_min", "pickup_hour", "pickup_month", "VendorID"]
TARGET   = "total_amount"

mlflow.set_experiment("/Users/laaurasaporski@gmail.com/nyc_taxi_data_quality_impact")

def treina_e_loga(pdf, nome_experimento):
    
    X = pdf[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float64")
    y = pdf[TARGET].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float64")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    with mlflow.start_run(run_name=nome_experimento):
        params = {"n_estimators": 100, "max_depth": 10, "random_state": SEED}
        mlflow.log_params(params)
        mlflow.log_param("dataset",          nome_experimento)
        mlflow.log_param("n_registros",      len(pdf))
        mlflow.log_param("total_amount_min", round(float(y.min()), 2))
        mlflow.log_param("total_amount_max", round(float(y.max()), 2))

        modelo = RandomForestRegressor(**params, n_jobs=-1)
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        mlflow.log_metric("rmse", round(rmse, 4))
        mlflow.log_metric("mae",  round(mae,  4))
        mlflow.log_metric("r2",   round(r2,   4))
        mlflow.sklearn.log_model(modelo, "model")

        print(f"\n{'='*45}")
        print(f"Experimento: {nome_experimento}")
        print(f"RMSE : ${rmse:.2f}  ← erro médio por corrida")
        print(f"MAE  : ${mae:.2f}")
        print(f"R²   : {r2:.4f}  ({r2*100:.1f}% da variação explicada)")
        print(f"{'='*45}")

        return {"nome": nome_experimento, "rmse": rmse, "mae": mae, "r2": r2}

# Treina os dois
print("Treinando modelo com DADO SUJO...")
res_sujo  = treina_e_loga(pdf_sujo,  "dado_sujo_bronze")

print("\nTreinando modelo com DADO LIMPO...")
res_limpo = treina_e_loga(pdf_limpo, "dado_limpo_gold")




# MAGIC %md
# MAGIC

# COMMAND ----------

delta_rmse = res_sujo["rmse"] - res_limpo["rmse"]
delta_mae  = res_sujo["mae"]  - res_limpo["mae"]
delta_r2   = res_limpo["r2"]  - res_sujo["r2"]

# Volume mensal estimado (média dos 5 meses)
corridas_mes = 15_335_554 / 5

print(f"""
╔══════════════════════════════════════════════════════╗
║     IMPACTO DA QUALIDADE DE DADOS NO MODELO ML       ║
╠══════════════════════════════════════════════════════╣
║                    SUJO         LIMPO      MELHORA   ║
║  RMSE ($/corrida)  $13.74       $12.23    -${delta_rmse:.2f}    ║
║  MAE  ($/corrida)  $5.94        $5.60     -${delta_mae:.2f}    ║
║  R²                66.0%        70.6%     +{delta_r2*100:.1f}pp   ║
╠══════════════════════════════════════════════════════╣
║  TRADUÇÃO EM NEGÓCIO                                  ║
╠══════════════════════════════════════════════════════╣
║  Erro por corrida reduzido em:  ${delta_rmse:.2f}              ║
║  Corridas/mês (média):          {corridas_mes:,.0f}       ║
║  Erro acumulado evitado/mês:    ${delta_rmse * corridas_mes:,.0f}  ║
╚══════════════════════════════════════════════════════╝

→ Mesmo modelo. Mesmos parâmetros. Mesma divisão treino/teste.
→ A única variável foi a QUALIDADE DO DADO.
→ Dado limpo = modelo ${delta_rmse:.2f} mais preciso por corrida.
""")

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

fig, axes = plt.subplots(1, 3, figsize=(14, 6))
fig.patch.set_facecolor("#FFFFFF")

fig.suptitle(
    "Impacto da Qualidade de Dados no Modelo ML",
    fontsize=15, fontweight="bold", color="#1A1A1A", y=1.04,
    fontfamily="DejaVu Sans"
)
fig.text(
    0.5, 0.98,
    "Mesmo algoritmo · Mesmos parâmetros · Dados diferentes",
    ha="center", fontsize=11, color="#717171"
)

metricas   = ["RMSE ($)\nmenor é melhor ↓", "MAE ($)\nmenor é melhor ↓", "R²\nmaior é melhor ↑"]
sujo_vals  = [res_sujo["rmse"],  res_sujo["mae"],  res_sujo["r2"]]
limpo_vals = [res_limpo["rmse"], res_limpo["mae"], res_limpo["r2"]]
deltas     = [f"-${delta_rmse:.2f}", f"-${delta_mae:.2f}", f"+{delta_r2*100:.1f}pp"]

COR_SUJO   = "#EA1D2C"  
COR_LIMPO  = "#1B8E3F"  
COR_BADGE  = "#E8F5ED"  
COR_BORDER = "#1B8E3F"

for i, (ax, metrica, sv, lv, delta) in enumerate(zip(axes, metricas, sujo_vals, limpo_vals, deltas)):
    ax.set_facecolor("#FAFAFA")
    for spine in ax.spines.values():
        spine.set_visible(False)

    
    x     = [0, 1]
    vals  = [sv, lv]
    cores = [COR_SUJO, COR_LIMPO]

    bars = ax.bar(x, vals, color=cores, width=0.45, zorder=3,
                  linewidth=0, edgecolor="none")

    
    ax.axhline(y=min(sv, lv), color="#E8E8E8", linewidth=1, zorder=2, linestyle="--")

    
    for bar, val, cor in zip(bars, vals, cores):
        label = f"${val:.2f}" if i < 2 else f"{val:.4f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(sv, lv) * 0.025,
            label,
            ha="center", va="bottom",
            fontweight="bold", fontsize=13, color="#1A1A1A"
        )

    
    ax.text(
        0.5, 0.96, f"Melhora: {delta}",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=11, fontweight="bold", color="#1B8E3F",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor=COR_BADGE,
            edgecolor=COR_BORDER,
            linewidth=1.5
        )
    )

    
    ax.set_title(metrica, fontsize=11, fontweight="bold",
                 color="#1A1A1A", pad=14, loc="center")

    # Eixo X
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Dado Sujo\n(Bronze)", "Dado Limpo\n(Gold)"],
                       fontsize=10.5, color="#3D3D3D")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, max(sv, lv) * 1.35)
    ax.set_facecolor("#FFFFFF")

    
    for spine in ["bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#E8E8E8")
        ax.spines[spine].set_linewidth(1)


for ax in axes[:-1]:
    ax.axvline(x=1.5, color="#E8E8E8", linewidth=1)


patch_sujo  = mpatches.Patch(
    color=COR_SUJO,
    label="Dado Sujo (Bronze) — com negativos, outliers e vazamentos"
)
patch_limpo = mpatches.Patch(
    color=COR_LIMPO,
    label="Dado Limpo (Gold) — validado com 11 regras DAMA-DMBOK"
)
fig.legend(
    handles=[patch_sujo, patch_limpo],
    loc="lower center", ncol=2,
    fontsize=10, frameon=False,
    bbox_to_anchor=(0.5, -0.06),
    labelcolor="#3D3D3D"
)


fig.text(
    0.5, -0.11,
    f"Erro acumulado evitado por mês: ${delta_rmse * (15_335_554/5):,.0f}  "
    f"(~3M corridas × ${delta_rmse:.2f} de melhora/corrida)",
    ha="center", fontsize=9.5, color="#717171", style="italic"
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("/tmp/ml_dq_impact_ifood.png", dpi=180,
            bbox_inches="tight", facecolor="#FFFFFF")
plt.show()
print("✅ Gráfico salvo!")
