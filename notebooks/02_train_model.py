# Databricks notebook source
#
# Model Training: Gold → MLflow Production
#
# Responsabilidade deste notebook:
#   1. Lê o dataset Gold do S3 (gerado pelo 01_etl_pipeline)
#   2. Monta o pipeline sklearn via build_preprocessor_pipeline()
#      (inclui AnomalyHandler + DomainFeatureBuilder + ColumnTransformer)
#   3. Treina com cross-validation estratificada via CreditTrainer
#   4. Registra o modelo no MLflow Model Registry e promove para Production
#
# O pipeline salvo no MLflow recebe dados brutos — a mesma entrada do Gold.
# Isso garante que a inferência na API não precisa de pré-processamento externo.

# COMMAND ----------

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks.sdk.runtime import dbutils  # noqa: F401

REPO_ROOT = "/Users/gdseabra@gmail.com/credit-score-analysis"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["AWS_ACCESS_KEY_ID"]     = dbutils.secrets.get("aws-credentials", "access-key-id")
os.environ["AWS_SECRET_ACCESS_KEY"] = dbutils.secrets.get("aws-credentials", "secret-access-key")
os.environ["AWS_DEFAULT_REGION"]    = "us-east-1"

BUCKET_DATALAKE   = dbutils.secrets.get("aws-credentials", "datalake-bucket")
MLFLOW_MODEL_NAME = "credit_score.default.credit_score_lightgbm"

import mlflow
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

import io
import boto3
import pandas as pd
import mlflow

from src.features.build_features import build_preprocessor_pipeline
from src.models.trainer import CreditTrainer

# COMMAND ----------

# Carrega dataset Gold do S3

s3 = boto3.client("s3", region_name="us-east-1")
obj = s3.get_object(Bucket=BUCKET_DATALAKE, Key="gold/application_train.parquet")
df = pd.read_parquet(io.BytesIO(obj["Body"].read()))

y = df["TARGET"]
X = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")

print(f"Gold carregado: {X.shape[1]} features | {len(y)} amostras")
print(f"Taxa de inadimplência: {y.mean():.4f}")

# COMMAND ----------

# Define colunas para o ColumnTransformer
#
# Inclui as colunas brutas originais + as colunas derivadas que o DomainFeatureBuilder
# cria internamente no pipeline (AGE_YEARS, CREDIT_INCOME_RATIO, etc.) — o ColumnTransformer
# as receberá DEPOIS que o DomainFeatureBuilder rodar no pipeline.

NUMERIC_COLS = [c for c in [
    # Campos brutos
    "AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "CNT_FAM_MEMBERS",
    # Criadas pelo DomainFeatureBuilder dentro do pipeline
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM_MONTHS",
    "INCOME_PER_FAMILY_MEMBER", "AGE_YEARS", "EMPLOYED_YEARS", "EMPLOYED_TO_AGE_RATIO",
    # Criada pelo AnomalyHandler dentro do pipeline
    "DAYS_EMPLOYED_ANOMALY",
    # Agregações do ETL (já no Gold)
    "bureau_loan_count", "bureau_active_loans",
    "bureau_total_credit_sum", "bureau_mean_days_credit",
    "bureau_bal_max_dpd", "bureau_bal_dpd_months_total",
    "cc_utilization_mean", "cc_drawings_total", "cc_months_with_balance",
    "install_payment_ratio_mean", "install_days_late_mean", "install_late_count",
    "prev_app_count", "prev_app_approval_rate",
    "prev_app_credit_ratio_mean", "prev_app_days_last_decision",
    "pos_dpd_mean", "pos_dpd_months_total",
] if c in X.columns or c in [
    # Colunas geradas internamente pelo pipeline (não estão em X, mas estarão após transform)
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM_MONTHS",
    "INCOME_PER_FAMILY_MEMBER", "AGE_YEARS", "EMPLOYED_YEARS",
    "EMPLOYED_TO_AGE_RATIO", "DAYS_EMPLOYED_ANOMALY",
]]

CATEGORICAL_COLS = [c for c in [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
] if c in X.columns]

print(f"Numéricas: {len(NUMERIC_COLS)} | Categóricas: {len(CATEGORICAL_COLS)}")

# COMMAND ----------

# Monta o pipeline completo:
#   AnomalyHandler → DomainFeatureBuilder → ColumnTransformer
#
# O modelo (LightGBM) é adicionado pelo CreditTrainer em cada fold do CV.

pipeline = build_preprocessor_pipeline(
    numeric_cols=NUMERIC_COLS,
    categorical_cols=CATEGORICAL_COLS,
)

# COMMAND ----------

# Treina com cross-validation estratificada e loga no MLflow

trainer = CreditTrainer(
    modelo_nome="lightgbm",
    mlflow_tracking_uri="databricks",
)

metricas = trainer.train(X, y, pipeline, cv_folds=5)

print(f"\nResultado CV:")
print(f"  AUC-ROC : {metricas['auc_roc_mean']:.4f} ± {metricas['auc_roc_std']:.4f}")
print(f"  Gini    : {metricas['gini_mean']:.4f}")
print(f"  KS      : {metricas['ks_stat_mean']:.4f}")
print(f"  F1      : {metricas['f1_mean']:.4f}")

# COMMAND ----------

# Treina modelo final no dataset completo e registra no MLflow Model Registry

from sklearn.base import clone
from mlflow.models import infer_signature
from src.models.classifiers import CreditClassifier

pipeline_final = clone(pipeline)
pipeline_final.steps.append(("modelo", CreditClassifier.get_model("lightgbm")))
pipeline_final.fit(X, y)

with mlflow.start_run(run_name="lightgbm-databricks-final"):
    mlflow.log_params({
        "modelo": "lightgbm",
        "cv_folds": 5,
        "n_features": X.shape[1],
        "n_amostras": len(y),
        "plataforma": "databricks",
    })
    for nome, valor in metricas.items():
        if isinstance(valor, float):
            mlflow.log_metric(nome, valor)

    signature = infer_signature(X, pipeline_final.predict(X))
    mlflow.sklearn.log_model(
        pipeline_final,
        artifact_path="pipeline",
        registered_model_name=MLFLOW_MODEL_NAME,
        signature=signature,
    )
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
    print(f"Modelo registrado: {MLFLOW_MODEL_NAME}")

# COMMAND ----------

# Promove a nova versão para Production via alias (Unity Catalog)

client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
latest = max(versions, key=lambda v: int(v.version))

print(f"Versão {latest.version} — definindo alias 'production'...")
client.set_registered_model_alias(MLFLOW_MODEL_NAME, "production", latest.version)
print(f"Modelo v{latest.version} com alias @production.")
print(f"Modelo registrado: {MLFLOW_MODEL_NAME}")
