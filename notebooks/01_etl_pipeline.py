# Databricks notebook source
#
# ETL Pipeline: Bronze → Silver → Gold
#
# Responsabilidade deste notebook:
#   Bronze  → lê CSVs do S3
#   Silver  → enriquece com AuxiliaryFeatureBuilder (joins das tabelas auxiliares)
#   Gold    → seleciona apenas as colunas necessárias para o treinamento
#
# AnomalyHandler e DomainFeatureBuilder ficam DENTRO do sklearn Pipeline (notebook 02)
# para que o modelo salvo no MLflow possa receber dados brutos na inferência.

# COMMAND ----------

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks.sdk.runtime import dbutils  # noqa: F401

# Permite importar src/ a partir do Databricks Repos
REPO_ROOT = "/Users/gdseabra@gmail.com/credit-score-analysis"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["AWS_ACCESS_KEY_ID"]     = dbutils.secrets.get("aws-credentials", "access-key-id")
os.environ["AWS_SECRET_ACCESS_KEY"] = dbutils.secrets.get("aws-credentials", "secret-access-key")
os.environ["AWS_DEFAULT_REGION"]    = "us-east-1"

BUCKET_DATALAKE = dbutils.secrets.get("aws-credentials", "datalake-bucket")

# COMMAND ----------

import io
import boto3
import pandas as pd

from src.data.loader import S3DataLoader
from src.features.build_features import AuxiliaryFeatureBuilder

# COMMAND ----------

# EXTRACT — lê todas as tabelas da camada Bronze

loader = S3DataLoader(bucket=BUCKET_DATALAKE)
tables = loader.load_all_core_tables()

df_train         = tables["application_train"]
df_bureau        = tables["bureau"]
df_bureau_bal    = tables["bureau_balance"]
df_previous_app  = tables["previous_application"]
df_installments  = tables["installments_payments"]
df_pos_cash      = tables["pos_cash_balance"]
df_credit_card   = tables["credit_card_balance"]

print(f"application_train : {df_train.shape}")
print(f"bureau            : {df_bureau.shape}")
print(f"bureau_balance    : {df_bureau_bal.shape}")
print(f"previous_app      : {df_previous_app.shape}")
print(f"installments      : {df_installments.shape}")
print(f"pos_cash          : {df_pos_cash.shape}")
print(f"credit_card       : {df_credit_card.shape}")
print(f"\nTARGET: {df_train['TARGET'].value_counts(normalize=True).to_dict()}")

# COMMAND ----------

# TRANSFORM — enriquece application_train com agregações das tabelas auxiliares
# AuxiliaryFeatureBuilder faz LEFT JOIN das agregações por SK_ID_CURR

aux_builder = AuxiliaryFeatureBuilder(
    bureau        = df_bureau,
    bureau_balance= df_bureau_bal,
    credit_card   = df_credit_card,
    installments  = df_installments,
    previous_app  = df_previous_app,
    pos_cash      = df_pos_cash,
)

df_silver = aux_builder.fit_transform(df_train)

print(f"Silver: {df_silver.shape} (original: {df_train.shape})")
novas_cols = [c for c in df_silver.columns if c not in df_train.columns]
print(f"Novas colunas ({len(novas_cols)}): {novas_cols}")

# COMMAND ----------

# LOAD — salva Silver no S3 (dataset completo, todas as colunas)

def write_parquet_to_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    s3 = boto3.client("s3")
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    print(f"Salvo: s3://{bucket}/{key} ({df.shape[0]} linhas, {df.shape[1]} colunas)")

write_parquet_to_s3(df_silver, BUCKET_DATALAKE, "silver/application_train.parquet")

# COMMAND ----------

# GOLD — seleciona apenas as colunas que entram no modelo
#
# Nota: AnomalyHandler e DomainFeatureBuilder criam colunas derivadas DENTRO
# do sklearn Pipeline — por isso os campos brutos (DAYS_EMPLOYED, DAYS_BIRTH,
# AMT_CREDIT, etc.) são mantidos aqui; as colunas derivadas (AGE_YEARS,
# CREDIT_INCOME_RATIO, etc.) não precisam estar no Gold.

NUMERIC_COLS = [
    # Campos brutos da tabela principal (transformados pelo pipeline)
    "AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "CNT_FAM_MEMBERS",
    # Agregações do bureau (geradas pelo AuxiliaryFeatureBuilder)
    "bureau_loan_count", "bureau_active_loans",
    "bureau_total_credit_sum", "bureau_mean_days_credit",
    "bureau_bal_max_dpd", "bureau_bal_dpd_months_total",
    # Agregações de cartão de crédito
    "cc_utilization_mean", "cc_drawings_total", "cc_months_with_balance",
    # Agregações de parcelas
    "install_payment_ratio_mean", "install_days_late_mean", "install_late_count",
    # Agregações de aplicações anteriores
    "prev_app_count", "prev_app_approval_rate",
    "prev_app_credit_ratio_mean", "prev_app_days_last_decision",
    # Agregações POS/empréstimo pessoal
    "pos_dpd_mean", "pos_dpd_months_total",
]

CATEGORICAL_COLS = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
]

cols_gold = NUMERIC_COLS + CATEGORICAL_COLS + ["TARGET", "SK_ID_CURR"]
cols_disponiveis = [c for c in cols_gold if c in df_silver.columns]
cols_ausentes = [c for c in cols_gold if c not in df_silver.columns]

if cols_ausentes:
    print(f"AVISO: colunas ausentes no Silver (serão ignoradas): {cols_ausentes}")

df_gold = df_silver[cols_disponiveis].copy()

nulos_pct = df_gold.isnull().mean().mean() * 100
print(f"Gold: {df_gold.shape} | {nulos_pct:.2f}% nulos (serão imputados pelo pipeline)")

write_parquet_to_s3(df_gold, BUCKET_DATALAKE, "gold/application_train.parquet")

# COMMAND ----------

print("ETL concluído.")
print(f"  Silver: s3://{BUCKET_DATALAKE}/silver/application_train.parquet")
print(f"  Gold  : s3://{BUCKET_DATALAKE}/gold/application_train.parquet")
print(f"  Colunas Gold: {len(cols_disponiveis)} ({len(NUMERIC_COLS)} numéricas, {len(CATEGORICAL_COLS)} categóricas)")
