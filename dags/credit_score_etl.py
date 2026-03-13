"""
DAG: credit_score_etl

Pipeline ETL para o dataset Home Credit Default Risk.

Fluxo:
    extract >> transform >> load

- extract   : Carrega os CSVs brutos via HomeCreditDataLoader e salva em
              data/interim/ como Parquet (formato eficiente para uso interno).
- transform : Aplica o pipeline de feature engineering (AnomalyHandler +
              DomainFeatureBuilder + ColumnTransformer) sobre os dados interim.
- load      : Salva o dataset transformado em data/processed/ pronto para
              treinamento de modelos.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relativas ao volume montado no container)
# ---------------------------------------------------------------------------
DATA_DIR = Path("/opt/airflow/data")
RAW_DIR = DATA_DIR           # CSVs brutos ficam direto em data/
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

INTERIM_FILE = INTERIM_DIR / "application_train_interim.parquet"
PROCESSED_FILE = PROCESSED_DIR / "application_train_processed.parquet"

# Colunas selecionadas para o modelo (baseado na EDA — notebook 01)
NUMERIC_COLS = [
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_INCOME_TOTAL",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "CNT_FAM_MEMBERS",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM_MONTHS",
    "INCOME_PER_FAMILY_MEMBER",
    "AGE_YEARS",
    "EMPLOYED_YEARS",
    "EMPLOYED_TO_AGE_RATIO",
]

CATEGORICAL_COLS = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
]


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

@dag(
    dag_id="credit_score_etl",
    description="ETL pipeline para o dataset Home Credit Default Risk",
    schedule="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["credit", "etl", "home-credit"],
)
def credit_score_etl():

    # -----------------------------------------------------------------------
    # EXTRACT
    # -----------------------------------------------------------------------
    @task(task_id="extract")
    def extract() -> str:
        """Carrega application_train.csv e salva em Parquet na pasta interim."""
        from src.data.loader import HomeCreditDataLoader

        INTERIM_DIR.mkdir(parents=True, exist_ok=True)

        loader = HomeCreditDataLoader(data_dir=str(RAW_DIR))
        df = loader.load_application_train()

        df.to_parquet(INTERIM_FILE, index=False)
        log.info("Extract concluído: %d linhas, %d colunas → %s", *df.shape, INTERIM_FILE)

        return str(INTERIM_FILE)

    # -----------------------------------------------------------------------
    # TRANSFORM
    # -----------------------------------------------------------------------
    @task(task_id="transform")
    def transform(interim_path: str) -> str:
        """Aplica o pipeline de feature engineering e salva em Parquet."""
        from src.features.build_features import (
            AnomalyHandler,
            DomainFeatureBuilder,
            build_preprocessor_pipeline,
        )

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(interim_path)
        target = df["TARGET"].copy() if "TARGET" in df.columns else None

        # Aplica apenas AnomalyHandler + DomainFeatureBuilder para manter o
        # DataFrame com nomes de colunas legíveis antes do ColumnTransformer.
        df = AnomalyHandler().fit_transform(df)
        df = DomainFeatureBuilder().fit_transform(df)

        # Filtra apenas colunas disponíveis no dataset para evitar KeyError
        num_cols = [c for c in NUMERIC_COLS if c in df.columns]
        cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

        pipeline = build_preprocessor_pipeline(num_cols, cat_cols)
        X = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
        X_transformed = pipeline.fit_transform(X)

        # Reconstrói DataFrame com nomes das colunas geradas pelo pipeline
        cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        cat_feature_names = cat_encoder.named_steps["encoder"].get_feature_names_out(cat_cols)
        feature_names = num_cols + list(cat_feature_names)

        df_processed = pd.DataFrame(X_transformed, columns=feature_names)

        if target is not None:
            df_processed["TARGET"] = target.values

        df_processed.to_parquet(PROCESSED_FILE, index=False)
        log.info(
            "Transform concluído: %d linhas, %d features → %s",
            *df_processed.shape,
            PROCESSED_FILE,
        )

        return str(PROCESSED_FILE)

    # -----------------------------------------------------------------------
    # LOAD
    # -----------------------------------------------------------------------
    @task(task_id="load")
    def load(processed_path: str) -> None:
        """Persiste o dataset processado no Postgres e loga métricas de qualidade."""
        from src.data.database import get_engine, CREDIT_SCHEMA

        df = pd.read_parquet(processed_path)

        null_pct = df.isnull().mean().mean() * 100
        if null_pct > 5:
            log.warning("Taxa de nulos acima de 5%% (%.2f%%). Verifique o pipeline.", null_pct)

        engine = get_engine()
        df.to_sql(
            name="features",
            con=engine,
            schema=CREDIT_SCHEMA,
            if_exists="replace",
            index=False,
            chunksize=10_000,
        )

        log.info(
            "Load concluído: %d linhas | %d colunas | %.2f%% nulos → %s.features",
            *df.shape,
            null_pct,
            CREDIT_SCHEMA,
        )

    # -----------------------------------------------------------------------
    # TRAIN MODEL
    # -----------------------------------------------------------------------
    @task(task_id="train_model")
    def train_model(processed_path: str) -> None:
        """Treina modelo LightGBM com cross-validation e persiste via MLflow."""
        from sklearn.base import clone

        from src.features.build_features import build_preprocessor_pipeline
        from src.models.classifiers import CreditClassifier
        from src.models.trainer import CreditTrainer

        df = pd.read_parquet(processed_path)

        y = df["TARGET"]
        X = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")

        # Separa colunas numéricas e categóricas disponíveis no arquivo processado
        num_cols = [c for c in NUMERIC_COLS if c in X.columns]
        cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]

        pipeline = build_preprocessor_pipeline(num_cols, cat_cols)

        trainer = CreditTrainer(modelo_nome="lightgbm")
        metricas = trainer.train(X, y, pipeline, cv_folds=5)

        log.info(
            "train_model | AUC=%.4f±%.4f | Gini=%.4f | KS=%.4f | F1=%.4f",
            metricas["auc_roc_mean"],
            metricas["auc_roc_std"],
            metricas["gini_mean"],
            metricas["ks_stat_mean"],
            metricas["f1_mean"],
        )

        # Ajusta pipeline completo (com modelo) nos dados completos antes de salvar
        modelo = CreditClassifier.get_model("lightgbm")
        pipeline_final = clone(pipeline)
        pipeline_final.steps.append(("modelo", modelo))
        pipeline_final.fit(X, y)

        trainer.save_model(pipeline_final, metricas)
        log.info("train_model concluído — modelo salvo com sucesso.")

    # -----------------------------------------------------------------------
    # Dependências
    # -----------------------------------------------------------------------
    interim = extract()
    processed = transform(interim)
    loaded = load(processed)
    loaded >> train_model(processed)


credit_score_etl()
