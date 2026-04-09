"""
Dependências compartilhadas da API REST.

Gerencia o ciclo de vida dos modelos ML com suporte a blue-green deployment:
- Carrega dois modelos do MLflow Registry: Production e Staging.
- Fallback para arquivo local (.joblib) se MLflow não estiver disponível.
- Dependency FastAPI para injeção do modelo nos endpoints via header X-Model-Stage.

O modelo esperado é um sklearn Pipeline completo (pré-processamento + classificador).
"""

import logging
import os
from pathlib import Path

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

DATABRICKS_HOST: str = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_TOKEN: str = os.getenv("DATABRICKS_TOKEN", "")
MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "credit_score.default.credit_score_lightgbm")
MODEL_PATH: Path = Path("models/lightgbm_pipeline.joblib")

# Colunas usadas pelo modelo (espelham as constantes do DAG)
NUMERIC_COLS: list[str] = [
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

CATEGORICAL_COLS: list[str] = [
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
# Estado global — blue-green
# ---------------------------------------------------------------------------

_models: dict[str, object] = {}       # {"production": pipeline, "staging": pipeline}
_shap_explainers: dict[str, object] = {}  # {"production": explainer, "staging": explainer}


def _load_mlflow_model(stage: str) -> object | None:
    """Tenta carregar um modelo do MLflow Registry (Databricks) por stage."""
    if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
        return None
    try:
        import mlflow
        os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
        os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
        mlflow.set_tracking_uri("databricks")
        model_uri = f"models:/{MLFLOW_MODEL_NAME}@{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Modelo carregado do MLflow: %s (%s)", MLFLOW_MODEL_NAME, stage)
        return model
    except Exception as exc:
        logger.warning("MLflow %s não disponível: %s", stage, exc)
        return None


def _init_shap_explainer(model: object, stage: str) -> None:
    """Inicializa SHAPExplainer para um modelo carregado."""
    try:
        from src.explainability.shap_explainer import SHAPExplainer
        _shap_explainers[stage] = SHAPExplainer(model, top_n=6)
    except Exception as exc:
        logger.warning("SHAPExplainer não inicializado (%s): %s", stage, exc)


def load_model() -> None:
    """Carrega os modelos Production e Staging no startup da aplicação.

    Ordem de tentativa para Production:
    1. MLflow Registry (stage=Production)
    2. Arquivo local (.joblib)

    Staging é carregado apenas do MLflow Registry.
    """
    # --- Production ---
    prod = _load_mlflow_model("production")
    if prod is None and MODEL_PATH.exists():
        try:
            import joblib
            prod = joblib.load(MODEL_PATH)
            logger.info("Modelo Production carregado do fallback local: %s", MODEL_PATH)
        except Exception as exc:
            logger.error("Falha ao carregar modelo local: %s", exc)

    if prod is not None:
        _models["production"] = prod
        _init_shap_explainer(prod, "production")
    else:
        logger.warning(
            "Nenhum modelo Production disponível. "
            "Configure DATABRICKS_HOST + DATABRICKS_TOKEN ou coloque o .joblib em '%s'.",
            MODEL_PATH,
        )

    # --- Staging ---
    staging = _load_mlflow_model("staging")
    if staging is not None:
        _models["staging"] = staging
        _init_shap_explainer(staging, "staging")
    else:
        logger.info("Nenhum modelo Staging disponível — shadow mode desabilitado.")


def get_model(x_model_stage: str = Header("production", alias="X-Model-Stage")):
    """Dependency FastAPI — injeta o modelo baseado no header X-Model-Stage.

    Headers:
        X-Model-Stage: "production" (default) ou "staging".

    Raises:
        HTTPException 503: Se o modelo do stage solicitado não estiver disponível.
    """
    stage = x_model_stage.lower()
    model = _models.get(stage)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Modelo '{stage}' não disponível. Stages carregados: {list(_models.keys())}",
        )
    return model


def get_model_by_stage(stage: str):
    """Retorna o modelo de um stage específico ou None."""
    return _models.get(stage)


def get_shap_explainer(x_model_stage: str = Header("production", alias="X-Model-Stage")):
    """Dependency FastAPI — injeta o SHAPExplainer baseado no header X-Model-Stage.

    Raises:
        HTTPException 503: Se o explainer do stage solicitado não estiver disponível.
    """
    stage = x_model_stage.lower()
    explainer = _shap_explainers.get(stage)
    if explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"SHAPExplainer '{stage}' não disponível.",
        )
    return explainer


def is_model_loaded() -> bool:
    """Retorna True se ao menos o modelo Production estiver carregado."""
    return "production" in _models


def get_loaded_stages() -> list[str]:
    """Retorna lista de stages com modelos carregados."""
    return list(_models.keys())
