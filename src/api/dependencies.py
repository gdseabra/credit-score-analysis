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

# Registro de modelos S3 (fonte de maior prioridade para Production)
DATALAKE_BUCKET: str = os.getenv("DATALAKE_BUCKET", "")
MODEL_POINTER_KEY: str = os.getenv("MODEL_POINTER_KEY", "models/current/pointer.json")

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
_current_run_id: str | None = None    # run_id do modelo Production vindo do registro S3


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


def _load_s3_model() -> object | None:
    """Tenta carregar o modelo Production do registro S3 (ponteiro → candidato).

    Lê ``models/current/pointer.json``, baixa o ``model.joblib`` do run_id apontado
    (com cache local) e registra o run_id ativo. Retorna ``None`` se o bucket não
    estiver configurado ou o registro não estiver acessível.
    """
    global _current_run_id
    if not DATALAKE_BUCKET:
        return None
    try:
        import joblib

        from src.registry import s3_registry

        pointer = s3_registry.read_pointer(DATALAKE_BUCKET, MODEL_POINTER_KEY)
        run_id = pointer["run_id"]
        model_path = s3_registry.download_model(DATALAKE_BUCKET, run_id)
        model = joblib.load(model_path)
        _current_run_id = run_id
        logger.info("Modelo Production carregado do registro S3: run_id=%s", run_id)
        return model
    except Exception as exc:
        logger.warning("Registro S3 não disponível: %s", exc)
        return None


def _init_shap_explainer(model: object, stage: str) -> None:
    """Inicializa SHAPExplainer para um modelo carregado.

    Construído sob demanda (na primeira chamada de ``/explain``) e não no cold
    start: importar ``shap`` dispara a compilação JIT do numba e a construção do
    TreeExplainer, o que estoura o orçamento de init do Lambda. O caminho de
    ``/predict`` não precisa de SHAP.
    """
    try:
        from src.explainability.shap_explainer import SHAPExplainer
        _shap_explainers[stage] = SHAPExplainer(model, top_n=6)
    except Exception as exc:
        logger.warning("SHAPExplainer não inicializado (%s): %s", stage, exc)


def load_model() -> None:
    """Carrega os modelos Production e Staging no startup da aplicação.

    Ordem de tentativa para Production:
    1. Registro S3 (ponteiro → candidato)
    2. MLflow Registry (stage=Production)
    3. Arquivo local (.joblib)

    Staging é carregado apenas do MLflow Registry.
    """
    # --- Production ---
    prod = _load_s3_model()
    if prod is None:
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
    if explainer is None and stage in _models:
        _init_shap_explainer(_models[stage], stage)
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


def get_current_run_id() -> str | None:
    """Retorna o run_id do modelo Production vindo do registro S3 (ou None)."""
    return _current_run_id


def get_loaded_stages() -> list[str]:
    """Retorna lista de stages com modelos carregados."""
    return list(_models.keys())
