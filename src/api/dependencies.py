"""
Dependências compartilhadas da API REST.

Gerencia o ciclo de vida do modelo ML:
- Carregamento lazy do pipeline treinado no startup da aplicação.
- Dependency FastAPI para injeção do modelo nos endpoints.

O modelo esperado é um sklearn Pipeline completo (pré-processamento + classificador),
salvo pelo Airflow DAG ``credit_score_etl`` via joblib.
"""

import logging
from pathlib import Path

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração de paths
# ---------------------------------------------------------------------------

MODEL_PATH: Path = Path("models/lightgbm_pipeline.joblib")
"""Caminho padrão do pipeline serializado. Gerado pelo DAG train_model."""

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
# Estado global da aplicação
# ---------------------------------------------------------------------------

_model = None        # Pipeline sklearn carregado no startup
_shap_explainer = None  # SHAPExplainer inicializado após o modelo


def load_model() -> None:
    """Carrega o pipeline treinado em memória no startup da aplicação.

    Chamado automaticamente pelo evento ``lifespan`` do FastAPI.
    Se o arquivo não existir, a API sobe normalmente mas retorna 503
    nos endpoints de predição.
    """
    global _model
    if MODEL_PATH.exists():
        try:
            import joblib

            _model = joblib.load(MODEL_PATH)
            logger.info("Modelo carregado com sucesso: %s", MODEL_PATH)
        except Exception as exc:
            logger.error("Falha ao carregar o modelo '%s': %s", MODEL_PATH, exc)
            _model = None
            return

        # Inicializa SHAPExplainer logo após carregar o modelo
        global _shap_explainer
        try:
            from src.explainability.shap_explainer import SHAPExplainer
            _shap_explainer = SHAPExplainer(_model, top_n=6)
        except Exception as exc:
            logger.warning("SHAPExplainer não pôde ser inicializado: %s", exc)
            _shap_explainer = None
    else:
        logger.warning(
            "Arquivo de modelo não encontrado em '%s'. "
            "Execute o DAG 'credit_score_etl' no Airflow para treinar o modelo.",
            MODEL_PATH,
        )


def get_model():
    """Dependency FastAPI que injeta o pipeline treinado nos endpoints.

    Returns:
        Pipeline sklearn carregado.

    Raises:
        HTTPException 503: Se o modelo não estiver disponível.
    """
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Modelo não disponível. "
                "Execute o DAG 'credit_score_etl' no Airflow (http://localhost:8080) "
                "para treinar e persistir o modelo."
            ),
        )
    return _model


def get_shap_explainer():
    """Dependency FastAPI que injeta o SHAPExplainer nos endpoints.

    Returns:
        SHAPExplainer inicializado.

    Raises:
        HTTPException 503: Se o explainer não estiver disponível.
    """
    if _shap_explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "SHAPExplainer não disponível. "
                "Verifique se o modelo foi carregado corretamente e se "
                "a biblioteca shap está instalada."
            ),
        )
    return _shap_explainer


def is_model_loaded() -> bool:
    """Retorna True se o modelo estiver carregado em memória."""
    return _model is not None
