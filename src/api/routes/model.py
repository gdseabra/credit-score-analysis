"""
Rota de informações do modelo — GET /model/info.

Retorna metadados do pipeline carregado: nome, path, features e status.
Exige token JWT válido.
"""

from fastapi import APIRouter, Depends

from src.api.auth import get_current_user
from src.api.dependencies import CATEGORICAL_COLS, MODEL_PATH, NUMERIC_COLS, is_model_loaded
from src.api.schemas import ModelInfoResponse

router = APIRouter(prefix="/model", tags=["Modelo"])


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Informações e status do modelo carregado",
    description=(
        "Retorna os metadados do pipeline ML em uso:\n\n"
        "- Nome e caminho do arquivo\n"
        "- Features numéricas e categóricas\n"
        "- Status de disponibilidade\n\n"
        "Requer token JWT no header `Authorization: Bearer <token>`."
    ),
)
def model_info(_user: dict = Depends(get_current_user)) -> ModelInfoResponse:
    """Metadados e disponibilidade do modelo de crédito.

    Args:
        _user: Usuário autenticado (injetado via JWT).

    Returns:
        ModelInfoResponse com metadados e status do pipeline.
    """
    loaded = is_model_loaded()
    model_status = (
        "disponível"
        if loaded
        else "indisponível — execute o DAG 'credit_score_etl' no Airflow"
    )

    return ModelInfoResponse(
        model_name="LightGBM Credit Classifier — Home Credit Default Risk",
        model_path=str(MODEL_PATH.resolve()),
        features_numeric=NUMERIC_COLS,
        features_categorical=CATEGORICAL_COLS,
        status=model_status,
        loaded=loaded,
    )
