"""
Rotas de predição de risco de crédito.

Endpoints:
    POST /predict/         — Predição individual (1 solicitação).
    POST /predict/batch    — Predição em lote (até 1.000 solicitações).

Ambos exigem token JWT válido no header ``Authorization: Bearer <token>``.
O modelo é injetado via dependency ``get_model()``.
"""

import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import get_current_user
from src.api.dependencies import get_model
from src.api.schemas import (
    ApplicationInput,
    BatchInput,
    BatchPredictionResponse,
    Decision,
    PredictionResponse,
    ScoreBand,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Predição"])

# ---------------------------------------------------------------------------
# Constantes de negócio
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD: float = 0.5
"""Limiar de probabilidade para decisão APROVADO / NEGADO."""


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------


def _score_band(probability: float) -> ScoreBand:
    """Converte probabilidade de inadimplência em faixa de risco.

    Args:
        probability: Probabilidade estimada de inadimplência [0, 1].

    Returns:
        ScoreBand correspondente (A = menor risco, E = maior risco).
    """
    if probability < 0.20:
        return ScoreBand.A
    if probability < 0.40:
        return ScoreBand.B
    if probability < 0.60:
        return ScoreBand.C
    if probability < 0.80:
        return ScoreBand.D
    return ScoreBand.E


def _build_dataframe(application: ApplicationInput) -> pd.DataFrame:
    """Converte ApplicationInput em DataFrame compatível com o pipeline sklearn.

    Args:
        application: Dados brutos da solicitação.

    Returns:
        DataFrame com uma linha pronto para ``pipeline.predict_proba()``.
    """
    return pd.DataFrame([application.model_dump()])


def _run_prediction(application: ApplicationInput, model) -> PredictionResponse:
    """Executa a predição para uma única solicitação.

    Args:
        application: Dados da solicitação de crédito.
        model: Pipeline sklearn carregado (pré-processamento + classificador).

    Returns:
        PredictionResponse com probabilidade, decisão e faixa de risco.

    Raises:
        HTTPException 422: Se o pipeline falhar ao processar os dados.
    """
    try:
        df = _build_dataframe(application)
        probability = float(model.predict_proba(df)[:, 1][0])
    except Exception as exc:
        logger.error("Erro durante predição: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Erro ao processar os dados de entrada: {exc}",
        ) from exc

    decision = Decision.NEGADO if probability >= DEFAULT_THRESHOLD else Decision.APROVADO

    return PredictionResponse(
        probability_default=round(probability, 4),
        decision=decision,
        score_band=_score_band(probability),
        threshold=DEFAULT_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/",
    response_model=PredictionResponse,
    summary="Predição individual de risco de crédito",
    description=(
        "Recebe as features brutas de uma solicitação de crédito e retorna:\n\n"
        "- **probability_default**: probabilidade estimada de inadimplência (0–1)\n"
        "- **decision**: `APROVADO` (prob < 0.5) ou `NEGADO` (prob ≥ 0.5)\n"
        "- **score_band**: faixa de risco A–E\n\n"
        "Requer token JWT no header `Authorization: Bearer <token>`."
    ),
)
def predict(
    application: ApplicationInput,
    model=Depends(get_model),
    _user: dict = Depends(get_current_user),
) -> PredictionResponse:
    """Predição de risco para uma solicitação de crédito.

    Args:
        application: Features brutas da solicitação.
        model: Pipeline sklearn injetado via dependency.
        _user: Usuário autenticado (injetado via JWT — usado para auditoria futura).

    Returns:
        Resultado da predição com probabilidade e decisão.
    """
    logger.info("Predição individual | user=%s", _user["username"])
    return _run_prediction(application, model)


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Predição em lote (até 1.000 solicitações)",
    description=(
        "Recebe uma lista de solicitações e retorna uma predição para cada uma.\n\n"
        "Limite máximo: **1.000 solicitações por requisição**.\n\n"
        "Requer token JWT no header `Authorization: Bearer <token>`."
    ),
)
def predict_batch(
    payload: BatchInput,
    model=Depends(get_model),
    _user: dict = Depends(get_current_user),
) -> BatchPredictionResponse:
    """Predição em lote de risco de crédito.

    Args:
        payload: Lista de solicitações de crédito.
        model: Pipeline sklearn injetado via dependency.
        _user: Usuário autenticado.

    Returns:
        Lista de predições com total processado.
    """
    logger.info(
        "Predição em lote | user=%s | n=%d",
        _user["username"],
        len(payload.applications),
    )
    predictions = [_run_prediction(app, model) for app in payload.applications]
    return BatchPredictionResponse(predictions=predictions, total=len(predictions))
