"""
Rota de explicabilidade — POST /explain/.

Recebe os dados de uma solicitação de crédito e retorna:
    - A predição completa (probabilidade, decisão, banda)
    - As top features por impacto SHAP com descrições de negócio
    - Uma explicação textual gerada pelo LLM via RAG

Requer token JWT válido no header ``Authorization: Bearer <token>``.
"""

import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import get_current_user
from src.api.dependencies import get_model, get_shap_explainer
from src.api.routes.predict import DEFAULT_THRESHOLD, _score_band
from src.api.schemas import (
    ApplicationInput,
    Decision,
    ExplanationResponse,
    ShapFeatureSchema,
)
from src.explainability.knowledge_base import get_description
from src.explainability.llm_explainer import generate_explanation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["Explicabilidade"])


@router.post(
    "/",
    response_model=ExplanationResponse,
    summary="Predição com explicação SHAP + LLM",
    description=(
        "Retorna a predição de risco de crédito enriquecida com:\n\n"
        "- **shap_features**: as features mais determinantes para a decisão\n"
        "- **explanation**: explicação em linguagem natural gerada por IA\n\n"
        "A explicação segue o padrão RAG: os SHAP values identificam as features "
        "relevantes, suas descrições de negócio são recuperadas da knowledge base "
        "e o LLM gera uma narrativa empática em português.\n\n"
        "Requer token JWT no header `Authorization: Bearer <token>`."
    ),
)
def explain(
    application: ApplicationInput,
    model=Depends(get_model),
    shap_explainer=Depends(get_shap_explainer),
    _user: dict = Depends(get_current_user),
) -> ExplanationResponse:
    """Predição + explicação SHAP + narrativa LLM para uma solicitação de crédito.

    Args:
        application: Dados brutos da solicitação.
        model: Pipeline sklearn injetado via dependency.
        shap_explainer: SHAPExplainer injetado via dependency.
        _user: Usuário autenticado (injetado via JWT).

    Returns:
        ExplanationResponse com predição, SHAP features e explicação textual.
    """
    logger.info("Explain request | user=%s", _user["username"])

    df = pd.DataFrame([application.model_dump()])

    # --- Predição ---
    try:
        probability = float(model.predict_proba(df)[:, 1][0])
    except Exception as exc:
        logger.error("Erro na predição: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Erro ao processar os dados de entrada: {exc}",
        ) from exc

    decision = Decision.NEGADO if probability >= DEFAULT_THRESHOLD else Decision.APROVADO
    score_band = _score_band(probability)

    # --- SHAP values ---
    try:
        shap_features = shap_explainer.explain(df)
    except Exception as exc:
        logger.error("Erro no cálculo SHAP: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Erro ao calcular SHAP values: {exc}",
        ) from exc

    # Converte para schema de resposta, enriquecendo com labels e descrições
    shap_schemas = []
    for feat in shap_features:
        desc = get_description(feat.name)
        shap_schemas.append(
            ShapFeatureSchema(
                name=feat.name,
                label=desc.label if desc else feat.name,
                shap_value=round(feat.shap_value, 4),
                feature_value=feat.feature_value,
                description=desc.description if desc else "",
            )
        )

    # --- Explicação LLM (RAG) ---
    try:
        explanation = generate_explanation(
            shap_features=shap_features,
            probability=probability,
            decision=decision.value,
            score_band=score_band.value,
        )
    except Exception as exc:
        logger.error("Erro ao gerar explicação LLM: %s", exc)
        explanation = (
            "Não foi possível gerar a explicação textual no momento. "
            "Consulte as features SHAP para entender os fatores da decisão."
        )

    return ExplanationResponse(
        probability_default=round(probability, 4),
        decision=decision,
        score_band=score_band,
        threshold=DEFAULT_THRESHOLD,
        shap_features=shap_schemas,
        explanation=explanation,
    )
