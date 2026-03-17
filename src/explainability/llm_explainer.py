"""
Geração de explicação textual via Claude API (RAG).

Padrão RAG aplicado:
    - Retrieve : seleciona as top-N features por SHAP value e recupera
                 suas descrições de negócio da knowledge_base.
    - Augment  : constrói um prompt estruturado com o resultado da
                 predição + contexto recuperado.
    - Generate : chama a Claude API para produzir narrativa em português
                 simples, sem jargão técnico.

Responsabilidade única: receber ShapFeature[] + metadados da predição
e devolver um texto explicativo ao usuário final.
"""

import logging
import os

from src.explainability.knowledge_base import get_description
from src.explainability.shap_explainer import ShapFeature

logger = logging.getLogger(__name__)

_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
_MODEL = "claude-haiku-4-5-20251001"   # rápido e econômico para explicações curtas

_SYSTEM_PROMPT = """Você é um analista de crédito experiente explicando uma decisão de crédito
para o próprio cliente. Sua linguagem deve ser:
- Clara e empática, sem jargão financeiro ou técnico
- Baseada estritamente nas informações fornecidas
- Objetiva: 3 a 4 frases no total
- Em português do Brasil

Nunca mencione nomes de variáveis técnicas (como AMT_CREDIT, DAYS_BIRTH).
Use sempre os rótulos amigáveis fornecidos.
Não invente informações além das fornecidas."""


def _build_context(
    shap_features: list[ShapFeature],
    probability: float,
    decision: str,
    score_band: str,
) -> str:
    """Recupera as descrições das features (Retrieve) e monta o contexto (Augment).

    Args:
        shap_features: Top features ordenadas por impacto SHAP.
        probability: Probabilidade de inadimplência (0-1).
        decision: "APROVADO" ou "NEGADO".
        score_band: Banda A–E.

    Returns:
        String formatada para ser inserida no prompt do LLM.
    """
    lines = [
        f"Resultado da análise: {decision}",
        f"Probabilidade estimada de inadimplência: {probability:.1%}",
        f"Faixa de risco: {score_band}",
        "",
        "Fatores que mais influenciaram esta decisão:",
    ]

    for feat in shap_features:
        desc = get_description(feat.name)
        if desc is None:
            continue

        impact = "aumenta o risco" if feat.shap_value > 0 else "reduz o risco"
        impact_magnitude = abs(feat.shap_value)

        # Formata o valor da feature de forma legível
        value_str = _format_value(feat)

        lines.append(
            f'- {desc.label} (valor: {value_str} | impacto: {impact_magnitude:.3f} → {impact}): '
            f"{desc.description}"
        )

    return "\n".join(lines)


def _format_value(feat: ShapFeature) -> str:
    """Formata o valor de uma feature para exibição legível no prompt."""
    if feat.feature_value is None:
        return "não informado"
    if isinstance(feat.feature_value, float):
        if abs(feat.feature_value) > 100:
            return f"R$ {feat.feature_value:,.0f}" if "AMT" in feat.name else f"{feat.feature_value:,.1f}"
        return f"{feat.feature_value:.2f}"
    return str(feat.feature_value)


def generate_explanation(
    shap_features: list[ShapFeature],
    probability: float,
    decision: str,
    score_band: str,
) -> str:
    """Gera explicação textual amigável via Claude API.

    Implementa o padrão RAG:
        1. Retrieve  — busca descrições das features na knowledge_base
        2. Augment   — constrói prompt com resultado + contexto recuperado
        3. Generate  — chama Claude para produzir a explicação final

    Args:
        shap_features: Features mais impactantes (saída do SHAPExplainer).
        probability: Probabilidade de inadimplência.
        decision: Decisão final ("APROVADO" / "NEGADO").
        score_band: Banda de risco (A–E).

    Returns:
        Texto explicativo em português simples, pronto para exibição ao cliente.

    Raises:
        RuntimeError: Se a ANTHROPIC_API_KEY não estiver configurada.
    """
    if not _ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY não configurada. "
            "Defina a variável de ambiente antes de iniciar a API."
        )

    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "Biblioteca anthropic não instalada. Execute: pip install anthropic"
        ) from exc

    context = _build_context(shap_features, probability, decision, score_band)

    user_message = (
        f"{context}\n\n"
        "Com base nessas informações, escreva uma explicação clara e empática "
        "para o cliente, sem usar nomes técnicos de variáveis."
    )

    logger.info(
        "Chamando Claude API | decision=%s | probability=%.3f | features=%d",
        decision,
        probability,
        len(shap_features),
    )

    client = anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY)

    message = client.messages.create(
        model=_MODEL,
        max_tokens=300,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    explanation = message.content[0].text.strip()
    logger.info("Explicação gerada com sucesso (%d caracteres)", len(explanation))

    return explanation
