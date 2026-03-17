"""
Componente de exibição do resultado da predição.

Responsabilidade única: renderizar o cartão de resultado
a partir de um ``PredictionResult``.
"""

import streamlit as st

from src.frontend.api_client import PredictionResult

# Mapeamento de faixa → descrição de risco
_BAND_LABELS = {
    "A": "Risco muito baixo (0–20%)",
    "B": "Risco baixo (20–40%)",
    "C": "Risco moderado (40–60%)",
    "D": "Risco alto (60–80%)",
    "E": "Risco muito alto (80–100%)",
}

_BAND_COLORS = {
    "A": "🟢",
    "B": "🟡",
    "C": "🟠",
    "D": "🔴",
    "E": "🔴",
}


def render_result(result: PredictionResult) -> None:
    """Exibe o cartão de resultado da análise de crédito.

    Args:
        result: Objeto retornado por :meth:`CreditScoreApiClient.predict`.
    """
    st.divider()
    st.subheader("Resultado da Análise")

    approved = result.decision == "APROVADO"
    decision_color = "green" if approved else "red"
    decision_icon = "✅" if approved else "❌"

    st.markdown(
        f"""
        <div style="
            border: 2px solid {decision_color};
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin-bottom: 16px;
        ">
            <h2 style="color: {decision_color}; margin: 0;">
                {decision_icon} {result.decision}
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Probabilidade de Inadimplência",
            value=f"{result.probability_default:.1%}",
        )

    with col2:
        band = result.score_band
        st.metric(
            label="Faixa de Risco",
            value=f"{_BAND_COLORS.get(band, '')} Banda {band}",
        )

    with col3:
        st.metric(
            label="Limiar de Decisão",
            value=f"{result.threshold:.0%}",
        )

    st.progress(
        value=result.probability_default,
        text=_BAND_LABELS.get(result.score_band, ""),
    )
