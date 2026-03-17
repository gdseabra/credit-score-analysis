"""
Painel de explicabilidade SHAP + narrativa LLM.

Responsabilidade única: renderizar o resultado do endpoint /explain/,
exibindo o gráfico de barras SHAP e o texto gerado pelo LLM.
"""

import streamlit as st

from src.frontend.api_client import ExplanationResult

_IMPACT_COLOR_POS = "#e74c3c"   # vermelho — aumenta risco
_IMPACT_COLOR_NEG = "#2ecc71"   # verde — reduz risco


def render_explanation_panel(result: ExplanationResult) -> None:
    """Exibe o painel de explicabilidade.

    Args:
        result: Objeto retornado por :meth:`CreditScoreApiClient.explain`.
    """
    st.divider()
    st.subheader("Por que essa decisão foi tomada?")

    # --- Explicação textual do LLM ---
    st.info(result.explanation, icon="🤖")

    # --- Gráfico de barras SHAP ---
    st.markdown("**Fatores que mais influenciaram a análise:**")

    if not result.shap_features:
        st.warning("SHAP values não disponíveis.")
        return

    features = result.shap_features

    # Ordena por impacto absoluto para exibição
    features_sorted = sorted(features, key=lambda f: abs(f.shap_value))

    labels = [f.label for f in features_sorted]
    values = [f.shap_value for f in features_sorted]
    colors = [_IMPACT_COLOR_POS if v > 0 else _IMPACT_COLOR_NEG for v in values]

    # Gráfico de barras horizontal usando dados nativos do Streamlit
    import pandas as pd

    chart_data = pd.DataFrame(
        {
            "Feature": labels,
            "Impacto SHAP": values,
            "Cor": colors,
            "Direção": ["↑ Aumenta risco" if v > 0 else "↓ Reduz risco" for v in values],
        }
    )

    # Exibe como barras horizontais via st.bar_chart não suporta horizontal,
    # então usamos uma tabela visual + indicadores de progresso
    for feat in reversed(features_sorted):
        col1, col2, col3 = st.columns([3, 2, 5])
        direction = "🔴" if feat.shap_value > 0 else "🟢"
        impact_text = f"{'+' if feat.shap_value > 0 else ''}{feat.shap_value:.3f}"

        with col1:
            st.markdown(f"**{feat.label}**")
        with col2:
            st.markdown(f"{direction} `{impact_text}`")
        with col3:
            st.caption(feat.description)

    # Legenda
    st.caption(
        "🔴 Valor positivo = aumenta a probabilidade de inadimplência  |  "
        "🟢 Valor negativo = reduz a probabilidade de inadimplência"
    )
