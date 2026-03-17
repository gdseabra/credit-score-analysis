"""Módulo de explicabilidade para o Credit Score API.

Componentes:
    knowledge_base  — descrições das features (documentos do RAG)
    shap_explainer  — cálculo de SHAP values via TreeExplainer
    llm_explainer   — geração de explicação textual via Claude API
"""
