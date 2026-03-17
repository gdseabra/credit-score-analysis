"""
Interface web para análise de crédito — Credit Score Frontend.

Ponto de entrada do Streamlit. Orquestra os componentes sem conter
lógica de negócio ou de comunicação HTTP (responsabilidades delegadas).

Execução local:
    streamlit run src/frontend/app.py

Execução via Docker:
    docker-compose up frontend
"""

import streamlit as st

from src.frontend.api_client import ApiError, CreditScoreApiClient
from src.frontend.components.explanation_panel import render_explanation_panel
from src.frontend.components.input_form import render_input_form
from src.frontend.components.result_card import render_result

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Credit Score — Análise de Crédito",
    page_icon="💳",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Inicialização do estado de sessão
# ---------------------------------------------------------------------------

if "token" not in st.session_state:
    st.session_state.token = None

client = CreditScoreApiClient()

# ---------------------------------------------------------------------------
# Barra lateral — login / status
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    with st.sidebar:
        st.title("💳 Credit Score")
        st.caption("Análise de risco de crédito")
        st.divider()

        api_ok = client.is_healthy()
        st.markdown(
            f"**API:** {'🟢 online' if api_ok else '🔴 offline'}"
        )

        if st.session_state.token:
            st.success("Autenticado")
            if st.button("Sair", use_container_width=True):
                st.session_state.token = None
                st.rerun()
        else:
            st.subheader("Login")
            username = st.text_input("Usuário", value="admin")
            password = st.text_input("Senha", type="password", value="admin123")
            if st.button("Entrar", use_container_width=True, type="primary"):
                _do_login(username, password)


def _do_login(username: str, password: str) -> None:
    try:
        auth = client.login(username, password)
        st.session_state.token = auth.token
        st.rerun()
    except ApiError as e:
        st.sidebar.error(f"Falha no login: {e.detail}")
    except Exception as e:
        st.sidebar.error(f"Erro de conexão: {e}")


# ---------------------------------------------------------------------------
# Tela principal
# ---------------------------------------------------------------------------


def _render_main() -> None:
    st.title("Análise de Risco de Crédito")
    st.markdown(
        "Preencha os dados do cliente e clique em **Analisar Crédito** "
        "para obter a predição do modelo."
    )

    if not st.session_state.token:
        st.info("Faça login na barra lateral para continuar.")
        return

    explain_mode = st.toggle(
        "Solicitar explicação detalhada (SHAP + IA)",
        value=False,
        help="Quando ativado, calcula os fatores determinantes da decisão e gera uma explicação em linguagem natural.",
    )

    payload = render_input_form()

    if payload is None:
        return

    with st.spinner("Consultando modelo..."):
        try:
            if explain_mode:
                result = client.explain(st.session_state.token, payload)
                render_result(result)
                render_explanation_panel(result)
            else:
                result = client.predict(st.session_state.token, payload)
                render_result(result)
        except ApiError as e:
            if e.status_code == 401:
                st.session_state.token = None
                st.error("Sessão expirada. Faça login novamente.")
            else:
                st.error(f"Erro da API ({e.status_code}): {e.detail}")
        except Exception as e:
            st.error(f"Erro inesperado: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_render_sidebar()
_render_main()
