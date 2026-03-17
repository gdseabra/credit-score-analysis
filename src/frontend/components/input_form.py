"""
Formulário de entrada de dados do cliente.

Responsabilidade única: renderizar o formulário e converter os valores
inseridos pelo usuário no payload esperado pelo endpoint POST /predict/.

Conversões aplicadas (UX → API):
  - Idade em anos         → DAYS_BIRTH        (negativo)
  - Anos de emprego       → DAYS_EMPLOYED      (negativo ou 365243)
  - Anos desde registro   → DAYS_REGISTRATION  (negativo)
  - Anos desde emissão ID → DAYS_ID_PUBLISH    (negativo)
"""

from typing import Optional

import streamlit as st

# Opções dos campos categóricos conforme o dataset Home Credit
_INCOME_TYPES = [
    "Working",
    "Commercial associate",
    "Pensioner",
    "State servant",
    "Unemployed",
    "Student",
    "Businessman",
    "Maternity leave",
]

_EDUCATION_TYPES = [
    "Secondary / secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]

_FAMILY_STATUS = [
    "Single / not married",
    "Married",
    "Civil marriage",
    "Widow",
    "Separated",
]

_HOUSING_TYPES = [
    "House / apartment",
    "Rented apartment",
    "With parents",
    "Municipal apartment",
    "Office apartment",
    "Co-op apartment",
]

_OCCUPATION_TYPES = [
    "Laborers",
    "Core staff",
    "Accountants",
    "Managers",
    "Drivers",
    "Sales staff",
    "Cleaning staff",
    "Cooking staff",
    "Private service staff",
    "Medicine staff",
    "Security staff",
    "High skill tech staff",
    "Waiters/barmen staff",
    "Low-skill Laborers",
    "Realty agents",
    "Secretaries",
    "IT staff",
    "HR staff",
]


def render_input_form() -> Optional[dict]:
    """Renderiza o formulário de solicitação de crédito.

    Returns:
        Dict com payload pronto para o endpoint ``POST /predict/``,
        ou ``None`` se o formulário ainda não foi submetido.
    """
    with st.form("credit_form"):
        st.subheader("Dados Financeiros")
        col1, col2 = st.columns(2)
        with col1:
            amt_credit = st.number_input(
                "Valor do crédito (R$)", min_value=1.0, value=450_000.0, step=1_000.0
            )
            amt_income = st.number_input(
                "Renda anual (R$)", min_value=1.0, value=180_000.0, step=1_000.0
            )
        with col2:
            amt_annuity = st.number_input(
                "Parcela mensal (R$) — opcional", min_value=0.0, value=25_000.0, step=500.0
            )
            amt_goods = st.number_input(
                "Preço do bem (R$) — opcional", min_value=0.0, value=450_000.0, step=1_000.0
            )

        st.subheader("Dados Pessoais")
        col3, col4 = st.columns(2)
        with col3:
            age_years = st.number_input("Idade (anos)", min_value=18, max_value=100, value=35)
            cnt_family = st.number_input("Membros da família", min_value=1, max_value=20, value=2)
            gender = st.selectbox("Gênero", ["M", "F"])
        with col4:
            own_car = st.selectbox("Possui veículo?", ["N", "Y"])
            own_realty = st.selectbox("Possui imóvel?", ["N", "Y"])
            family_status = st.selectbox("Estado civil", _FAMILY_STATUS)

        col5, col6 = st.columns(2)
        with col5:
            education = st.selectbox("Escolaridade", _EDUCATION_TYPES)
        with col6:
            housing = st.selectbox("Tipo de moradia", _HOUSING_TYPES)

        st.subheader("Dados Profissionais")
        col7, col8 = st.columns(2)
        with col7:
            income_type = st.selectbox("Tipo de renda", _INCOME_TYPES)
            occupation = st.selectbox("Ocupação", ["(não informado)"] + _OCCUPATION_TYPES)
        with col8:
            employed = st.checkbox("Possui vínculo empregatício atual?", value=True)
            years_employed = st.number_input(
                "Anos no emprego atual",
                min_value=0,
                max_value=50,
                value=5,
                disabled=not employed,
            )

        st.subheader("Documentação")
        col9, col10 = st.columns(2)
        with col9:
            years_registration = st.number_input(
                "Anos desde o último registro de documento", min_value=0, max_value=60, value=10
            )
        with col10:
            years_id = st.number_input(
                "Anos desde a emissão do documento de identidade",
                min_value=0,
                max_value=30,
                value=5,
            )

        submitted = st.form_submit_button("Analisar Crédito", use_container_width=True, type="primary")

    if not submitted:
        return None

    return {
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": amt_annuity if amt_annuity > 0 else None,
        "AMT_INCOME_TOTAL": amt_income,
        "AMT_GOODS_PRICE": amt_goods if amt_goods > 0 else None,
        "DAYS_BIRTH": -int(age_years * 365),
        "DAYS_EMPLOYED": -int(years_employed * 365) if employed else 365243,
        "DAYS_REGISTRATION": -float(years_registration * 365),
        "DAYS_ID_PUBLISH": -int(years_id * 365),
        "CNT_FAM_MEMBERS": float(cnt_family),
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing,
        "OCCUPATION_TYPE": occupation if occupation != "(não informado)" else None,
    }
