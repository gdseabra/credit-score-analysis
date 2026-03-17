"""
Knowledge Base do RAG de explicabilidade de crédito.

Contém as descrições de negócio de cada feature usada pelo modelo.
É o conjunto de "documentos" recuperados pelo RAG antes de chamar o LLM.

Fonte das descrições das features brutas:
    data/HomeCredit_columns_description.csv

As features de domínio (engenhadas) têm descrições próprias documentadas
em src/features/build_features.py (DomainFeatureBuilder._FEATURE_DESCRIPTIONS).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureDescription:
    """Metadados de uma feature para uso no prompt do LLM."""

    name: str
    label: str
    description: str
    high_risk_direction: str  # "high" = valor alto aumenta risco | "low" = valor baixo aumenta risco | "none"


# ---------------------------------------------------------------------------
# Descrições das features brutas (do dataset Home Credit)
# ---------------------------------------------------------------------------

_RAW_FEATURES: list[FeatureDescription] = [
    FeatureDescription(
        name="AMT_CREDIT",
        label="Valor do crédito",
        description="Valor total do crédito solicitado.",
        high_risk_direction="high",
    ),
    FeatureDescription(
        name="AMT_ANNUITY",
        label="Parcela mensal",
        description="Valor da parcela mensal do empréstimo.",
        high_risk_direction="high",
    ),
    FeatureDescription(
        name="AMT_INCOME_TOTAL",
        label="Renda anual",
        description="Renda anual declarada pelo cliente.",
        high_risk_direction="low",
    ),
    FeatureDescription(
        name="AMT_GOODS_PRICE",
        label="Preço do bem",
        description="Preço do bem financiado (para empréstimos ao consumidor).",
        high_risk_direction="high",
    ),
    FeatureDescription(
        name="DAYS_BIRTH",
        label="Idade do cliente",
        description="Idade do cliente no momento da solicitação (em dias, valor negativo).",
        high_risk_direction="low",  # mais jovem (menos negativo) = mais risco
    ),
    FeatureDescription(
        name="DAYS_EMPLOYED",
        label="Tempo de emprego",
        description=(
            "Há quantos dias o cliente está no emprego atual. "
            "Valor 365243 indica ausência de vínculo empregatício formal."
        ),
        high_risk_direction="low",  # menos tempo empregado = mais risco
    ),
    FeatureDescription(
        name="DAYS_REGISTRATION",
        label="Tempo desde último registro",
        description="Há quantos dias o cliente atualizou seu último registro de documento.",
        high_risk_direction="none",
    ),
    FeatureDescription(
        name="DAYS_ID_PUBLISH",
        label="Validade do documento de identidade",
        description="Há quantos dias o documento de identidade foi emitido.",
        high_risk_direction="none",
    ),
    FeatureDescription(
        name="CNT_FAM_MEMBERS",
        label="Membros da família",
        description="Número de membros da família do cliente.",
        high_risk_direction="high",
    ),
    FeatureDescription(
        name="CODE_GENDER",
        label="Gênero",
        description="Gênero do cliente.",
        high_risk_direction="none",
    ),
    FeatureDescription(
        name="FLAG_OWN_CAR",
        label="Possui veículo",
        description="Indica se o cliente possui veículo próprio.",
        high_risk_direction="none",
    ),
    FeatureDescription(
        name="FLAG_OWN_REALTY",
        label="Possui imóvel",
        description="Indica se o cliente possui imóvel próprio.",
        high_risk_direction="low",  # não ter imóvel = mais risco
    ),
    FeatureDescription(
        name="NAME_INCOME_TYPE",
        label="Tipo de renda",
        description="Categoria da fonte de renda do cliente (trabalhador, aposentado, servidor público, etc.).",
        high_risk_direction="none",
    ),
    FeatureDescription(
        name="NAME_EDUCATION_TYPE",
        label="Escolaridade",
        description="Nível de escolaridade mais alto concluído pelo cliente.",
        high_risk_direction="none",
    ),
    FeatureDescription(
        name="NAME_FAMILY_STATUS",
        label="Estado civil",
        description="Situação familiar do cliente (casado, solteiro, divorciado, etc.).",
        high_risk_direction="none",
    ),
    FeatureDescription(
        name="NAME_HOUSING_TYPE",
        label="Tipo de moradia",
        description="Situação de moradia do cliente (casa própria, aluguel, com pais, etc.).",
        high_risk_direction="none",
    ),
    FeatureDescription(
        name="OCCUPATION_TYPE",
        label="Ocupação",
        description="Tipo de ocupação profissional do cliente.",
        high_risk_direction="none",
    ),
]

# ---------------------------------------------------------------------------
# Descrições das features de domínio (engenhadas)
# ---------------------------------------------------------------------------

_DOMAIN_FEATURES: list[FeatureDescription] = [
    FeatureDescription(
        name="CREDIT_INCOME_RATIO",
        label="Proporção crédito/renda",
        description=(
            "Razão entre o valor do crédito e a renda anual. "
            "Valores altos indicam alto comprometimento da capacidade de pagamento."
        ),
        high_risk_direction="high",
    ),
    FeatureDescription(
        name="ANNUITY_INCOME_RATIO",
        label="Comprometimento de renda com parcelas",
        description=(
            "Razão entre a parcela mensal e a renda anual. "
            "Mede o quanto da renda será comprometido com o pagamento."
        ),
        high_risk_direction="high",
    ),
    FeatureDescription(
        name="CREDIT_TERM_MONTHS",
        label="Prazo do empréstimo (meses)",
        description="Prazo implícito do empréstimo calculado como crédito dividido pela parcela.",
        high_risk_direction="high",
    ),
    FeatureDescription(
        name="AGE_YEARS",
        label="Idade (anos)",
        description="Idade do cliente em anos.",
        high_risk_direction="low",  # mais jovem = mais risco
    ),
    FeatureDescription(
        name="EMPLOYED_YEARS",
        label="Tempo de emprego (anos)",
        description="Tempo no emprego atual em anos. Clientes sem vínculo têm valor nulo.",
        high_risk_direction="low",  # menos tempo = mais risco
    ),
    FeatureDescription(
        name="EMPLOYED_TO_AGE_RATIO",
        label="Estabilidade no emprego relativa à idade",
        description=(
            "Razão entre o tempo de emprego e a idade do cliente. "
            "Indica estabilidade profissional relativa."
        ),
        high_risk_direction="low",
    ),
    FeatureDescription(
        name="INCOME_PER_FAMILY_MEMBER",
        label="Renda per capita familiar",
        description="Renda anual dividida pelo número de membros da família.",
        high_risk_direction="low",
    ),
]

# ---------------------------------------------------------------------------
# Índice unificado: nome da feature → FeatureDescription
# ---------------------------------------------------------------------------

FEATURE_INDEX: dict[str, FeatureDescription] = {
    feat.name: feat for feat in _RAW_FEATURES + _DOMAIN_FEATURES
}


def get_description(feature_name: str) -> FeatureDescription | None:
    """Recupera a descrição de uma feature pelo nome exato ou pelo prefixo (one-hot).

    Suporta nomes de features one-hot encoded como ``CODE_GENDER_M`` → ``CODE_GENDER``.

    Args:
        feature_name: Nome da feature (pode ser one-hot como ``FLAG_OWN_CAR_Y``).

    Returns:
        FeatureDescription correspondente, ou None se não encontrada.
    """
    if feature_name in FEATURE_INDEX:
        return FEATURE_INDEX[feature_name]

    # Tenta encontrar a feature pai (remove sufixo do one-hot encoding)
    for known_name in FEATURE_INDEX:
        if feature_name.startswith(known_name + "_"):
            return FEATURE_INDEX[known_name]

    return None
