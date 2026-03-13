"""
Schemas Pydantic para a API REST de Credit Score.

Define os contratos de entrada e saída de todos os endpoints,
garantindo validação automática de dados e documentação OpenAPI.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums de domínio
# ---------------------------------------------------------------------------


class ScoreBand(str, Enum):
    """Faixa de risco de crédito, de A (menor risco) a E (maior risco)."""

    A = "A"  # 0–20 % probabilidade de inadimplência
    B = "B"  # 20–40 %
    C = "C"  # 40–60 %
    D = "D"  # 60–80 %
    E = "E"  # 80–100 %


class Decision(str, Enum):
    """Decisão de crédito baseada no limiar de probabilidade."""

    APROVADO = "APROVADO"
    NEGADO = "NEGADO"


# ---------------------------------------------------------------------------
# Schemas de autenticação
# ---------------------------------------------------------------------------


class TokenRequest(BaseModel):
    """Credenciais para obtenção de token JWT."""

    username: str = Field(..., description="Nome de usuário")
    password: str = Field(..., description="Senha do usuário")

    model_config = {
        "json_schema_extra": {
            "example": {"username": "admin", "password": "admin123"}
        }
    }


class TokenResponse(BaseModel):
    """Token JWT retornado após autenticação bem-sucedida."""

    access_token: str = Field(..., description="Token JWT de acesso")
    token_type: str = Field("bearer", description="Tipo do token")
    expires_in: int = Field(1800, description="Expiração em segundos")


# ---------------------------------------------------------------------------
# Schemas de predição
# ---------------------------------------------------------------------------


class ApplicationInput(BaseModel):
    """Features brutas de uma solicitação de crédito.

    Os campos correspondem às colunas originais do dataset Home Credit.
    Campos opcionais serão imputados pelo pipeline de pré-processamento.
    """

    # --- Financeiro ---
    AMT_CREDIT: float = Field(..., gt=0, description="Valor total do crédito solicitado (R$)")
    AMT_ANNUITY: Optional[float] = Field(None, gt=0, description="Valor da parcela mensal (R$)")
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Renda anual declarada (R$)")
    AMT_GOODS_PRICE: Optional[float] = Field(None, gt=0, description="Preço do bem financiado (R$)")

    # --- Temporal (valores negativos = dias antes do pedido) ---
    DAYS_BIRTH: int = Field(..., lt=0, description="Dias desde o nascimento (negativo)")
    DAYS_EMPLOYED: int = Field(
        ..., description="Dias de emprego atual (negativo = empregado; 365243 = sem vínculo)"
    )
    DAYS_REGISTRATION: float = Field(
        ..., description="Dias desde o último registro de documento (negativo)"
    )
    DAYS_ID_PUBLISH: int = Field(
        ..., lt=0, description="Dias desde a emissão do documento de identidade (negativo)"
    )

    # --- Demográfico ---
    CNT_FAM_MEMBERS: Optional[float] = Field(None, ge=1, description="Número de membros da família")

    # --- Categórico ---
    CODE_GENDER: Optional[str] = Field(None, description="Gênero: 'M' ou 'F'")
    FLAG_OWN_CAR: Optional[str] = Field(None, description="Possui veículo: 'Y' ou 'N'")
    FLAG_OWN_REALTY: Optional[str] = Field(None, description="Possui imóvel: 'Y' ou 'N'")
    NAME_INCOME_TYPE: Optional[str] = Field(
        None, description="Tipo de renda (ex: 'Working', 'Pensioner')"
    )
    NAME_EDUCATION_TYPE: Optional[str] = Field(
        None, description="Escolaridade (ex: 'Higher education')"
    )
    NAME_FAMILY_STATUS: Optional[str] = Field(
        None, description="Estado civil (ex: 'Married', 'Single / not married')"
    )
    NAME_HOUSING_TYPE: Optional[str] = Field(
        None, description="Tipo de moradia (ex: 'House / apartment')"
    )
    OCCUPATION_TYPE: Optional[str] = Field(
        None, description="Ocupação (ex: 'Laborers', 'Core staff')"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "AMT_CREDIT": 450000.0,
                "AMT_ANNUITY": 25000.0,
                "AMT_INCOME_TOTAL": 180000.0,
                "AMT_GOODS_PRICE": 450000.0,
                "DAYS_BIRTH": -12000,
                "DAYS_EMPLOYED": -3000,
                "DAYS_REGISTRATION": -5000.0,
                "DAYS_ID_PUBLISH": -2000,
                "CNT_FAM_MEMBERS": 2.0,
                "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "Y",
                "FLAG_OWN_REALTY": "Y",
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Higher education",
                "NAME_FAMILY_STATUS": "Married",
                "NAME_HOUSING_TYPE": "House / apartment",
                "OCCUPATION_TYPE": "Laborers",
            }
        }
    }


class PredictionResponse(BaseModel):
    """Resultado da predição de risco de crédito."""

    probability_default: float = Field(
        ..., ge=0.0, le=1.0, description="Probabilidade estimada de inadimplência (0–1)"
    )
    decision: Decision = Field(..., description="Decisão de crédito")
    score_band: ScoreBand = Field(..., description="Faixa de risco (A = menor, E = maior)")
    threshold: float = Field(..., description="Limiar de probabilidade utilizado na decisão")


class BatchInput(BaseModel):
    """Lista de solicitações para predição em lote."""

    applications: list[ApplicationInput] = Field(
        ..., min_length=1, max_length=1000, description="Lista de solicitações (máx. 1000)"
    )


class BatchPredictionResponse(BaseModel):
    """Resultado da predição em lote."""

    predictions: list[PredictionResponse]
    total: int = Field(..., description="Total de predições realizadas")


# ---------------------------------------------------------------------------
# Schemas de informações do modelo
# ---------------------------------------------------------------------------


class ModelInfoResponse(BaseModel):
    """Metadados e status do modelo carregado."""

    model_config = {"protected_namespaces": ()}

    model_name: str
    model_path: str
    features_numeric: list[str]
    features_categorical: list[str]
    status: str
    loaded: bool
