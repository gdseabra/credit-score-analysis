"""
Cliente HTTP para o Credit Score API.

Responsabilidade única: toda comunicação com a API REST fica aqui.
O restante da aplicação nunca chama `requests` diretamente.
"""

import os
from dataclasses import dataclass

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Tipos de retorno
# ---------------------------------------------------------------------------


@dataclass
class AuthResult:
    token: str
    token_type: str
    expires_in: int


@dataclass
class ShapFeatureResult:
    name: str
    label: str
    shap_value: float
    feature_value: float | str | None
    description: str


@dataclass
class PredictionResult:
    probability_default: float
    decision: str
    score_band: str
    threshold: float


@dataclass
class ExplanationResult:
    probability_default: float
    decision: str
    score_band: str
    threshold: float
    shap_features: list[ShapFeatureResult]
    explanation: str


# ---------------------------------------------------------------------------
# Exceções de domínio
# ---------------------------------------------------------------------------


class ApiError(Exception):
    """Erro retornado pela API (4xx / 5xx)."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


# ---------------------------------------------------------------------------
# Cliente
# ---------------------------------------------------------------------------


class CreditScoreApiClient:
    """Encapsula todas as chamadas ao Credit Score API."""

    def __init__(self, base_url: str = API_URL):
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Autenticação
    # ------------------------------------------------------------------

    def login(self, username: str, password: str) -> AuthResult:
        """Autentica e retorna o token JWT.

        Raises:
            ApiError: Se as credenciais forem inválidas.
        """
        response = self._session.post(
            f"{self._base_url}/auth/token",
            json={"username": username, "password": password},
            timeout=10,
        )
        self._raise_for_status(response)
        data = response.json()
        return AuthResult(
            token=data["access_token"],
            token_type=data["token_type"],
            expires_in=data["expires_in"],
        )

    # ------------------------------------------------------------------
    # Predição
    # ------------------------------------------------------------------

    def predict(self, token: str, payload: dict) -> PredictionResult:
        """Envia uma solicitação de crédito e retorna a predição.

        Args:
            token: Token JWT obtido via :meth:`login`.
            payload: Dict com os campos de ``ApplicationInput``.

        Raises:
            ApiError: Se a API retornar erro.
        """
        response = self._session.post(
            f"{self._base_url}/predict/",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        self._raise_for_status(response)
        data = response.json()
        return PredictionResult(
            probability_default=data["probability_default"],
            decision=data["decision"],
            score_band=data["score_band"],
            threshold=data["threshold"],
        )

    # ------------------------------------------------------------------
    # Explicação SHAP + LLM
    # ------------------------------------------------------------------

    def explain(self, token: str, payload: dict) -> ExplanationResult:
        """Envia solicitação ao endpoint /explain/ e retorna predição + SHAP + narrativa.

        Args:
            token: Token JWT obtido via :meth:`login`.
            payload: Dict com os campos de ``ApplicationInput``.

        Raises:
            ApiError: Se a API retornar erro.
        """
        response = self._session.post(
            f"{self._base_url}/explain/",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )
        self._raise_for_status(response)
        data = response.json()

        shap_features = [
            ShapFeatureResult(
                name=f["name"],
                label=f["label"],
                shap_value=f["shap_value"],
                feature_value=f.get("feature_value"),
                description=f["description"],
            )
            for f in data.get("shap_features", [])
        ]

        return ExplanationResult(
            probability_default=data["probability_default"],
            decision=data["decision"],
            score_band=data["score_band"],
            threshold=data["threshold"],
            shap_features=shap_features,
            explanation=data["explanation"],
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_healthy(self) -> bool:
        """Retorna True se a API estiver online e o modelo carregado."""
        try:
            response = self._session.get(f"{self._base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    # ------------------------------------------------------------------
    # Auxiliar
    # ------------------------------------------------------------------

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise ApiError(status_code=response.status_code, detail=str(detail))
