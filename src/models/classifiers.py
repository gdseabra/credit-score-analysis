"""
Fábrica de classificadores para o projeto Home Credit Default Risk.

Encapsula a criação e configuração dos modelos disponíveis, garantindo que
todos utilizem ``class_weight="balanced"`` (ou equivalente) para lidar com
o desbalanceamento severo (~8% de inadimplentes).
"""

import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Constantes de domínio
# ---------------------------------------------------------------------------

MODELOS_DISPONIVEIS: list[str] = [
    "logistic_regression",
    "random_forest",
    "lightgbm",
    "xgboost",
]
"""Lista de identificadores válidos para uso em CreditClassifier.get_model()."""

N_ESTIMATORS_RF: int = 200
"""Número de árvores para o RandomForest."""

MAX_ITER_LR: int = 1000
"""Máximo de iterações para convergência da Regressão Logística."""

RANDOM_STATE: int = 42
"""Semente global para reprodutibilidade dos experimentos."""


logger = logging.getLogger(__name__)


class CreditClassifier:
    """Fábrica de modelos de classificação de risco de crédito.

    Centraliza a instanciação dos classificadores com hiperparâmetros padrão
    adequados ao problema (desbalanceamento, escala, tipo de dados).

    Todos os modelos retornados são compatíveis com a API do Scikit-Learn
    (``fit``, ``predict``, ``predict_proba``).

    Examples:
        >>> modelo = CreditClassifier.get_model("lightgbm")
        >>> modelo.fit(X_train, y_train)
        >>> proba = modelo.predict_proba(X_val)[:, 1]
    """

    @staticmethod
    def get_model(nome: str, **kwargs):
        """Retorna um classificador instanciado com parâmetros padrão.

        Os kwargs fornecidos sobrescrevem os defaults — útil para tuning ou
        para passar ``scale_pos_weight`` ao XGBoost.

        Args:
            nome: Identificador do modelo. Deve ser um dos valores em
                ``MODELOS_DISPONIVEIS``.
            **kwargs: Parâmetros adicionais que sobrescrevem os defaults
                do estimador selecionado.

        Returns:
            Estimador Scikit-Learn instanciado e pronto para ``fit()``.

        Raises:
            ValueError: Se ``nome`` não estiver em ``MODELOS_DISPONIVEIS``.
            ImportError: Se a biblioteca do modelo solicitado não estiver
                instalada (LightGBM, XGBoost).

        Examples:
            >>> lr = CreditClassifier.get_model("logistic_regression")
            >>> rf = CreditClassifier.get_model("random_forest", n_estimators=100)
            >>> lgbm = CreditClassifier.get_model("lightgbm", n_estimators=500)
            >>> xgb = CreditClassifier.get_model("xgboost", scale_pos_weight=11.5)
        """
        if nome not in MODELOS_DISPONIVEIS:
            raise ValueError(
                f"Modelo '{nome}' não disponível. "
                f"Opções válidas: {MODELOS_DISPONIVEIS}"
            )

        logger.info("Instanciando modelo: %s", nome)

        if nome == "logistic_regression":
            params = dict(
                class_weight="balanced",
                max_iter=MAX_ITER_LR,
                random_state=RANDOM_STATE,
            )
            params.update(kwargs)
            return LogisticRegression(**params)

        if nome == "random_forest":
            params = dict(
                n_estimators=N_ESTIMATORS_RF,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
            params.update(kwargs)
            return RandomForestClassifier(**params)

        if nome == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError as exc:
                raise ImportError(
                    "LightGBM não está instalado. Execute: pip install lightgbm"
                ) from exc

            params = dict(
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=-1,
            )
            params.update(kwargs)
            return lgb.LGBMClassifier(**params)

        if nome == "xgboost":
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise ImportError(
                    "XGBoost não está instalado. Execute: pip install xgboost"
                ) from exc

            params = dict(
                eval_metric="auc",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbosity=0,
            )
            params.update(kwargs)
            return xgb.XGBClassifier(**params)
