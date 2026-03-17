"""
Cálculo de SHAP values para o pipeline de crédito.

Responsabilidade única: dado um pipeline sklearn treinado, extrair o
classificador LightGBM, calcular SHAP values com TreeExplainer e
retornar as top-N features mais impactantes com seus valores.

Por que TreeExplainer (e não KernelExplainer)?
    - TreeExplainer é nativo para modelos baseados em árvore (LightGBM, XGBoost, RF).
    - É ~100x mais rápido que KernelExplainer para inferência em produção.
    - Fornece SHAP values exatos (não aproximados).
"""

import logging
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ShapFeature:
    """Contribuição de uma feature para a predição."""

    name: str
    shap_value: float
    feature_value: float | str | None


class SHAPExplainer:
    """Calcula SHAP values para um sklearn Pipeline com LightGBM como estimador final.

    O pipeline é dividido em duas partes:
    - ``preprocessor``: todos os steps exceto o último (aplica transformações).
    - ``model``: o estimador final (LightGBM), usado pelo TreeExplainer.

    Attributes:
        top_n: Número de features a retornar (por impacto absoluto).

    Examples:
        >>> explainer = SHAPExplainer(pipeline, top_n=5)
        >>> features = explainer.explain(df_raw)
    """

    def __init__(self, pipeline, top_n: int = 6):
        import shap

        self.top_n = top_n
        self._preprocessor = pipeline[:-1]   # todos os steps antes do modelo
        self._model = pipeline[-1]            # LGBMClassifier

        # Inicializa TreeExplainer com o modelo bruto (não o pipeline)
        self._explainer = shap.TreeExplainer(self._model)

        # Obtém nomes das features após o preprocessamento
        self._feature_names = self._get_feature_names()

        logger.info(
            "SHAPExplainer inicializado | features=%d | top_n=%d",
            len(self._feature_names),
            top_n,
        )

    def _get_feature_names(self) -> list[str]:
        """Extrai os nomes das features geradas pelo ColumnTransformer."""
        try:
            column_transformer = self._preprocessor.named_steps["preprocessor"]
            return list(column_transformer.get_feature_names_out())
        except Exception as exc:
            logger.warning("Não foi possível obter feature names: %s", exc)
            return []

    def explain(self, df_raw: pd.DataFrame) -> list[ShapFeature]:
        """Calcula SHAP values e retorna as top-N features mais impactantes.

        Para features categóricas (one-hot encoded), os valores SHAP são
        agregados por feature pai (ex: CODE_GENDER_M + CODE_GENDER_F → CODE_GENDER).

        Args:
            df_raw: DataFrame com uma linha de dados brutos (antes do preprocessing).

        Returns:
            Lista de ShapFeature ordenada por impacto absoluto decrescente.
        """
        # Preprocessa os dados
        X_preprocessed = self._preprocessor.transform(df_raw)

        # Calcula SHAP values — para classificação binária retorna lista [classe_0, classe_1]
        raw_shap = self._explainer.shap_values(X_preprocessed)

        # Pega SHAP values para a classe positiva (inadimplência)
        if isinstance(raw_shap, list):
            shap_values = raw_shap[1][0]   # shape: (n_features,)
        else:
            shap_values = raw_shap[0]

        # Agrega features one-hot pelo nome pai
        aggregated = self._aggregate_shap(shap_values, df_raw)

        # Ordena por impacto absoluto e retorna top-N
        sorted_features = sorted(aggregated, key=lambda f: abs(f.shap_value), reverse=True)
        return sorted_features[: self.top_n]

    def _aggregate_shap(
        self, shap_values: np.ndarray, df_raw: pd.DataFrame
    ) -> list[ShapFeature]:
        """Agrega SHAP values de colunas one-hot na feature pai.

        Args:
            shap_values: Array de SHAP values com shape (n_features,).
            df_raw: DataFrame original para recuperar o valor real da feature.

        Returns:
            Lista de ShapFeature com uma entrada por feature original (não por dummy).
        """
        from src.explainability.knowledge_base import get_description

        # Mapeia feature_name → shap_value acumulado
        aggregated: dict[str, float] = defaultdict(float)

        for feature_name, sv in zip(self._feature_names, shap_values):
            # Remove prefixo "num__" ou "cat__" do ColumnTransformer
            clean_name = feature_name.split("__", 1)[-1] if "__" in feature_name else feature_name

            # Descobre o nome pai (para one-hot: CODE_GENDER_M → CODE_GENDER)
            desc = get_description(clean_name)
            parent_name = desc.name if desc else clean_name

            aggregated[parent_name] += sv

        # Constrói lista de ShapFeature com o valor real da feature
        result = []
        for parent_name, sv in aggregated.items():
            raw_value = df_raw[parent_name].iloc[0] if parent_name in df_raw.columns else None
            result.append(ShapFeature(name=parent_name, shap_value=sv, feature_value=raw_value))

        return result
