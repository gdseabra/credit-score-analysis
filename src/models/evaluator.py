"""
Módulo de avaliação de modelos de risco de crédito.

Implementa métricas de negócio (AUC-ROC, Gini, KS Statistic) e utilitários
de visualização seguindo o padrão estatístico da indústria de crédito.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Constantes de domínio
# ---------------------------------------------------------------------------

THRESHOLD_PADRAO: float = 0.5
"""Limiar padrão para conversão de probabilidade em classe binária."""

COR_ROC: str = "steelblue"
"""Cor da curva ROC no gráfico."""

COR_DIAGONAL: str = "gray"
"""Cor da linha diagonal (classificador aleatório) no gráfico ROC."""

ALPHA_DIAGONAL: float = 0.5
"""Transparência da linha diagonal."""

FIGSIZE_PADRAO: tuple[int, int] = (8, 6)
"""Tamanho padrão da figura matplotlib."""


logger = logging.getLogger(__name__)


class CreditEvaluator:
    """Avalia modelos de classificação de risco de crédito.

    Calcula métricas de negócio relevantes para a indústria de crédito:
    AUC-ROC, coeficiente de Gini, estatística KS, F1, Precisão, Recall e
    Log-Loss.

    Todos os métodos são estáticos — a classe funciona como namespace de
    funções relacionadas à avaliação.

    Examples:
        >>> metricas = CreditEvaluator.evaluate(y_true, y_pred_proba)
        >>> print(f"AUC-ROC: {metricas['auc_roc']:.4f}")
        >>> fig = CreditEvaluator.plot_roc_curve(y_true, y_pred_proba, label="LightGBM")
    """

    @staticmethod
    def evaluate(y_true: pd.Series | np.ndarray, y_pred_proba: np.ndarray) -> dict:
        """Calcula o conjunto completo de métricas de avaliação.

        Args:
            y_true: Rótulos verdadeiros (0 = adimplente, 1 = inadimplente).
            y_pred_proba: Probabilidades preditas para a classe positiva (inadimplência).

        Returns:
            Dicionário com as seguintes chaves:
            - ``auc_roc``: Área sob a curva ROC.
            - ``gini``: Coeficiente de Gini (2 × AUC − 1).
            - ``ks_stat``: Estatística KS (máx. separação entre TPR e FPR).
            - ``f1``: F1-score com threshold de 0.5.
            - ``precision``: Precisão com threshold de 0.5.
            - ``recall``: Recall com threshold de 0.5.
            - ``log_loss``: Log-loss (entropia cruzada binária).

        Raises:
            ValueError: Se y_true e y_pred_proba tiverem tamanhos diferentes.

        Examples:
            >>> metricas = CreditEvaluator.evaluate(y_true, proba)
            >>> metricas["gini"]
            0.6243
        """
        auc_roc = roc_auc_score(y_true, y_pred_proba)
        gini = 2.0 * auc_roc - 1.0

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        ks_stat = float(np.max(np.abs(tpr - fpr)))

        y_pred_bin = (y_pred_proba >= THRESHOLD_PADRAO).astype(int)

        return {
            "auc_roc": float(auc_roc),
            "gini": float(gini),
            "ks_stat": float(ks_stat),
            "f1": float(f1_score(y_true, y_pred_bin, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred_bin, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred_bin, zero_division=0)),
            "log_loss": float(log_loss(y_true, y_pred_proba)),
        }

    @staticmethod
    def plot_roc_curve(
        y_true: pd.Series | np.ndarray,
        y_pred_proba: np.ndarray,
        label: str = "",
    ):
        """Plota a curva ROC com AUC e KS Statistic anotados.

        Args:
            y_true: Rótulos verdadeiros.
            y_pred_proba: Probabilidades preditas para a classe positiva.
            label: Rótulo descritivo exibido na legenda (ex: "LightGBM").

        Returns:
            matplotlib.figure.Figure com a curva ROC plotada.
            O chamador é responsável por exibir ou salvar a figura.

        Examples:
            >>> fig = CreditEvaluator.plot_roc_curve(y_true, proba, label="RF")
            >>> fig.savefig("roc_curve.png")
        """
        import matplotlib.pyplot as plt

        metricas = CreditEvaluator.evaluate(y_true, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

        nome_curva = f"{label} " if label else ""
        rotulo = (
            f"{nome_curva}(AUC={metricas['auc_roc']:.4f}, "
            f"KS={metricas['ks_stat']:.4f})"
        )

        fig, ax = plt.subplots(figsize=FIGSIZE_PADRAO)
        ax.plot(fpr, tpr, color=COR_ROC, lw=2, label=rotulo)
        ax.plot(
            [0, 1], [0, 1],
            color=COR_DIAGONAL, lw=1, linestyle="--", alpha=ALPHA_DIAGONAL,
            label="Classificador Aleatório",
        )

        # Anotação do ponto de KS máximo
        idx_ks = int(np.argmax(np.abs(tpr - fpr)))
        ax.annotate(
            f"KS={metricas['ks_stat']:.3f}",
            xy=(fpr[idx_ks], tpr[idx_ks]),
            xytext=(fpr[idx_ks] + 0.05, tpr[idx_ks] - 0.08),
            arrowprops={"arrowstyle": "->", "color": "black"},
            fontsize=9,
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Taxa de Falsos Positivos (FPR)")
        ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)")
        ax.set_title("Curva ROC — Risco de Crédito")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        return fig
