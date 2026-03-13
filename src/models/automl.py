"""
AutoML para otimização automática de modelos de risco de crédito.

Utiliza FLAML (Fast and Lightweight AutoML) para exploração automática
do espaço de modelos e hiperparâmetros, com rastreabilidade via MLflow.
"""

import logging

import pandas as pd

# ---------------------------------------------------------------------------
# Constantes de domínio
# ---------------------------------------------------------------------------

TAREFA_CLASSIFICACAO: str = "classification"
"""Tipo de tarefa para o AutoML (classificação binária)."""

METRICA_AUTOML: str = "roc_auc"
"""Métrica de otimização para o AutoML."""

EXPERIMENTO_MLFLOW: str = "credit-score"
"""Nome do experimento no MLflow."""

TEMPO_PADRAO_SEGUNDOS: int = 120
"""Orçamento de tempo padrão para busca AutoML (em segundos)."""

MLFLOW_URI_PADRAO: str = "http://mlflow:5000"
"""URI padrão do servidor MLflow."""


logger = logging.getLogger(__name__)


def rodar_automl(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    tempo_segundos: int = TEMPO_PADRAO_SEGUNDOS,
    mlflow_tracking_uri: str = MLFLOW_URI_PADRAO,
) -> dict:
    """Executa busca automática de modelos e hiperparâmetros com FLAML.

    Treina e avalia modelos automaticamente dentro do orçamento de tempo
    fornecido, otimizando para AUC-ROC. Os resultados e configuração do
    melhor modelo são logados no MLflow.

    Args:
        X_train: Features de treino.
        y_train: Rótulos de treino (0 = adimplente, 1 = inadimplente).
        X_val: Features de validação (usadas para cálculo do val_auc_roc).
        y_val: Rótulos de validação.
        tempo_segundos: Orçamento de tempo em segundos para a busca.
        mlflow_tracking_uri: URI do servidor de rastreamento MLflow.

    Returns:
        Dicionário com:
        - ``best_estimator``: nome do melhor tipo de modelo encontrado.
        - ``best_config``: dict com os melhores hiperparâmetros.
        - ``val_auc_roc``: AUC-ROC do melhor modelo na validação.
        - ``automl``: objeto AutoML ajustado (para predict_proba, etc.).

    Raises:
        ImportError: Se FLAML não estiver instalado.

    Examples:
        >>> resultado = rodar_automl(X_train, y_train, X_val, y_val,
        ...                          tempo_segundos=300)
        >>> print(f"Melhor modelo: {resultado['best_estimator']}")
        >>> print(f"AUC-ROC val: {resultado['val_auc_roc']:.4f}")
    """
    try:
        from flaml import AutoML
    except ImportError as exc:
        raise ImportError(
            "Instale flaml: pip install flaml[automl]"
        ) from exc

    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow não está instalado. Execute: pip install mlflow"
        ) from exc

    from sklearn.metrics import roc_auc_score

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENTO_MLFLOW)

    logger.info(
        "Iniciando AutoML | tempo=%ds | amostras_treino=%d | features=%d",
        tempo_segundos, len(y_train), X_train.shape[1],
    )

    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task=TAREFA_CLASSIFICACAO,
        metric=METRICA_AUTOML,
        time_budget=tempo_segundos,
        verbose=0,
    )

    proba_val = automl.predict_proba(X_val)[:, 1]
    val_auc_roc = float(roc_auc_score(y_val, proba_val))

    best_estimator = automl.best_estimator
    best_config = automl.best_config

    logger.info(
        "AutoML concluído | best_estimator=%s | val_auc_roc=%.4f",
        best_estimator, val_auc_roc,
    )

    with mlflow.start_run():
        mlflow.set_tag("source", "automl")
        mlflow.log_param("best_estimator", best_estimator)
        mlflow.log_param("tempo_budget_segundos", tempo_segundos)
        mlflow.log_params({
            f"config_{k}": v for k, v in best_config.items()
        })
        mlflow.log_metric("val_auc_roc", val_auc_roc)

    return {
        "best_estimator": best_estimator,
        "best_config": best_config,
        "val_auc_roc": val_auc_roc,
        "automl": automl,
    }
