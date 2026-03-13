"""
Módulo de treinamento e rastreamento de experimentos com MLflow.

Implementa cross-validation estratificada, tuning com RandomizedSearchCV e
persistência de modelos, integrando com MLflow para rastreabilidade completa
dos experimentos.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.models.classifiers import CreditClassifier
from src.models.evaluator import CreditEvaluator

# ---------------------------------------------------------------------------
# Constantes de domínio
# ---------------------------------------------------------------------------

EXPERIMENTO_MLFLOW: str = "credit-score"
"""Nome do experimento no MLflow."""

N_ITER_RANDOM_SEARCH: int = 20
"""Número de iterações para o RandomizedSearchCV."""

SCORING_PADRAO: str = "roc_auc"
"""Métrica de scoring para seleção de modelos durante o tuning."""

RANDOM_STATE: int = 42
"""Semente global para reprodutibilidade."""

MODELS_DIR_PADRAO: str = "models/"
"""Diretório padrão para salvar modelos serializados."""

TAG_CROSS_VALIDATION: str = "cross_validation"
"""Tag MLflow para runs de cross-validation."""

TAG_HYPERPARAMETER_TUNING: str = "hyperparameter_tuning"
"""Tag MLflow para runs de tuning de hiperparâmetros."""


class ModelTrainingError(Exception):
    """Exceção customizada para erros durante o treinamento de modelos."""
    pass


class CreditTrainer:
    """Orquestra o treinamento, avaliação e persistência de modelos de crédito.

    Integra com MLflow para rastreabilidade de experimentos: parâmetros,
    métricas, tags e artefatos são logados automaticamente em cada operação.

    Attributes:
        modelo_nome (str): Identificador do modelo (ver CreditClassifier).
        mlflow_tracking_uri (str): URI do servidor MLflow.

    Examples:
        >>> trainer = CreditTrainer(modelo_nome="lightgbm")
        >>> metricas = trainer.train(X, y, pipeline, cv_folds=5)
        >>> best_params = trainer.tune(X, y, pipeline, param_grid)
        >>> caminho = trainer.save_model(pipeline_fitted, metricas)
    """

    def __init__(
        self,
        modelo_nome: str,
        mlflow_tracking_uri: str = "http://mlflow:5000",
    ) -> None:
        """Inicializa o trainer e configura o experimento no MLflow.

        Args:
            modelo_nome: Nome do modelo a utilizar (ver MODELOS_DISPONIVEIS).
            mlflow_tracking_uri: URI do servidor de rastreamento MLflow.

        Raises:
            ImportError: Se mlflow não estiver instalado.
        """
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                "MLflow não está instalado. Execute: pip install mlflow"
            ) from exc

        self.modelo_nome = modelo_nome
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.logger = logging.getLogger(self.__class__.__name__)

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(EXPERIMENTO_MLFLOW)

        self.logger.info(
            "CreditTrainer inicializado | modelo=%s | mlflow=%s",
            modelo_nome, mlflow_tracking_uri,
        )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pipeline: Pipeline,
        cv_folds: int = 5,
    ) -> dict:
        """Treina o modelo via cross-validation estratificada e loga no MLflow.

        Em cada fold, o pipeline completo é ajustado no conjunto de treino e
        avaliado no conjunto de validação, garantindo ausência de data leakage.

        Args:
            X: Features de entrada (sem TARGET e SK_ID_CURR).
            y: Variável-alvo (0 = adimplente, 1 = inadimplente).
            pipeline: Pipeline Scikit-Learn de pré-processamento (será clonado
                internamente em cada fold — o objeto original não é modificado).
            cv_folds: Número de folds da cross-validation estratificada.

        Returns:
            Dicionário com métricas agregadas (mean ± std) por fold:
            - ``<metrica>_mean``: média da métrica ao longo dos folds.
            - ``<metrica>_std``: desvio padrão da métrica ao longo dos folds.
            - ``n_folds``: número de folds executados.

        Raises:
            ModelTrainingError: Se ocorrer erro durante o ajuste do pipeline.

        Examples:
            >>> metricas = trainer.train(X_train, y_train, pipeline)
            >>> print(f"AUC-ROC médio: {metricas['auc_roc_mean']:.4f}")
        """
        import mlflow
        from sklearn.base import clone

        modelo = CreditClassifier.get_model(self.modelo_nome)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

        historico: dict[str, list[float]] = {
            "auc_roc": [], "gini": [], "ks_stat": [],
            "f1": [], "precision": [], "recall": [], "log_loss": [],
        }

        self.logger.info(
            "Iniciando CV | modelo=%s | folds=%d | amostras=%d | features=%d",
            self.modelo_nome, cv_folds, len(y), X.shape[1],
        )

        taxa_inadimplencia = float(y.mean())

        try:
            for fold_idx, (idx_treino, idx_val) in enumerate(skf.split(X, y), start=1):
                X_treino, X_val = X.iloc[idx_treino], X.iloc[idx_val]
                y_treino, y_val = y.iloc[idx_treino], y.iloc[idx_val]

                pipeline_fold = clone(pipeline)
                # Adiciona o modelo como etapa final do pipeline clonado
                pipeline_fold.steps.append(("modelo", clone(modelo)))

                pipeline_fold.fit(X_treino, y_treino)
                proba_val = pipeline_fold.predict_proba(X_val)[:, 1]

                metricas_fold = CreditEvaluator.evaluate(y_val, proba_val)
                for metrica, valor in metricas_fold.items():
                    historico[metrica].append(valor)

                self.logger.info(
                    "Fold %d/%d | AUC=%.4f | KS=%.4f",
                    fold_idx, cv_folds,
                    metricas_fold["auc_roc"], metricas_fold["ks_stat"],
                )

        except Exception as exc:
            raise ModelTrainingError(
                f"Erro no treinamento do modelo '{self.modelo_nome}': {exc}"
            ) from exc

        # Agrega métricas
        metricas_agregadas: dict = {"n_folds": cv_folds}
        for metrica, valores in historico.items():
            metricas_agregadas[f"{metrica}_mean"] = float(np.mean(valores))
            metricas_agregadas[f"{metrica}_std"] = float(np.std(valores))

        # Loga no MLflow
        with mlflow.start_run():
            mlflow.set_tag("source", TAG_CROSS_VALIDATION)
            mlflow.log_params({
                "modelo_nome": self.modelo_nome,
                "cv_folds": cv_folds,
                "n_features": X.shape[1],
                "n_amostras": len(y),
                "taxa_inadimplencia": round(taxa_inadimplencia, 4),
            })
            mlflow.log_metrics({
                "auc_roc_mean": metricas_agregadas["auc_roc_mean"],
                "auc_roc_std": metricas_agregadas["auc_roc_std"],
                "gini_mean": metricas_agregadas["gini_mean"],
                "ks_stat_mean": metricas_agregadas["ks_stat_mean"],
                "f1_mean": metricas_agregadas["f1_mean"],
            })

        self.logger.info(
            "CV concluída | AUC=%.4f±%.4f | Gini=%.4f | KS=%.4f",
            metricas_agregadas["auc_roc_mean"],
            metricas_agregadas["auc_roc_std"],
            metricas_agregadas["gini_mean"],
            metricas_agregadas["ks_stat_mean"],
        )

        return metricas_agregadas

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pipeline: Pipeline,
        param_grid: dict,
        cv_folds: int = 5,
    ) -> dict:
        """Realiza tuning de hiperparâmetros com RandomizedSearchCV.

        Args:
            X: Features de entrada.
            y: Variável-alvo.
            pipeline: Pipeline de pré-processamento. O modelo será adicionado
                como etapa final para o search.
            param_grid: Dicionário com os parâmetros e distribuições para
                busca aleatória. As chaves devem seguir o padrão
                ``"modelo__<param>"`` para referenciar o modelo no pipeline.
            cv_folds: Número de folds na cross-validation interna.

        Returns:
            Dicionário com os melhores hiperparâmetros encontrados
            (``best_params_`` do RandomizedSearchCV).

        Raises:
            ModelTrainingError: Se ocorrer erro durante a busca.

        Examples:
            >>> param_grid = {"modelo__n_estimators": [100, 200, 500],
            ...               "modelo__max_depth": [3, 5, 7]}
            >>> best_params = trainer.tune(X, y, pipeline, param_grid)
        """
        import mlflow
        from sklearn.base import clone

        self.logger.info(
            "Iniciando RandomizedSearchCV | modelo=%s | n_iter=%d | folds=%d",
            self.modelo_nome, N_ITER_RANDOM_SEARCH, cv_folds,
        )

        try:
            modelo = CreditClassifier.get_model(self.modelo_nome)
            pipeline_tune = clone(pipeline)
            pipeline_tune.steps.append(("modelo", modelo))

            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

            search = RandomizedSearchCV(
                estimator=pipeline_tune,
                param_distributions=param_grid,
                n_iter=N_ITER_RANDOM_SEARCH,
                scoring=SCORING_PADRAO,
                cv=skf,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
            search.fit(X, y)

        except Exception as exc:
            raise ModelTrainingError(
                f"Erro durante o tuning do modelo '{self.modelo_nome}': {exc}"
            ) from exc

        best_auc = float(search.best_score_)
        best_params = search.best_params_

        # Loga no MLflow
        with mlflow.start_run():
            mlflow.set_tag("source", TAG_HYPERPARAMETER_TUNING)
            mlflow.log_params({"modelo_nome": self.modelo_nome, **best_params})
            mlflow.log_metric("best_auc_roc", best_auc)

        self.logger.info(
            "Tuning concluído | best_auc_roc=%.4f | best_params=%s",
            best_auc, best_params,
        )

        return best_params

    def save_model(
        self,
        pipeline: Pipeline,
        metricas: dict,
        path: str = MODELS_DIR_PADRAO,
    ) -> str:
        """Persiste o pipeline treinado em disco e loga como artefato no MLflow.

        Args:
            pipeline: Pipeline Scikit-Learn já ajustado com ``fit()``.
            metricas: Dicionário de métricas para rastreabilidade (logado
                como parâmetros no MLflow).
            path: Diretório onde o arquivo será salvo.

        Returns:
            Caminho absoluto do arquivo ``.joblib`` salvo.

        Raises:
            ModelTrainingError: Se ocorrer erro ao salvar o arquivo.

        Examples:
            >>> caminho = trainer.save_model(pipeline_fitted, metricas)
            >>> print(f"Modelo salvo em: {caminho}")
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                "joblib não está instalado. Execute: pip install joblib"
            ) from exc

        import mlflow
        import mlflow.sklearn

        model_dir = Path(path)
        model_dir.mkdir(parents=True, exist_ok=True)

        nome_arquivo = f"{self.modelo_nome}_pipeline.joblib"
        caminho_arquivo = str(model_dir / nome_arquivo)

        try:
            joblib.dump(pipeline, caminho_arquivo)
        except Exception as exc:
            raise ModelTrainingError(
                f"Erro ao salvar o modelo em '{caminho_arquivo}': {exc}"
            ) from exc

        with mlflow.start_run():
            mlflow.log_params({
                f"metricas_{k}": round(v, 6) if isinstance(v, float) else v
                for k, v in metricas.items()
            })
            mlflow.sklearn.log_model(pipeline, artifact_path="pipeline")
            mlflow.log_artifact(caminho_arquivo)

        self.logger.info("Modelo salvo em: %s", caminho_arquivo)
        return caminho_arquivo
