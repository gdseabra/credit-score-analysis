"""
Modelo de deep learning para classificação de risco de crédito.

Implementa uma MLP (Multi-Layer Perceptron) com BatchNorm e Dropout usando
PyTorch. O PyTorch é importado apenas dentro das funções/métodos para não
quebrar o projeto em ambientes sem GPU ou sem a biblioteca instalada.

Nota: pos_weight ≈ n_negativos / n_positivos ≈ 92 / 8 ≈ 11.5 para o
dataset Home Credit, onde ~8% dos clientes são inadimplentes.
"""

import logging

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes de domínio
# ---------------------------------------------------------------------------

HIDDEN_DIMS_PADRAO: list[int] = [256, 128, 64]
"""Dimensões das camadas ocultas da MLP."""

DROPOUT_PADRAO: float = 0.3
"""Taxa de dropout padrão entre as camadas ocultas."""

LEARNING_RATE_PADRAO: float = 1e-3
"""Taxa de aprendizado padrão para o otimizador Adam."""

POS_WEIGHT_PADRAO: float = 11.0
"""Peso da classe positiva para BCEWithLogitsLoss (≈ n_neg / n_pos)."""

EPOCHS_PADRAO: int = 50
"""Número de épocas de treinamento padrão."""

BATCH_SIZE_PADRAO: int = 1024
"""Tamanho do mini-batch padrão."""


logger = logging.getLogger(__name__)


class CreditMLP:
    """Rede neural MLP para classificação binária de risco de crédito.

    Arquitetura: Linear → BatchNorm1d → ReLU → Dropout (repetido por camada)
    seguido de Linear(hidden_dims[-1], 1) + Sigmoid na saída.

    O PyTorch é importado durante a inicialização — se não estiver instalado,
    um ImportError claro é levantado.

    Attributes:
        input_dim (int): Dimensão do vetor de entrada.
        hidden_dims (list[int]): Dimensões das camadas ocultas.
        dropout (float): Taxa de dropout.
        network: Módulo nn.Sequential do PyTorch.

    Examples:
        >>> mlp = CreditMLP(input_dim=50, hidden_dims=[256, 128, 64])
        >>> mlp_trainer = CreditMLPTrainer(model=mlp)
        >>> historico = mlp_trainer.fit(X_train, y_train, X_val, y_val)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = None,
        dropout: float = DROPOUT_PADRAO,
    ) -> None:
        """Constrói a arquitetura MLP com BatchNorm e Dropout.

        Args:
            input_dim: Número de features de entrada.
            hidden_dims: Lista com as dimensões de cada camada oculta.
                Default: [256, 128, 64].
            dropout: Taxa de dropout aplicada após cada camada oculta.

        Raises:
            ImportError: Se PyTorch não estiver instalado.
        """
        try:
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError(
                "PyTorch não está instalado. Execute: pip install torch"
            ) from exc

        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS_PADRAO

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        camadas = []
        dim_entrada = input_dim

        for dim_saida in hidden_dims:
            camadas.extend([
                nn.Linear(dim_entrada, dim_saida),
                nn.BatchNorm1d(dim_saida),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ])
            dim_entrada = dim_saida

        # Camada de saída — Sigmoid para probabilidade
        camadas.append(nn.Linear(dim_entrada, 1))
        camadas.append(nn.Sigmoid())

        self.network = nn.Sequential(*camadas)

        logger.info(
            "CreditMLP criada | input_dim=%d | hidden_dims=%s | dropout=%.2f",
            input_dim, hidden_dims, dropout,
        )


class CreditMLPTrainer:
    """Treina e avalia um CreditMLP usando PyTorch.

    Utiliza BCEWithLogitsLoss com pos_weight para lidar com o desbalanceamento
    severo da base (~8% inadimplentes → pos_weight ≈ 11.5).

    Attributes:
        model (CreditMLP): Modelo a ser treinado.
        lr (float): Taxa de aprendizado.
        pos_weight (float): Peso da classe positiva na loss.

    Examples:
        >>> mlp = CreditMLP(input_dim=50)
        >>> mlp_trainer = CreditMLPTrainer(model=mlp, pos_weight=11.5)
        >>> historico = mlp_trainer.fit(X_train, y_train, X_val, y_val, epochs=30)
        >>> proba = mlp_trainer.predict_proba(X_test)
    """

    def __init__(
        self,
        model: "CreditMLP",
        lr: float = LEARNING_RATE_PADRAO,
        pos_weight: float = POS_WEIGHT_PADRAO,
    ) -> None:
        """Inicializa o trainer com otimizador e função de loss.

        Args:
            model: Instância de CreditMLP a ser treinada.
            lr: Taxa de aprendizado para o otimizador Adam.
            pos_weight: Peso da classe positiva (BCEWithLogitsLoss).
                Calculado como n_negativos / n_positivos.

        Raises:
            ImportError: Se PyTorch não estiver instalado.
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError as exc:
            raise ImportError(
                "PyTorch não está instalado. Execute: pip install torch"
            ) from exc

        self.model = model
        self.lr = lr
        self.pos_weight = pos_weight
        self.logger = logging.getLogger(self.__class__.__name__)

        self._optimizer = optim.Adam(model.network.parameters(), lr=lr)
        self._criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: np.ndarray | pd.DataFrame,
        y_val: np.ndarray | pd.Series,
        epochs: int = EPOCHS_PADRAO,
        batch_size: int = BATCH_SIZE_PADRAO,
    ) -> dict:
        """Treina a MLP e avalia a validação ao final de cada época.

        Args:
            X_train: Features de treino.
            y_train: Rótulos de treino (0/1).
            X_val: Features de validação.
            y_val: Rótulos de validação (0/1).
            epochs: Número de épocas de treinamento.
            batch_size: Tamanho do mini-batch.

        Returns:
            Dicionário com histórico de treinamento:
            - ``train_loss``: lista de loss por época no treino.
            - ``val_loss``: lista de loss por época na validação.
            - ``val_auc_roc``: AUC-ROC final na validação.

        Examples:
            >>> historico = mlp_trainer.fit(X_train, y_train, X_val, y_val)
            >>> print(f"AUC-ROC val: {historico['val_auc_roc']:.4f}")
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import roc_auc_score

        # Converte para FloatTensor
        def _to_tensor(arr):
            if isinstance(arr, (pd.DataFrame, pd.Series)):
                arr = arr.values
            return torch.FloatTensor(arr)

        X_tr_t = _to_tensor(X_train)
        y_tr_t = _to_tensor(y_train).unsqueeze(1)
        X_vl_t = _to_tensor(X_val)
        y_vl_t = _to_tensor(y_val).unsqueeze(1)

        dataset_treino = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(dataset_treino, batch_size=batch_size, shuffle=True)

        historico: dict[str, list] = {"train_loss": [], "val_loss": []}

        self.logger.info(
            "Iniciando treinamento MLP | epochs=%d | batch_size=%d",
            epochs, batch_size,
        )

        for epoca in range(1, epochs + 1):
            # --- Treino ---
            self.model.network.train()
            loss_treino_total = 0.0

            for X_batch, y_batch in loader:
                self._optimizer.zero_grad()
                # Usa a rede sem Sigmoid final para BCEWithLogitsLoss
                logits = self.model.network[:-1](X_batch)
                loss = self._criterion(logits, y_batch)
                loss.backward()
                self._optimizer.step()
                loss_treino_total += loss.item() * len(X_batch)

            loss_treino_media = loss_treino_total / len(X_tr_t)

            # --- Validação ---
            self.model.network.eval()
            with torch.no_grad():
                logits_val = self.model.network[:-1](X_vl_t)
                loss_val = self._criterion(logits_val, y_vl_t).item()

            historico["train_loss"].append(loss_treino_media)
            historico["val_loss"].append(loss_val)

            if epoca % 10 == 0 or epoca == 1:
                self.logger.info(
                    "Época %d/%d | train_loss=%.4f | val_loss=%.4f",
                    epoca, epochs, loss_treino_media, loss_val,
                )

        # AUC-ROC final
        proba_val = self.predict_proba(X_val)
        y_val_np = y_val.values if isinstance(y_val, pd.Series) else np.array(y_val)
        val_auc = float(roc_auc_score(y_val_np, proba_val))
        historico["val_auc_roc"] = val_auc

        self.logger.info("Treinamento concluído | val_auc_roc=%.4f", val_auc)
        return historico

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Retorna as probabilidades de inadimplência como array numpy.

        Args:
            X: Features de entrada (mesma dimensão usada no treino).

        Returns:
            Array numpy de shape (n_amostras,) com probabilidades [0, 1].

        Examples:
            >>> proba = mlp_trainer.predict_proba(X_test)
            >>> auc = roc_auc_score(y_test, proba)
        """
        import torch

        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        X_tensor = torch.FloatTensor(X)

        self.model.network.eval()
        with torch.no_grad():
            proba = self.model.network(X_tensor).squeeze(1).numpy()

        return proba
