"""
Análises de clustering e detecção de anomalias para segmentação de clientes.

Funções utilitárias de análise pontual — não são modelos de produção.
Utilizadas para exploração e segmentação da base de crédito.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Constantes de domínio
# ---------------------------------------------------------------------------

N_CLUSTERS_PADRAO: int = 5
"""Número padrão de segmentos de clientes."""

CONTAMINACAO_PADRAO: float = 0.05
"""Proporção esperada de anomalias na base (5% por padrão)."""

RANDOM_STATE_PADRAO: int = 42
"""Semente para reprodutibilidade."""

N_COMPONENTES_PCA: int = 2
"""Número de componentes PCA para visualização 2D."""

PERPLEXIDADE_TSNE: int = 30
"""Perplexidade padrão do t-SNE (recomendado entre 5 e 50)."""

MAX_ITER_TSNE: int = 1000
"""Número máximo de iterações do t-SNE."""

TAMANHO_PONTO: int = 10
"""Tamanho dos pontos no scatter plot."""

ALPHA_SCATTER: float = 0.5
"""Transparência dos pontos no scatter plot."""

FIGSIZE_CLUSTERS: tuple[int, int] = (10, 7)
"""Tamanho da figura do scatter plot de clusters."""


logger = logging.getLogger(__name__)


def segmentar_clientes(
    X: np.ndarray | pd.DataFrame,
    n_clusters: int = N_CLUSTERS_PADRAO,
    random_state: int = RANDOM_STATE_PADRAO,
) -> pd.Series:
    """Segmenta clientes em grupos usando KMeans.

    Args:
        X: Matriz de features (sem nulos — pré-processar antes de chamar).
        n_clusters: Número de segmentos desejados.
        random_state: Semente para reprodutibilidade.

    Returns:
        pd.Series com o label do cluster (0 a n_clusters-1) para cada cliente.
        O índice é preservado caso X seja um DataFrame.

    Examples:
        >>> labels = segmentar_clientes(X_scaled, n_clusters=5)
        >>> df["segmento"] = labels.values
    """
    logger.info("Iniciando KMeans | n_clusters=%d", n_clusters)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X)

    indice = X.index if isinstance(X, pd.DataFrame) else None
    resultado = pd.Series(labels, index=indice, name="cluster", dtype=int)

    logger.info(
        "Segmentação concluída | distribuição: %s",
        resultado.value_counts().to_dict(),
    )
    return resultado


def detectar_anomalias(
    X: np.ndarray | pd.DataFrame,
    contamination: float = CONTAMINACAO_PADRAO,
) -> pd.Series:
    """Detecta clientes anômalos usando Isolation Forest.

    O caller é responsável por passar apenas colunas numéricas sem nulos.

    Args:
        X: Matriz de features numéricas sem valores ausentes.
        contamination: Proporção esperada de anomalias na base (entre 0 e 0.5).

    Returns:
        pd.Series com 1 = normal e -1 = anomalia para cada cliente.
        O índice é preservado caso X seja um DataFrame.

    Examples:
        >>> flags = detectar_anomalias(X_numerico, contamination=0.05)
        >>> n_anomalias = (flags == -1).sum()
    """
    logger.info(
        "Iniciando IsolationForest | contamination=%.2f | amostras=%d",
        contamination, len(X),
    )

    iso = IsolationForest(
        contamination=contamination,
        random_state=RANDOM_STATE_PADRAO,
        n_jobs=-1,
    )
    predicoes = iso.fit_predict(X)

    indice = X.index if isinstance(X, pd.DataFrame) else None
    resultado = pd.Series(predicoes, index=indice, name="anomalia", dtype=int)

    n_anomalias = (resultado == -1).sum()
    logger.info(
        "Detecção concluída | anomalias=%d (%.1f%%)",
        n_anomalias, 100 * n_anomalias / len(resultado),
    )
    return resultado


def plot_clusters_2d(
    X: np.ndarray | pd.DataFrame,
    labels: pd.Series,
    title: str = "Segmentação de Clientes",
):
    """Visualiza clusters em 2D após redução de dimensionalidade com PCA.

    Args:
        X: Matriz de features (sem nulos). Deve ter ao menos 2 colunas.
        labels: Series com os labels de cluster (saída de segmentar_clientes).
        title: Título do gráfico.

    Returns:
        matplotlib.figure.Figure com o scatter plot colorido por cluster.
        O chamador é responsável por exibir ou salvar a figura.

    Examples:
        >>> fig = plot_clusters_2d(X_scaled, labels, title="Segmentos de Risco")
        >>> fig.savefig("clusters.png")
    """
    import matplotlib.pyplot as plt

    logger.info("Reduzindo dimensionalidade com PCA para visualização 2D")

    pca = PCA(n_components=N_COMPONENTES_PCA, random_state=RANDOM_STATE_PADRAO)
    X_2d = pca.fit_transform(X)

    variancia_explicada = pca.explained_variance_ratio_

    clusters_unicos = sorted(labels.unique())
    cmap = plt.cm.get_cmap("tab10", len(clusters_unicos))

    fig, ax = plt.subplots(figsize=FIGSIZE_CLUSTERS)

    for i, cluster_id in enumerate(clusters_unicos):
        mascara = labels.values == cluster_id
        ax.scatter(
            X_2d[mascara, 0],
            X_2d[mascara, 1],
            s=TAMANHO_PONTO,
            alpha=ALPHA_SCATTER,
            color=cmap(i),
            label=f"Cluster {cluster_id}",
        )

    ax.set_xlabel(f"PC1 ({variancia_explicada[0]:.1%} var. explicada)")
    ax.set_ylabel(f"PC2 ({variancia_explicada[1]:.1%} var. explicada)")
    ax.set_title(title)
    ax.legend(markerscale=3, loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_clusters_tsne(
    X: np.ndarray | pd.DataFrame,
    labels: pd.Series,
    title: str = "Segmentação de Clientes (t-SNE)",
    perplexity: int = PERPLEXIDADE_TSNE,
    max_iter: int = MAX_ITER_TSNE,
):
    """Visualiza clusters em 2D usando t-SNE.

    t-SNE preserva estrutura local melhor que PCA, sendo mais adequado para
    identificar agrupamentos não-lineares. Para datasets grandes (>10k linhas)
    considere amostrar antes de chamar esta função.

    Args:
        X: Matriz de features (sem nulos). Deve ter ao menos 2 colunas.
        labels: Series com os labels de cluster (saída de segmentar_clientes).
        title: Título do gráfico.
        perplexity: Perplexidade do t-SNE. Valores típicos: 5–50.
        max_iter: Número máximo de iterações de otimização (mínimo 250).

    Returns:
        matplotlib.figure.Figure com o scatter plot colorido por cluster.
        O chamador é responsável por exibir ou salvar a figura.

    Examples:
        >>> fig = plot_clusters_tsne(X_scaled, labels, title="Segmentos de Risco")
        >>> fig.savefig("clusters_tsne.png")
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    logger.info(
        "Iniciando t-SNE | amostras=%d | perplexity=%d | max_iter=%d",
        len(X), perplexity, max_iter,
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=RANDOM_STATE_PADRAO,
    )
    X_2d = tsne.fit_transform(X)

    clusters_unicos = sorted(labels.unique())
    cmap = plt.cm.get_cmap("tab10", len(clusters_unicos))

    fig, ax = plt.subplots(figsize=FIGSIZE_CLUSTERS)

    for i, cluster_id in enumerate(clusters_unicos):
        mascara = labels.values == cluster_id
        ax.scatter(
            X_2d[mascara, 0],
            X_2d[mascara, 1],
            s=TAMANHO_PONTO,
            alpha=ALPHA_SCATTER,
            color=cmap(i),
            label=f"Cluster {cluster_id}",
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend(markerscale=3, loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig
