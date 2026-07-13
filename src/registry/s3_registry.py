"""
Registro de modelos leve, apoiado em S3.

Em vez de um servidor MLflow sempre ligado, o modelo "promovido" é indicado por
um único objeto JSON (o *ponteiro*):

    models/current/pointer.json  →  {"run_id", "metrics", "promoted_at"}
    models/candidates/<run_id>/model.joblib
    models/candidates/<run_id>/metrics.json

A camada de serving lê o ponteiro no cold start, baixa o ``model.joblib`` do
candidato correspondente e o mantém em cache local. Promover um novo modelo é
apenas reescrever o ponteiro — nenhum redeploy é necessário.

Todas as funções aceitam um ``client`` boto3 injetável para facilitar testes;
quando omitido, um cliente S3 padrão é criado sob demanda.
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_POINTER_KEY: str = os.getenv("MODEL_POINTER_KEY", "models/current/pointer.json")
CANDIDATES_PREFIX: str = "models/candidates"
_LOCAL_CACHE_DIR: Path = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/credit-score-models"))

_downloaded: dict[str, Path] = {}


def _default_client():
    """Cria um cliente S3 boto3 sob demanda (import tardio para não exigir boto3 nos testes)."""
    import boto3

    return boto3.client("s3")


def candidate_model_key(run_id: str) -> str:
    """Chave S3 do artefato de modelo de um candidato."""
    return f"{CANDIDATES_PREFIX}/{run_id}/model.joblib"


def candidate_metrics_key(run_id: str) -> str:
    """Chave S3 do arquivo de métricas de um candidato."""
    return f"{CANDIDATES_PREFIX}/{run_id}/metrics.json"


def read_pointer(
    bucket: str,
    key: str = DEFAULT_POINTER_KEY,
    client: Optional[Any] = None,
) -> dict:
    """Lê o ponteiro do modelo promovido.

    Args:
        bucket: Nome do bucket do data lake.
        key: Chave do objeto ponteiro.
        client: Cliente S3 (injetável). Se ``None``, cria um cliente padrão.

    Returns:
        Dict com ``run_id``, ``metrics`` e ``promoted_at``.
    """
    client = client or _default_client()
    obj = client.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())


def write_pointer(
    bucket: str,
    run_id: str,
    metrics: dict,
    key: str = DEFAULT_POINTER_KEY,
    client: Optional[Any] = None,
) -> dict:
    """Aponta o registro para ``run_id`` (promove o modelo).

    Args:
        bucket: Nome do bucket do data lake.
        run_id: Identificador do candidato a promover.
        metrics: Métricas de avaliação associadas ao modelo.
        key: Chave do objeto ponteiro.
        client: Cliente S3 (injetável).

    Returns:
        O payload do ponteiro que foi gravado.
    """
    client = client or _default_client()
    payload = {
        "run_id": run_id,
        "metrics": metrics,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info("Ponteiro do registro atualizado: run_id=%s", run_id)
    return payload


def upload_candidate(
    bucket: str,
    run_id: str,
    model_path: str | Path,
    metrics: dict,
    client: Optional[Any] = None,
) -> None:
    """Envia o modelo e as métricas de um candidato para o S3 (sem promover).

    Args:
        bucket: Nome do bucket do data lake.
        run_id: Identificador do candidato.
        model_path: Caminho local do ``.joblib`` treinado.
        metrics: Métricas de avaliação a persistir.
        client: Cliente S3 (injetável).
    """
    client = client or _default_client()
    model_bytes = Path(model_path).read_bytes()
    client.put_object(Bucket=bucket, Key=candidate_model_key(run_id), Body=model_bytes)
    client.put_object(
        Bucket=bucket,
        Key=candidate_metrics_key(run_id),
        Body=json.dumps(metrics, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info("Candidato enviado ao registro: run_id=%s", run_id)


def download_model(
    bucket: str,
    run_id: str,
    client: Optional[Any] = None,
) -> Path:
    """Baixa o ``model.joblib`` de um candidato, com cache local por ``run_id``.

    Warm starts do Lambda reaproveitam o arquivo já em ``/tmp`` sem novo download.

    Args:
        bucket: Nome do bucket do data lake.
        run_id: Identificador do candidato.
        client: Cliente S3 (injetável).

    Returns:
        Caminho local do modelo baixado.
    """
    cached = _downloaded.get(run_id)
    if cached is not None and cached.exists():
        return cached

    client = client or _default_client()
    dest = _LOCAL_CACHE_DIR / run_id / "model.joblib"
    dest.parent.mkdir(parents=True, exist_ok=True)
    obj = client.get_object(Bucket=bucket, Key=candidate_model_key(run_id))
    dest.write_bytes(obj["Body"].read())

    _downloaded[run_id] = dest
    logger.info("Modelo baixado do registro: run_id=%s → %s", run_id, dest)
    return dest


def read_metrics(bucket: str, run_id: str, client: Optional[Any] = None) -> dict:
    """Lê as métricas persistidas de um candidato."""
    client = client or _default_client()
    obj = client.get_object(Bucket=bucket, Key=candidate_metrics_key(run_id))
    return json.loads(obj["Body"].read())


def _reset_cache() -> None:
    """Limpa o cache de downloads em memória (uso em testes)."""
    _downloaded.clear()


def _bytesio(data: bytes) -> io.BytesIO:
    """Helper para stubs de teste que precisam de um Body com ``.read()``."""
    return io.BytesIO(data)
