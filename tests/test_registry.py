"""Testes do registro de modelos baseado em S3 (com cliente S3 falso em memória)."""

import io
import json
from pathlib import Path

import pytest

from src.registry import s3_registry


class FakeS3:
    """Cliente S3 mínimo em memória: implementa apenas get_object/put_object."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}

    def put_object(self, Bucket, Key, Body, **_kwargs):  # noqa: N803 (assinatura boto3)
        self.store[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, Bucket, Key):  # noqa: N803 (assinatura boto3)
        if (Bucket, Key) not in self.store:
            raise KeyError(f"NoSuchKey: {Key}")
        return {"Body": io.BytesIO(self.store[(Bucket, Key)])}


BUCKET = "credit-score-datalake-test"


@pytest.fixture
def client() -> FakeS3:
    s3_registry._reset_cache()
    return FakeS3()


def test_pointer_round_trip(client):
    """write_pointer seguido de read_pointer recupera run_id e métricas."""
    metrics = {"auc": 0.78, "ks": 0.41}
    written = s3_registry.write_pointer(BUCKET, "run-42", metrics, client=client)

    pointer = s3_registry.read_pointer(BUCKET, client=client)

    assert pointer["run_id"] == "run-42"
    assert pointer["metrics"] == metrics
    assert pointer["promoted_at"] == written["promoted_at"]
    assert "promoted_at" in pointer


def test_upload_candidate_persiste_modelo_e_metricas(client, tmp_path):
    """upload_candidate grava tanto o .joblib quanto o metrics.json."""
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"fake-model-bytes")
    metrics = {"auc": 0.80}

    s3_registry.upload_candidate(BUCKET, "run-1", model_file, metrics, client=client)

    assert (BUCKET, "models/candidates/run-1/model.joblib") in client.store
    assert s3_registry.read_metrics(BUCKET, "run-1", client=client) == metrics


def test_download_model_recupera_bytes(client, tmp_path, monkeypatch):
    """download_model baixa os bytes exatos que foram enviados."""
    monkeypatch.setattr(s3_registry, "_LOCAL_CACHE_DIR", Path(tmp_path))
    model_file = tmp_path / "src.joblib"
    model_file.write_bytes(b"weights-123")
    s3_registry.upload_candidate(BUCKET, "run-7", model_file, {}, client=client)

    dest = s3_registry.download_model(BUCKET, "run-7", client=client)

    assert dest.read_bytes() == b"weights-123"


def test_download_model_usa_cache_no_segundo_acesso(client, tmp_path, monkeypatch):
    """O segundo download do mesmo run_id não toca o S3 (warm start)."""
    monkeypatch.setattr(s3_registry, "_LOCAL_CACHE_DIR", Path(tmp_path))
    model_file = tmp_path / "src.joblib"
    model_file.write_bytes(b"weights-123")
    s3_registry.upload_candidate(BUCKET, "run-7", model_file, {}, client=client)

    first = s3_registry.download_model(BUCKET, "run-7", client=client)
    client.store.clear()  # se o cache falhar, o segundo acesso quebraria
    second = s3_registry.download_model(BUCKET, "run-7", client=client)

    assert first == second


def test_candidate_keys_seguem_convencao():
    """As chaves de candidato seguem o layout documentado."""
    assert s3_registry.candidate_model_key("abc") == "models/candidates/abc/model.joblib"
    assert s3_registry.candidate_metrics_key("abc") == "models/candidates/abc/metrics.json"


def test_read_pointer_json_valido(client):
    """O ponteiro é serializado como JSON válido e parseável."""
    s3_registry.write_pointer(BUCKET, "run-x", {"auc": 0.5}, client=client)
    raw = client.store[(BUCKET, s3_registry.DEFAULT_POINTER_KEY)]
    assert json.loads(raw)["run_id"] == "run-x"
