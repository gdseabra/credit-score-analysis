"""
Semeia o registro de modelos S3 a partir do ``.joblib`` local existente.

Envia o modelo atual como o primeiro candidato e aponta o ponteiro para ele,
de modo que a camada de serving tenha um modelo promovido para carregar antes de
o pipeline de retreino (Fase 3) existir.

Uso:
    python scripts/seed_registry.py --bucket <datalake-bucket>
    python scripts/seed_registry.py --bucket <bucket> --run-id seed-0001 \
        --model-path models/lightgbm_pipeline.joblib
"""

import argparse
import json
import sys
from pathlib import Path

# Permite rodar como `python scripts/seed_registry.py` sem exportar PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.registry import s3_registry  # noqa: E402 (após ajuste do sys.path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semeia o registro de modelos S3.")
    parser.add_argument("--bucket", required=True, help="Bucket do data lake.")
    parser.add_argument("--run-id", default="seed-0001", help="Identificador do candidato semente.")
    parser.add_argument(
        "--model-path",
        default="models/lightgbm_pipeline.joblib",
        help="Caminho local do modelo a enviar.",
    )
    parser.add_argument(
        "--metrics",
        default="{}",
        help="JSON com métricas conhecidas do modelo (opcional).",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"Modelo não encontrado: {model_path}")

    metrics = json.loads(args.metrics)

    s3_registry.upload_candidate(args.bucket, args.run_id, model_path, metrics)
    pointer = s3_registry.write_pointer(args.bucket, args.run_id, metrics)

    print("Registro semeado com sucesso:")
    print(json.dumps(pointer, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
