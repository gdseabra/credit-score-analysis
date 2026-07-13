#!/usr/bin/env bash
# ============================================================
# push_serving_image.sh — build + push da imagem de serving para o ECR
#
# Pré-requisito: o repositório ECR já existe
#   (terraform apply -target=aws_ecr_repository.serving)
#
# Uso:
#   ./scripts/push_serving_image.sh [tag]        # tag padrão: latest
# ============================================================
set -euo pipefail

TAG="${1:-latest}"
REGION="$(terraform -chdir=infra output -raw aws_region 2>/dev/null || echo us-east-1)"
REPO="$(terraform -chdir=infra output -raw serving_ecr_repo)"
REGISTRY="${REPO%/*}"

echo "→ Login no ECR ($REGISTRY)"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

echo "→ Build da imagem (linux/arm64 — Lambda Graviton, build nativo em Apple Silicon)"
docker build --platform linux/arm64 -f Dockerfile.lambda -t "${REPO}:${TAG}" .

echo "→ Push para o ECR"
docker push "${REPO}:${TAG}"

echo "✓ Imagem publicada: ${REPO}:${TAG}"
echo "  Agora rode: terraform -chdir=infra apply"
