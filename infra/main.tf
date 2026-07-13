terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    databricks = {
      source  = "databricks/databricks"
      version = "~> 1.40"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

locals {
  bucket_datalake = "${var.project_name}-datalake-${random_id.suffix.hex}"
  bucket_mlflow   = "${var.project_name}-mlflow-${random_id.suffix.hex}"
}

# Sufixo aleatorio para garantir nomes unicos de bucket
resource "random_id" "suffix" {
  byte_length = 4
}

# ============================================================
# FASE 1 — S3 (Data Lake + MLflow Artifacts)
# ============================================================

resource "aws_s3_bucket" "datalake" {
  bucket = local.bucket_datalake

  tags = {
    Project = var.project_name
    Layer   = "data-lake"
  }
}

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = local.bucket_mlflow

  tags = {
    Project = var.project_name
    Layer   = "mlflow"
  }
}

# Bloqueia acesso publico nos dois buckets
resource "aws_s3_bucket_public_access_block" "datalake" {
  bucket = aws_s3_bucket.datalake.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "mlflow" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Cria os prefixos (camadas) do Data Lake
resource "aws_s3_object" "bronze" {
  bucket = aws_s3_bucket.datalake.id
  key    = "bronze/"
}

resource "aws_s3_object" "silver" {
  bucket = aws_s3_bucket.datalake.id
  key    = "silver/"
}

resource "aws_s3_object" "gold" {
  bucket = aws_s3_bucket.datalake.id
  key    = "gold/"
}

# ============================================================
# Acesso ao S3
# ============================================================
# Design AWS-native: cada compute (Lambda de serving, futuro Fargate/Lambda do
# pipeline) recebe uma IAM Role com permissões mínimas — sem usuário IAM de
# chaves estáticas. A role da Lambda de serving está em serving.tf.
# O acesso local (CLI/Terraform) usa o profile de bootstrap do desenvolvedor.
