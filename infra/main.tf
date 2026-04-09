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
# IAM — User para Databricks/API acessarem o S3
# ============================================================

resource "aws_iam_user" "s3_access" {
  name = "${var.project_name}-s3-access"

  tags = {
    Project = var.project_name
  }
}

resource "aws_iam_user_policy" "s3_access" {
  name = "S3CreditScoreAccess"
  user = aws_iam_user.s3_access.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
        ]
        Resource = [
          aws_s3_bucket.datalake.arn,
          "${aws_s3_bucket.datalake.arn}/*",
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*",
        ]
      }
    ]
  })
}

resource "aws_iam_access_key" "s3_access" {
  user = aws_iam_user.s3_access.name
}
