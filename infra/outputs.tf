# ============================================================
# Outputs — valores necessarios para as proximas fases
# ============================================================

# --- S3 ---

output "datalake_bucket" {
  description = "Nome do bucket S3 do Data Lake"
  value       = aws_s3_bucket.datalake.id
}

output "mlflow_bucket" {
  description = "Nome do bucket S3 para artefatos MLflow"
  value       = aws_s3_bucket.mlflow_artifacts.id
}

# --- Databricks ---

output "databricks_job_id" {
  description = "ID do Databricks Job de re-treinamento (vazio se databricks_enabled = false)"
  value       = var.databricks_enabled ? databricks_job.retrain_pipeline[0].id : ""
}
