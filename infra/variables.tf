variable "project_name" {
  description = "Nome base para todos os recursos AWS"
  type        = string
  default     = "credit-score"
}

variable "aws_region" {
  description = "Regiao AWS (us-east-1 tem mais servicos no Free Tier)"
  type        = string
  default     = "us-east-1"
}

# ============================================================
# DATABRICKS
# ============================================================

variable "databricks_host" {
  description = "URL do workspace Databricks (ex: https://adb-1234567890.azuredatabricks.net)"
  type        = string
  default     = ""
}

variable "databricks_token" {
  description = "Personal Access Token do Databricks (User Settings → Access Tokens)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "databricks_enabled" {
  description = "Se false, destrói todos os recursos Databricks (clusters + jobs). Use no fallback para community."
  type        = bool
  default     = true
}

variable "databricks_notebook_etl_path" {
  description = "Caminho do notebook ETL no workspace Databricks (ex: /Shared/01_etl_pipeline)"
  type        = string
  default     = "/Shared/01_etl_pipeline"
}

variable "databricks_notebook_train_path" {
  description = "Caminho do notebook de treinamento no workspace Databricks (ex: /Shared/02_train_model)"
  type        = string
  default     = "/Shared/02_train_model"
}

variable "alert_email" {
  description = "E-mail para notificação de falhas nos Databricks Jobs. Deixe vazio para desativar."
  type        = string
  default     = ""
}
