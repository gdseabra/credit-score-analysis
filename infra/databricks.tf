# ============================================================
# DATABRICKS — Serverless Workflows (Jobs API)
#
# Controlado por:
#   databricks_enabled = true/false  → cria ou destrói o job
#
# Sem cluster dedicado — cada execução usa compute serverless.
# Dependências instaladas via environment spec (client = "2").
# ============================================================

provider "databricks" {
  host  = var.databricks_host
  token = var.databricks_token
}

# ============================================================
# JOB — ETL + Treinamento (serverless)
#
# Executa semanalmente (domingo 03:00 UTC):
#   Tarefa 1: 01_etl_pipeline  (Bronze → Silver → Gold)
#   Tarefa 2: 02_train_model   (Gold → MLflow Production)
# ============================================================

resource "databricks_job" "retrain_pipeline" {
  count = var.databricks_enabled ? 1 : 0

  name = "${var.project_name}-retrain-pipeline"

  # ---- Ambiente serverless com dependências Python ----
  environment {
    environment_key = "default"
    spec {
      client = "2"
      dependencies = [
        "lightgbm==4.3.0",
        "mlflow==2.14.3",
        "boto3",
        "shap==0.45.1",
      ]
    }
  }

  # ---- Tarefa 1: ETL ----
  task {
    task_key        = "etl_pipeline"
    environment_key = "default"

    notebook_task {
      notebook_path = var.databricks_notebook_etl_path
    }

    retry_on_timeout = true
    max_retries      = 1
  }

  # ---- Tarefa 2: Treinamento (depende do ETL) ----
  task {
    task_key        = "train_model"
    environment_key = "default"

    depends_on {
      task_key = "etl_pipeline"
    }

    notebook_task {
      notebook_path = var.databricks_notebook_train_path
    }

    retry_on_timeout = true
    max_retries      = 1
  }

  # ---- Agendamento semanal ----
  schedule {
    quartz_cron_expression = "0 0 3 ? * 1"   # domingo, 03:00 UTC
    timezone_id            = "UTC"
    pause_status           = "UNPAUSED"
  }

  # Notificação por e-mail em caso de falha
  email_notifications {
    on_failure = var.alert_email != "" ? [var.alert_email] : []
  }

  tags = {
    Project = var.project_name
  }
}
