# ============================================================
# DATABRICKS — Job com cluster efêmero (new_cluster)
#
# Controlado por:
#   databricks_enabled = true/false  → cria ou destrói o job
#
# O cluster é criado no início de cada execução e destruído
# ao final — sem custo idle. Ambas as tarefas reusam o mesmo
# cluster via job_cluster_key.
#
# Motivo: serverless bloqueia boto3 e spark.hadoop.fs.s3a por
# design (requer Unity Catalog External Location). O new_cluster
# roda no VPC do workspace e acessa S3 normalmente.
# ============================================================

provider "databricks" {
  host  = var.databricks_host
  token = var.databricks_token
}

# ============================================================
# JOB — ETL + Treinamento
#
# Executa semanalmente (domingo 03:00 UTC):
#   Tarefa 1: 01_etl_pipeline  (Bronze → Silver → Gold)
#   Tarefa 2: 02_train_model   (Gold → MLflow Production)
# ============================================================

resource "databricks_job" "retrain_pipeline" {
  count = var.databricks_enabled ? 1 : 0

  name = "${var.project_name}-retrain-pipeline"

  # ---- Cluster efêmero compartilhado entre as tarefas ----
  job_cluster {
    job_cluster_key = "training"

    new_cluster {
      spark_version = "15.4.x-scala2.12"
      node_type_id  = "i3.xlarge"
      num_workers   = 0

      # Requerido com Unity Catalog habilitado
      data_security_mode = "SINGLE_USER"

      spark_conf = {
        "spark.databricks.cluster.profile" = "singleNode"
        "spark.master"                     = "local[*]"
      }

      custom_tags = {
        "ResourceClass" = "SingleNode"
        "Project"       = var.project_name
      }
    }
  }

  # ---- Tarefa 1: ETL ----
  task {
    task_key        = "etl_pipeline"
    job_cluster_key = "training"

    notebook_task {
      notebook_path = var.databricks_notebook_etl_path
    }

    library {
      pypi { package = "lightgbm==4.3.0" }
    }
    library {
      pypi { package = "mlflow==2.14.3" }
    }
    library {
      pypi { package = "boto3" }
    }
    library {
      pypi { package = "shap==0.45.1" }
    }

    retry_on_timeout = true
    max_retries      = 1
  }

  # ---- Tarefa 2: Treinamento (depende do ETL) ----
  task {
    task_key        = "train_model"
    job_cluster_key = "training"

    depends_on {
      task_key = "etl_pipeline"
    }

    notebook_task {
      notebook_path = var.databricks_notebook_train_path
    }

    library {
      pypi { package = "lightgbm==4.3.0" }
    }
    library {
      pypi { package = "mlflow==2.14.3" }
    }
    library {
      pypi { package = "boto3" }
    }
    library {
      pypi { package = "shap==0.45.1" }
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
