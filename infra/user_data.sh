#!/bin/bash
# Bootstrap script — executado automaticamente na primeira inicializacao do EC2.
# Instala o MLflow e configura como servico systemd.

set -euo pipefail
exec > /var/log/user_data.log 2>&1

echo "=== Atualizando pacotes ==="
yum update -y
yum install python3.11 python3.11-pip -y

echo "=== Instalando MLflow + boto3 ==="
python3.11 -m pip install --upgrade pip
python3.11 -m pip install mlflow==2.14.3 boto3

echo "=== Criando diretorios ==="
mkdir -p /home/ec2-user/mlflow/db
chown -R ec2-user:ec2-user /home/ec2-user/mlflow

echo "=== Configurando MLflow como servico systemd ==="
cat > /etc/systemd/system/mlflow.service << 'SERVICE'
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/mlflow
ExecStart=/usr/bin/python3.11 -m mlflow server \
  --backend-store-uri sqlite:////home/ec2-user/mlflow/db/mlflow.db \
  --default-artifact-root ${s3_artifact_uri} \
  --host 0.0.0.0 \
  --port ${mlflow_port}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable mlflow
systemctl start mlflow

echo "=== MLflow Server iniciado na porta ${mlflow_port} ==="
