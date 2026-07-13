# ============================================================
# FASE 1 — Serving (scale-to-zero)
#
# API Gateway (HTTP API) → Lambda (imagem de container) → modelo do registro S3.
#
# Ordem de aplicação (a imagem precisa existir antes do Lambda):
#   1. terraform apply -target=aws_ecr_repository.serving
#   2. ./scripts/push_serving_image.sh          # build + push :latest
#   3. terraform apply                           # cria Lambda + API Gateway
#
# Custo em repouso: ~$0 (Lambda e API Gateway HTTP cobram por requisição).
# ============================================================

variable "anthropic_api_key" {
  description = "Chave da API Anthropic (Claude) para as narrativas de explicação."
  type        = string
  sensitive   = true
  default     = ""
}

variable "serving_image_tag" {
  description = "Tag da imagem de serving no ECR."
  type        = string
  default     = "latest"
}

data "aws_caller_identity" "current" {}

# ---- ECR: repositório da imagem de serving ----
resource "aws_ecr_repository" "serving" {
  name                 = "${var.project_name}-serving"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project = var.project_name
    Layer   = "serving"
  }
}

# ---- IAM: role de execução do Lambda (sem chaves estáticas) ----
resource "aws_iam_role" "serving_lambda" {
  name = "${var.project_name}-serving-lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = {
    Project = var.project_name
  }
}

# Logs no CloudWatch
resource "aws_iam_role_policy_attachment" "serving_logs" {
  role       = aws_iam_role.serving_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Leitura do registro de modelos no data lake
resource "aws_iam_role_policy" "serving_s3_read" {
  name = "S3RegistryRead"
  role = aws_iam_role.serving_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.datalake.arn, "${aws_s3_bucket.datalake.arn}/*"]
      }
    ]
  })
}

# ---- Lambda: API FastAPI via Mangum ----
resource "aws_lambda_function" "serving" {
  function_name = "${var.project_name}-serving"
  role          = aws_iam_role.serving_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.serving.repository_url}:${var.serving_image_tag}"
  architectures = ["arm64"] # Graviton — build nativo em Apple Silicon, ~20% mais barato
  timeout       = 30
  memory_size   = 3008 # mais memória = mais CPU no Lambda → cold start (imports + download S3) dentro do init budget

  environment {
    variables = {
      DATALAKE_BUCKET   = aws_s3_bucket.datalake.id
      MODEL_POINTER_KEY = "models/current/pointer.json"
      ANTHROPIC_API_KEY = var.anthropic_api_key
    }
  }

  tags = {
    Project = var.project_name
    Layer   = "serving"
  }
}

# ---- API Gateway HTTP API (proxy total para o Lambda) ----
resource "aws_apigatewayv2_api" "serving" {
  name          = "${var.project_name}-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "serving" {
  api_id                 = aws_apigatewayv2_api.serving.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.serving.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.serving.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.serving.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.serving.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.serving.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.serving.execution_arn}/*/*"
}

# ---- Outputs ----
output "serving_api_url" {
  description = "URL pública da API (aponte o frontend Streamlit para cá)."
  value       = aws_apigatewayv2_stage.default.invoke_url
}

output "serving_ecr_repo" {
  description = "URL do repositório ECR da imagem de serving."
  value       = aws_ecr_repository.serving.repository_url
}

output "aws_region" {
  description = "Região AWS em uso (consumida pelo push_serving_image.sh)."
  value       = var.aws_region
}
