# AWS Production-Ready ML Pipeline — Execution Plan

A phased plan to turn this project into a production-ready, AWS-native credit-scoring
pipeline that simulates a real deployment: users get live predictions, synthetic data
is ingested continuously, and models are retrained + validated + promoted automatically.

**Decisions locked in:**
- **Serving:** scale-to-zero — Lambda (container image) + API Gateway HTTP API
- **Training/orchestration:** AWS-native — Step Functions + Fargate/Lambda (replaces Databricks)
- **Ingestion:** scheduled batch — EventBridge Scheduler → generator Lambda
- **Registry:** lightweight S3-pointer convention (no always-on MLflow server)
- **Frontend:** Streamlit stays on Railway, pointed at the API Gateway URL

Each **step** ends with a **✅ Checkpoint** — a concrete command + expected result you run
before moving on. Do not proceed past a failing checkpoint.

---

## Conventions (read once)

**S3 layout** (inside the existing `…-datalake-<suffix>` bucket):
```
bronze/dt=YYYY-MM-DD/HH/applications-<uuid>.parquet   ← raw synthetic applications
silver/dt=YYYY-MM-DD/…                                 ← cleaned (AnomalyHandler)
gold/dt=YYYY-MM-DD/features.parquet                    ← model-ready features + TARGET
reference/train_reference.parquet                      ← frozen distribution for PSI baseline
models/candidates/<run_id>/model.joblib + metrics.json ← every training run
models/current/pointer.json  → { run_id, metrics, promoted_at }
predictions/dt=YYYY-MM-DD/…                             ← logged live predictions (for drift)
```

**Naming:** all resources prefixed `credit-score-` (matches `var.project_name`).

**Env vars used by the app code:**
| Var | Meaning | Default |
|---|---|---|
| `DATALAKE_BUCKET` | data lake bucket name | (from Terraform output) |
| `MODEL_POINTER_KEY` | pointer object key | `models/current/pointer.json` |
| `DRIFT_STEP` | drift knob for the generator (0 = no drift) | `0.0` |
| `ANTHROPIC_API_KEY` | Claude narrative | (secret) |

**Git:** work on a branch per phase (`feat/aws-phase-1-serving`, etc.), open a PR, let CI pass, merge.

---

## Prerequisites (one-time)

- [ ] AWS account with the existing Terraform stack applied (`infra/`): datalake + mlflow buckets exist.
- [ ] AWS CLI v2 configured locally (`aws sts get-caller-identity` returns your account).
- [ ] Terraform ≥ 1.5, Docker, and `jq` installed locally.
- [ ] Decide an AWS region and confirm it matches `var.aws_region`.

**✅ Checkpoint (prereqs):**
```bash
aws sts get-caller-identity            # → your account id, no error
terraform -chdir=infra output          # → datalake/mlflow bucket names
docker version && jq --version         # → both print versions
```

---

# Phase 1 — Model Registry + Serving on AWS

**Goal:** a public HTTPS endpoint (API Gateway) that returns a live credit-score prediction,
backed by a Lambda that loads the *currently promoted* model from S3. Streamlit (on Railway)
points at it. No training changes yet.

**Outcome demo:** `curl` the API Gateway URL → JSON decision + probability + band.

### Step 1.1 — Seed the registry from the existing model
Upload the current `models/lightgbm_pipeline.joblib` as the first "promoted" model so
serving has something to load.

- Write `src/registry/s3_registry.py` with:
  - `read_pointer(bucket, key) -> dict`
  - `write_pointer(bucket, key, run_id, metrics)` (atomic-ish: put JSON)
  - `download_model(bucket, run_id, dest_path) -> Path` (with local `/tmp` cache)
  - `upload_candidate(bucket, run_id, model_path, metrics)`
- One-off script/CLI: upload the joblib to `models/candidates/seed-0001/model.joblib`,
  write `metrics.json` (use whatever eval numbers you have, or `{}`), and point
  `models/current/pointer.json` at `seed-0001`.

**✅ Checkpoint 1.1:**
```bash
aws s3 ls s3://$DATALAKE_BUCKET/models/candidates/seed-0001/   # → model.joblib + metrics.json
aws s3 cp s3://$DATALAKE_BUCKET/models/current/pointer.json - | jq .run_id   # → "seed-0001"
```
Also add a unit test for `read_pointer`/`write_pointer` round-trip (moto or a stubbed client).
`pytest tests/test_registry.py -v` → green.

### Step 1.2 — Teach the serving layer to load from S3
Extend [src/api/dependencies.py](../src/api/dependencies.py) `load_model()` with a new
**first-priority** source: S3 pointer → download → `joblib.load`. Keep MLflow + local `.joblib`
as fallbacks (order: S3 pointer → MLflow → local). Cache the downloaded file in `/tmp` keyed by
`run_id` so warm Lambda invocations skip the download.

- Reuse the existing `NUMERIC_COLS` / `CATEGORICAL_COLS` — do **not** redefine them.
- Add `src/registry/s3_registry.py` calls behind the existing `_load_*` pattern.

**✅ Checkpoint 1.2 (local, before any Lambda):**
```bash
export DATALAKE_BUCKET=<your-bucket>
export PYTHONPATH=$PWD
uvicorn src.api.main:app --port 8000 &
curl -s localhost:8000/health | jq        # → {"status": "...", model loaded: true}
# grab a token, then predict (use an example from schemas.py):
curl -s -X POST localhost:8000/predict/ -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" -d @examples/predict_sample.json | jq
# → decision + probability + band, model sourced from S3 (check logs say "S3 pointer")
```
Confirm the log line proves it loaded from S3, not the local `.joblib` fallback.

### Step 1.3 — Package the API as a Lambda container image
- Add `Dockerfile.lambda` (base `public.ecr.aws/lambda/python:3.11`), install
  `requirements.api.txt` + `mangum`, copy `src/`, set handler to a new
  `src/serving/lambda_handler.py` that wraps the FastAPI app with `Mangum(app)`.
- Verify the image size fits Lambda (< 10 GB; it will be ~1–2 GB with shap/lightgbm).

**✅ Checkpoint 1.3 (local container):**
```bash
docker build -f Dockerfile.lambda -t credit-score-lambda .
docker run --rm -p 9000:8080 \
  -e DATALAKE_BUCKET=$DATALAKE_BUCKET -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... \
  credit-score-lambda &
# Lambda RIE emulator — invoke the health route:
curl -s "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -d '{"requestContext":{"http":{"method":"GET","path":"/health"}},"rawPath":"/health","version":"2.0"}' | jq
# → 200 body with model loaded
```

### Step 1.4 — Terraform: ECR + Lambda + API Gateway + IAM role
Add to `infra/` (new file `infra/serving.tf`):
- `aws_ecr_repository.serving`
- `aws_iam_role` for the Lambda with a policy allowing `s3:GetObject`/`ListBucket` on the
  datalake bucket + `bedrock`/none (Claude is HTTPS out) + `logs:*`. **Use a role, not the
  static IAM user** currently in `main.tf`.
- `aws_lambda_function.serving` (package_type = `Image`, `image_uri` = ECR digest, `timeout=30`,
  `memory_size=2048`, env vars).
- `aws_apigatewayv2_api` (HTTP API) + integration + `$default` route + stage.
- `aws_lambda_permission` for API Gateway to invoke.
- Output the invoke URL.

Push the image to ECR first (manually or via a small `make push`), then `terraform apply`.

**✅ Checkpoint 1.4:**
```bash
aws ecr describe-images --repository-name credit-score-serving   # → your pushed tag
API_URL=$(terraform -chdir=infra output -raw serving_api_url)
curl -s $API_URL/health | jq                                     # → 200, model loaded
# Cold start first call may take 2–5s; second call should be <500ms.
time curl -s -X POST $API_URL/predict/ -H "Authorization: Bearer <token>" \
     -d @examples/predict_sample.json | jq .decision             # → APROVADO / NEGADO
```
Check CloudWatch Logs group `/aws/lambda/credit-score-serving` shows the request.

### Step 1.5 — Point Streamlit at the AWS endpoint
Set `API_URL` on Railway to the API Gateway URL. No code change if
[src/frontend/api_client.py](../src/frontend/api_client.py) already reads `API_URL`.

**✅ Checkpoint 1.5 (end-to-end, manual/UI):**
- [ ] Open the Railway Streamlit URL.
- [ ] Fill the applicant form, submit.
- [ ] A decision + probability + SHAP panel + Claude narrative render.
- [ ] CloudWatch shows the Lambda invocation from that submission.

> **Phase 1 done when:** a human using the web UI gets a prediction served by AWS Lambda,
> and the model came from the S3 registry pointer.

---

# Phase 2 — Continuous Synthetic Ingestion

**Goal:** fresh, realistic applications land in `bronze/` on a schedule, with a tunable
drift knob so you can later force distribution shift and watch PSI react.

**Outcome demo:** `bronze/` fills up hourly; you can dial drift up and see the data change.

### Step 2.1 — Synthetic generator module
Write `src/ingestion/synthetic_generator.py`:
- `generate_batch(n: int, drift_step: float, seed=None) -> pd.DataFrame` producing all raw
  columns the transform expects (the pre-engineering source columns — reuse the schema, not the
  engineered ones, which the transform derives).
- The `drift_step` shifts a few distributions over time (e.g. mean `AMT_INCOME_TOTAL`,
  `DAYS_EMPLOYED`) so PSI climbs when you raise it.
- Include a plausible `TARGET` (default 0/1) so downstream training has labels.

**✅ Checkpoint 2.1 (local):**
```bash
python -c "from src.ingestion.synthetic_generator import generate_batch; \
  df=generate_batch(1000, drift_step=0.0); print(df.shape); print(df.dtypes)"
# → (1000, N) with expected raw columns and a TARGET
```
Unit test: `drift_step=0` vs `drift_step=1.0` produces a measurably higher PSI on
`AMT_INCOME_TOTAL` (uses your `population_stability_index`). `pytest tests/test_generator.py -v`.

### Step 2.2 — Generator Lambda + EventBridge schedule
- `src/ingestion/lambda_handler.py`: read `DRIFT_STEP` from env (or SSM param for live tuning),
  call `generate_batch`, write parquet to `bronze/dt=…/HH/…`.
- Terraform `infra/ingestion.tf`: Lambda (zip or image), IAM role (`s3:PutObject` on bronze),
  `aws_scheduler_schedule` (EventBridge Scheduler) at `rate(1 hour)`, and an SSM parameter
  `/credit-score/drift_step` the Lambda reads.

**✅ Checkpoint 2.2:**
```bash
aws lambda invoke --function-name credit-score-generator /dev/stdout | jq   # manual fire → ok
aws s3 ls s3://$DATALAKE_BUCKET/bronze/ --recursive | tail                   # → a new parquet
# wait for one scheduled tick (or check after an hour):
aws s3 ls s3://$DATALAKE_BUCKET/bronze/ --recursive | wc -l                  # → grows over time
```

### Step 2.3 — Freeze the PSI reference distribution
Snapshot the *current* training feature distribution to `reference/train_reference.parquet`.
This is the baseline every future PSI comparison uses.

**✅ Checkpoint 2.3:**
```bash
aws s3 ls s3://$DATALAKE_BUCKET/reference/train_reference.parquet   # → exists, non-zero size
```

> **Phase 2 done when:** bronze grows automatically every hour, and raising the SSM `drift_step`
> demonstrably shifts the generated distributions (verified with a quick PSI calc).

---

# Phase 3 — Automated Retrain + Validation + Promotion Gate

**Goal:** the star of the show. On a schedule, the pipeline transforms new data, trains a
candidate, validates it (metrics **+ PSI**), and **promotes it only if it beats the incumbent
and drift is within bounds** — otherwise it keeps the old model and alerts you.

**Outcome demo:** force drift → pipeline runs → candidate rejected with an SNS alert; or a clean
run → candidate promoted → serving picks up the new model on next cold start (no redeploy).

### Step 3.1 — Transform step (bronze → silver → gold) as a Lambda
Refactor the logic from [dags/credit_score_etl.py](../dags/credit_score_etl.py) `transform`
task and [src/features/build_features.py](../src/features/build_features.py) into
`src/pipeline/transform.py` (`run_transform(bucket) -> gold_uri`). **Drop the Postgres `load`
step** — write gold parquet to S3 instead.

**✅ Checkpoint 3.1 (local against S3):**
```bash
python -c "from src.pipeline.transform import run_transform; print(run_transform('$DATALAKE_BUCKET'))"
aws s3 ls s3://$DATALAKE_BUCKET/gold/ --recursive | tail   # → features.parquet written
```
Sanity: gold row count ≈ sum of bronze rows consumed; engineered columns present.

### Step 3.2 — Train step on Fargate
`src/pipeline/train.py` (`run_training(bucket) -> run_id`): read gold, train via
[src/models/trainer.py](../src/models/trainer.py) `CreditTrainer`, then
`s3_registry.upload_candidate(run_id, model, metrics)`. Package as a container, run as an
ECS Fargate task (training is heavier than Lambda's limits).

- Terraform `infra/pipeline.tf`: ECS cluster, Fargate task definition (the training image),
  task role (`s3:Get/Put` on datalake), CloudWatch log group.

**✅ Checkpoint 3.2:**
```bash
aws ecs run-task --cluster credit-score --task-definition credit-score-train …   # manual run
# watch logs, then:
aws s3 ls s3://$DATALAKE_BUCKET/models/candidates/ --recursive | tail   # → new run_id dir + metrics.json
aws s3 cp s3://$DATALAKE_BUCKET/models/candidates/<run_id>/metrics.json - | jq   # → AUC/KS/Gini present
```

### Step 3.3 — Validate + promotion gate (Lambda)
`src/pipeline/validate.py` (`run_validation(bucket, candidate_run_id) -> {promote: bool, reason}`):
1. Load candidate; score a holdout; compute metrics via
   [src/models/evaluator.py](../src/models/evaluator.py) `CreditEvaluator.evaluate`.
2. Compute PSI on the candidate's training features vs `reference/train_reference.parquet`
   using [src/monitoring/stability.py](../src/monitoring/stability.py) `stability_report`.
3. Read incumbent metrics from `models/current/pointer.json`.
4. **Gate:** promote iff `candidate.AUC >= incumbent.AUC - epsilon` **and** max PSI
   `< PSI_SIGNIFICANT_THRESHOLD`. Return decision + reason.

`src/pipeline/promote.py`: if gate passes, `write_pointer` to the candidate; else publish to SNS.

**✅ Checkpoint 3.3 (two cases, run locally against S3 first):**
```bash
# Case A — healthy candidate should promote:
python -c "from src.pipeline.validate import run_validation; print(run_validation('$DATALAKE_BUCKET','<good_run_id>'))"
# → {'promote': True, ...}
# Case B — degraded/drifted candidate should be rejected:
#   (temporarily set reference to a very different distribution, or feed a bad model)
# → {'promote': False, 'reason': 'PSI 0.31 > 0.25'}
```
Unit-test the gate logic in isolation (`tests/test_gate.py`) with synthetic metric dicts —
promote-if-better and drift-reject branches both covered.

### Step 3.4 — Orchestrate with Step Functions
Terraform state machine: `Transform(Lambda) → Train(Fargate task, .sync) → Validate(Lambda)
→ Choice(promote?) → [Promote(Lambda) | Alert(SNS)]`. EventBridge Scheduler triggers it daily.

**✅ Checkpoint 3.4:**
```bash
SM_ARN=$(terraform -chdir=infra output -raw pipeline_state_machine_arn)
aws stepfunctions start-execution --state-machine-arn $SM_ARN   # manual full run
# in the console (or describe-execution) confirm every state = SUCCEEDED and the
# Choice branch matches expectation.
aws s3 cp s3://$DATALAKE_BUCKET/models/current/pointer.json - | jq .run_id  # → new run_id if promoted
```

### Step 3.5 — Prove the loop end-to-end
```bash
# 1. Raise drift, let the generator run, then trigger the pipeline:
aws ssm put-parameter --name /credit-score/drift_step --value 1.0 --overwrite
aws lambda invoke --function-name credit-score-generator /dev/null
aws stepfunctions start-execution --state-machine-arn $SM_ARN
```
**✅ Checkpoint 3.5:**
- [ ] Drifted run → candidate **rejected**, SNS email received, pointer unchanged.
- [ ] Reset `drift_step=0.0`, run again → candidate **promoted**, pointer updated.
- [ ] Hit the serving API after a promotion → new model served on cold start (check `run_id`
      in a `/model/info` response), **no redeploy**.

> **Phase 3 done when:** you can force a bad model to be rejected and a good one to be promoted,
> and serving reflects promotions automatically.

---

# Phase 4 — Monitoring & Observability

**Goal:** you can *see* the system's health and get alerted when it degrades — the last piece
that makes it read as a real deployment.

### Step 4.1 — Log live predictions
In the serving path, write each request + prediction to `predictions/dt=…/` (async/batched
so it doesn't slow the response).

**✅ Checkpoint 4.1:**
```bash
# make a few predictions via the API, then:
aws s3 ls s3://$DATALAKE_BUCKET/predictions/ --recursive | tail   # → records appearing
```

### Step 4.2 — Drift on live traffic
A scheduled Lambda computes PSI of `predictions/` inputs vs the reference, publishes the value
as a **CloudWatch custom metric** (`CreditScore/InputPSI`), and alerts via SNS if it crosses
`PSI_SIGNIFICANT_THRESHOLD`.

**✅ Checkpoint 4.2:**
```bash
aws cloudwatch get-metric-statistics --namespace CreditScore --metric-name InputPSI \
  --start-time <...> --end-time <...> --period 3600 --statistics Average   # → datapoints
```
Force drift (Phase 3 knob) and confirm the metric rises and an alert fires.

### Step 4.3 — Dashboard + alarms
Terraform `aws_cloudwatch_dashboard`: serving latency/errors, Lambda invocations, pipeline
success/failure, input PSI over time, current model AUC. Alarms → SNS for: pipeline failure,
serving 5xx rate, input PSI breach.

**✅ Checkpoint 4.3 (manual):**
- [ ] Open the CloudWatch dashboard — all widgets have data.
- [ ] Trip each alarm once (drift breach, a forced 5xx, a failed pipeline run) → SNS email each time.

> **Phase 4 done when:** one dashboard shows serving + pipeline + drift health, and every
> failure mode you care about pages you.

---

## Cross-cutting: cost hygiene (keep it a portfolio, not a bill)
- [ ] Everything scales to zero: Lambda (per-request), Fargate (per-run), Step Functions (per-transition).
- [ ] No RDS (gold is parquet on S3), no always-on MLflow server, no NAT gateway.
- [ ] S3 lifecycle rule: expire `bronze/` and `predictions/` after N days.
- [ ] Extend the existing **teardown workflow** to cover the new resources for demo cleanup.
- [ ] A single `terraform destroy` in `infra/` removes the whole stack.

## Suggested PR sequence
1. `feat/aws-phase-1-serving` — Steps 1.1–1.5
2. `feat/aws-phase-2-ingestion` — Steps 2.1–2.3
3. `feat/aws-phase-3-pipeline` — Steps 3.1–3.5
4. `feat/aws-phase-4-monitoring` — Steps 4.1–4.3

Each PR must pass CI (ruff + pytest) and include tests for the new modules before merge.
