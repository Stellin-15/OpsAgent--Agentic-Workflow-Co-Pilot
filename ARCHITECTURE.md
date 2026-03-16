# OpsAgent — Architecture Reference

This document describes the design, data flow, database schema, AI pipeline internals, security model, and infrastructure of OpsAgent. It is intended for contributors, interviewers, and anyone who wants to understand how the system is built.

---

## Table of contents

1. [System overview](#1-system-overview)
2. [Request lifecycle](#2-request-lifecycle)
3. [Incident state machine](#3-incident-state-machine)
4. [Database schema](#4-database-schema)
5. [AI pipeline](#5-ai-pipeline)
6. [Safe execution engine](#6-safe-execution-engine)
7. [Observability](#7-observability)
8. [SaaS layer: auth, rate limiting, billing](#8-saas-layer-auth-rate-limiting-billing)
9. [Frontend](#9-frontend)
10. [Infrastructure: Kubernetes + Terraform](#10-infrastructure-kubernetes--terraform)
11. [CI/CD](#11-cicd)
12. [Security model](#12-security-model)
13. [Where to make changes](#13-where-to-make-changes)

---

## 1. System overview

```
Alert Sources                  React Dashboard (port 5173)   Slack
Prometheus/AlertManager  ──►   ┌──────────────────────────┐  ◄──►
Grafana webhook          ──►   │   FastAPI Gateway          │
PagerDuty webhook        ──►   │   (JWT auth, rate limit,   │
Manual POST              ──►   │    correlation IDs, OTEL)  │
                               └──────────┬───────────────┘
                                          │ Celery task (< 5 ms response)
                               ┌──────────▼───────────────┐
                               │    Celery Workers         │
                               │    (Redis broker)         │
                               └──────────┬───────────────┘
                                          │
               ┌──────────────────────────┼─────────────────────────┐
               │                          │                          │
         ┌─────▼──────┐         ┌─────────▼──────┐        ┌────────▼──────┐
         │ PostgreSQL  │         │     Redis       │        │  pgvector     │
         │ incidents   │         │ broker + cache  │        │ runbook_chunks│
         │ audit_log   │         │ rate limiter    │        │ (HNSW index)  │
         │ action_log  │         │ embed cache     │        └───────────────┘
         └─────┬──────┘         └────────────────┘
               │
         ┌─────▼───────────────────────────────────┐
         │              AI Pipeline                 │
         │  LLM Router: Gemini → Claude → GPT-4o    │
         │  Hybrid retrieval: dense + BM25 + RRF    │
         │  RAGAS eval on every draft               │
         │  MLflow experiment tracking (:5001)       │
         └─────────────────────────────────────────┘
               │
         ┌─────▼───────────────────────────────────┐
         │         Safe Execution Engine            │
         │  YAML action catalog (kubectl, AWS CLI)  │
         │  dry-run → approval gate → live run      │
         │  shlex + subprocess_exec (no shell=True)  │
         │  auto-rollback + action_log audit trail   │
         └─────────────────────────────────────────┘
               │
  Prometheus (:9090) + Grafana (:3000) + Jaeger (:16686) — observability
  GitHub Actions CI/CD → Kubernetes + Terraform (GCP)
```

---

## 2. Request lifecycle

### Alert webhook → draft in Slack (< 5 ms gateway, ~10 s total)

```
POST /api/alerts/webhook/{source}
  │
  ├─ AlertAdapter.normalise(raw_payload) → Incident schema
  ├─ IncidentRepo.create() — write FIRING row to DB
  ├─ celery: process_incident.delay(incident_id)  ← returns here, 200 OK
  │
  └─ [Celery worker process]
       ├─ IncidentRepo.update_status(PROCESSING)
       ├─ RagPipeline.generate_draft_v4(incident_id, ...)
       │     ├─ EmbeddingProvider.embed(alert_description)
       │     ├─ PgVectorStore.search_dense(embedding, k=20)
       │     ├─ BM25Retriever.search(alert_description, k=20)
       │     ├─ RRF fusion → top 5 chunks
       │     ├─ LLMRouter.generate(prompt + chunks)
       │     │     tries: GeminiProvider → AnthropicProvider → OpenAIProvider
       │     └─ RagasEvaluator.score(query, contexts, answer) → scores dict
       ├─ DraftRepo.create(content, ragas_scores)
       ├─ IncidentRepo.update_status(DRAFT_READY)
       └─ SlackNotifier.send_incident_draft(incident, draft, ragas_scores)
            └─ Block Kit message: title + draft + RAGAS badge + Approve/Reject buttons
```

### Approve action in Slack

```
Slack sends POST to /api/alerts/slack/actions (interactive endpoint)
  │
  ├─ Verify Slack signing secret (HMAC-SHA256)
  ├─ Parse action_id: "approve_{incident_id}" or "reject_{incident_id}"
  ├─ IncidentRepo.update_status(APPROVED)
  └─ SlackNotifier.update_message(ts, "Approved by @user")
```

### Execute action via API

```
POST /api/actions/{name}/execute
  │
  ├─ require_auth (JWT) + require_tier("starter")
  ├─ ActionCatalog.get(name)  ← loaded from actions/*.yaml at startup
  ├─ action.validate_params(params)
  ├─ SafeExecutor.run(request, session)
  │     ├─ [if dry_run] fill_dry_run_command()
  │     ├─ shlex.split(command)
  │     ├─ asyncio.create_subprocess_exec(*tokens)  ← no shell=True
  │     ├─ wait_for(communicate(), timeout=...)
  │     ├─ [on failure] rollback_command
  │     └─ ActionLogRepo.log_action(...)
  └─ return ExecutionResult (JSON)
```

---

## 3. Incident state machine

```
FIRING
  └──► PROCESSING        (Celery task picked up)
         ├──► DRAFT_READY    (RAG pipeline completed)
         │      ├──► APPROVED    (human approved in Slack or API)
         │      │      └──► EXECUTING  (action running)
         │      │              ├──► RESOLVED   (exit_code == 0)
         │      │              └──► FAILED      (non-zero + rollback attempted)
         │      └──► REJECTED    (human rejected — rejection_reason stored)
         └──► FAILED           (RAG pipeline error)
```

State transitions are written to `audit_log` (append-only) so the full history of every incident is preserved.

---

## 4. Database schema

PostgreSQL 16 with the `pgvector` extension.

```sql
-- Core incident record
incidents (
  id           VARCHAR(26) PRIMARY KEY,  -- ULID
  alert_name   VARCHAR(255) NOT NULL,
  labels       JSONB,                    -- raw alert labels
  description  TEXT,
  status       VARCHAR(50),              -- state machine value
  severity     VARCHAR(20),
  source       VARCHAR(50),              -- alertmanager | grafana | pagerduty | manual
  fired_at     TIMESTAMPTZ DEFAULT now(),
  resolved_at  TIMESTAMPTZ
)

-- AI-generated remediation draft
drafts (
  id               VARCHAR(26) PRIMARY KEY,
  incident_id      VARCHAR(26) REFERENCES incidents(id),
  content          TEXT,
  model_used       VARCHAR(100),
  ragas_scores     JSONB,       -- {context_precision, faithfulness, answer_relevance, confidence}
  retrieval_chunks JSONB,       -- IDs + similarity scores of top-k chunks used
  created_at       TIMESTAMPTZ DEFAULT now()
)

-- Append-only execution audit
action_log (
  id           BIGSERIAL PRIMARY KEY,
  incident_id  VARCHAR(26) REFERENCES incidents(id),
  action_type  VARCHAR(100),   -- e.g. "rolling_restart" or "dry_run:rolling_restart"
  command      TEXT,           -- exact command string executed
  stdout       TEXT,
  exit_code    INT,
  approved_by  VARCHAR(100),   -- Slack user ID or API caller
  executed_at  TIMESTAMPTZ DEFAULT now()
)

-- Append-only state transition log
audit_log (
  id           BIGSERIAL PRIMARY KEY,
  incident_id  VARCHAR(26) REFERENCES incidents(id),
  event_type   VARCHAR(50),    -- status_changed | draft_created | action_executed ...
  actor        VARCHAR(100),
  payload      JSONB,
  occurred_at  TIMESTAMPTZ DEFAULT now()
)

-- pgvector runbook chunks (Migration 0002)
runbook_chunks (
  id           BIGSERIAL PRIMARY KEY,
  source_file  VARCHAR(255),
  chunk_index  INT,
  content      TEXT,
  embedding    vector(768),    -- Google text-embedding-004 dimensions
  created_at   TIMESTAMPTZ DEFAULT now()
)
-- HNSW index for approximate nearest neighbour search:
-- CREATE INDEX ON runbook_chunks USING hnsw (embedding vector_cosine_ops)
--   WITH (m = 16, ef_construction = 64);
```

---

## 5. AI pipeline

All AI code lives in `opsagent/ai/`.

### LLM router (`ai/llm/router.py`)

```python
# Provider selection: deterministic A/B by incident_id hash, with fallback chain
provider = hash(incident_id) % 2 == 0 ? gemini : anthropic
try:
    return await provider.generate(prompt)
except Exception:
    return await openai_provider.generate(prompt)  # final fallback
```

Each provider (`GeminiProvider`, `AnthropicProvider`, `OpenAIProvider`) implements the `LLMProvider` protocol: `async def generate(prompt: str) -> str`.

### Embeddings + cache (`ai/embeddings/`)

- `GoogleEmbeddingProvider`: calls `text-embedding-004` (768 dimensions)
- `EmbeddingCache` (Redis): key = `sha256(model_name + text)`, TTL = 24h — avoids re-embedding identical chunks on every re-index

### Hybrid retrieval (`ai/retrieval/`)

```
Query text
  │
  ├─ dense:  embed(query) → pgvector cosine similarity → top-20 chunks
  ├─ sparse: BM25 (rank-bm25) over all chunk content → top-20 chunks
  └─ fuse:   Reciprocal Rank Fusion (k=60) → merged top-5 chunks
```

RRF score: `1 / (k + rank_dense) + 1 / (k + rank_sparse)`

The fused chunks are the context window passed to the LLM.

### RAGAS evaluation (`ai/evaluation/ragas_eval.py`)

Scores computed per draft:

| Metric | What it measures |
|---|---|
| `context_precision` | Are the retrieved chunks actually relevant? |
| `faithfulness` | Does the answer only contain claims supported by context? |
| `answer_relevance` | Does the answer address the original question? |
| `confidence` | Composite (mean of above three) |

If `confidence < 0.6`, the Slack message includes a warning badge: "Low confidence — recommend manual review".

Scores are stored in `drafts.ragas_scores` (JSONB) and logged to MLflow for experiment tracking.

### MLflow (`ai/evaluation/mlflow_logger.py`)

Each RAG pipeline run creates an MLflow run in the `opsagent-rag` experiment, logging:
- Parameters: `embedding_model`, `chunk_size`, `top_k`, `llm_model`
- Metrics: RAGAS scores
- Tags: `incident_id`, `alert_name`

Access the MLflow UI at `http://localhost:5001`.

### Chunking strategy (`ai/retrieval/chunking.py`)

Markdown runbooks are chunked by heading boundaries first, then by token count (max 512 tokens per chunk, 64 token overlap). Heading-boundary chunking prevents splitting a procedure step across two chunks.

---

## 6. Safe execution engine

All execution code lives in `opsagent/execution/` and `actions/*.yaml`.

### Action catalog (YAML)

```yaml
# actions/kubernetes.yaml
actions:
  - name: rolling_restart
    description: Perform a rolling restart of a Kubernetes deployment
    labels: [kubernetes, restart, safe]
    safety_level: medium
    requires_approval: true
    timeout_seconds: 120
    parameters:
      - name: deployment_name
        required: true
        description: Name of the deployment to restart
      - name: namespace
        required: false
        default: default
    command: "kubectl rollout restart deployment/{deployment_name} -n {namespace}"
    dry_run_command: "kubectl rollout status deployment/{deployment_name} -n {namespace}"
    rollback_command: "kubectl rollout undo deployment/{deployment_name} -n {namespace}"
```

**Safety checks applied at catalog load time:**
- Blocked patterns: `rm -rf`, `DROP TABLE`, `DROP DATABASE`, `TRUNCATE`, `format C:`, `> /dev/sd*`, `mkfs`
- Any action whose command, dry_run_command, or rollback_command matches a blocked pattern is rejected and skipped with an error log

**Runtime safety:**
- `shlex.split(command)` tokenises the filled command string
- `asyncio.create_subprocess_exec(*tokens)` — `shell=False` by default; tokens are arguments, not shell input
- User-supplied parameter values are interpolated via string replace into fixed YAML templates — not shell-interpreted
- Max timeout: `min(action.timeout_seconds, 300)` — hard cap at 5 minutes

---

## 7. Observability

### Prometheus metrics (`observability/metrics.py`)

| Metric | Type | Labels |
|---|---|---|
| `opsagent_alerts_total` | Counter | `source`, `severity` |
| `opsagent_incident_resolution_seconds` | Histogram | `severity` |
| `opsagent_llm_tokens_total` | Counter | `model`, `direction` (input/output) |
| `opsagent_llm_call_duration_seconds` | Histogram | `provider` |
| `opsagent_rag_retrieval_duration_seconds` | Histogram | — |
| `opsagent_action_executions_total` | Counter | `action_name`, `exit_code` |
| `opsagent_draft_decisions_total` | Counter | `decision` (approved/rejected) |
| `opsagent_celery_queue_depth` | Gauge | — |
| `opsagent_active_incidents` | Gauge | `status` |

Exposed at `GET /metrics` (prometheus_client text format).

### OpenTelemetry tracing (`observability/tracing.py`)

Single incident trace with spans:
```
webhook_receive
  └─ db_write
       └─ celery_task_enqueue
            └─ [worker] process_incident
                 ├─ embedding_cache_lookup
                 ├─ pgvector_search
                 ├─ bm25_search
                 ├─ rrf_fusion
                 ├─ llm_call (GeminiProvider)
                 ├─ ragas_eval
                 ├─ draft_stored
                 └─ slack_notify
```

OTLP exporter → Jaeger at `http://jaeger:4317`. View traces at `http://localhost:16686`.

### Grafana dashboard (`grafana/dashboards/opsagent.json`)

10 panels provisioned automatically:
1. Alert ingestion rate by source
2. MTTR (mean time to resolution) trend
3. Draft approval/rejection rate
4. LLM token usage by model
5. RAG retrieval P95 latency
6. Action execution success/failure rate
7. Active incidents by status
8. Celery queue depth
9. LLM call duration P99
10. Error rate by endpoint

---

## 8. SaaS layer: auth, rate limiting, billing

### JWT auth (`auth/jwt.py`)

```python
# FastAPI dependency
async def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not settings.jwt_secret:
        return {"sub": "dev", "tier": "enterprise"}  # auth disabled in dev
    payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    return payload

async def require_tier(min_tier: str):
    # Checks payload["tier"] is at or above the required tier
    ...
```

Setting `JWT_SECRET=` (empty) in `.env` disables auth entirely — safe for local development.

### Rate limiting (`auth/rate_limiter.py`)

Redis sliding window per `team_id`:

| Tier | Daily limit |
|---|---|
| free | 100 alerts |
| starter | 500 alerts |
| team | 5,000 alerts |
| enterprise | unlimited |

Window = 86,400 seconds. Each webhook request increments a Redis counter with TTL. Exceeding the limit returns HTTP 429.

### Stripe billing (`billing/stripe_client.py`)

- `POST /api/billing/checkout` — creates a Stripe Checkout session for a given price tier
- `POST /api/billing/webhook` — receives Stripe events; on `checkout.session.completed` it activates the team's subscription and stores their tier in the DB
- `GET /api/billing/status` — returns current subscription tier for the authenticated team

Price IDs are configured via `STRIPE_PRICE_STARTER`, `STRIPE_PRICE_TEAM`, `STRIPE_PRICE_ENTERPRISE` in `.env`.

---

## 9. Frontend

Stack: React 18, TypeScript, Vite, TanStack Query, Zustand, Tailwind CSS, Recharts, Playwright.

### Key components

| Component | File | What it does |
|---|---|---|
| `IncidentFeed` | `components/IncidentFeed/` | Real-time list; polls `GET /api/incidents` every 5s via TanStack Query |
| `DraftViewer` | `components/DraftViewer/` | Connects to `GET /api/incidents/{id}/stream` (SSE); renders tokens as they arrive; shows RAGAS confidence badge |
| `ActionApproval` | `components/ActionApproval/` | Lists available actions; triggers dry-run; shows output; sends execute request |
| `AnalyticsDashboard` | `components/AnalyticsDashboard/` | Recharts area/bar charts for MTTR, approval rate, cost-per-incident |

### State management

`uiStore.ts` (Zustand) holds:
- `selectedIncidentId` — which incident is open in the detail pane
- `sidebarOpen` — mobile sidebar state
- `streamStatus` — `idle | streaming | done | error` for the active SSE connection

Server state (incidents list, draft content) is managed by TanStack Query with appropriate `staleTime` and `refetchInterval`.

### SSE streaming (`api/streaming.ts`)

```typescript
const es = new EventSource(`/api/incidents/${id}/stream`);
es.onmessage = (e) => appendToken(e.data);
es.addEventListener('done', () => { es.close(); setStreamStatus('done'); });
```

The FastAPI `StreamingResponse` yields `data: {token}\n\n` for each LLM token, then `event: done\ndata: {}\n\n` at the end.

### Builds to `/static`

`docker compose` runs `npm run build` in the frontend container; the output is written to `/app/static` which FastAPI serves via `StaticFiles`. In development, `vite dev` runs on port 5173 and proxies API requests to port 8000.

---

## 10. Infrastructure: Kubernetes + Terraform

### Kubernetes (`k8s/`)

| File | Resource |
|---|---|
| `namespace.yaml` | `opsagent` namespace |
| `configmap.yaml` | Non-secret env vars |
| `secret.yaml` | Secret template (populate with `kubectl create secret`) |
| `api-deployment.yaml` | FastAPI app, 2 replicas, readiness/liveness probes |
| `worker-deployment.yaml` | Celery worker, 2 replicas |
| `api-hpa.yaml` | HPA: scale API on `celery_queue_depth` (custom metric via kube-metrics-adapter) |
| `ingress.yaml` | Nginx ingress; includes `nginx.ingress.kubernetes.io/proxy-buffering: "off"` for SSE |

To deploy locally (minikube or k3d):
```bash
kubectl apply -f k8s/
```

### Terraform (`terraform/`)

GCP modules:

```
terraform/
  modules/
    postgres/   Cloud SQL Postgres 15 + pgvector extension + private IP
    redis/      Memorystore Redis 7 (standard tier)
  environments/
    prod/main.tf  wires modules together, outputs connection strings to Secret Manager
```

```bash
cd terraform/environments/prod
terraform init
terraform plan -var="project_id=my-gcp-project"
terraform apply
```

---

## 11. CI/CD

`.github/workflows/ci.yml` runs on every push to `main` and all pull requests.

```
lint          ruff check + ruff format --check
type-check    mypy opsagent/
test-unit     pytest tests/unit/ (no external services)
test-int      docker compose -f docker-compose.test.yml up -d postgres redis
              pytest tests/integration/
              docker compose down
build         docker build + push to GHCR (on main only)
deploy        gcloud run deploy (on main only, requires GCP_SA_KEY secret)
```

---

## 12. Security model

### What is safe to put in this repo

- `.env.example` — template with placeholder values (no real secrets)
- All source code — no secrets are hardcoded anywhere

### What must never be committed

- `.env` — contains real API keys (already in `.gitignore`)
- `terraform/environments/prod/terraform.tfvars` — may contain real project IDs

### Subprocess safety

Action commands from the YAML catalog are interpolated with user-supplied parameters via string replacement, then processed by `shlex.split()`, then passed as an argument list to `asyncio.create_subprocess_exec(*tokens)`. Because `shell=False` (the default), the OS exec syscall receives the argument list directly — no shell interprets it. A parameter value containing shell metacharacters (`;`, `&&`, `$(...)`) becomes a literal argument to the subprocess, not a shell command.

### SQL injection

All database access uses SQLAlchemy ORM methods or `select(Model).where(Model.column == value)` parameterised queries. The one literal SQL call is `text("SELECT 1")` in the health check, which takes no user input.

### Authentication bypass

`JWT_SECRET` is intentionally empty in `.env.example`. The `require_auth` dependency checks `if not settings.jwt_secret: return dev_payload` — this makes local development frictionless while ensuring that setting a non-empty `JWT_SECRET` in production enables real auth with no code changes.

---

## 13. Where to make changes

| Goal | Files to change |
|---|---|
| Add a new alert source | `opsagent/integrations/alert_sources/` — implement `AlertAdapter` protocol |
| Add a new LLM provider | `opsagent/ai/llm/` — implement `LLMProvider` protocol, register in `router.py` |
| Add a new action | `actions/*.yaml` — add an entry; restart the API container |
| Change chunking strategy | `opsagent/ai/retrieval/chunking.py` + re-run `scripts/index_runbooks.py` |
| Add a Prometheus metric | `opsagent/observability/metrics.py` — define counter/histogram; increment at call site |
| Add a new API endpoint | `opsagent/api/` — new router; register in `opsagent/main.py` |
| Change pricing | `opsagent/auth/rate_limiter.py` (limits) + `opsagent/billing/stripe_client.py` (price IDs) |
| Add a frontend page | `frontend/src/pages/` — new page component; add route in `App.tsx` |
