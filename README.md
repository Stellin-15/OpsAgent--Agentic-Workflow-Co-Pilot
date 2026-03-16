# OpsAgent — Open-Source AI SRE Co-Pilot

> **Alert fires → AI reads your runbooks → drafts a fix → human approves → actions run.**

OpsAgent is a production-grade AI incident-response assistant. It ingests alerts from Prometheus/AlertManager, Grafana, and PagerDuty, runs RAG over your runbooks to generate a step-by-step remediation plan, delivers that plan to Slack with one-click approve/reject buttons, and can execute the approved action (kubectl, AWS CLI, custom script) safely with a full audit trail.

**Business model:** MIT-licensed OSS core (self-hostable with `docker compose up`) + hosted cloud SaaS at $29–99/month per team.

---

## Demo flow

```
1.  Prometheus alert fires → POST /api/alerts/webhook/alertmanager
2.  Celery worker retrieves relevant runbook chunks (pgvector + BM25 hybrid search)
3.  LLM router (Gemini → Claude → GPT-4o fallback) generates a draft fix
4.  Draft streams token-by-token to the React dashboard (SSE)
5.  Slack message appears with Approve / Reject buttons
6.  Human clicks Approve
7.  SafeExecutor runs kubectl rollout restart (dry-run first, then live)
8.  Execution result + approver logged to action_log; incident marked RESOLVED
```

---

## Quickstart (Docker Compose)

**Prerequisites:** Docker Desktop, a [Google AI Studio API key](https://aistudio.google.com/app/apikey), and optionally a Slack bot token.

```bash
git clone https://github.com/your-org/opsagent.git
cd opsagent

cp .env.example .env
# Edit .env — fill in GOOGLE_API_KEY and (optionally) SLACK_BOT_TOKEN

docker compose up --build

# First run only — apply DB migrations
docker compose exec api alembic upgrade head
```

| Service | URL | Notes |
|---|---|---|
| API + docs | http://localhost:8000/docs | FastAPI auto-generated Swagger |
| React dashboard | http://localhost:5173 | Vite dev server |
| Grafana | http://localhost:3000 | admin / admin |
| MLflow | http://localhost:5001 | RAG experiment tracking |
| Flower | http://localhost:5555 | Celery task monitor |
| Jaeger | http://localhost:16686 | Distributed traces |
| Prometheus | http://localhost:9090 | Raw metrics |

### Fire a test alert

```bash
curl -X POST http://localhost:8000/api/alerts/webhook/manual \
  -H "Content-Type: application/json" \
  -d '{"alert_name": "HighCPU", "severity": "warning",
       "description": "CPU usage above 90% for 5 minutes on api-server-1"}'
```

---

## Features

### Alert ingestion
- **Adapter pattern** normalises Prometheus AlertManager, Grafana v8/v9, PagerDuty, and manual webhooks into a unified `Incident` model
- Webhook returns **< 5 ms** — all RAG processing is async via Celery + Redis

### AI pipeline
- **pgvector** hybrid retrieval: dense embeddings + BM25 keyword search fused via Reciprocal Rank Fusion (RRF)
- **Multi-model router**: Gemini (primary) → Claude (fallback) → GPT-4o (fallback), A/B routed by hashed incident ID
- **RAGAS evaluation** (`context_precision`, `faithfulness`, `answer_relevance`) on every draft — low-confidence drafts are flagged in Slack
- **MLflow** experiment tracking for chunking strategy and embedding model comparisons
- **SSE streaming** — draft tokens stream to the dashboard in real-time

### Safe execution engine
- Actions defined in YAML (`actions/*.yaml`) — catalog validates blocked patterns at load time (`rm -rf`, `DROP TABLE`, etc.)
- **Dry-run first**: show diff, require explicit approval before live execution
- **Subprocess isolation**: `shlex.split()` + `asyncio.create_subprocess_exec(*tokens)` — no `shell=True`
- **Auto-rollback** on non-zero exit code
- Every execution appended to `action_log` with stdout, exit code, approver

### Observability
- 9 custom Prometheus metrics: alert ingestion rate, MTTR histogram, LLM token counters, RAG latency, action success rate
- 10-panel Grafana dashboard (provisioned automatically)
- OpenTelemetry → Jaeger: single trace per incident spanning webhook, RAG, LLM call, Slack notify

### SaaS infrastructure
- JWT Bearer auth (`JWT_SECRET=` disables auth in dev — safe for local docker compose)
- Redis sliding-window rate limiter by tier (free/starter/team/enterprise)
- Stripe Checkout sessions + webhook handler for subscription management
- Kubernetes manifests with HPA (scales on `celery_queue_depth` custom metric)
- Terraform modules for GCP Cloud SQL (pgvector) + Memorystore
- GitHub Actions CI: ruff lint → mypy → unit tests → integration tests (real PG + Redis) → Docker build → Cloud Run deploy

### React dashboard
- Real-time incident feed (SSE)
- Token-streaming draft viewer with RAGAS confidence badges
- Action dry-run output inline before final approval
- Analytics: MTTR trend, approval rate, cost-per-incident charts (Recharts)
- Playwright e2e test: alert → stream draft → approve → resolved

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed breakdown of every component, data-flow diagrams, database schema, AI pipeline internals, and security model.

---

## Project layout

```
opsagent/               Python package (FastAPI app + domain logic)
  api/                  HTTP routers (alerts, incidents, actions, streaming, billing, health)
  ai/                   LLM router, embeddings, hybrid retrieval, RAGAS eval, MLflow
  auth/                 JWT validation, Redis rate limiter
  billing/              Stripe client
  execution/            Safe action executor + YAML catalog loader
  integrations/         Alert adapters (AlertManager, Grafana, PagerDuty) + Slack bot
  models/               SQLAlchemy ORM models
  observability/        Prometheus metrics, OpenTelemetry tracing, request middleware
  repositories/         DB access layer (repository pattern)
  schemas/              Pydantic I/O schemas
  worker/               Celery app, tasks, worker-process state singleton

actions/                YAML action catalog (kubectl, AWS CLI, custom scripts)
runbooks/               Markdown runbooks indexed into pgvector
alembic/                Database migrations
frontend/               React 18 + TypeScript + Vite dashboard
  src/api/              HTTP + SSE client
  src/components/       IncidentFeed, DraftViewer, ActionApproval, AnalyticsDashboard
  src/stores/           Zustand UI state
  e2e/                  Playwright end-to-end tests
k8s/                    Kubernetes manifests
terraform/              GCP infrastructure (Cloud SQL pgvector + Memorystore)
grafana/                Dashboard JSON + provisioning config
prometheus/             Scrape config
.github/workflows/      CI/CD pipeline
tests/                  pytest unit + integration tests
```

---

## Running tests

```bash
# Unit tests only (no external services needed)
make test-unit

# Full integration tests (requires docker compose up postgres redis)
make test

# E2E (requires the full stack running)
make test-e2e
```

---

## Pricing tiers

| Tier | Price | Limits |
|---|---|---|
| OSS / Self-hosted | Free | Unlimited — bring your own infra |
| Starter | $29/month | 500 alerts/month, 3 users |
| Team | $79/month | 5,000 alerts/month, 15 users |
| Enterprise | $299/month | Unlimited, SSO, SLA |

---

## Contributing

1. Fork the repo and create a feature branch
2. `cp .env.example .env` and fill in required keys
3. `docker compose up postgres redis -d` then `make test`
4. Open a PR — CI must pass before review

See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions and where to make specific types of changes.

---

## License

MIT
