# Runbook: Database Connection Pool Exhausted

**Alert:** `DBConnectionPoolExhausted`, `DBConnectionsHigh`
**Severity:** critical
**Tags:** database, postgresql, performance

## When does this fire?

- Postgres `max_connections` is >90% utilised
- Application reports `too many connections` errors
- `pg_stat_activity` shows many `idle in transaction` connections

## Step 1 — Check current connection count (1 min)

```sql
-- Connect to the database
psql -h <host> -U postgres -d <database>

-- Total connections by state
SELECT state, count(*)
FROM pg_stat_activity
GROUP BY state
ORDER BY count DESC;

-- Connections by application name
SELECT application_name, count(*), max(now() - query_start) as max_duration
FROM pg_stat_activity
GROUP BY application_name;

-- Long-running queries (> 5 minutes)
SELECT pid, now() - query_start AS duration, query, state
FROM pg_stat_activity
WHERE now() - query_start > interval '5 minutes'
ORDER BY duration DESC;
```

## Step 2 — Kill idle/stuck connections

```sql
-- Kill idle connections older than 10 minutes (safe for most apps)
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle'
  AND now() - state_change > interval '10 minutes'
  AND pid <> pg_backend_pid();

-- Kill specific long-running query (get PID from Step 1)
SELECT pg_cancel_backend(<pid>);   -- graceful
SELECT pg_terminate_backend(<pid>); -- forceful
```

## Step 3 — Restart PgBouncer (if used)

```bash
sudo systemctl restart pgbouncer
# Check pool stats
psql -h localhost -p 6432 -U pgbouncer pgbouncer -c "SHOW POOLS;"
```

## Step 4 — Temporarily increase max_connections (emergency only)

```bash
# Edit postgresql.conf
sudo nano /etc/postgresql/<version>/main/postgresql.conf
# Increase: max_connections = 200  → 300

# Reload config (no restart needed)
sudo systemctl reload postgresql
# or
psql -c "SELECT pg_reload_conf();"
```

**⚠️ Warning:** Increasing max_connections raises shared memory usage.
Restart the application connection pools after changing this.

## Step 5 — Application-side fix

If the application is not using a connection pool:
```python
# SQLAlchemy — set pool size and max overflow
engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,       # base pool
    max_overflow=20,    # burst capacity
    pool_pre_ping=True, # verify connections before use
)
```

## Post-incident

- Review `pool_size` and `max_overflow` settings across all services
- Consider PgBouncer if multiple services share one PostgreSQL instance
- Add connection pool metrics to Grafana dashboard
- Alert at 70% (warning) not 90% (too late)
