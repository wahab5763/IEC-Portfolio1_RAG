# Production Incident Response Runbook

## Severity Classification

| Severity | Definition | Acknowledgement SLA | Update Cadence |
| --- | --- | --- | --- |
| SEV-1 | Complete service outage or data loss | 10 minutes | Every 30 minutes |
| SEV-2 | Major feature degradation (>20% users) | 30 minutes | Every 2 hours |
| SEV-3 | Minor bug with known workaround | 4 business hours | Daily |

## Database Connection Exhaustion Triage

### Step 1: Verify Neon Postgres Pool Metrics
Check active connections and idle-in-transaction sessions using `pg_stat_activity`:

```sql
SELECT state, count(*) FROM pg_stat_activity GROUP BY state;
```

### Step 2: Recovery Procedure
1. Verify `pool_pre_ping=True` and `pool_recycle=300` are enabled in `backend/app/db/session.py`.
2. If Neon compute auto-suspended, trigger a warm-up `GET /api/v1/health` probe.
3. Escalate to the Infrastructure Lead (`infra-oncall@acmecloud.internal`) if errors persist > 15 minutes.

## Post-Mortem Policy
A blameless post-mortem document must be published within 48 hours of resolving any SEV-1 or SEV-2 incident.
