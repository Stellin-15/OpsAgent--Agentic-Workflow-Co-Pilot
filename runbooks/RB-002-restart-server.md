# Runbook: Server / Service Restart

**Alert:** `ServiceDown`, `InstanceUnreachable`
**Severity:** critical
**Tags:** availability, restart

## When does this fire?

- A health check endpoint returns non-2xx for > 2 minutes
- The instance fails Prometheus `up` metric checks
- TCP connection to the service port times out

## Step 1 — Verify the instance is actually down (1 min)

```bash
# From your laptop or a bastion host
ping <instance-ip>
curl -s -o /dev/null -w "%{http_code}" http://<instance-ip>:<port>/health
```

If ping responds but HTTP fails → service crash, not network issue. Go to Step 3.
If ping fails → possible instance failure. Go to Step 2.

## Step 2 — Restart the EC2/VM instance (if unreachable)

```bash
# AWS
aws ec2 reboot-instances --instance-ids <instance-id>
# Wait 60 seconds, then verify
aws ec2 describe-instance-status --instance-ids <instance-id>
```

Expected time: 2-3 minutes

## Step 3 — Restart the application service (if instance responds)

```bash
ssh <instance> "sudo systemctl restart <service-name>"

# Verify it restarted cleanly
ssh <instance> "sudo systemctl status <service-name> --no-pager"
ssh <instance> "sudo journalctl -u <service-name> -n 50 --no-pager"
```

Expected time: 1-2 minutes

## Step 4 — For Kubernetes pods

```bash
# Restart the affected deployment (rolling restart)
kubectl rollout restart deployment/<deployment-name> -n <namespace>

# Watch the rollout
kubectl rollout status deployment/<deployment-name> -n <namespace>

# If the rollout hangs, check pod events
kubectl describe pod -l app=<app-label> -n <namespace> | tail -30
```

Expected time: 2-5 minutes depending on pod count

## Step 5 — Verify recovery (1 min)

```bash
# Check health endpoint
curl http://<instance-ip>:<port>/health

# Check Prometheus — wait for up=1
# promtool query instant http://prometheus:9090 'up{instance="<instance>"}'
```

## Announcement

Post to `#incidents`:
```
✅ Service <service-name> restarted on <instance>.
Health check: OK. Monitoring for 10 minutes.
```

## Escalation

If the service crashes again within 30 minutes:
- Capture core dump if available: `sudo coredumpctl info`
- Pull the last 200 lines of app logs and share in the incident thread
- Page the on-call backend engineer
