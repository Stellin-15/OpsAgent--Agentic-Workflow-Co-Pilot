# Runbook: High CPU Usage

**Alert:** `HighCPUUsage`
**Severity:** warning / critical
**Tags:** performance, compute

## When does this fire?

This alert fires when CPU utilisation on a server exceeds 85% (warning) or 95%
(critical) for more than 5 consecutive minutes.

## Immediate triage (< 2 minutes)

1. **Identify the offending process**
   ```bash
   ssh <instance> "top -bn1 | head -20"
   # or
   ssh <instance> "ps aux --sort=-%cpu | head -10"
   ```
   Expected time: 1 minute

2. **Check if this is a known deployment or cron job**
   - Recent deploys? Check Slack `#deployments` or `git log --oneline -10`
   - Scheduled batch job? Check `crontab -l` on the instance
   - If yes → wait for the process to finish (set a 10-minute timer)

## If not a known cause

3. **Capture the process details**
   ```bash
   ssh <instance> "sudo strace -p <PID> -e trace=all -c -S calls -f 2>&1 | head -30"
   ```

4. **Restart the offending application service**
   ```bash
   ssh <instance> "sudo systemctl restart <service-name>"
   # Verify it came back healthy
   ssh <instance> "sudo systemctl status <service-name>"
   ```
   Expected time: 2 minutes

5. **If restart doesn't help — scale horizontally**
   ```bash
   # Kubernetes
   kubectl scale deployment <deployment-name> --replicas=<current+1> -n <namespace>

   # AWS Auto Scaling (increase desired capacity by 1)
   aws autoscaling set-desired-capacity \
     --auto-scaling-group-name <asg-name> \
     --desired-capacity <current+1>
   ```

6. **Check CPU after 3 minutes**
   ```bash
   ssh <instance> "sar -u 1 5"
   ```

## Escalation

If CPU remains above 85% after all steps above:
- Page the on-call engineer via PagerDuty
- Open a P1 incident in the `#incidents` Slack channel
- Attach: `top` output, `dmesg | tail -50`, and application logs

## Post-incident

- File a post-mortem if severity was critical
- Review if CPU alerting threshold needs tuning
- Check if horizontal pod autoscaler (HPA) should be lowered
