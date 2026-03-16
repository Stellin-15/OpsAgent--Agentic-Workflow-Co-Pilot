# Runbook: Disk Full / Low Disk Space

**Alert:** `DiskSpaceLow`, `DiskFull`
**Severity:** warning (> 80%), critical (> 95%)
**Tags:** storage, disk

## When does this fire?

Disk utilisation on a monitored filesystem exceeds the threshold.
Common culprits: application logs, database WAL files, Docker images, core dumps.

## Step 1 — Find what's consuming space (2 min)

```bash
ssh <instance>

# Top-level directories
du -sh /* 2>/dev/null | sort -rh | head -10

# If /var is large, check logs first
du -sh /var/log/* 2>/dev/null | sort -rh | head -10

# If /var/lib is large, check Docker
du -sh /var/lib/docker 2>/dev/null
```

## Step 2 — Clean up application logs (if safe)

```bash
# Rotate and compress logs immediately
sudo logrotate -f /etc/logrotate.conf

# Remove logs older than 7 days
sudo find /var/log -name "*.log" -mtime +7 -delete
sudo find /var/log -name "*.gz" -mtime +14 -delete

# Check freed space
df -h <mount-point>
```

Expected time: 1-2 minutes

## Step 3 — Clean up Docker (if Docker is installed)

```bash
# Remove stopped containers, unused images, build cache
docker system prune -f

# Remove unused volumes (WARNING: verify none are needed)
docker volume prune -f
```

## Step 4 — Clean up core dumps

```bash
sudo find /var/crash /var/core -name "core*" -delete 2>/dev/null
sudo find / -name "core" -size +100M 2>/dev/null -delete
```

## Step 5 — If disk is still critical, expand volume

**AWS EBS:**
```bash
# Increase EBS volume size (replace vol-xxx with actual volume ID)
aws ec2 modify-volume --volume-id <vol-id> --size <new-size-gb>

# After resize (may take a minute):
sudo growpart /dev/xvda 1
sudo resize2fs /dev/xvda1
df -h
```

**Kubernetes PVC:**
```bash
kubectl patch pvc <pvc-name> -n <namespace> \
  -p '{"spec":{"resources":{"requests":{"storage":"<new-size>Gi"}}}}'
```

## Prevention

- Ensure log rotation is configured (`/etc/logrotate.d/<app>`)
- Set log retention policy (max 7 days or 1 GB, whichever comes first)
- Add disk usage Prometheus recording rules to alert at 70% (not 85%)
- Consider external log shipping (CloudWatch Logs, Datadog, Loki)
