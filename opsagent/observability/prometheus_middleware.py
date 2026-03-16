"""HTTP-level Prometheus metrics (separate to avoid import cycles)."""

from prometheus_client import Counter, Histogram

http_requests_total = Counter(
    "opsagent_http_requests_total",
    "Total HTTP requests",
    labelnames=["method", "path", "status_code"],
)

http_request_duration_seconds = Histogram(
    "opsagent_http_request_duration_seconds",
    "HTTP request duration",
    labelnames=["method", "path"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)


def record_request(
    *,
    method: str,
    path: str,
    status_code: int,
    duration_seconds: float,
) -> None:
    http_requests_total.labels(
        method=method,
        path=path,
        status_code=str(status_code),
    ).inc()
    http_request_duration_seconds.labels(
        method=method,
        path=path,
    ).observe(duration_seconds)
