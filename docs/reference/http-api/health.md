# GET /health

Health check endpoint for monitoring and uptime checking.

---

## Overview

The `/health` endpoint provides a simple health check for monitoring server status and uptime.

```bash
GET /health
```

---

## Request

### Method

`GET`

### URL

```
http://localhost:8080/health
```

### Headers

None required

### Parameters

None

---

## Response

### Success Response

**Status**: `200 OK`

**Content-Type**: `application/json`

**Body**:
```json
{
  "status": "healthy",
  "model": "sentiment-analysis"
}
```

**Fields**:

- `status` (`string`): Server health status (always `"healthy"` if responding)
- `model` (`string`): Name of the loaded model

---

## Examples

### cURL

```bash
curl http://localhost:8080/health
```

**Response**:
```json
{
  "status": "healthy",
  "model": "sentiment-analysis"
}
```

### Python

```python
import requests

response = requests.get("http://localhost:8080/health")
data = response.json()

if data["status"] == "healthy":
    print(f"Server is healthy, model: {data['model']}")
else:
    print("Server unhealthy")
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8080/health');
const data = await response.json();

if (data.status === 'healthy') {
  console.log(`Server healthy, model: ${data.model}`);
}
```

---

## Use Cases

### 1. Uptime Monitoring

Monitor server availability:

```bash
#!/bin/bash
while true; do
  if curl -sf http://localhost:8080/health > /dev/null; then
    echo "Server is healthy"
  else
    echo "Server is down!"
    # Send alert
  fi
  sleep 60
done
```

### 2. Load Balancer Health Check

Configure load balancer health checks:

**Nginx**:
```nginx
upstream gpux_backend {
  server localhost:8080;

  # Health check
  health_check uri=/health interval=10s;
}
```

**AWS ALB**:
```yaml
HealthCheck:
  Path: /health
  Interval: 30
  Timeout: 5
  HealthyThreshold: 2
  UnhealthyThreshold: 3
```

### 3. Kubernetes Liveness Probe

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
```

### 4. Cron Job Monitoring

```cron
*/5 * * * * curl -f http://localhost:8080/health || echo "GPUX server down" | mail -s "Alert" admin@example.com
```

---

## Best Practices

!!! tip "Use for Monitoring"
    Integrate `/health` endpoint with monitoring tools:
    - Prometheus
    - Grafana
    - Datadog
    - New Relic

!!! tip "Set Appropriate Timeouts"
    Configure reasonable timeout values:
    ```python
    response = requests.get(
        "http://localhost:8080/health",
        timeout=5  # 5 second timeout
    )
    ```

!!! tip "Check Regularly"
    Poll health endpoint at regular intervals (30-60 seconds)

---

## See Also

- [Info Endpoint](info.md)
- [Metrics Endpoint](../http-api/endpoints.md#get-metrics)
- [All Endpoints](endpoints.md)
