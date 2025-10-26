# `gpux serve`

Start HTTP server for model serving from registries or local projects.

---

## Overview

The `gpux serve` command starts a FastAPI server that provides REST API endpoints for model inference. It supports both registry models (pulled from Hugging Face) and local models with `gpux.yml` configuration.

```bash
gpux serve MODEL_NAME [OPTIONS]
```

---

## Arguments

### `MODEL_NAME` *(required)*

Name of the model to serve. Can be:

- **Registry model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Local model**: `sentiment-analysis` (requires `gpux.yml`)
- **Model path**: `./models/bert` or `/path/to/model`

**Examples**:
```bash
# Registry models
gpux serve distilbert-base-uncased-finetuned-sst-2-english
gpux serve facebook/opt-125m
gpux serve sentence-transformers/all-MiniLM-L6-v2

# Local models
gpux serve sentiment-analysis
gpux serve image-classifier
gpux serve ./models/bert
```

---

## Options

### Server Options

#### `--port`, `-p`

Port to serve on.

- **Type**: `integer`
- **Default**: `8080`

```bash
gpux serve sentiment --port 9000
gpux serve sentiment -p 3000
```

#### `--host`, `-h`

Host to bind to.

- **Type**: `string`
- **Default**: `0.0.0.0`

```bash
gpux serve sentiment --host 127.0.0.1
gpux serve sentiment -h localhost
```

#### `--workers`

Number of worker processes.

- **Type**: `integer`
- **Default**: `1`

```bash
gpux serve sentiment --workers 4
```

### Configuration Options

#### `--config`, `-c`

Configuration file name.

- **Type**: `string`
- **Default**: `gpux.yml`

```bash
gpux serve sentiment --config custom.yml
```

#### `--provider`

Preferred execution provider.

- **Type**: `string`
- **Choices**: `cuda`, `coreml`, `rocm`, `directml`, `openvino`, `tensorrt`, `cpu`

```bash
gpux serve sentiment --provider cuda
```

### Other Options

#### `--verbose`

Enable verbose output.

- **Type**: `boolean`
- **Default**: `false`

```bash
gpux serve sentiment --verbose
```

---

## API Endpoints

The server exposes the following REST API endpoints:

### `POST /predict`

Run inference on input data.

**Request**:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

**Response**:
```json
{
  "sentiment": [0.1, 0.9]
}
```

### `GET /health`

Health check endpoint.

**Request**:
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

### `GET /info`

Get model information.

**Request**:
```bash
curl http://localhost:8080/info
```

**Response**:
```json
{
  "name": "sentiment-analysis",
  "version": "1.0.0",
  "format": "onnx",
  "inputs": [
    {
      "name": "text",
      "type": "string",
      "required": true
    }
  ],
  "outputs": [
    {
      "name": "sentiment",
      "type": "float32",
      "shape": [2]
    }
  ]
}
```

### `GET /metrics`

Get performance metrics and provider information.

**Request**:
```bash
curl http://localhost:8080/metrics
```

**Response**:
```json
{
  "provider": {
    "name": "CUDAExecutionProvider",
    "available": true,
    "platform": "NVIDIA CUDA"
  },
  "available_providers": [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
  ]
}
```

---

## Examples

### Basic Server

Start server on default port (8080):

```bash
gpux serve sentiment-analysis
```

**Output**:
```
Model Information
┌──────────┬────────────────────┐
│ Property │ Value              │
├──────────┼────────────────────┤
│ Name     │ sentiment-analysis │
│ Version  │ 1.0.0              │
│ Inputs   │ 1                  │
│ Outputs  │ 1                  │
└──────────┴────────────────────┘

Server Configuration
┌──────────┬──────────────────────────┐
│ Property │ Value                    │
├──────────┼──────────────────────────┤
│ Host     │ 0.0.0.0                  │
│ Port     │ 8080                     │
│ Workers  │ 1                        │
│ URL      │ http://0.0.0.0:8080      │
└──────────┴──────────────────────────┘

API Endpoints
┌────────┬───────────┬─────────────────────┐
│ Method │ Path      │ Description         │
├────────┼───────────┼─────────────────────┤
│ POST   │ /predict  │ Run inference       │
│ GET    │ /health   │ Health check        │
│ GET    │ /info     │ Model information   │
│ GET    │ /metrics  │ Performance metrics │
└────────┴───────────┴─────────────────────┘

🚀 Starting GPUX server...
Server will be available at: http://0.0.0.0:8080
Press Ctrl+C to stop the server
```

### Custom Port

Serve on a custom port:

```bash
gpux serve sentiment --port 9000
```

**Test**:
```bash
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great product!"}'
```

### Localhost Only

Serve on localhost only (not accessible externally):

```bash
gpux serve sentiment --host 127.0.0.1 --port 8080
```

### Multiple Workers

Use multiple workers for better throughput:

```bash
gpux serve sentiment --workers 4
```

!!! warning "GPU Memory with Multiple Workers"
    Each worker loads the model into GPU memory. Ensure you have enough GPU memory:
    - 1 worker: ~256 MB
    - 4 workers: ~1 GB
    - 8 workers: ~2 GB

### With Specific Provider

Serve with CUDA provider:

```bash
gpux serve sentiment --provider cuda --port 8080
```

---

## Making Requests

### Using cURL

**Single Inference**:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love GPUX!"}'
```

**Health Check**:
```bash
curl http://localhost:8080/health
```

### Using Python

```python
import requests

# Predict
response = requests.post(
    "http://localhost:8080/predict",
    json={"text": "I love GPUX!"}
)
result = response.json()
print(result)  # {"sentiment": [0.1, 0.9]}

# Health check
health = requests.get("http://localhost:8080/health")
print(health.json())  # {"status": "healthy", "model": "sentiment-analysis"}
```

### Using JavaScript

```javascript
// Predict
const response = await fetch('http://localhost:8080/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'I love GPUX!' })
});
const result = await response.json();
console.log(result);  // {sentiment: [0.1, 0.9]}

// Health check
const health = await fetch('http://localhost:8080/health');
const healthData = await health.json();
console.log(healthData);  // {status: "healthy", model: "sentiment-analysis"}
```

---

## OpenAPI Documentation

The server automatically generates interactive API documentation:

### Swagger UI

Visit `http://localhost:8080/docs` for interactive API documentation.

### ReDoc

Visit `http://localhost:8080/redoc` for alternative API documentation.

---

## Production Deployment

### Behind Nginx

Use Nginx as a reverse proxy:

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### With Systemd

Create a systemd service:

```ini
[Unit]
Description=GPUX Model Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDir=/opt/models/sentiment
ExecStart=/usr/local/bin/gpux serve sentiment --port 8080 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

### Docker Deployment

See [Docker Deployment Guide](../../deployment/docker.md) for containerized deployment.

---

## Error Handling

### Model Not Found

```bash
Error: Model 'sentiment-analysis' not found
```

**Solution**: Ensure the model exists and `gpux.yml` is properly configured.

### Port Already in Use

```bash
Error: [Errno 48] Address already in use
```

**Solution**: Use a different port or stop the process using the port:
```bash
gpux serve sentiment --port 9000
```

### Missing Dependencies

```bash
Error: FastAPI and uvicorn are required for serving
Install with: pip install fastapi uvicorn
```

**Solution**: Install FastAPI dependencies:
```bash
uv add fastapi uvicorn
```

---

## Best Practices

!!! tip "Use Multiple Workers"
    For production, use multiple workers to handle concurrent requests:
    ```bash
    gpux serve model --workers 4
    ```

!!! tip "Health Check Monitoring"
    Monitor the `/health` endpoint for uptime monitoring:
    ```bash
    */5 * * * * curl -f http://localhost:8080/health || alert
    ```

!!! tip "Use Process Manager"
    In production, use a process manager like systemd, supervisord, or PM2.

!!! warning "Bind to 0.0.0.0 with Caution"
    Only bind to `0.0.0.0` if you need external access. For local development, use `127.0.0.1`:
    ```bash
    gpux serve model --host 127.0.0.1
    ```

!!! tip "Set Resource Limits"
    Configure timeout and memory limits in `gpux.yml`:
    ```yaml
    runtime:
      timeout: 30
      gpu:
        memory: 2GB
    ```

---

## Performance Tips

1. **Multiple Workers**: Use `--workers` for concurrent request handling
2. **GPU Provider**: Use GPU providers (cuda, coreml) for best performance
3. **Batch Requests**: Send batch requests when possible
4. **Connection Pooling**: Use HTTP connection pooling in clients
5. **Load Balancing**: Use multiple server instances behind a load balancer

---

## Related Commands

- [`gpux run`](run.md) - Run inference directly
- [`gpux build`](build.md) - Build models before serving
- [`gpux inspect`](inspect.md) - Inspect model details

---

## See Also

- [Serving Tutorial](../../tutorial/serving.md)
- [HTTP API Reference](../http-api/endpoints.md)
- [Docker Deployment](../../deployment/docker.md)
- [Production Best Practices](../../advanced/production.md)
