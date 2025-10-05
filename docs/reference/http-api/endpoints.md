# HTTP API Endpoints

Complete REST API reference for GPUX serving.

---

## Overview

GPUX provides a FastAPI-based HTTP server with REST endpoints for inference, health checks, and model information.

**Base URL**: `http://localhost:8080` (default)

---

## Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Run inference |
| GET | `/health` | Health check |
| GET | `/info` | Model information |
| GET | `/metrics` | Performance metrics |
| GET | `/docs` | Swagger UI documentation |
| GET | `/redoc` | ReDoc documentation |

---

## POST /predict

Run inference on input data.

### Request

**Content-Type**: `application/json`

**Body**:
```json
{
  "input_name": "value",
  ...
}
```

**Example**:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -D '{"text": "I love this product!"}'
```

### Response

**Status**: `200 OK`

**Content-Type**: `application/json`

**Body**:
```json
{
  "output_name": [values],
  ...
}
```

**Example**:
```json
{
  "sentiment": [0.1, 0.9]
}
```

### Error Responses

**400 Bad Request** - Invalid input
```json
{
  "detail": "Missing required input: 'text'"
}
```

**500 Internal Server Error** - Inference failed
```json
{
  "detail": "Inference failed: ..."
}
```

---

## GET /health

Health check endpoint for monitoring.

### Request

```bash
curl http://localhost:8080/health
```

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "status": "healthy",
  "model": "sentiment-analysis"
}
```

---

## GET /info

Get model information and specifications.

### Request

```bash
curl http://localhost:8080/info
```

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "name": "sentiment-analysis",
  "version": "1.0.0",
  "format": "onnx",
  "size_mb": 256.0,
  "inputs": [
    {
      "name": "input_ids",
      "type": "int64",
      "shape": [1, 128],
      "required": true,
      "description": "Tokenized input IDs"
    }
  ],
  "outputs": [
    {
      "name": "logits",
      "type": "float32",
      "shape": [1, 2],
      "labels": ["negative", "positive"]
    }
  ],
  "metadata": {}
}
```

---

## GET /metrics

Get performance metrics and provider information.

### Request

```bash
curl http://localhost:8080/metrics
```

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "provider": {
    "name": "CUDAExecutionProvider",
    "available": true,
    "platform": "NVIDIA CUDA",
    "description": "NVIDIA CUDA GPU acceleration"
  },
  "available_providers": [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
  ]
}
```

---

## Interactive Documentation

### Swagger UI

Interactive API documentation at `/docs`:

```
http://localhost:8080/docs
```

Features:
- Try out endpoints
- View request/response schemas
- Authentication testing
- Download OpenAPI spec

### ReDoc

Alternative documentation at `/redoc`:

```
http://localhost:8080/redoc
```

Features:
- Clean, readable format
- API structure overview
- Code samples
- Search functionality

---

## Example Requests

### Python

```python
import requests

# Predict
response = requests.post(
    "http://localhost:8080/predict",
    json={"text": "I love GPUX!"}
)
result = response.json()
print(result["sentiment"])

# Health check
health = requests.get("http://localhost:8080/health")
print(health.json())

# Model info
info = requests.get("http://localhost:8080/info")
print(info.json()["name"])
```

### JavaScript

```javascript
// Predict
const response = await fetch('http://localhost:8080/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'I love GPUX!' })
});
const result = await response.json();
console.log(result.sentiment);

// Health check
const health = await fetch('http://localhost:8080/health');
const healthData = await health.json();
console.log(healthData.status);
```

### cURL

```bash
# Predict
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love GPUX!"}'

# Health check
curl http://localhost:8080/health

# Model info
curl http://localhost:8080/info

# Metrics
curl http://localhost:8080/metrics
```

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 404 | Not Found |
| 500 | Internal Server Error |

### Error Response Format

```json
{
  "detail": "Error message here"
}
```

### Common Errors

**Missing Input**:
```json
{
  "detail": "Missing required input: 'text'"
}
```

**Invalid Input Type**:
```json
{
  "detail": "Invalid input type for 'image': expected float32"
}
```

**Inference Timeout**:
```json
{
  "detail": "Inference timeout after 30 seconds"
}
```

---

## See Also

- [Predict Endpoint](predict.md)
- [Health Endpoint](health.md)
- [Info Endpoint](info.md)
- [Serving Tutorial](../../tutorial/serving.md)
- [gpux serve Command](../cli/serve.md)
