# Serving Models

Deploy models with HTTP APIs for production use.

---

## 🎯 What You'll Learn

- ✅ Starting HTTP server
- ✅ Making API requests
- ✅ API endpoints
- ✅ Production deployment
- ✅ Scaling strategies

---

## 🚀 Quick Start

Start HTTP server:

```bash
gpux serve model-name --port 8080
```

Output:
```
INFO: Started server on http://0.0.0.0:8080
INFO: Using provider: CoreMLExecutionProvider
INFO: Model loaded: model-name v1.0.0
```

---

## 📡 API Endpoints

### Health Check

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy"
}
```

### Model Info

```bash
curl http://localhost:8080/info
```

Response:
```json
{
  "name": "model-name",
  "version": "1.0.0",
  "provider": "CoreMLExecutionProvider"
}
```

### Prediction

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!"}'
```

Response:
```json
{
  "sentiment": [0.1, 0.9]
}
```

---

## 🔧 Server Configuration

Configure in `gpux.yml`:

```yaml
serving:
  port: 8080
  host: 0.0.0.0
  batch_size: 1
  timeout: 5
  max_workers: 4
```

### Command-Line Options

```bash
# Custom port
gpux serve model --port 9000

# Bind to localhost only
gpux serve model --host 127.0.0.1

# Multiple workers
gpux serve model --workers 4
```

---

## 📊 OpenAPI Documentation

Automatic API documentation:

- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`
- OpenAPI spec: `http://localhost:8080/openapi.json`

---

## 🐍 Python Client

Make requests programmatically:

```python
import requests

url = "http://localhost:8080/predict"
data = {"text": "This is great!"}

response = requests.post(url, json=data)
result = response.json()
print(result)
```

---

## 🚀 Production Deployment

### Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install GPUX
RUN pip install gpux

# Copy model and config
COPY model.onnx .
COPY gpux.yml .

# Expose port
EXPOSE 8080

# Start server
CMD ["gpux", "serve", "model-name", "--port", "8080"]
```

Build and run:

```bash
docker build -t my-model .
docker run -p 8080:8080 my-model
```

### Reverse Proxy (nginx)

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

### Load Balancing

Use multiple workers:

```bash
gpux serve model --workers 4
```

Or use external load balancer (nginx, HAProxy).

---

## 📈 Monitoring

### Metrics

Track performance:
- Request latency
- Throughput (requests/sec)
- Error rates
- Memory usage

### Logging

Enable verbose logging:

```bash
gpux serve model --verbose
```

---

## 💡 Key Takeaways

!!! success "What You Learned"
    ✅ Starting HTTP server
    ✅ API endpoints and usage
    ✅ Configuration options
    ✅ Production deployment with Docker
    ✅ Scaling and monitoring

---

## 🎉 Congratulations!

You've completed the GPUX tutorial! You now know how to:

- ✅ Install and configure GPUX
- ✅ Build and run models
- ✅ Optimize performance
- ✅ Deploy to production

**Next steps:**
- [User Guide](../guide/index.md) - Deep dive into concepts
- [Examples](../examples/index.md) - Real-world use cases
- [Deployment](../deployment/index.md) - Production guides

---

**Previous:** [Benchmarking](benchmarking.md)
