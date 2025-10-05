# Serving Configuration

HTTP server configuration in `gpux.yml`.

---

## Overview

The `serving` section configures the FastAPI HTTP server.

```yaml
serving:
  port: int               # Server port (default: 8080)
  host: string            # Server host (default: "0.0.0.0")
  batch_size: int         # Serving batch size (default: 1)
  timeout: int            # Request timeout (default: 5)
  max_workers: int        # Max worker processes (default: 4)
```

---

## Fields

### `port`

HTTP server port.

- **Type**: `integer`
- **Required**: No
- **Default**: `8080`

```yaml
serving:
  port: 8080   # Default
  port: 9000   # Custom port
```

### `host`

Server host/address.

- **Type**: `string`
- **Required**: No
- **Default**: `0.0.0.0` (all interfaces)

```yaml
serving:
  host: 0.0.0.0      # All interfaces (public)
  host: 127.0.0.1    # Localhost only (private)
  host: localhost    # Localhost alias
```

### `batch_size`

Maximum batch size for serving.

- **Type**: `integer`
- **Required**: No
- **Default**: `1`

```yaml
serving:
  batch_size: 1    # Single inference
  batch_size: 32   # Batch up to 32
```

### `timeout`

Request timeout in seconds.

- **Type**: `integer`
- **Required**: No
- **Default**: `5`

```yaml
serving:
  timeout: 5    # 5 second timeout
  timeout: 30   # 30 second timeout
```

### `max_workers`

Maximum number of worker processes.

- **Type**: `integer`
- **Required**: No
- **Default**: `4`

```yaml
serving:
  max_workers: 1   # Single worker
  max_workers: 8   # 8 workers
```

---

## Examples

### Minimal

```yaml
serving:
  port: 8080
```

### Development

```yaml
serving:
  port: 8080
  host: 127.0.0.1  # Localhost only
  max_workers: 1   # Single worker for debugging
```

### Production

```yaml
serving:
  port: 8080
  host: 0.0.0.0
  batch_size: 16
  timeout: 10
  max_workers: 8
```

### High-Throughput

```yaml
serving:
  port: 8080
  host: 0.0.0.0
  batch_size: 64
  timeout: 30
  max_workers: 16
```

---

## Complete Example

```yaml
name: sentiment-api
version: 1.0.0

model:
  source: ./model.onnx

inputs:
  - name: text
    type: string

outputs:
  - name: sentiment
    type: float32
    shape: [2]

serving:
  port: 9000
  host: 0.0.0.0
  batch_size: 32
  timeout: 10
  max_workers: 4
```

---

## Best Practices

!!! tip "Use Localhost in Development"
    Bind to localhost for local development:
    ```yaml
    serving:
      host: 127.0.0.1
    ```

!!! tip "Multiple Workers for Production"
    Use multiple workers for concurrency:
    ```yaml
    serving:
      max_workers: 8  # Based on CPU cores
    ```

!!! warning "GPU Memory with Workers"
    Each worker loads the model. With 4 workers:
    - Model size: 256 MB
    - GPU memory: 256 MB × 4 = 1 GB

!!! tip "Adjust Batch Size for Throughput"
    Larger batches improve throughput:
    ```yaml
    serving:
      batch_size: 32  # Process 32 items at once
    ```

---

## See Also

- [Configuration Schema](schema.md)
- [Serving Tutorial](../../tutorial/serving.md)
- [HTTP API Reference](../http-api/endpoints.md)
- [Production Guide](../../advanced/production.md)
