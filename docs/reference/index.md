# API Reference

Complete API reference for GPUX.

---

## Command-Line Interface

Documentation for all CLI commands:

- **[gpux build](cli/build.md)** - Build and optimize models
- **[gpux run](cli/run.md)** - Run inference on models
- **[gpux serve](cli/serve.md)** - Start HTTP server
- **[gpux inspect](cli/inspect.md)** - Inspect models and runtime

---

## Configuration Reference

Complete `gpux.yml` configuration reference:

- **[Schema Overview](configuration/schema.md)** - Complete schema reference
- **[Model](configuration/model.md)** - Model source and format
- **[Inputs](configuration/inputs.md)** - Input specifications
- **[Outputs](configuration/outputs.md)** - Output specifications
- **[Runtime](configuration/runtime.md)** - GPU and runtime settings
- **[Serving](configuration/serving.md)** - HTTP server configuration
- **[Preprocessing](configuration/preprocessing.md)** - Data preprocessing

---

## Python API

Python API reference for programmatic usage:

- **[GPUXRuntime](python-api/runtime.md)** - Main runtime class
- **[ProviderManager](python-api/providers.md)** - Execution provider management
- **[ModelInspector](python-api/models.md)** - Model introspection
- **[Configuration](python-api/config.md)** - Configuration parsing

---

## HTTP API

REST API reference for serving:

- **[Endpoints Overview](http-api/endpoints.md)** - All endpoints
- **[POST /predict](http-api/predict.md)** - Run inference
- **[GET /health](http-api/health.md)** - Health check
- **[GET /info](http-api/info.md)** - Model information

---

## Quick Reference

### Common Commands

```bash
# Build model
gpux build .

# Run inference
gpux run model-name --input '{"data": [1,2,3]}'

# Start server
gpux serve model-name --port 8080

# Inspect model
gpux inspect model-name
```

### Configuration Template

```yaml
name: model-name
version: 1.0.0

model:
  source: ./model.onnx

inputs:
  - name: input
    type: float32
    shape: [1, 10]

outputs:
  - name: output
    type: float32
    shape: [1, 2]

runtime:
  gpu:
    backend: auto
    memory: 2GB
```

### Python API Example

```python
from gpux import GPUXRuntime

runtime = GPUXRuntime(model_path="model.onnx")
result = runtime.infer({"input": data})
print(result["output"])
```

---

## See Also

- [User Guide](../guide/index.md)
- [Tutorial](../tutorial/index.md)
- [Examples](../examples/index.md)
