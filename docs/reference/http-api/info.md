# GET /info

Get model information and specifications.

---

## Overview

The `/info` endpoint returns detailed information about the loaded model including inputs, outputs, and metadata.

```bash
GET /info
```

---

## Request

### Method

`GET`

### URL

```
http://localhost:8080/info
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
  "name": "string",
  "version": "string",
  "format": "string",
  "size_mb": number,
  "path": "string",
  "inputs": [
    {
      "name": "string",
      "type": "string",
      "shape": [int],
      "required": boolean,
      "description": "string"
    }
  ],
  "outputs": [
    {
      "name": "string",
      "type": "string",
      "shape": [int],
      "labels": ["string"],
      "description": "string"
    }
  ],
  "metadata": {}
}
```

### Error Response

**500 Internal Server Error**:
```json
{
  "detail": "Model not loaded"
}
```

---

## Examples

### cURL

```bash
curl http://localhost:8080/info
```

**Response**:
```json
{
  "name": "sentiment-analysis",
  "version": "1.0.0",
  "format": "onnx",
  "size_mb": 256.0,
  "path": "./model.onnx",
  "inputs": [
    {
      "name": "input_ids",
      "type": "int64",
      "shape": [1, 128],
      "required": true,
      "description": "Tokenized input IDs"
    },
    {
      "name": "attention_mask",
      "type": "int64",
      "shape": [1, 128],
      "required": true,
      "description": "Attention mask"
    }
  ],
  "outputs": [
    {
      "name": "logits",
      "type": "float32",
      "shape": [1, 2],
      "labels": ["negative", "positive"],
      "description": "Sentiment logits"
    }
  ],
  "metadata": {
    "author": "GPUX Team",
    "dataset": "SST-2"
  }
}
```

### Python

```python
import requests

response = requests.get("http://localhost:8080/info")
info = response.json()

print(f"Model: {info['name']} v{info['version']}")
print(f"Size: {info['size_mb']:.1f} MB")
print(f"Inputs: {len(info['inputs'])}")
print(f"Outputs: {len(info['outputs'])}")

# Print input specifications
for inp in info['inputs']:
    print(f"  - {inp['name']}: {inp['type']} {inp['shape']}")

# Print output specifications
for out in info['outputs']:
    print(f"  - {out['name']}: {out['type']} {out['shape']}")
    if 'labels' in out:
        print(f"    Labels: {out['labels']}")
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8080/info');
const info = await response.json();

console.log(`Model: ${info.name} v${info.version}`);
console.log(`Size: ${info.size_mb.toFixed(1)} MB`);

// Print inputs
info.inputs.forEach(input => {
  console.log(`Input: ${input.name} (${input.type})`);
});

// Print outputs
info.outputs.forEach(output => {
  console.log(`Output: ${output.name} (${output.type})`);
  if (output.labels) {
    console.log(`  Labels: ${output.labels.join(', ')}`);
  }
});
```

---

## Response Fields

### Model Information

- `name` (`string`): Model name
- `version` (`string`): Model version
- `format` (`string`): Model format (e.g., `"onnx"`)
- `size_mb` (`number`): Model size in megabytes
- `path` (`string`): Path to model file

### Input Specification

- `name` (`string`): Input name
- `type` (`string`): Data type (`float32`, `int64`, etc.)
- `shape` (`list[int]`): Tensor shape (`-1` for dynamic)
- `required` (`boolean`): Whether input is required
- `description` (`string`, optional): Input description

### Output Specification

- `name` (`string`): Output name
- `type` (`string`): Data type
- `shape` (`list[int]`): Tensor shape
- `labels` (`list[string]`, optional): Class labels
- `description` (`string`, optional): Output description

### Metadata

- `metadata` (`object`): Custom metadata fields

---

## Use Cases

### 1. Dynamic Client Configuration

Use model info to configure client inputs:

```python
import requests
import numpy as np

# Get model info
info = requests.get("http://localhost:8080/info").json()

# Extract input specs
input_spec = info['inputs'][0]
input_shape = input_spec['shape']
input_type = input_spec['type']

# Create input data matching spec
data = np.zeros(input_shape, dtype=input_type)

# Run prediction
result = requests.post(
    "http://localhost:8080/predict",
    json={input_spec['name']: data.tolist()}
)
```

### 2. Validation

Validate inputs before sending:

```python
def validate_input(input_data, model_info):
    for inp in model_info['inputs']:
        if inp['required'] and inp['name'] not in input_data:
            raise ValueError(f"Missing required input: {inp['name']}")
    return True

info = requests.get("http://localhost:8080/info").json()
validate_input({"text": "test"}, info)
```

### 3. Documentation Generation

Generate API docs from model info:

```python
info = requests.get("http://localhost:8080/info").json()

print(f"# {info['name']} API")
print(f"\n## Inputs")
for inp in info['inputs']:
    print(f"- **{inp['name']}** ({inp['type']}): {inp.get('description', 'N/A')}")

print(f"\n## Outputs")
for out in info['outputs']:
    print(f"- **{out['name']}** ({out['type']}): {out.get('description', 'N/A')}")
```

---

## Best Practices

!!! tip "Cache Model Info"
    Cache model info on client startup:
    ```python
    class GPUXClient:
        def __init__(self, base_url):
            self.base_url = base_url
            self.model_info = self.get_info()

        def get_info(self):
            response = requests.get(f"{self.base_url}/info")
            return response.json()
    ```

!!! tip "Validate Before Prediction"
    Use model info to validate inputs before sending prediction requests

!!! tip "Display to Users"
    Show model info to users for transparency:
    - Model version
    - Expected inputs
    - Output format

---

## See Also

- [Predict Endpoint](predict.md)
- [Health Endpoint](health.md)
- [All Endpoints](endpoints.md)
- [Model Inspector](../python-api/models.md)
