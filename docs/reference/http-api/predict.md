# POST /predict

Run inference on input data.

---

## Overview

The `/predict` endpoint runs model inference on provided input data.

```bash
POST /predict
```

---

## Request

### Method

`POST`

### URL

```
http://localhost:8080/predict
```

### Headers

- `Content-Type: application/json` (required)

### Body

JSON object mapping input names to values:

```json
{
  "input_name": value,
  ...
}
```

**Input Types**:

- **Arrays**: For tensor inputs
- **Strings**: For text inputs
- **Numbers**: For scalar inputs

---

## Response

### Success Response

**Status**: `200 OK`

**Content-Type**: `application/json`

**Body**:
```json
{
  "output_name": [values],
  ...
}
```

### Error Responses

**400 Bad Request**:
```json
{
  "detail": "Error message"
}
```

**500 Internal Server Error**:
```json
{
  "detail": "Inference failed: ..."
}
```

---

## Examples

### Sentiment Analysis

**Request**:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input_ids": [101, 1045, 2293, 2023, 102],
    "attention_mask": [1, 1, 1, 1, 1]
  }'
```

**Response**:
```json
{
  "logits": [0.1, 0.9]
}
```

### Image Classification

**Request**:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image": [[[[0.5, 0.3, ...]]]]
  }'
```

**Response**:
```json
{
  "probabilities": [0.001, 0.002, ..., 0.85]
}
```

### Text Input

**Request**:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

**Response**:
```json
{
  "sentiment": [0.05, 0.95]
}
```

---

## Client Examples

### Python

```python
import requests
import numpy as np

# Text input
response = requests.post(
    "http://localhost:8080/predict",
    json={"text": "I love GPUX!"}
)
result = response.json()
print(f"Sentiment: {result['sentiment']}")

# Tensor input
response = requests.post(
    "http://localhost:8080/predict",
    json={
        "input_ids": [101, 2054, 2003, 102],
        "attention_mask": [1, 1, 1, 1]
    }
)
print(response.json())
```

### JavaScript

```javascript
// Fetch API
const response = await fetch('http://localhost:8080/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'I love GPUX!'
  })
});
const result = await response.json();
console.log(result.sentiment);

// Axios
const { data } = await axios.post('http://localhost:8080/predict', {
  text: 'I love GPUX!'
});
console.log(data.sentiment);
```

### Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "net/http"
)

type PredictRequest struct {
    Text string `json:"text"`
}

type PredictResponse struct {
    Sentiment []float64 `json:"sentiment"`
}

func predict(text string) (*PredictResponse, error) {
    reqBody, _ := json.Marshal(PredictRequest{Text: text})

    resp, err := http.Post(
        "http://localhost:8080/predict",
        "application/json",
        bytes.NewBuffer(reqBody),
    )
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result PredictResponse
    json.NewDecoder(resp.Body).Decode(&result)
    return &result, nil
}
```

---

## Error Handling

### Missing Input

**Request**:
```json
{
  "attention_mask": [1, 1, 1]
}
```

**Response** (400):
```json
{
  "detail": "Missing required input: 'input_ids'"
}
```

### Invalid Input Type

**Request**:
```json
{
  "input_ids": "not an array"
}
```

**Response** (400):
```json
{
  "detail": "Invalid input type for 'input_ids': expected array"
}
```

### Inference Failure

**Response** (500):
```json
{
  "detail": "Inference failed: Model execution error"
}
```

---

## Best Practices

!!! tip "Batch Requests"
    Send multiple items in a batch for better throughput:
    ```json
    {
      "text": ["item1", "item2", "item3", ...]
    }
    ```

!!! tip "Connection Pooling"
    Use connection pooling for high-throughput applications:
    ```python
    import requests
    from requests.adapters import HTTPAdapter

    session = requests.Session()
    session.mount('http://', HTTPAdapter(pool_connections=10, pool_maxsize=10))

    response = session.post("http://localhost:8080/predict", json={...})
    ```

!!! tip "Error Handling"
    Always handle errors gracefully:
    ```python
    try:
        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.Timeout:
        print("Request timed out")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e.response.json()['detail']}")
    ```

---

## See Also

- [All Endpoints](endpoints.md)
- [Health Check](health.md)
- [Model Info](info.md)
- [Serving Tutorial](../../tutorial/serving.md)
