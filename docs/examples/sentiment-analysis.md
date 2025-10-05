# Sentiment Analysis

BERT-based text classification for sentiment analysis.

---

## 🎯 What You'll Build

A sentiment classifier that determines if text is positive or negative using BERT.

**Example:**
- Input: "I love this product!"
- Output: `{sentiment: [0.1, 0.9]}` (90% positive)

---

## 📦 Model Preparation

### Export BERT Model

```python
from transformers import AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Export to ONNX
model = ORTModelForSequenceClassification.from_pretrained(
    model_name,
    export=True
)

model.save_pretrained("./sentiment-model")
```

This creates `sentiment-model/model.onnx`.

---

## ⚙️ Configuration

Create `gpux.yml`:

```yaml
name: sentiment-analysis
version: 1.0.0
description: "BERT sentiment classification"

model:
  source: ./sentiment-model/model.onnx
  format: onnx

inputs:
  input_ids:
    type: int64
    shape: [1, 128]
    required: true

  attention_mask:
    type: int64
    shape: [1, 128]
    required: true

outputs:
  logits:
    type: float32
    shape: [1, 2]
    labels: [negative, positive]

runtime:
  gpu:
    backend: auto
    memory: 2GB
```

---

## 🚀 Running Inference

### Prepare Input

```python
from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

text = "I love this product!"
tokens = tokenizer(text, padding="max_length", max_length=128, return_tensors="np")

# Save as JSON
input_data = {
    "input_ids": tokens["input_ids"].tolist(),
    "attention_mask": tokens["attention_mask"].tolist()
}

with open("input.json", "w") as f:
    json.dump(input_data, f)
```

### Build and Run

```bash
# Build
gpux build .

# Run inference
gpux run sentiment-analysis --file input.json
```

**Output:**
```json
{
  "logits": [[0.1, 0.9]]
}
```

90% positive! ✅

---

## 🐍 Python API

```python
from transformers import AutoTokenizer
from gpux import GPUXRuntime
import numpy as np

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Initialize runtime
runtime = GPUXRuntime("sentiment-model/model.onnx")

# Tokenize
text = "This is amazing!"
tokens = tokenizer(text, padding="max_length", max_length=128, return_tensors="np")

# Inference
result = runtime.infer({
    "input_ids": tokens["input_ids"],
    "attention_mask": tokens["attention_mask"]
})

# Get probabilities
logits = result["logits"][0]
probs = np.exp(logits) / np.sum(np.exp(logits))

print(f"Negative: {probs[0]:.2%}")
print(f"Positive: {probs[1]:.2%}")
```

---

## 🌐 Production Deployment

### HTTP Server

```bash
gpux serve sentiment-analysis --port 8080
```

### Client

```python
import requests

response = requests.post(
    "http://localhost:8080/predict",
    json={
        "input_ids": [[101, 1045, ...]],  # Tokenized
        "attention_mask": [[1, 1, ...]]
    }
)

print(response.json())
```

---

## 💡 Key Takeaways

!!! success
    ✅ BERT model export to ONNX
    ✅ Text tokenization
    ✅ Multi-input models
    ✅ Probability calculation
    ✅ Production serving

---

**Next:** [Image Classification →](image-classification.md)
