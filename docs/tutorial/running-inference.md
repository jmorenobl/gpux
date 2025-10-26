# Running Inference

Master the `gpux run` command with registry models and different input formats.

---

## 🎯 What You'll Learn

- ✅ Running inference on registry models
- ✅ Different ways to provide input data
- ✅ Using JSON files and inline data
- ✅ Saving inference results
- ✅ Batch processing
- ✅ Using the Python API

---

## 🚀 Basic Usage

### Registry Models

Run inference on models pulled from registries:

```bash
# Sentiment analysis
gpux run distilbert-base-uncased-finetuned-sst-2-english --input '{"inputs": "I love this product!"}'

# Text generation
gpux run facebook/opt-125m --input '{"inputs": "The future of AI is"}'

# Embeddings
gpux run sentence-transformers/all-MiniLM-L6-v2 --input '{"inputs": "Hello world"}'
```

### Local Models

Run inference on local models with `gpux.yml`:

```bash
gpux run model-name --input '{"data": [1,2,3,4,5]}'
```

---

## 📥 Input Methods

### 1. Inline JSON

Provide input directly on the command line:

```bash
# Registry model
gpux run distilbert-base-uncased-finetuned-sst-2-english --input '{"inputs": "I love this product!"}'

# Local model
gpux run sentiment-analysis --input '{"text": "I love this product!"}'
```

### 2. JSON File

Save input to a file:

**For sentiment analysis:**
```json
{
  "inputs": "This is amazing!"
}
```

**For text generation:**
```json
{
  "inputs": "The future of AI is",
  "max_length": 50
}
```

**For embeddings:**
```json
{
  "inputs": "Hello world"
}
```

Run with file:

```bash
gpux run distilbert-base-uncased-finetuned-sst-2-english --file input.json
```

### 3. File Reference (@ prefix)

```bash
gpux run distilbert-base-uncased-finetuned-sst-2-english --input @input.json
```

---

## 📤 Output Options

### Print to Console (Default)

```bash
gpux run model-name --input data.json
```

Output:
```json
{
  "result": [0.2, 0.8]
}
```

### Save to File

```bash
gpux run model-name --input data.json --output result.json
```

---

## 🔁 Batch Inference

Process multiple inputs:

**For sentiment analysis:**
```json
[
  {"inputs": "I love this product!"},
  {"inputs": "This is terrible."},
  {"inputs": "It's okay, nothing special."}
]
```

**For text generation:**
```json
[
  {"inputs": "The future of AI is"},
  {"inputs": "Machine learning will"},
  {"inputs": "Deep learning models"}
]
```

```bash
gpux run distilbert-base-uncased-finetuned-sst-2-english --file batch_input.json
```

---

## 🐍 Python API

Use GPUX programmatically with registry models:

```python
from gpux import GPUXRuntime
import numpy as np

# Initialize runtime with registry model
runtime = GPUXRuntime(model_id="distilbert-base-uncased-finetuned-sst-2-english")

# Prepare input
input_data = {"inputs": "I love this product!"}

# Run inference
result = runtime.infer(input_data)
print(result)

# Cleanup
runtime.cleanup()
```

### Context Manager

```python
from gpux import GPUXRuntime

with GPUXRuntime(model_id="distilbert-base-uncased-finetuned-sst-2-english") as runtime:
    result = runtime.infer({"inputs": "This is amazing!"})
    print(result)
```

### Local Models

```python
from gpux import GPUXRuntime
import numpy as np

# Initialize runtime with local model
runtime = GPUXRuntime(model_path="model.onnx")

# Prepare input
input_data = {"data": np.array([[1, 2, 3, 4, 5]])}

# Run inference
result = runtime.infer(input_data)
print(result)

# Cleanup
runtime.cleanup()
```

---

## ⚙️ Advanced Options

### Specify Provider

```bash
# Force CPU provider
gpux run distilbert-base-uncased-finetuned-sst-2-english --input '{"inputs": "test"}' --provider cpu

# Force specific GPU provider
gpux run distilbert-base-uncased-finetuned-sst-2-english --input '{"inputs": "test"}' --provider cuda
```

### Verbose Output

```bash
gpux run distilbert-base-uncased-finetuned-sst-2-english --input '{"inputs": "test"}' --verbose
```

### Custom Model Path

```bash
# Use specific model path
gpux run /path/to/model --input '{"inputs": "test"}'
```

---

## 💡 Key Takeaways

!!! success "What You Learned"
    ✅ Running inference on registry models (Hugging Face)
    ✅ Multiple input methods (inline, file, @-prefix)
    ✅ Different input formats for different model types
    ✅ Saving output to files
    ✅ Batch processing with arrays
    ✅ Using the Python API with both registry and local models
    ✅ Advanced command-line options

---

**Previous:** [Pulling Models](pulling-models.md) | **Next:** [Benchmarking](benchmarking.md)
