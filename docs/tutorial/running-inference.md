# Running Inference

Master the `gpux run` command with different input formats and advanced techniques.

---

## 🎯 What You'll Learn

- ✅ Different ways to provide input data
- ✅ Using JSON files and inline data
- ✅ Saving inference results
- ✅ Batch processing
- ✅ Using the Python API

---

## 🚀 Basic Usage

Run inference on your model:

```bash
gpux run model-name --input '{"data": [1,2,3,4,5]}'
```

---

## 📥 Input Methods

### 1. Inline JSON

Provide input directly on the command line:

```bash
gpux run sentiment-analysis --input '{"text": "I love this product!"}'
```

### 2. JSON File

Save input to a file:

```json
{
  "text": "This is amazing!"
}
```

Run with file:

```bash
gpux run sentiment-analysis --file input.json
```

### 3. File Reference (@ prefix)

```bash
gpux run sentiment-analysis --input @input.json
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

```json
[
  {"text": "First review"},
  {"text": "Second review"},
  {"text": "Third review"}
]
```

```bash
gpux run sentiment-analysis --file batch_input.json
```

---

## 🐍 Python API

Use GPUX programmatically:

```python
from gpux import GPUXRuntime
import numpy as np

# Initialize runtime
runtime = GPUXRuntime(model_path="model.onnx")

# Prepare input
input_data = {"data": np.array([[1, 2, 3, 4, 5]])}

# Run inference
result = runtime.infer(input_data)
print(result)

# Cleanup
runtime.cleanup()
```

### Context Manager

```python
from gpux import GPUXRuntime
import numpy as np

with GPUXRuntime(model_path="model.onnx") as runtime:
    result = runtime.infer({"data": np.array([[1, 2, 3]])})
    print(result)
```

---

## ⚙️ Advanced Options

### Specify Provider

```bash
gpux run model-name --input data.json --provider cuda
```

### Verbose Output

```bash
gpux run model-name --input data.json --verbose
```

---

## 💡 Key Takeaways

!!! success "What You Learned"
    ✅ Multiple input methods (inline, file, @-prefix)
    ✅ Saving output to files
    ✅ Batch processing
    ✅ Using the Python API
    ✅ Advanced command-line options

---

**Previous:** [Configuration](configuration.md) | **Next:** [Benchmarking](benchmarking.md)
