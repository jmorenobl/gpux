# Working with Models

Deep dive into ONNX models, conversion, optimization, and inspection.

---

## 🎯 What You'll Learn

- ✅ Understanding ONNX format
- ✅ Converting models from PyTorch, TensorFlow, etc.
- ✅ Model optimization techniques
- ✅ Inspecting and debugging models
- ✅ Model versioning and management

---

## 🧠 Understanding ONNX

### What is ONNX?

**ONNX (Open Neural Network Exchange)** is an open format for representing machine learning models. It enables interoperability between different ML frameworks.

**Benefits:**
- 🔄 Framework agnostic (PyTorch, TensorFlow, scikit-learn)
- ⚡ Optimized runtime performance
- 🌍 Cross-platform compatibility
- 🔧 Hardware-specific optimizations

### ONNX Model Structure

```
model.onnx
├── Graph
│   ├── Inputs (tensors)
│   ├── Outputs (tensors)
│   ├── Nodes (operations)
│   └── Initializers (weights)
├── Metadata
│   ├── Model version
│   ├── Producer name
│   └── Domain
└── Opset version
```

---

## 🔄 Converting Models to ONNX

### From PyTorch

```python
import torch
import torch.nn as nn

# Define model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

# Create and export model
model = MyModel()
model.eval()

dummy_input = torch.randn(1, 10)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

### From TensorFlow

```python
import tensorflow as tf
import tf2onnx

# Load TensorFlow model
model = tf.keras.models.load_model('my_model.h5')

# Convert to ONNX
spec = (tf.TensorSpec((None, 10), tf.float32, name="input"),)
output_path = "model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=14,
    output_path=output_path
)
```

### From scikit-learn

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, 10]))]
onx = convert_sklearn(clf, initial_types=initial_type)

with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

### From Hugging Face Transformers

```python
from transformers import AutoModel, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Load and export
model = ORTModelForSequenceClassification.from_pretrained(
    model_name,
    export=True
)

# Save ONNX model
model.save_pretrained("./onnx_model")
```

---

## 🔍 Inspecting Models

### Using GPUX

```bash
gpux inspect model-name
```

### Using ONNX Tools

```python
import onnx

# Load model
model = onnx.load("model.onnx")

# Check model
onnx.checker.check_model(model)

# Print model info
print(f"IR Version: {model.ir_version}")
print(f"Producer: {model.producer_name}")
print(f"Opset: {model.opset_import[0].version}")

# Print inputs
for input in model.graph.input:
    print(f"Input: {input.name}")
    print(f"  Type: {input.type.tensor_type.elem_type}")
    print(f"  Shape: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")

# Print outputs
for output in model.graph.output:
    print(f"Output: {output.name}")
```

### Using Netron (Visual Inspector)

```bash
# Install Netron
pip install netron

# Visualize model
netron model.onnx
```

Opens interactive visualization in browser.

---

## ⚡ Model Optimization

### ONNX Optimizer

```python
import onnx
from onnxoptimizer import optimize

# Load model
model = onnx.load("model.onnx")

# Optimize
optimized_model = optimize(model, ['eliminate_identity'])

# Save optimized model
onnx.save(optimized_model, "model_optimized.onnx")
```

### Quantization (INT8)

Reduce model size and improve performance:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'model.onnx'
model_quant = 'model_quant.onnx'

quantize_dynamic(
    model_fp32,
    model_quant,
    weight_type=QuantType.QUInt8
)
```

**Results:**
- 🗜️ ~4x smaller model size
- ⚡ ~2-4x faster inference
- 📊 Minimal accuracy loss (~1-2%)

### Graph Optimization

ONNX Runtime performs automatic optimizations:

```yaml
runtime:
  enable_profiling: true  # See what gets optimized
```

Common optimizations:
- Constant folding
- Operator fusion
- Layout optimization
- Dead code elimination

---

## 📊 Model Formats

### Supported Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `.onnx` | Standard ONNX | General use |
| `.onnx.gz` | Compressed ONNX | Reduce storage/transfer |
| `.onnx.tar` | Archived ONNX | Multiple files |

### Compressing Models

```bash
# Compress with gzip
gzip model.onnx
# Creates model.onnx.gz

# Decompress
gunzip model.onnx.gz
```

---

## 🏷️ Model Versioning

### Semantic Versioning

Use semantic versioning for models:

```yaml
name: sentiment-model
version: 2.1.0  # Major.Minor.Patch

model:
  source: ./model_v2.1.0.onnx
  format: onnx
  version: 2.1.0
```

**Version scheme:**
- **Major**: Breaking changes (new inputs/outputs)
- **Minor**: New features (backward compatible)
- **Patch**: Bug fixes, optimizations

### Model Registry

Organize models:

```
models/
├── sentiment/
│   ├── v1.0.0/
│   │   ├── model.onnx
│   │   └── gpux.yml
│   ├── v2.0.0/
│   │   ├── model.onnx
│   │   └── gpux.yml
│   └── latest -> v2.0.0
└── image-classifier/
    └── v1.0.0/
```

---

## 🐛 Debugging Models

### Common Issues

#### Shape Mismatch

```python
# Check actual model shapes
import onnx
model = onnx.load("model.onnx")
for input in model.graph.input:
    print(input)
```

#### Missing Operators

Some operators may not be supported:

```bash
# Check model opset
python -c "import onnx; print(onnx.load('model.onnx').opset_import)"
```

#### Numerical Differences

Compare ONNX vs original framework:

```python
import torch
import onnxruntime as ort
import numpy as np

# PyTorch inference
model_pt = torch.load('model.pt')
input_pt = torch.randn(1, 10)
output_pt = model_pt(input_pt).detach().numpy()

# ONNX inference
sess = ort.InferenceSession('model.onnx')
input_ort = {'input': input_pt.numpy()}
output_ort = sess.run(None, input_ort)[0]

# Compare
diff = np.abs(output_pt - output_ort).max()
print(f"Max difference: {diff}")  # Should be < 1e-5
```

---

## 📚 Best Practices

### 1. Test After Conversion

Always validate converted models:

```python
# Test with sample data
test_input = np.random.rand(1, 10).astype(np.float32)
result = runtime.infer({"input": test_input})
assert result is not None
```

### 2. Use Dynamic Shapes

Enable flexible batch sizes:

```python
dynamic_axes = {
    'input': {0: 'batch_size'},
    'output': {0: 'batch_size'}
}
```

### 3. Optimize Before Deployment

```bash
# Quantize for production
python -m onnxruntime.quantization.preprocess --input model.onnx --output model_prep.onnx
```

### 4. Version Control

Track model files with DVC or Git LFS:

```bash
# Using Git LFS
git lfs track "*.onnx"
git add .gitattributes
git add model.onnx
git commit -m "Add model v1.0.0"
```

---

## 💡 Key Takeaways

!!! success "What You Learned"
    ✅ ONNX format and structure
    ✅ Converting from PyTorch, TensorFlow, scikit-learn
    ✅ Model inspection and visualization
    ✅ Optimization techniques (quantization, compression)
    ✅ Model versioning and management
    ✅ Debugging converted models

---

**Next:** [GPU Providers →](providers.md)
