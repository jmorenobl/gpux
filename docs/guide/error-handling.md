# Error Handling

Common errors, solutions, and debugging techniques.

---

## 🎯 Overview

Learn how to handle and debug common issues.

---

## 🐛 Common Errors

### Model Not Found

**Error:** `FileNotFoundError: Model file not found`

**Solution:**
```bash
# Check file exists
ls -lh model.onnx

# Use absolute path
gpux build /full/path/to/project
```

### Input Validation Failed

**Error:** `Input mismatch. Missing: {'input'}`

**Solution:**
```bash
# Check input names
gpux inspect model-name

# Match input names in JSON
{
  "correct_input_name": [data]
}
```

### Shape Mismatch

**Error:** `Shape mismatch: expected [1,10], got [1,5]`

**Solution:**
```python
# Check expected shape
gpux inspect model

# Provide correct shape
data = np.zeros((1, 10))
```

### Provider Not Available

**Error:** `RuntimeError: No execution providers available`

**Solution:**
```bash
# Install GPU runtime
pip install onnxruntime-gpu

# Or fallback to CPU
gpux build . --provider cpu
```

### Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```yaml
# Reduce memory limit
runtime:
  gpu:
    memory: 1GB
  batch_size: 1
```

---

## 🔍 Debugging

### Enable Verbose Logging

```bash
gpux run model --verbose
```

### Check Model Details

```bash
gpux inspect model --json > info.json
cat info.json
```

### Test with CPU

```bash
# Rule out GPU issues
gpux build . --provider cpu
gpux run model --input data.json
```

### Validate Configuration

```bash
python -c "from gpux.config.parser import GPUXConfigParser; GPUXConfigParser().parse_file('gpux.yml')"
```

---

## 🛡️ Exception Handling

### Python

```python
from gpux import GPUXRuntime

try:
    runtime = GPUXRuntime("model.onnx")
    result = runtime.infer(data)
except FileNotFoundError:
    print("Model not found")
except ValueError:
    print("Invalid input data")
except RuntimeError as e:
    print(f"Runtime error: {e}")
finally:
    runtime.cleanup()
```

---

## 💡 Key Takeaways

!!! success
    ✅ Common errors and solutions
    ✅ Debugging techniques
    ✅ Exception handling
    ✅ Validation methods

---

**Previous:** [Python API](python-api.md)
