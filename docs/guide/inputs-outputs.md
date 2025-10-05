# Inputs & Outputs

Understanding input/output handling, data types, shapes, and validation.

---

## 🎯 Overview

Learn how to work with model inputs and outputs effectively.

---

## 📥 Input Handling

### Data Types

Supported types:
- `float32`, `float64`
- `int32`, `int64`
- `bool`, `string`

### Shapes

```yaml
inputs:
  data:
    type: float32
    shape: [1, 10]  # Fixed: batch=1, features=10
    # or
    shape: [-1, 10]  # Dynamic batch
```

### JSON Format

```json
{
  "input_name": [[1.0, 2.0, 3.0]]
}
```

---

## 📤 Output Handling

### Output Format

```json
{
  "output_name": [[0.2, 0.8]]
}
```

### With Labels

```yaml
outputs:
  sentiment:
    labels: [negative, positive]
```

Returns:
```json
{
  "sentiment": [[0.2, 0.8]],
  "labels": ["negative", "positive"]
}
```

---

## ✅ Validation

GPUX automatically validates:
- Input names match
- Data types match
- Shapes compatible

---

## 💡 Key Takeaways

!!! success
    ✅ Input data types and shapes
    ✅ JSON format
    ✅ Output handling
    ✅ Automatic validation

---

**Previous:** [Providers](providers.md) | **Next:** [Preprocessing →](preprocessing.md)
