# Model Configuration

Model source and format configuration in `gpux.yml`.

---

## Overview

The `model` section specifies the model file path and format.

```yaml
model:
  source: string    # Required: Path to model file
  format: string    # Optional: Model format (default: "onnx")
  version: string   # Optional: Model version
```

---

## Fields

### `source` *(required)*

Path to the model file.

- **Type**: `string` or `Path`
- **Required**: Yes
- **Supports**: Relative and absolute paths

**Examples**:
```yaml
model:
  source: ./model.onnx              # Relative path
  source: ./models/sentiment.onnx   # Subdirectory
  source: /opt/models/model.onnx    # Absolute path
```

### `format`

Model format specification.

- **Type**: `string`
- **Required**: No
- **Default**: `onnx`
- **Supported**: `onnx` (others planned)

```yaml
model:
  format: onnx
```

### `version`

Model version (distinct from project version).

- **Type**: `string`
- **Required**: No

```yaml
model:
  version: 1.0.0
```

---

## Examples

### Minimal

```yaml
model:
  source: ./model.onnx
```

### Complete

```yaml
model:
  source: ./models/sentiment-v2.onnx
  format: onnx
  version: 2.0.0
```

---

## Path Resolution

Paths are resolved relative to `gpux.yml`:

```
project/
├── gpux.yml
├── model.onnx         # source: ./model.onnx
└── models/
    └── model.onnx     # source: ./models/model.onnx
```

---

## See Also

- [Configuration Schema](schema.md)
- [Working with Models](../../guide/models.md)
