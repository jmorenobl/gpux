# Output Configuration

Output specifications in `gpux.yml`.

---

## Overview

The `outputs` section defines model output specifications.

```yaml
outputs:
  - name: string          # Required: Output name
    type: string          # Required: Data type
    shape: [int]          # Optional: Tensor shape
    labels: [string]      # Optional: Class labels
    description: string   # Optional: Description
```

---

## Fields

### `name` *(required)*

Output name matching the model's output.

- **Type**: `string`
- **Required**: Yes

```yaml
outputs:
  - name: logits
  - name: probabilities
```

### `type` *(required)*

Data type of the output tensor.

- **Type**: `string`
- **Required**: Yes
- **Values**: `float32`, `float64`, `int32`, `int64`, `uint8`, `bool`, `string`

```yaml
outputs:
  - name: logits
    type: float32
  - name: classes
    type: int64
```

### `shape`

Tensor shape specification.

- **Type**: `list[int]`
- **Required**: No
- **Dynamic**: Use `-1` for dynamic dimensions

```yaml
outputs:
  - name: logits
    type: float32
    shape: [1, 2]        # Fixed shape
  - name: boxes
    type: float32
    shape: [-1, 4]       # Dynamic number of boxes
```

### `labels`

Class labels for classification outputs.

- **Type**: `list[string]`
- **Required**: No

```yaml
outputs:
  - name: sentiment
    type: float32
    shape: [1, 2]
    labels: [negative, positive]
```

### `description`

Human-readable description.

- **Type**: `string`
- **Required**: No

```yaml
outputs:
  - name: logits
    type: float32
    description: Raw classification logits before softmax
```

---

## Examples

### Classification Output

```yaml
outputs:
  - name: logits
    type: float32
    shape: [1, 2]
    labels: [negative, positive]
    description: Sentiment classification logits
```

### Detection Outputs

```yaml
outputs:
  - name: boxes
    type: float32
    shape: [-1, 4]
    description: Bounding boxes [x1, y1, x2, y2]
  - name: scores
    type: float32
    shape: [-1]
    description: Confidence scores
  - name: classes
    type: int64
    shape: [-1]
    description: Class indices
```

### Multi-Label Classification

```yaml
outputs:
  - name: probabilities
    type: float32
    shape: [1, 1000]
    labels: [cat, dog, bird, ...]  # 1000 ImageNet classes
```

---

## Alternative Syntax

Dict-style outputs:

```yaml
outputs:
  logits:
    type: float32
    shape: [1, 2]
    labels: [negative, positive]
  embeddings:
    type: float32
    shape: [1, 768]
```

---

## See Also

- [Configuration Schema](schema.md)
- [Inputs](inputs.md)
- [Input/Output Guide](../../guide/inputs-outputs.md)
