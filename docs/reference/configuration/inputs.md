# Input Configuration

Input specifications in `gpux.yml`.

---

## Overview

The `inputs` section defines model input specifications.

```yaml
inputs:
  - name: string          # Required: Input name
    type: string          # Required: Data type
    shape: [int]          # Optional: Tensor shape
    required: bool        # Optional: Required (default: true)
    max_length: int       # Optional: Maximum length
    description: string   # Optional: Description
```

---

## Fields

### `name` *(required)*

Input name matching the model's expected input.

- **Type**: `string`
- **Required**: Yes

```yaml
inputs:
  - name: input_ids
  - name: attention_mask
```

### `type` *(required)*

Data type of the input tensor.

- **Type**: `string`
- **Required**: Yes
- **Values**: `float32`, `float64`, `int32`, `int64`, `uint8`, `bool`, `string`

```yaml
inputs:
  - name: image
    type: float32
  - name: text
    type: string
```

### `shape`

Tensor shape specification.

- **Type**: `list[int]`
- **Required**: No
- **Dynamic**: Use `-1` for dynamic dimensions

```yaml
inputs:
  - name: image
    type: float32
    shape: [1, 3, 224, 224]  # Fixed shape
  - name: text
    type: int64
    shape: [1, -1]           # Dynamic length
```

### `required`

Whether the input is required.

- **Type**: `boolean`
- **Required**: No
- **Default**: `true`

```yaml
inputs:
  - name: text
    type: string
    required: true
  - name: metadata
    type: string
    required: false
```

### `max_length`

Maximum length for variable-length inputs.

- **Type**: `integer`
- **Required**: No

```yaml
inputs:
  - name: text
    type: int64
    max_length: 512
```

### `description`

Human-readable description.

- **Type**: `string`
- **Required**: No

```yaml
inputs:
  - name: attention_mask
    type: int64
    description: Binary mask indicating valid tokens (1) and padding (0)
```

---

## Examples

### Image Input

```yaml
inputs:
  - name: image
    type: float32
    shape: [1, 3, 224, 224]
    required: true
    description: RGB image tensor normalized to [0, 1]
```

### Text Input

```yaml
inputs:
  - name: input_ids
    type: int64
    shape: [1, 128]
    required: true
    description: Tokenized input IDs
  - name: attention_mask
    type: int64
    shape: [1, 128]
    required: true
    description: Attention mask for input
```

### Multiple Inputs

```yaml
inputs:
  - name: image
    type: float32
    shape: [1, 3, 224, 224]
  - name: text
    type: int64
    shape: [1, 77]
```

---

## Alternative Syntax

Dict-style inputs:

```yaml
inputs:
  input_ids:
    type: int64
    shape: [1, 128]
  attention_mask:
    type: int64
    shape: [1, 128]
```

---

## See Also

- [Configuration Schema](schema.md)
- [Outputs](outputs.md)
- [Input/Output Guide](../../guide/inputs-outputs.md)
