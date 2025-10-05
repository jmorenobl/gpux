# ModelInspector

ONNX model introspection and metadata extraction.

---

## Overview

`ModelInspector` extracts comprehensive information from ONNX models including inputs, outputs, shapes, and metadata.

```python
from gpux.core.models import ModelInspector

inspector = ModelInspector()
info = inspector.inspect("model.onnx")
```

---

## Dataclasses

### `InputSpec`

Input specification.

```python
@dataclass
class InputSpec:
    name: str
    type: str
    shape: list[int]
    required: bool = True
    description: str | None = None
```

### `OutputSpec`

Output specification.

```python
@dataclass
class OutputSpec:
    name: str
    type: str
    shape: list[int]
    labels: list[str] | None = None
    description: str | None = None
```

### `ModelInfo`

Complete model information.

```python
@dataclass
class ModelInfo:
    name: str
    version: str
    format: str
    path: Path
    size_bytes: int
    inputs: list[InputSpec]
    outputs: list[OutputSpec]
    metadata: dict[str, Any]
```

---

## Class: `ModelInspector`

### Constructor

```python
ModelInspector()
```

**Example**:
```python
inspector = ModelInspector()
```

---

## Methods

### `inspect()`

Inspect an ONNX model.

```python
inspect(model_path: str | Path) -> ModelInfo
```

**Parameters**:

- `model_path` (`str | Path`): Path to ONNX model

**Returns**:

- `ModelInfo`: Model information

**Raises**:

- `FileNotFoundError`: If model not found
- `ValueError`: If invalid model

**Example**:
```python
info = inspector.inspect("sentiment.onnx")
print(f"Model: {info.name}")
print(f"Size: {info.size_mb:.1f} MB")
print(f"Inputs: {len(info.inputs)}")
print(f"Outputs: {len(info.outputs)}")
```

---

## ModelInfo Methods

### `to_dict()`

Convert to dictionary.

```python
to_dict() -> dict[str, Any]
```

**Returns**:

- `dict`: Model information as dictionary

**Example**:
```python
info_dict = info.to_dict()
print(info_dict["size_mb"])
```

### `save()`

Save model info to JSON file.

```python
save(path: str | Path) -> None
```

**Parameters**:

- `path` (`str | Path`): Output file path

**Example**:
```python
info.save("model_info.json")
```

### `from_dict()`

Create from dictionary.

```python
@classmethod
from_dict(data: dict[str, Any]) -> ModelInfo
```

**Parameters**:

- `data` (`dict`): Dictionary data

**Returns**:

- `ModelInfo`: Model information

**Example**:
```python
info = ModelInfo.from_dict(data)
```

---

## Complete Example

```python
from gpux.core.models import ModelInspector

# Create inspector
inspector = ModelInspector()

# Inspect model
info = inspector.inspect("sentiment.onnx")

# Model information
print(f"Name: {info.name}")
print(f"Version: {info.version}")
print(f"Format: {info.format}")
print(f"Size: {info.size_mb:.1f} MB")

# Inputs
print("\nInputs:")
for inp in info.inputs:
    print(f"  - {inp.name}: {inp.type} {inp.shape}")

# Outputs
print("\nOutputs:")
for out in info.outputs:
    print(f"  - {out.name}: {out.type} {out.shape}")
    if out.labels:
        print(f"    Labels: {out.labels}")

# Metadata
print(f"\nMetadata: {info.metadata}")

# Save to file
info.save("model_info.json")
```

---

## See Also

- [GPUXRuntime](runtime.md)
- [Model Guide](../../guide/models.md)
