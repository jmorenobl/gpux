# Configuration Classes

Configuration parsing and validation classes.

---

## Overview

GPUX uses Pydantic models for configuration parsing and validation.

```python
from gpux.config.parser import GPUXConfigParser

parser = GPUXConfigParser()
config = parser.parse_file("gpux.yml")
```

---

## Class: `GPUXConfigParser`

### Constructor

```python
GPUXConfigParser()
```

**Example**:
```python
parser = GPUXConfigParser()
```

---

## Methods

### `parse_file()`

Parse configuration from file.

```python
parse_file(config_path: str | Path) -> GPUXConfig
```

**Parameters**:

- `config_path` (`str | Path`): Path to `gpux.yml`

**Returns**:

- `GPUXConfig`: Parsed configuration

**Raises**:

- `FileNotFoundError`: If config file not found
- `ValueError`: If config invalid

**Example**:
```python
config = parser.parse_file("gpux.yml")
print(config.name)
```

### `parse_string()`

Parse configuration from string.

```python
parse_string(config_str: str) -> GPUXConfig
```

**Parameters**:

- `config_str` (`str`): YAML configuration string

**Returns**:

- `GPUXConfig`: Parsed configuration

**Example**:
```python
yaml_str = """
name: model
version: 1.0.0
model:
  source: ./model.onnx
inputs:
  - name: input
    type: float32
outputs:
  - name: output
    type: float32
"""
config = parser.parse_string(yaml_str)
```

### `get_model_path()`

Get absolute model path.

```python
get_model_path(base_path: str | Path | None = None) -> Path | None
```

**Parameters**:

- `base_path` (`str | Path`, optional): Base path for relative paths

**Returns**:

- `Path | None`: Absolute model path

**Example**:
```python
model_path = parser.get_model_path(Path("./project"))
```

---

## Configuration Models

### `GPUXConfig`

Main configuration model.

```python
class GPUXConfig(BaseModel):
    name: str
    version: str = "1.0.0"
    description: str | None = None
    model: ModelConfig
    inputs: list[InputConfig]
    outputs: list[OutputConfig]
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    serving: ServingConfig | None = None
    preprocessing: PreprocessingConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### `ModelConfig`

Model configuration.

```python
class ModelConfig(BaseModel):
    source: str | Path
    format: str = "onnx"
    version: str | None = None
```

### `RuntimeConfig`

Runtime configuration.

```python
class RuntimeConfig(BaseModel):
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    timeout: int = 30
    batch_size: int = 1
    enable_profiling: bool = False
```

### `ServingConfig`

Serving configuration.

```python
class ServingConfig(BaseModel):
    port: int = 8080
    host: str = "0.0.0.0"
    batch_size: int = 1
    timeout: int = 5
    max_workers: int = 4
```

---

## Complete Example

```python
from gpux.config.parser import GPUXConfigParser
from pathlib import Path

# Create parser
parser = GPUXConfigParser()

# Parse configuration
config = parser.parse_file("gpux.yml")

# Access configuration
print(f"Model: {config.name} v{config.version}")
print(f"Source: {config.model.source}")

# Runtime settings
print(f"GPU Memory: {config.runtime.gpu.memory}")
print(f"Backend: {config.runtime.gpu.backend}")
print(f"Batch Size: {config.runtime.batch_size}")

# Inputs
for inp in config.inputs:
    print(f"Input: {inp.name} ({inp.type})")

# Outputs
for out in config.outputs:
    print(f"Output: {out.name} ({out.type})")

# Get model path
model_path = parser.get_model_path(Path("."))
print(f"Model path: {model_path}")
```

---

## See Also

- [Configuration Schema](../configuration/schema.md)
- [GPUXRuntime](runtime.md)
