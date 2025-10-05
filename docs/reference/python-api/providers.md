# ProviderManager

Execution provider management and selection.

---

## Overview

`ProviderManager` handles automatic selection of the best GPU execution provider based on platform and availability.

```python
from gpux.core.providers import ProviderManager, ExecutionProvider

manager = ProviderManager()
provider = manager.get_best_provider()
```

---

## Enum: `ExecutionProvider`

Available execution providers.

```python
class ExecutionProvider(Enum):
    TENSORRT = "TensorrtExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    ROCM = "ROCmExecutionProvider"
    COREML = "CoreMLExecutionProvider"
    DIRECTML = "DirectMLExecutionProvider"
    OPENVINO = "OpenVINOExecutionProvider"
    CPU = "CPUExecutionProvider"
```

---

## Class: `ProviderManager`

### Constructor

```python
ProviderManager()
```

Automatically detects available providers and determines priority.

**Example**:
```python
manager = ProviderManager()
```

---

## Methods

### `get_best_provider()`

Get the best available execution provider.

```python
get_best_provider(preferred: str | None = None) -> ExecutionProvider
```

**Parameters**:

- `preferred` (`str`, optional): Preferred provider name (`cuda`, `coreml`, etc.)

**Returns**:

- `ExecutionProvider`: Best available provider

**Example**:
```python
# Auto-select best provider
provider = manager.get_best_provider()

# Prefer CUDA if available
provider = manager.get_best_provider("cuda")
```

### `get_available_providers()`

Get list of available provider names.

```python
get_available_providers() -> list[str]
```

**Returns**:

- `list[str]`: Available provider names

**Example**:
```python
providers = manager.get_available_providers()
print(f"Available: {providers}")
# Output: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### `get_provider_config()`

Get provider configuration.

```python
get_provider_config(provider: ExecutionProvider) -> list[tuple[str, dict]]
```

**Parameters**:

- `provider` (`ExecutionProvider`): Provider to configure

**Returns**:

- `list[tuple[str, dict]]`: Provider configuration

**Example**:
```python
config = manager.get_provider_config(ExecutionProvider.CUDA)
```

### `get_provider_info()`

Get provider information.

```python
get_provider_info(provider: ExecutionProvider) -> dict[str, Any]
```

**Parameters**:

- `provider` (`ExecutionProvider`): Provider to query

**Returns**:

- `dict`: Provider information
  - `name` (`str`): Provider name
  - `available` (`bool`): Availability
  - `platform` (`str`): Platform/hardware type
  - `description` (`str`): Description

**Example**:
```python
info = manager.get_provider_info(ExecutionProvider.CUDA)
print(f"Provider: {info['name']}")
print(f"Available: {info['available']}")
print(f"Platform: {info['platform']}")
```

---

## Provider Priority

Providers are prioritized based on platform and performance:

### Default Priority

1. TensorRT (NVIDIA - best performance)
2. CUDA (NVIDIA)
3. ROCm (AMD)
4. CoreML (Apple Silicon)
5. DirectML (Windows)
6. OpenVINO (Intel)
7. CPU (Universal fallback)

### Platform-Specific

**Apple Silicon**:
1. CoreML
2. CPU

**Windows**:
1. TensorRT
2. CUDA
3. DirectML
4. OpenVINO
5. CPU

---

## Complete Example

```python
from gpux.core.providers import ProviderManager, ExecutionProvider

# Create manager
manager = ProviderManager()

# Check available providers
available = manager.get_available_providers()
print(f"Available providers: {available}")

# Get best provider
provider = manager.get_best_provider()
print(f"Selected provider: {provider.value}")

# Get provider info
info = manager.get_provider_info(provider)
print(f"Platform: {info['platform']}")
print(f"Description: {info['description']}")

# Prefer specific provider
cuda_provider = manager.get_best_provider("cuda")
if cuda_provider == ExecutionProvider.CUDA:
    print("Using CUDA acceleration")
else:
    print("CUDA not available, falling back to:", cuda_provider.value)
```

---

## See Also

- [GPUXRuntime](runtime.md)
- [Providers Guide](../../guide/providers.md)
