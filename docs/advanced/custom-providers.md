# Custom Providers

Extend GPUX with custom execution providers.

---

## 🎯 Overview

Learn how to add custom execution providers to GPUX.

!!! info "Advanced Topic"
    This is an advanced topic. Ensure you understand [GPU Providers](../guide/providers.md) first.

---

## 🔧 Creating a Custom Provider

### Step 1: Define Provider

```python
from gpux.core.providers import ExecutionProvider
from enum import Enum

class CustomProvider(Enum):
    CUSTOM = "CustomExecutionProvider"
```

### Step 2: Register Provider

```python
from gpux.core.providers import ProviderManager

class CustomProviderManager(ProviderManager):
    def _get_provider_priority(self):
        priority = super()._get_provider_priority()
        priority.insert(0, CustomProvider.CUSTOM)
        return priority
```

### Step 3: Configure Provider

```yaml
runtime:
  gpu:
    backend: custom
```

---

## 💡 Key Takeaways

!!! success
    ✅ Custom provider registration
    ✅ Priority configuration
    ✅ Integration with GPUX

---

**Previous:** [Advanced Index](index.md) | **Next:** [Optimization →](optimization.md)
