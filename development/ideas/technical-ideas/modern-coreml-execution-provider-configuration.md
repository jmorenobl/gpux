# [PROPOSED] Modern CoreML Execution Provider Configuration

**Date**: 2025-01-27
**Author**: Jorge MB
**Category**: Technical Idea
**Priority**: High
**Estimated Effort**: Small

## Summary

Update the CoreML execution provider configuration to use modern ONNX Runtime options (`ModelFormat` and `MLComputeUnits`) instead of the deprecated `coreml_flags` option. This eliminates the EP Error and ensures optimal CoreML performance on Apple Silicon devices.

## Problem Statement

GPUX currently uses deprecated CoreML configuration options that cause execution provider errors:

```
*************** EP Error ***************
EP Error /Users/runner/work/1/s/onnxruntime/core/providers/coreml/coreml_options.cc:68 void
onnxruntime::CoreMLOptions::ValidateAndParseProviderOption(const ProviderOptions &) Unknown option: coreml_flags
 when using [('CoreMLExecutionProvider', {'coreml_flags': 0})]
Falling back to ['CPUExecutionProvider'] and retrying.
****************************************
```

**Root Cause**: The `coreml_flags` option is deprecated in modern ONNX Runtime versions (1.16.0+).

## Proposed Solution

Replace the deprecated CoreML configuration with modern options:

### Current Configuration (Deprecated)
```python
elif provider == ExecutionProvider.COREML:
    config = {
        "coreml_flags": 0,  # Use default settings - DEPRECATED
    }
```

### Modern Configuration
```python
elif provider == ExecutionProvider.COREML:
    config = {
        "ModelFormat": "MLProgram",  # Modern CoreML format
        "MLComputeUnits": "ALL",    # Use all available compute units
    }
```

### Enhanced Implementation

```python
def get_provider_config(self, provider: ExecutionProvider) -> dict[str, Any]:
    """Get configuration for a specific provider with modern options."""
    config: dict[str, Any] = {}

    if provider == ExecutionProvider.TENSORRT:
        config = {
            "trt_max_workspace_size": 1 << 30,  # 1GB
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
        }
    elif provider == ExecutionProvider.CUDA:
        config = {
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }
    elif provider == ExecutionProvider.COREML:
        # Modern CoreML configuration
        config = {
            "ModelFormat": "MLProgram",  # Use modern MLProgram format
            "MLComputeUnits": "ALL",     # Use all available compute units
        }
    elif provider == ExecutionProvider.DIRECTML:
        config = {
            "device_id": 0,
        }

    return config
```

### Advanced Configuration Options

For more control, we can add environment-based configuration:

```python
def get_coreml_config(self) -> dict[str, Any]:
    """Get CoreML configuration based on environment and preferences."""
    config = {
        "ModelFormat": "MLProgram",  # Modern format
    }

    # Compute units configuration
    compute_units = os.getenv("COREML_COMPUTE_UNITS", "ALL")
    if compute_units in ["CPUOnly", "CPUAndNeuralEngine", "ALL"]:
        config["MLComputeUnits"] = compute_units
    else:
        config["MLComputeUnits"] = "ALL"  # Default to all

    # Optional: Enable profiling if requested
    if self._config.get("enable_profiling", False):
        config["enable_profiling"] = True

    return config
```

## Benefits

- **Eliminates EP Error**: No more "Unknown option: coreml_flags" errors
- **Better Performance**: MLProgram format provides better optimization
- **Modern Compatibility**: Works with ONNX Runtime 1.16.0+
- **Apple Silicon Optimization**: Properly utilizes Neural Engine and GPU
- **Future-Proof**: Uses current CoreML API standards

## Implementation Considerations

### Technical Requirements

- **ONNX Runtime Version**: Requires ONNX Runtime 1.16.0+ (already satisfied)
- **CoreML Compatibility**: Works with iOS 13+ and macOS 10.15+
- **Model Format**: MLProgram format for better performance
- **Compute Units**: Configurable compute unit selection

### Dependencies

- **ONNX Runtime**: Version 1.16.0+ (already in pyproject.toml)
- **CoreML Framework**: Available on Apple platforms
- **Existing ProviderManager**: Integration with current provider system

### Risks and Challenges

- **Platform Limitation**: CoreML only works on Apple devices
  - **Mitigation**: Graceful fallback to CPU on non-Apple platforms
- **Model Compatibility**: Some models might not support MLProgram format
  - **Mitigation**: Fallback to NeuralNetwork format if needed
- **Performance Variation**: Different compute unit configurations
  - **Mitigation**: Allow user configuration via environment variables

## Alternatives Considered

- **Remove CoreML Configuration**: Use default CoreML settings
  - **Rejected**: Loses optimization opportunities
- **Keep Deprecated Options**: Continue using coreml_flags
  - **Rejected**: Causes EP errors and poor user experience
- **Version Detection**: Detect ONNX Runtime version and use appropriate options
  - **Considered**: Good approach but modern options work universally

## Success Criteria

1. **Error Elimination**: No more "coreml_flags" EP errors
2. **Performance Improvement**: Better CoreML performance on Apple Silicon
3. **Compatibility**: Works with ONNX Runtime 1.16.0+
4. **User Experience**: Cleaner output without EP errors
5. **Backward Compatibility**: Graceful fallback for unsupported configurations

## Next Steps

- [ ] **Research Phase**: Verify CoreML configuration options for ONNX Runtime 1.16.0+
- [ ] **Implementation Phase**: Update ProviderManager with modern CoreML config
- [ ] **Testing Phase**: Test on Apple Silicon devices
- [ ] **Validation Phase**: Ensure no EP errors occur
- [ ] **Documentation Phase**: Update CoreML configuration documentation

## Related Resources

- [ONNX Runtime CoreML Execution Provider Documentation](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [CoreML Tools Documentation](https://apple.github.io/coremltools/docs-guides/source/faqs.html)
- [ONNX Runtime Compatibility Matrix](https://onnxruntime.ai/docs/reference/compatibility.html)

## Notes

### Current Error Analysis

The specific error occurs in the `get_provider_config` method:

```python
# Current problematic code (src/gpux/core/providers.py:162-164)
elif provider == ExecutionProvider.COREML:
    config = {
        "coreml_flags": 0,  # DEPRECATED - causes EP error
    }
```

### Modern CoreML Options

Based on ONNX Runtime documentation, the modern CoreML options are:

1. **ModelFormat**:
   - `"MLProgram"` (recommended) - Modern format with better optimization
   - `"NeuralNetwork"` (legacy) - Older format for compatibility

2. **MLComputeUnits**:
   - `"ALL"` (recommended) - Use all available compute units
   - `"CPUOnly"` - CPU only
   - `"CPUAndNeuralEngine"` - CPU + Neural Engine

### Implementation Strategy

1. **Phase 1**: Replace deprecated `coreml_flags` with modern options
2. **Phase 2**: Add environment variable configuration
3. **Phase 3**: Add fallback handling for unsupported configurations
4. **Phase 4**: Add performance monitoring and optimization

### Future Enhancements

- **Dynamic Configuration**: Detect optimal settings based on model characteristics
- **Performance Monitoring**: Track CoreML performance metrics
- **User Configuration**: Allow users to specify CoreML preferences
- **Model-Specific Optimization**: Optimize CoreML settings per model type
