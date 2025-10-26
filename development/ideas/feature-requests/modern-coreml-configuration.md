# [PLANNING] Modern CoreML Execution Provider Configuration

**Date**: 2025-01-27
**Author**: Jorge MB
**Category**: Feature Request
**Priority**: High
**Estimated Effort**: Small

## Summary

Update GPUX's CoreML execution provider configuration to use modern ONNX Runtime options (`ModelFormat` and `MLComputeUnits`) instead of the deprecated `coreml_flags` option. This eliminates execution provider errors and ensures optimal CoreML performance on Apple Silicon devices, providing users with a seamless experience when running models on Mac hardware. The `coreml_flags` API was deprecated in ONNX Runtime 1.20.0, making this update critical for maintaining compatibility and performance.

## Problem Statement

GPUX currently uses deprecated CoreML configuration options that cause execution provider errors and degrade user experience:

```
*************** EP Error ***************
EP Error /Users/runner/work/1/s/onnxruntime/core/providers/coreml/coreml_options.cc:68 void
onnxruntime::CoreMLOptions::ValidateAndParseProviderOption(const ProviderOptions &) Unknown option: coreml_flags
 when using [('CoreMLExecutionProvider', {'coreml_flags': 0})]
Falling back to ['CPUExecutionProvider'] and retrying.
****************************************
```

**Root Cause**: The `coreml_flags` API has been deprecated in ONNX Runtime 1.20.0, and users are encouraged to transition to the new configuration options to avoid compatibility issues. This causes:
- Execution provider errors in logs
- Suboptimal CoreML performance
- Poor user experience on Apple Silicon devices
- Potential fallback to CPU instead of utilizing Neural Engine and GPU

## Proposed Solution

### Core Configuration Update

Replace the deprecated CoreML configuration with modern, optimized options:

```python
# Current problematic configuration (deprecated)
elif provider == ExecutionProvider.COREML:
    config = {
        "coreml_flags": 0,  # DEPRECATED - causes EP error
    }

# Modern configuration
elif provider == ExecutionProvider.COREML:
    config = {
        "ModelFormat": "MLProgram",  # Modern CoreML format with better optimization
        "MLComputeUnits": "ALL",    # Use all available compute units (CPU + Neural Engine + GPU)
    }
```

### Enhanced Configuration with Environment Support

```python
def get_coreml_config(self) -> dict[str, Any]:
    """Get CoreML configuration based on environment and user preferences."""
    config = {
        "ModelFormat": "MLProgram",  # Modern format for better performance
    }

    # Configurable compute units via environment variable
    compute_units = os.getenv("COREML_COMPUTE_UNITS", "ALL")
    if compute_units in ["CPUOnly", "CPUAndNeuralEngine", "ALL"]:
        config["MLComputeUnits"] = compute_units
    else:
        config["MLComputeUnits"] = "ALL"  # Default to optimal performance

    # Optional profiling support
    if self._config.get("enable_profiling", False):
        config["enable_profiling"] = True

    return config
```

### CLI Configuration Support

Add CoreML-specific configuration options to `gpux.yml`:

```yaml
runtime:
  gpu:
    backend: coreml
    coreml:
      model_format: MLProgram  # MLProgram or NeuralNetwork
      compute_units: ALL       # ALL, CPUOnly, CPUAndNeuralEngine
      enable_profiling: false
```

### Advanced Configuration Options

```python
class CoreMLConfig:
    """CoreML-specific configuration options."""

    MODEL_FORMATS = ["MLProgram", "NeuralNetwork"]
    COMPUTE_UNITS = ["ALL", "CPUOnly", "CPUAndNeuralEngine"]

    def __init__(self, config: dict):
        self.model_format = config.get("model_format", "MLProgram")
        self.compute_units = config.get("compute_units", "ALL")
        self.enable_profiling = config.get("enable_profiling", False)

    def to_provider_config(self) -> dict[str, Any]:
        """Convert to ONNX Runtime provider configuration."""
        return {
            "ModelFormat": self.model_format,
            "MLComputeUnits": self.compute_units,
            **({"enable_profiling": True} if self.enable_profiling else {})
        }
```

## Benefits

### For Users
- **Error-free experience** - No more execution provider errors in logs
- **Better performance** - MLProgram format provides superior optimization
- **Apple Silicon optimization** - Properly utilizes Neural Engine and GPU
- **Configurable options** - Users can fine-tune CoreML behavior
- **Future-proof** - Uses current CoreML API standards

### For GPUX
- **Professional appearance** - Clean logs without EP errors
- **Better Apple ecosystem support** - Optimal performance on Mac hardware
- **Modern compatibility** - Works with ONNX Runtime 1.20.0+
- **Competitive advantage** - Better CoreML support than other ML runtimes
- **User satisfaction** - Improved experience for Mac users

### Technical Benefits
- **MLProgram format** - Modern CoreML format with better optimization
- **Compute unit flexibility** - Configurable CPU/Neural Engine/GPU usage
- **Performance monitoring** - Optional profiling support
- **Graceful fallback** - Fallback to NeuralNetwork format if needed
- **Environment configuration** - Runtime configuration via environment variables

## Implementation Considerations

### Technical Requirements
- **ONNX Runtime Version**: Requires ONNX Runtime 1.20.0+ (already satisfied)
- **CoreML Compatibility**: Works with iOS 13+ and macOS 10.15+
- **Model Format**: MLProgram format for better performance
- **Compute Units**: Configurable compute unit selection
- **Fallback Support**: Graceful fallback for unsupported configurations

### Dependencies
- **ONNX Runtime**: Version 1.20.0+ (already in pyproject.toml)
- **CoreML Framework**: Available on Apple platforms
- **Existing ProviderManager**: Integration with current provider system
- **Configuration System**: Integration with existing `gpux.yml` parser

### Risks and Challenges
- **Platform Limitation**: CoreML only works on Apple devices
  - *Mitigation*: Graceful fallback to CPU on non-Apple platforms
- **Model Compatibility**: Some models might not support MLProgram format
  - *Mitigation*: Fallback to NeuralNetwork format if needed
- **Performance Variation**: Different compute unit configurations
  - *Mitigation*: Allow user configuration via environment variables
- **Configuration Complexity**: Additional configuration options
  - *Mitigation*: Sensible defaults, clear documentation

## Alternatives Considered

- **Remove CoreML Configuration**: Use default CoreML settings
  - *Rejected*: Loses optimization opportunities and causes EP errors
- **Keep Deprecated Options**: Continue using coreml_flags
  - *Rejected*: Causes EP errors and poor user experience
- **Version Detection**: Detect ONNX Runtime version and use appropriate options
  - *Considered*: Good approach but modern options work universally
- **Manual Configuration**: Require users to configure CoreML manually
  - *Rejected*: Adds complexity and reduces usability

## Success Criteria

### User Experience
- Zero execution provider errors in logs
- Elimination of all deprecation warnings
- Improved inference performance on Apple Silicon
- Seamless CoreML integration without user intervention
- Clear configuration options for advanced users

### Technical Metrics
- 100% elimination of "coreml_flags" EP errors
- Clean build without deprecation warnings
- Performance improvement on Apple Silicon devices
- Successful fallback to NeuralNetwork format when needed
- Environment variable configuration working correctly

### Compatibility
- Works with ONNX Runtime 1.20.0+
- Compatible with iOS 13+ and macOS 10.15+
- Graceful fallback on non-Apple platforms
- Backward compatibility with existing configurations

## Next Steps

### Phase 1: Core Configuration Update (Week 1)
- [ ] Research modern CoreML configuration options for ONNX Runtime 1.20.0+
- [ ] Update `ProviderManager.get_provider_config()` method
- [ ] Replace deprecated `coreml_flags` with modern options
- [ ] Test basic CoreML functionality on Apple Silicon
- [ ] Verify elimination of EP errors

### Phase 2: Enhanced Configuration (Week 2)
- [ ] Add environment variable support for CoreML configuration
- [ ] Implement `CoreMLConfig` class for advanced options
- [ ] Add CoreML-specific options to `gpux.yml` schema
- [ ] Update configuration parser to handle CoreML options
- [ ] Add fallback handling for unsupported configurations

### Phase 3: Testing and Validation (Week 3)
- [ ] Comprehensive testing on Apple Silicon devices
- [ ] Performance benchmarking with different configurations
- [ ] Test fallback scenarios and error handling
- [ ] Validate compatibility with various ONNX models
- [ ] Update documentation with new configuration options

### Phase 4: Documentation and Polish (Week 4)
- [ ] Update CoreML configuration documentation
- [ ] Add examples for different CoreML configurations
- [ ] Create troubleshooting guide for CoreML issues
- [ ] Update CLI help text and error messages
- [ ] Performance optimization and final testing

## Related Resources

- [ONNX Runtime CoreML Execution Provider Documentation](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [CoreML Tools Documentation](https://apple.github.io/coremltools/docs-guides/source/faqs.html)
- [ONNX Runtime Compatibility Matrix](https://onnxruntime.ai/docs/reference/compatibility.html)
- [Apple CoreML Framework Documentation](https://developer.apple.com/documentation/coreml)
- [MLProgram vs NeuralNetwork Format Comparison](https://developer.apple.com/documentation/coreml/mlprogram)

## Notes

### Deprecation Timeline

- **ONNX Runtime 1.20.0**: The `coreml_flags` API was officially deprecated
- **Current Status**: Users are encouraged to transition to new configuration options
- **Future**: The deprecated API may be removed in future ONNX Runtime versions
- **Impact**: Immediate compatibility issues and performance degradation

### Current Error Analysis

The specific error occurs in the `get_provider_config` method in `src/gpux/core/providers.py`:

```python
# Current problematic code (lines 162-164)
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
- **User Configuration**: Allow users to specify CoreML preferences via CLI
- **Model-Specific Optimization**: Optimize CoreML settings per model type
- **Benchmarking**: Compare performance across different CoreML configurations

This feature request addresses a critical user experience issue while providing significant performance improvements for Apple Silicon users. The implementation is straightforward but has high impact on user satisfaction and GPUX's professional appearance. Given the deprecation of `coreml_flags` in ONNX Runtime 1.20.0, this update is essential for maintaining compatibility and avoiding future breaking changes.
