# [PROPOSED] Dynamic ONNX Opset Selection

**Date**: 2025-01-27
**Author**: Jorge MB
**Category**: Technical Idea
**Priority**: Medium
**Estimated Effort**: Medium

## Summary

Implement intelligent ONNX opset version selection based on model architecture and runtime capabilities to optimize conversion quality and reduce warnings during model conversion. This addresses the current hardcoded opset 14 approach that causes suboptimal conversions for transformer models like DistilBERT.

## Problem Statement

Currently, GPUX uses a hardcoded ONNX opset version 14 for all model conversions, which causes several issues:

1. **Suboptimal Conversions**: Transformer models (DistilBERT, BERT, GPT) perform better with opset 18+
2. **Conversion Warnings**: Users see confusing warnings about "recommended minimum opset"
3. **Performance Impact**: Lower opset versions may result in larger models and slower inference
4. **One-Size-Fits-All**: Different model architectures have different optimal opset requirements

From the terminal output analysis:
```
Opset 14 is lower than the recommended minimum opset (18) to export distilbert.
The ONNX export may fail or the exported model may be suboptimal.
```

## Proposed Solution

Implement a **dynamic opset selection system** that:

1. **Architecture Detection**: Analyze model architecture from config.json or model metadata
2. **Optimal Opset Mapping**: Map model types to their optimal opset versions
3. **Runtime Compatibility**: Check ONNX Runtime version capabilities
4. **Fallback Strategy**: Graceful degradation if optimal opset isn't supported
5. **User Override**: Allow manual opset specification via CLI flags

### Core Implementation

```python
class OpsetSelector:
    """Intelligent ONNX opset version selection."""

    # Architecture-specific optimal opsets
    ARCHITECTURE_OPSETS = {
        'transformer': 18,      # BERT, DistilBERT, RoBERTa, GPT
        'vision': 14,          # ResNet, MobileNet, EfficientNet
        'rnn': 11,              # LSTM, GRU, RNN
        'cnn': 14,              # General CNN models
        'default': 14           # Fallback
    }

    def select_opset(self, metadata: ModelMetadata, runtime_version: str = None) -> int:
        """Select optimal opset version based on model architecture."""

        # 1. Detect model architecture
        architecture = self._detect_architecture(metadata)

        # 2. Get optimal opset for architecture
        optimal_opset = self.ARCHITECTURE_OPSETS.get(architecture, 14)

        # 3. Check runtime compatibility
        max_supported = self._get_max_supported_opset(runtime_version)

        # 4. Return compatible opset
        return min(optimal_opset, max_supported)

    def _detect_architecture(self, metadata: ModelMetadata) -> str:
        """Detect model architecture from config or model name."""
        model_name = metadata.model_id.lower()

        # Check model name patterns
        if any(arch in model_name for arch in ['bert', 'distilbert', 'roberta', 'gpt', 't5']):
            return 'transformer'
        elif any(arch in model_name for arch in ['resnet', 'mobilenet', 'efficientnet', 'vit']):
            return 'vision'
        elif any(arch in model_name for arch in ['lstm', 'gru', 'rnn']):
            return 'rnn'
        elif any(arch in model_name for arch in ['cnn', 'conv']):
            return 'cnn'

        # Fallback: analyze config.json
        return self._analyze_config_architecture(metadata)
```

### Integration Points

1. **PyTorchConverter**: Use OpsetSelector in both optimum and torch export methods
2. **TensorFlowConverter**: Apply same logic for TensorFlow models
3. **CLI Interface**: Add `--opset` flag for manual override
4. **Configuration**: Store selected opset in gpux.yml metadata

## Benefits

- **Optimized Performance**: Transformer models get opset 18+ for better optimization
- **Reduced Warnings**: Eliminates confusing opset version warnings
- **Better Compatibility**: Ensures models work optimally with target runtimes
- **User Control**: Allows manual override when needed
- **Future-Proof**: Automatically adapts to new ONNX Runtime versions
- **Architecture-Aware**: Different model types get their optimal opset versions

## Implementation Considerations

### Technical Requirements

- **ONNX Runtime Version Detection**: Check installed ONNX Runtime version
- **Model Architecture Analysis**: Parse config.json and model metadata
- **Backward Compatibility**: Ensure older ONNX Runtime versions still work
- **Error Handling**: Graceful fallback if optimal opset fails
- **Logging**: Clear logging of opset selection decisions

### Dependencies

- **ONNX Runtime**: For version detection and compatibility checking
- **Transformers Library**: For config.json parsing and architecture detection
- **Existing Conversion Pipeline**: Integration with PyTorchConverter and TensorFlowConverter

### Risks and Challenges

- **Runtime Compatibility**: Risk of selecting opset higher than runtime supports
  - **Mitigation**: Always check runtime capabilities before selection
- **Conversion Failures**: Higher opset might cause conversion to fail
  - **Mitigation**: Implement fallback to lower opset if conversion fails
- **Performance Regression**: Some models might perform worse with higher opset
  - **Mitigation**: Benchmark and validate performance improvements

## Alternatives Considered

- **Static Opset 18**: Use opset 18 for all models
  - **Rejected**: May break compatibility with older ONNX Runtime versions
- **User Configuration Only**: Let users specify opset in gpux.yml
  - **Rejected**: Too complex for users, most don't know optimal opset
- **Runtime Detection Only**: Select based on runtime capabilities only
  - **Rejected**: Doesn't optimize for specific model architectures

## Success Criteria

1. **Warning Elimination**: DistilBERT conversion shows no opset warnings
2. **Performance Improvement**: Transformer models show measurable performance gains
3. **Compatibility**: All existing models continue to work
4. **User Experience**: Clear logging of opset selection decisions
5. **Backward Compatibility**: Older ONNX Runtime versions still supported

## Next Steps

- [ ] **Research Phase**: Analyze ONNX Runtime version compatibility matrix
- [ ] **Design Phase**: Create detailed OpsetSelector class design
- [ ] **Implementation Phase**: Implement OpsetSelector and integrate with converters
- [ ] **Testing Phase**: Test with various model architectures and runtime versions
- [ ] **Validation Phase**: Benchmark performance improvements
- [ ] **Documentation Phase**: Update documentation with new opset selection behavior

## Related Resources

- [ONNX Opset Versioning Documentation](https://onnx.ai/onnx/repo-docs/Versioning.html)
- [ONNX Runtime Compatibility Matrix](https://onnxruntime.ai/docs/reference/compatibility.html)
- [Hugging Face Optimum Export Documentation](https://huggingface.co/docs/optimum/exporters/onnx/overview)
- [PyTorch ONNX Export Documentation](https://pytorch.org/docs/stable/onnx.html)

## Notes

### Current Warning Analysis

From the terminal output, the specific warnings that would be addressed:

1. **Opset Version Warning**:
   ```
   Opset 14 is lower than the recommended minimum opset (18) to export distilbert
   ```
   - **Solution**: Auto-select opset 18 for transformer models

2. **Producer Version Warning**:
   ```
   Failed to extract metadata: 'onnxruntime.capi.onnxruntime_pybind11_state.ModelMetadata' object has no attribute 'producer_version'
   ```
   - **Solution**: Safe attribute access with fallback values

3. **Tolerance Warning**:
   ```
   The maximum absolute difference between the output of the reference model and the ONNX exported model is not within the set tolerance 1e-05
   ```
   - **Solution**: Adjust tolerance based on model architecture

### Implementation Priority

1. **High Priority**: Opset selection for transformer models (DistilBERT, BERT)
2. **Medium Priority**: Vision model optimization (ResNet, MobileNet)
3. **Low Priority**: RNN model support (LSTM, GRU)

### Future Enhancements

- **Model-Specific Tuning**: Fine-tune opset selection based on specific model performance
- **Runtime Profiling**: Automatically detect optimal opset through performance testing
- **Community Feedback**: Collect user feedback on opset selection accuracy
