# [PROPOSED] Automatic Text Preprocessing for Human-Friendly Inputs

**Date**: 2025-01-27
**Author**: Jorge MB
**Category**: Technical Idea
**Priority**: High
**Estimated Effort**: Large

## Summary

Implement automatic text preprocessing in GPUX to convert human-friendly text inputs (like "Hello world") into the tokenized format required by transformer models (input_ids, attention_mask), eliminating the need for users to manually handle tokenization when using models like DistilBERT, BERT, or other text-based models.

## Problem Statement

Currently, users must provide complex, non-human-friendly inputs when running text models:

```bash
# Current complex input required:
uv run gpux run distilbert-base-uncased --input '{"input_ids": [[101, 7592, 2088, 102]], "attention_mask": [[1, 1, 1, 1]]}'
```

This creates several pain points:
- **Poor UX**: Users must manually tokenize text or know the exact input format
- **Error-prone**: Easy to make mistakes in token IDs or attention masks
- **Not scalable**: Difficult to use for batch processing or production systems
- **Barrier to adoption**: Non-technical users can't easily use text models

## Proposed Solution

Implement a comprehensive preprocessing system that automatically converts human-friendly inputs to model-ready formats:

### 1. Enhanced Config Generation
- Detect text-based models during `gpux pull`
- Automatically add preprocessing configuration to `gpux.yml`
- Include tokenizer information and model-specific settings

### 2. Preprocessing Engine
- Create `TextPreprocessor` class using HuggingFace tokenizers
- Support for different model types (BERT, DistilBERT, GPT, etc.)
- Handle padding, truncation, and special tokens automatically

### 3. Runtime Integration
- Modify `GPUXRuntime` to detect when preprocessing is needed
- Automatically apply preprocessing before inference
- Maintain backward compatibility with existing tokenized inputs

### 4. Human-Friendly Interface
```bash
# New human-friendly usage:
uv run gpux run distilbert-base-uncased --input '{"text": "Hello world"}'

# Or even simpler:
uv run gpux run distilbert-base-uncased --text "Hello world"
```

## Benefits

- **Improved UX**: Users can provide natural text inputs instead of tokenized arrays
- **Reduced Errors**: Eliminates manual tokenization mistakes
- **Faster Development**: Developers can focus on model logic instead of preprocessing
- **Better Adoption**: Makes GPUX accessible to non-technical users
- **Production Ready**: Easier to integrate into production systems
- **Consistent Interface**: Standardized input format across all text models

## Implementation Considerations

### Technical Requirements
- Integrate HuggingFace `transformers` library for tokenization
- Extend `ConfigGenerator` to detect text models and add preprocessing config
- Create `TextPreprocessor` class with model-specific tokenization logic
- Modify `GPUXRuntime.infer()` to handle preprocessing pipeline
- Add input format detection and validation
- Support for batch text processing

### Dependencies
- `transformers` library (already used in conversion pipeline)
- HuggingFace tokenizer models (downloaded on-demand)
- Enhanced model metadata from HuggingFace Hub
- Updated configuration schema for preprocessing

### Risks and Challenges
- **Performance Impact**: Tokenization adds overhead - mitigate with caching and optimization
- **Memory Usage**: Tokenizer models require additional memory - implement lazy loading
- **Model Detection**: Not all models are text-based - implement robust detection logic
- **Backward Compatibility**: Existing workflows must continue working - maintain dual input support
- **Tokenizer Availability**: Some models may not have tokenizers - implement fallback mechanisms

## Alternatives Considered

- **Manual Tokenization Scripts**: Rejected - doesn't solve the core UX problem
- **External Preprocessing Service**: Rejected - adds complexity and latency
- **Model-Specific Input Formats**: Rejected - creates inconsistency across models
- **Configuration-Only Approach**: Rejected - doesn't provide automatic conversion

## Success Criteria

- Users can run text models with simple text inputs: `{"text": "Hello world"}`
- Automatic tokenizer detection and configuration during `gpux pull`
- Backward compatibility maintained for existing tokenized inputs
- Performance impact < 10ms per inference for preprocessing overhead
- Support for at least 5 major text model families (BERT, DistilBERT, GPT, RoBERTa, etc.)
- Comprehensive test coverage for preprocessing pipeline

## Next Steps

- [ ] Research tokenizer requirements for major model families
- [ ] Design preprocessing configuration schema
- [ ] Implement `TextPreprocessor` class with HuggingFace integration
- [ ] Enhance `ConfigGenerator` to detect text models
- [ ] Modify `GPUXRuntime` to support preprocessing pipeline
- [ ] Add input format detection and validation
- [ ] Create comprehensive test suite
- [ ] Update documentation with new usage examples
- [ ] Performance testing and optimization

## Related Resources

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)
- [ONNX Runtime Preprocessing Examples](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/python/tools/transformers)
- [GPUX Preprocessing Configuration Docs](../docs/reference/configuration/preprocessing.md)
- [Current GPUX Runtime Implementation](../src/gpux/core/runtime.py)

## Notes

This feature aligns with GPUX's goal of providing a Docker-like UX for ML models. Just as Docker abstracts away containerization complexity, this preprocessing system abstracts away tokenization complexity, making ML models more accessible to a broader audience.

The implementation should follow GPUX's modular architecture, with preprocessing as a separate, pluggable component that can be extended for other data types (images, audio) in the future.

Consider implementing this as part of the broader "Human-Friendly Inputs" initiative that could eventually support:
- Text preprocessing (this idea)
- Image preprocessing (resize, normalize, etc.)
- Audio preprocessing (resampling, feature extraction)
- Multi-modal preprocessing (text + images, etc.)
