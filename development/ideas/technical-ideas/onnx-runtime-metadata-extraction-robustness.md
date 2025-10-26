# [PROPOSED] ONNX Runtime Metadata Extraction Robustness

**Date**: 2025-01-27
**Author**: Jorge MB
**Category**: Technical Idea
**Priority**: Medium
**Estimated Effort**: Small

## Summary

Implement robust metadata extraction from ONNX models with safe attribute access and fallback values to eliminate the "producer_version" attribute error and improve compatibility across different ONNX Runtime versions. This addresses the specific error seen during DistilBERT conversion where the ModelMetadata object lacks certain attributes.

## Problem Statement

During model conversion, GPUX encounters metadata extraction errors due to ONNX Runtime API inconsistencies:

```
WARNING  Failed to extract metadata: 'onnxruntime.capi.onnxruntime_pybind11_state.ModelMetadata' object has no attribute 'producer_version'
```

This error occurs because:

1. **API Inconsistency**: Different ONNX Runtime versions have different ModelMetadata attributes
2. **Missing Attributes**: Some attributes like `producer_version` may not exist in all versions
3. **Hardcoded Access**: Current code assumes all attributes exist
4. **Poor Error Handling**: Exceptions during metadata extraction cause warnings

## Proposed Solution

Implement **robust metadata extraction** with:

1. **Safe Attribute Access**: Use `getattr()` with fallback values
2. **Version Detection**: Detect ONNX Runtime version and adapt accordingly
3. **Graceful Degradation**: Continue operation even if metadata extraction fails
4. **Comprehensive Fallbacks**: Provide sensible default values for missing attributes
5. **Better Error Handling**: Catch and handle specific attribute errors

### Core Implementation

```python
class RobustMetadataExtractor:
    """Robust ONNX model metadata extraction with safe attribute access."""

    # Default values for missing metadata attributes
    DEFAULT_METADATA = {
        'producer_name': 'unknown',
        'producer_version': 'unknown',
        'domain': 'unknown',
        'model_version': 'unknown',
        'doc_string': '',
        'ir_version': 1,
        'opset_version': 11
    }

    def extract_metadata(self, session: ort.InferenceSession) -> dict[str, Any]:
        """Extract metadata with safe attribute access."""
        metadata = {}

        try:
            model_metadata = session.get_modelmeta()
            if model_metadata:
                # Safe attribute access with fallbacks
                metadata.update({
                    'producer_name': self._safe_getattr(model_metadata, 'producer_name'),
                    'producer_version': self._safe_getattr(model_metadata, 'producer_version'),
                    'domain': self._safe_getattr(model_metadata, 'domain'),
                    'model_version': self._safe_getattr(model_metadata, 'model_version'),
                    'doc_string': self._safe_getattr(model_metadata, 'doc_string'),
                })

                # Extract additional model information
                metadata.update(self._extract_model_info(session))

        except (RuntimeError, ImportError, AttributeError) as e:
            logger.warning("Failed to extract metadata: %s", e)
            # Use default values
            metadata.update(self.DEFAULT_METADATA)

        return metadata

    def _safe_getattr(self, obj: Any, attr_name: str, default: str = 'unknown') -> str:
        """Safely get attribute with fallback."""
        try:
            value = getattr(obj, attr_name, default)
            return value if value is not None else default
        except (AttributeError, TypeError):
            return default

    def _extract_model_info(self, session: ort.InferenceSession) -> dict[str, Any]:
        """Extract additional model information."""
        info = {}

        try:
            # Get execution providers
            providers = session.get_providers()
            info['execution_providers'] = providers

            # Get input/output information
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            info['input_count'] = len(inputs)
            info['output_count'] = len(outputs)

            # Extract input/output details
            info['inputs'] = [
                {
                    'name': inp.name,
                    'type': inp.type,
                    'shape': list(inp.shape) if inp.shape else []
                }
                for inp in inputs
            ]

            info['outputs'] = [
                {
                    'name': out.name,
                    'type': out.type,
                    'shape': list(out.shape) if out.shape else []
                }
                for out in outputs
            ]

        except Exception as e:
            logger.warning("Failed to extract model info: %s", e)

        return info
```

### Integration Points

1. **ModelInspector**: Replace current metadata extraction with robust version
2. **ConfigGenerator**: Use robust metadata for configuration generation
3. **Conversion Pipeline**: Apply robust extraction during model conversion
4. **Error Handling**: Improve error handling throughout the pipeline

## Benefits

- **Eliminates Warnings**: No more "producer_version" attribute errors
- **Better Compatibility**: Works across different ONNX Runtime versions
- **Robust Operation**: Continues working even if metadata extraction fails
- **Comprehensive Data**: Extracts more model information safely
- **Future-Proof**: Adapts to ONNX Runtime API changes

## Implementation Considerations

### Technical Requirements

- **Safe Attribute Access**: Use `getattr()` with fallbacks
- **Exception Handling**: Catch and handle specific attribute errors
- **Default Values**: Provide sensible defaults for missing attributes
- **Logging**: Log metadata extraction issues for debugging

### Dependencies

- **ONNX Runtime**: For model metadata access
- **Existing ModelInspector**: Integration with current metadata extraction
- **Logging System**: For warning and error logging

### Risks and Challenges

- **Information Loss**: Some metadata might be lost with fallback values
  - **Mitigation**: Use informative default values, log missing attributes
- **Performance Impact**: Safe attribute access might be slower
  - **Mitigation**: Minimal overhead, only during metadata extraction
- **API Changes**: ONNX Runtime API might change significantly
  - **Mitigation**: Use flexible attribute access, regular testing

## Alternatives Considered

- **No Metadata Extraction**: Skip metadata extraction entirely
  - **Rejected**: Metadata is useful for model information and debugging
- **Hardcoded Values**: Use fixed values for all metadata
  - **Rejected**: Loses valuable model-specific information
- **Exception Suppression**: Catch and ignore all metadata errors
  - **Rejected**: Might hide real issues, better to handle gracefully

## Success Criteria

1. **Warning Elimination**: No more "producer_version" attribute errors
2. **Compatibility**: Works with different ONNX Runtime versions
3. **Data Preservation**: Extracts available metadata successfully
4. **Graceful Degradation**: Continues operation if metadata extraction fails
5. **Performance**: No measurable impact on conversion speed

## Next Steps

- [ ] **Analysis Phase**: Identify all metadata extraction points in codebase
- [ ] **Design Phase**: Design robust metadata extraction architecture
- [ ] **Implementation Phase**: Implement RobustMetadataExtractor
- [ ] **Testing Phase**: Test with different ONNX Runtime versions
- [ ] **Validation Phase**: Ensure metadata extraction works reliably
- [ ] **Documentation Phase**: Document metadata extraction behavior

## Related Resources

- [ONNX Runtime Python API documentation](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [ONNX Runtime ModelMetadata documentation](https://onnxruntime.ai/docs/api/python/api_summary.html#modelmetadata)
- [Python getattr documentation](https://docs.python.org/3/library/functions.html#getattr)

## Notes

### Current Error Analysis

From the terminal output, the specific error:

```
WARNING  Failed to extract metadata: 'onnxruntime.capi.onnxruntime_pybind11_state.ModelMetadata' object has no attribute 'producer_version'
```

This occurs in the `_extract_metadata` method of `ModelInspector`:

```python
# Current problematic code
metadata.update({
    "producer_name": model_metadata.producer_name,
    "producer_version": model_metadata.producer_version,  # This fails
    "domain": model_metadata.domain,
    "model_version": model_metadata.model_version,
    "doc_string": model_metadata.doc_string,
})
```

### Implementation Strategy

1. **Phase 1**: Replace hardcoded attribute access with safe `getattr()` calls
2. **Phase 2**: Add comprehensive fallback values
3. **Phase 3**: Enhance with additional model information extraction
4. **Phase 4**: Add ONNX Runtime version detection and adaptation

### Future Enhancements

- **Version-Specific Handling**: Different extraction logic for different ONNX Runtime versions
- **Metadata Validation**: Validate extracted metadata for consistency
- **Performance Optimization**: Cache metadata extraction results
- **User Configuration**: Allow users to configure metadata extraction behavior
