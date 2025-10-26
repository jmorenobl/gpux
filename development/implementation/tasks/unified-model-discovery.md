# [COMPLETED] Implement Unified Model Discovery and Local Project Support

**Task ID**: TASK-2025-05
**Created**: 2025-10-25
**Completed**: 2025-10-26
**Assigned To**: Jorge MB
**Priority**: High
**Size**: M
**Estimated Hours**: 8 hours
**Actual Hours**: 6 hours
**Status**: COMPLETED

## Description

Implement unified model discovery system that supports both registry models (downloaded via `gpux pull`) and local projects (with custom `gpux.yml` files). This creates a Docker-like experience where all commands (`run`, `inspect`, `serve`) work seamlessly with both registry and local models.

## Acceptance Criteria

- [x] `ModelDiscovery` class with unified search logic
- [x] Support for local project paths (`./my-model`, `./path/to/model`)
- [x] Support for registry models (cached in `~/.gpux/models/`)
- [x] Support for current directory (`./gpux.yml`)
- [x] Support for `.gpux/` directory structure
- [x] Consistent model discovery across all commands (`run`, `inspect`, `serve`)
- [x] Clear error messages with helpful suggestions
- [x] Comprehensive test coverage for all discovery scenarios

## Technical Requirements

- Create `ModelDiscovery` class in `src/gpux/core/discovery.py`
- Implement search priority: Local projects → Cache directory → Current directory → .gpux directory
- Support both absolute and relative paths for local projects
- Validate `gpux.yml` files and extract model metadata
- Handle edge cases (missing files, invalid configs, permission errors)
- Provide clear error messages with actionable suggestions
- Maintain backward compatibility with existing functionality

## Dependencies

### Prerequisites
- `gpux pull` command (TASK-2025-03) ✅ Complete
- ONNX conversion pipeline (TASK-2025-04) ✅ Complete
- Config generation system ✅ Complete

### Blockers
- None (can be implemented in parallel with other tasks)

## Implementation Plan

### Step 1: Core Discovery Class
- Create `ModelDiscovery` class with static methods
- Implement `find_model_config()` method
- Add helper methods for different search locations
- Implement config validation logic

### Step 2: Search Logic Implementation
- Implement local project detection (`./` paths)
- Implement cache directory search (`~/.gpux/models/`)
- Implement current directory search (`./gpux.yml`)
- Implement `.gpux/` directory search
- Add proper error handling for each search location

### Step 3: Command Integration
- Update `gpux run` command to use `ModelDiscovery`
- Update `gpux inspect` command to use `ModelDiscovery`
- Update `gpux serve` command to use `ModelDiscovery`
- Ensure consistent behavior across all commands

### Step 4: Error Handling and UX
- Implement helpful error messages
- Add suggestions for common issues
- Test error scenarios and user feedback
- Validate user experience

## Testing Strategy

### Unit Tests
- Test `ModelDiscovery.find_model_config()` with various inputs
- Test each search location independently
- Test error handling and edge cases
- Test config validation logic

### Integration Tests
- Test with real local projects
- Test with registry models
- Test mixed usage scenarios
- Test error scenarios and user feedback

### Manual Testing
- Test with various project structures
- Test with different path formats
- Test error messages and suggestions
- Validate Docker-like user experience

## Files to Modify

- `src/gpux/core/discovery.py` - New ModelDiscovery class
- `src/gpux/cli/run.py` - Update to use ModelDiscovery
- `src/gpux/cli/inspect.py` - Update to use ModelDiscovery
- `src/gpux/cli/serve.py` - Update to use ModelDiscovery
- `tests/test_model_discovery.py` - New test file
- `tests/test_cli_run.py` - Update tests
- `tests/test_cli_inspect.py` - Update tests
- `tests/test_cli_serve.py` - Update tests

## Resources

### Documentation
- [Docker CLI Patterns](https://docs.docker.com/engine/reference/commandline/)
- [Pathlib Documentation](https://docs.python.org/3/library/pathlib.html)

### Code References
- Existing CLI commands for integration patterns
- Current model discovery logic in `inspect.py`

### External Resources
- [Docker Compose Project Structure](https://docs.docker.com/compose/gettingstarted/)
- [Docker CLI Discovery Patterns](https://docs.docker.com/engine/reference/commandline/)

## Progress Log

### 2025-10-25
- Task created and planned
- Ready to begin implementation

## Completion Notes

### What was implemented
- TBD

### What was not implemented
- TBD

### Lessons learned
- TBD

### Future improvements
- Add support for model aliases and shortcuts
- Implement model dependency resolution
- Add support for model templates and scaffolding
- Implement model validation and integrity checking

## Related Resources

- [Multi-Registry Model Integration Phase](../phases/current/multi-registry-phase-1.md)
- [GPUX Pull Command Task](./gpux-pull-command.md)
- [ONNX Conversion Pipeline Task](./onnx-conversion-pipeline.md)
- [Multi-Registry Model Integration Idea](../../ideas/feature-requests/huggingface-integration.md)

## Notes

This task creates the foundation for a Docker-like user experience where all commands work seamlessly with both registry and local models. The unified discovery system ensures consistent behavior across all GPUX commands and provides clear, helpful feedback to users.
