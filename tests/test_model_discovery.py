"""Tests for ModelDiscovery class."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from gpux.core.discovery import ModelDiscovery
from gpux.core.managers.exceptions import ModelNotFoundError


class TestModelDiscovery:
    """Test cases for ModelDiscovery class."""

    def test_find_model_config_explicit_path(self, tmp_path: Path) -> None:
        """Test finding model config with explicit path."""
        # Create a test model directory
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()

        # Create gpux.yml
        config_file = model_dir / "gpux.yml"
        config_data = {
            "name": "test-model",
            "version": "1.0.0",
            "model": {"source": "./model.onnx", "format": "onnx"},
            "inputs": {},
            "outputs": {},
            "runtime": {"gpu": {"memory": "1GB", "backend": "auto"}},
        }
        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        # Test finding the model
        result = ModelDiscovery.find_model_config(str(model_dir))
        assert result == model_dir

    def test_find_model_config_explicit_file_path(self, tmp_path: Path) -> None:
        """Test finding model config with explicit file path."""
        # Create a test model directory
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()

        # Create gpux.yml
        config_file = model_dir / "gpux.yml"
        config_data = {
            "name": "test-model",
            "version": "1.0.0",
            "model": {"source": "./model.onnx", "format": "onnx"},
            "inputs": {},
            "outputs": {},
            "runtime": {"gpu": {"memory": "1GB", "backend": "auto"}},
        }
        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        # Test finding the model by file path
        result = ModelDiscovery.find_model_config(str(config_file))
        assert result == model_dir

    def test_find_model_config_current_directory(self, tmp_path: Path) -> None:
        """Test finding model config in current directory."""
        # Create gpux.yml in current directory
        config_file = tmp_path / "gpux.yml"
        config_data = {
            "name": "current-model",
            "version": "1.0.0",
            "model": {"source": "./model.onnx", "format": "onnx"},
            "inputs": {},
            "outputs": {},
            "runtime": {"gpu": {"memory": "1GB", "backend": "auto"}},
        }
        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        # Mock Path() to return the tmp_path for current directory
        with patch("gpux.core.discovery.Path") as mock_path:
            mock_path.return_value = tmp_path
            mock_path.cwd.return_value = tmp_path

            result = ModelDiscovery.find_model_config("current-model")
            assert result == tmp_path

    def test_find_model_config_cache_directory(self, tmp_path: Path) -> None:
        """Test finding model config in cache directory."""
        # Create cache structure with revision subdirectory
        cache_dir = tmp_path / ".gpux" / "models" / "huggingface" / "bert-base" / "main"
        cache_dir.mkdir(parents=True)

        # Create gpux.yml in cache
        config_file = cache_dir / "gpux.yml"
        config_data = {
            "name": "bert-base",
            "version": "1.0.0",
            "model": {"source": "./model.onnx", "format": "onnx"},
            "inputs": {},
            "outputs": {},
            "runtime": {"gpu": {"memory": "1GB", "backend": "auto"}},
        }
        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        # Mock home directory
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = ModelDiscovery.find_model_config("bert-base")
            assert result == cache_dir

    def test_find_model_config_cache_directory_direct(self, tmp_path: Path) -> None:
        """Test finding model config in cache directory without revision."""
        # Create cache structure without revision subdirectory
        cache_dir = tmp_path / ".gpux" / "models" / "huggingface" / "gpt2"
        cache_dir.mkdir(parents=True)

        # Create gpux.yml in cache
        config_file = cache_dir / "gpux.yml"
        config_data = {
            "name": "gpt2",
            "version": "1.0.0",
            "model": {"source": "./model.onnx", "format": "onnx"},
            "inputs": {},
            "outputs": {},
            "runtime": {"gpu": {"memory": "1GB", "backend": "auto"}},
        }
        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        # Mock home directory
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = ModelDiscovery.find_model_config("gpt2")
            assert result == cache_dir

    def test_find_model_config_build_directory(self, tmp_path: Path) -> None:
        """Test finding model config in build directory."""
        # Create .gpux directory structure
        gpux_dir = tmp_path / ".gpux"
        gpux_dir.mkdir()

        # Create model info file
        info_file = gpux_dir / "models" / "test-model" / "model_info.json"
        info_file.parent.mkdir(parents=True)

        info_data = {
            "name": "test-model",
            "version": "1.0.0",
            "path": str(gpux_dir.parent),
        }
        with info_file.open("w") as f:
            json.dump(info_data, f)

        # Mock Path() to return the tmp_path for current directory
        with patch("gpux.core.discovery.Path") as mock_path:
            mock_path.return_value = tmp_path
            mock_path.cwd.return_value = tmp_path

            result = ModelDiscovery.find_model_config("test-model")
            assert result == gpux_dir / "models"

    def test_find_model_config_search_priority(self, tmp_path: Path) -> None:
        """Test that search priority is respected."""
        # Create multiple locations with same model name
        explicit_dir = tmp_path / "explicit-model"
        explicit_dir.mkdir()

        cache_dir = tmp_path / ".gpux" / "models" / "huggingface" / "priority-model"
        cache_dir.mkdir(parents=True)

        # Create configs in both locations
        config_data = {
            "name": "priority-model",
            "version": "1.0.0",
            "model": {"source": "./model.onnx", "format": "onnx"},
            "inputs": {},
            "outputs": {},
            "runtime": {"gpu": {"memory": "1GB", "backend": "auto"}},
        }

        with (explicit_dir / "gpux.yml").open("w") as f:
            yaml.dump(config_data, f)

        with (cache_dir / "gpux.yml").open("w") as f:
            yaml.dump(config_data, f)

        # Mock home directory
        with patch("pathlib.Path.home", return_value=tmp_path):
            # Explicit path should be found first
            result = ModelDiscovery.find_model_config(str(explicit_dir))
            assert result == explicit_dir

    def test_find_model_config_not_found(self) -> None:
        """Test ModelNotFoundError when model is not found."""
        with pytest.raises(ModelNotFoundError) as exc_info:
            ModelDiscovery.find_model_config("nonexistent-model")

        error = exc_info.value
        assert error.model_name == "nonexistent-model"
        assert len(error.search_locations) == 4  # All search locations
        assert not any(loc["found"] for loc in error.search_locations)
        assert len(error.suggestions) > 0

    def test_find_model_config_custom_config_file(self, tmp_path: Path) -> None:
        """Test finding model config with custom config file name."""
        # Create a test model directory
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()

        # Create custom config file
        config_file = model_dir / "custom.yml"
        config_data = {
            "name": "test-model",
            "version": "1.0.0",
            "model": {"source": "./model.onnx", "format": "onnx"},
            "inputs": {},
            "outputs": {},
            "runtime": {"gpu": {"memory": "1GB", "backend": "auto"}},
        }
        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        # Test finding the model with custom config file
        result = ModelDiscovery.find_model_config(str(model_dir), "custom.yml")
        assert result == model_dir

    def test_find_model_config_json_format(self, tmp_path: Path) -> None:
        """Test finding model config with JSON format."""
        # Create a test model directory
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()

        # Create JSON config file
        config_file = model_dir / "gpux.json"
        config_data = {
            "name": "test-model",
            "version": "1.0.0",
            "model": {"source": "./model.onnx", "format": "onnx"},
            "inputs": {},
            "outputs": {},
            "runtime": {"gpu": {"memory": "1GB", "backend": "auto"}},
        }
        with config_file.open("w") as f:
            json.dump(config_data, f)

        # Test finding the model with JSON config
        result = ModelDiscovery.find_model_config(str(model_dir), "gpux.json")
        assert result == model_dir

    def test_is_model_match_yaml(self, tmp_path: Path) -> None:
        """Test _is_model_match with YAML config."""
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()

        config_file = model_dir / "gpux.yml"
        config_data = {
            "name": "test-model",
            "version": "1.0.0",
        }
        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        # Test matching
        assert ModelDiscovery._is_model_match(model_dir, "test-model", "gpux.yml")
        assert not ModelDiscovery._is_model_match(model_dir, "other-model", "gpux.yml")

    def test_is_model_match_json(self, tmp_path: Path) -> None:
        """Test _is_model_match with JSON config."""
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()

        config_file = model_dir / "gpux.json"
        config_data = {
            "name": "test-model",
            "version": "1.0.0",
        }
        with config_file.open("w") as f:
            json.dump(config_data, f)

        # Test matching
        assert ModelDiscovery._is_model_match(model_dir, "test-model", "gpux.json")
        assert not ModelDiscovery._is_model_match(model_dir, "other-model", "gpux.json")

    def test_is_model_match_directory_name_fallback(self, tmp_path: Path) -> None:
        """Test _is_model_match fallback to directory name."""
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()

        # Create invalid config file
        config_file = model_dir / "gpux.yml"
        config_file.write_text("invalid: yaml: content: [")

        # Should fallback to directory name
        assert ModelDiscovery._is_model_match(model_dir, "test-model", "gpux.yml")
        assert not ModelDiscovery._is_model_match(model_dir, "other-model", "gpux.yml")

    def test_generate_suggestions_huggingface_model(self) -> None:
        """Test suggestion generation for Hugging Face model names."""
        suggestions = ModelDiscovery._generate_suggestions("microsoft/DialoGPT-medium")

        assert any(
            "gpux pull microsoft/DialoGPT-medium" in suggestion
            for suggestion in suggestions
        )
        assert any("gpux pull" in suggestion for suggestion in suggestions)

    def test_generate_suggestions_general(self) -> None:
        """Test suggestion generation for general model names."""
        suggestions = ModelDiscovery._generate_suggestions("my-model")

        assert any("explicit path" in suggestion.lower() for suggestion in suggestions)
        assert any("spelling" in suggestion.lower() for suggestion in suggestions)
        assert any("gpux pull" in suggestion for suggestion in suggestions)

    def test_get_cache_directory(self) -> None:
        """Test getting cache directory path."""
        cache_dir = ModelDiscovery.get_cache_directory()

        assert cache_dir.name == "models"
        assert cache_dir.parent.name == ".gpux"
        assert cache_dir.parent.parent.name == str(Path.home().name)

    def test_list_cached_models_empty(self, tmp_path: Path) -> None:
        """Test listing cached models when cache is empty."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            cached_models = ModelDiscovery.list_cached_models()
            assert cached_models == {}

    def test_list_cached_models_with_models(self, tmp_path: Path) -> None:
        """Test listing cached models with actual models."""
        # Create cache structure
        cache_dir = tmp_path / ".gpux" / "models"
        cache_dir.mkdir(parents=True)

        # Create registry directories
        hf_dir = cache_dir / "huggingface"
        hf_dir.mkdir()

        onnx_dir = cache_dir / "onnx-model-zoo"
        onnx_dir.mkdir()

        # Create model directories
        (hf_dir / "bert-base").mkdir()
        (hf_dir / "gpt2").mkdir()
        (onnx_dir / "resnet50").mkdir()

        with patch("pathlib.Path.home", return_value=tmp_path):
            cached_models = ModelDiscovery.list_cached_models()

            assert "huggingface" in cached_models
            assert "onnx-model-zoo" in cached_models
            assert "bert-base" in cached_models["huggingface"]
            assert "gpt2" in cached_models["huggingface"]
            assert "resnet50" in cached_models["onnx-model-zoo"]

    def test_model_not_found_error_formatting(self) -> None:
        """Test ModelNotFoundError error message formatting."""
        search_locations = [
            {"location": "Explicit path: test-model", "found": False, "path": None},
            {"location": "Current directory: ./gpux.yml", "found": False, "path": None},
            {
                "location": "Cache directory: ~/.gpux/models/",
                "found": False,
                "path": None,
            },
            {"location": "Build directory: ./.gpux/", "found": False, "path": None},
        ]
        suggestions = [
            "Pull from Hugging Face: gpux pull test-model",
            "Use explicit path: gpux run ./my-model/",
            "Check model name spelling",
        ]

        error = ModelNotFoundError(
            model_name="test-model",
            search_locations=search_locations,
            suggestions=suggestions,
        )

        error_message = error.format_error_message()

        assert "Error: Model 'test-model' not found" in error_message
        assert "Searched locations:" in error_message
        assert "✗ Explicit path: test-model" in error_message
        assert "✗ Current directory: ./gpux.yml" in error_message
        assert "Suggestions:" in error_message
        assert "• Pull from Hugging Face: gpux pull test-model" in error_message

    def test_model_not_found_error_minimal(self) -> None:
        """Test ModelNotFoundError with minimal information."""
        error = ModelNotFoundError(model_name="test-model")

        assert error.model_name == "test-model"
        assert error.search_locations == []
        assert error.suggestions == []
        assert str(error) == "Model 'test-model' not found"

    def test_model_not_found_error_custom_message(self) -> None:
        """Test ModelNotFoundError with custom message."""
        error = ModelNotFoundError(
            model_name="test-model",
            message="Custom error message",
        )

        assert str(error) == "Custom error message"
        assert error.model_name == "test-model"
