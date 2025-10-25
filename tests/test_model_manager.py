"""Tests for model manager interface."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from typing import Any

from gpux.core.managers.base import ModelManager, ModelMetadata, RegistryConfig
from gpux.core.managers.exceptions import (
    AuthenticationError,
    ConversionError,
    ModelNotFoundError,
    NetworkError,
    RegistryError,
)


class TestModelManager(ModelManager):
    """Test implementation of ModelManager for testing."""

    def pull_model(
        self,
        model_id: str,
        revision: str = "main",
        cache_dir: Path | None = None,  # noqa: ARG002
    ) -> ModelMetadata:
        """Test implementation of pull_model."""
        return ModelMetadata(
            registry="test",
            model_id=model_id,
            revision=revision,
            format="onnx",
            files={"model.onnx": Path("/tmp/model.onnx")},  # noqa: S108
            size_bytes=1024,
        )

    def search_models(
        self,
        query: str,  # noqa: ARG002
        limit: int = 10,  # noqa: ARG002
        **filters: Any,  # noqa: ARG002
    ) -> list[ModelMetadata]:
        """Test implementation of search_models."""
        return []

    def get_model_info(
        self,
        model_id: str,
        revision: str = "main",
    ) -> ModelMetadata:
        """Test implementation of get_model_info."""
        return self.pull_model(model_id, revision)

    def list_model_files(
        self,
        model_id: str,  # noqa: ARG002
        revision: str = "main",  # noqa: ARG002
    ) -> list[str]:
        """Test implementation of list_model_files."""
        return ["model.onnx", "config.json"]


class TestRegistryConfig:
    """Test RegistryConfig dataclass."""

    def test_basic_config(self):
        """Test basic config creation."""
        config = RegistryConfig(
            name="test-registry",
            api_url="https://api.test.com",
        )

        assert config.name == "test-registry"
        assert config.api_url == "https://api.test.com"
        assert config.auth_token is None
        assert config.cache_dir is None
        assert config.timeout == 30
        assert config.max_retries == 3

    def test_full_config(self):
        """Test config with all parameters."""
        config = RegistryConfig(
            name="test-registry",
            api_url="https://api.test.com",
            auth_token="token123",  # noqa: S106
            cache_dir=Path("/tmp/cache"),  # noqa: S108
            timeout=60,
            max_retries=5,
        )

        assert config.auth_token == "token123"  # noqa: S105
        assert config.cache_dir == Path("/tmp/cache")  # noqa: S108
        assert config.timeout == 60
        assert config.max_retries == 5


class TestModelMetadata:
    """Test ModelMetadata dataclass."""

    def test_basic_metadata(self):
        """Test basic metadata creation."""
        metadata = ModelMetadata(
            registry="huggingface",
            model_id="microsoft/DialoGPT-medium",
            revision="main",
            format="pytorch",
            files={"model.bin": Path("/tmp/model.bin")},  # noqa: S108
            size_bytes=2048,
        )

        assert metadata.registry == "huggingface"
        assert metadata.model_id == "microsoft/DialoGPT-medium"
        assert metadata.revision == "main"
        assert metadata.format == "pytorch"
        assert metadata.size_bytes == 2048
        assert metadata.description is None
        assert metadata.tags is None
        assert metadata.metadata is None

    def test_full_metadata(self):
        """Test metadata with all parameters."""
        metadata = ModelMetadata(
            registry="huggingface",
            model_id="microsoft/DialoGPT-medium",
            revision="main",
            format="pytorch",
            files={"model.bin": Path("/tmp/model.bin")},  # noqa: S108
            size_bytes=2048,
            description="A conversational AI model",
            tags=["text", "conversation"],
            metadata={"license": "mit", "author": "microsoft"},
        )

        assert metadata.description == "A conversational AI model"
        assert metadata.tags == ["text", "conversation"]
        assert metadata.metadata == {"license": "mit", "author": "microsoft"}


class TestModelManagerInterface:
    """Test ModelManager abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that ModelManager cannot be instantiated directly."""
        config = RegistryConfig(name="test", api_url="https://api.test.com")

        with pytest.raises(TypeError):
            ModelManager(config)

    def test_concrete_implementation(self):
        """Test concrete implementation works."""
        config = RegistryConfig(name="test", api_url="https://api.test.com")
        manager = TestModelManager(config)

        assert manager.config == config
        assert isinstance(manager.logger, type(manager.logger))

    def test_get_cache_dir_default(self):
        """Test default cache directory."""
        config = RegistryConfig(name="test", api_url="https://api.test.com")
        manager = TestModelManager(config)

        cache_dir = manager.get_cache_dir()
        expected = Path.home() / ".gpux" / "models" / "test"
        assert cache_dir == expected

    def test_get_cache_dir_custom(self):
        """Test custom cache directory."""
        config = RegistryConfig(
            name="test",
            api_url="https://api.test.com",
            cache_dir=Path("/custom/cache"),
        )
        manager = TestModelManager(config)

        cache_dir = manager.get_cache_dir()
        assert cache_dir == Path("/custom/cache")

    def test_get_cache_dir_override(self):
        """Test cache directory override."""
        config = RegistryConfig(
            name="test",
            api_url="https://api.test.com",
            cache_dir=Path("/config/cache"),
        )
        manager = TestModelManager(config)

        cache_dir = manager.get_cache_dir(Path("/override/cache"))
        assert cache_dir == Path("/override/cache")

    def test_ensure_cache_dir(self):
        """Test cache directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RegistryConfig(name="test", api_url="https://api.test.com")
            manager = TestModelManager(config)

            cache_dir = manager.ensure_cache_dir(Path(tmpdir) / "test-cache")
            assert cache_dir.exists()
            assert cache_dir.is_dir()

    def test_get_model_cache_path(self):
        """Test model cache path generation."""
        config = RegistryConfig(name="test", api_url="https://api.test.com")
        manager = TestModelManager(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            model_path = manager.get_model_cache_path(
                "microsoft/DialoGPT-medium",
                "main",
                cache_dir,
            )

            expected = cache_dir / "microsoft--DialoGPT-medium" / "main"
            assert model_path == expected

    def test_is_model_cached_false(self):
        """Test model cache check when not cached."""
        config = RegistryConfig(name="test", api_url="https://api.test.com")
        manager = TestModelManager(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            is_cached = manager.is_model_cached(
                "microsoft/DialoGPT-medium",
                "main",
                cache_dir,
            )
            assert not is_cached

    def test_is_model_cached_true(self):
        """Test model cache check when cached."""
        config = RegistryConfig(name="test", api_url="https://api.test.com")
        manager = TestModelManager(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            model_cache = manager.get_model_cache_path(
                "microsoft/DialoGPT-medium",
                "main",
                cache_dir,
            )
            model_cache.mkdir(parents=True)

            # Create a dummy file
            (model_cache / "model.onnx").touch()

            is_cached = manager.is_model_cached(
                "microsoft/DialoGPT-medium",
                "main",
                cache_dir,
            )
            assert is_cached

    def test_get_cached_model_metadata_not_found(self):
        """Test getting cached metadata when not found."""
        config = RegistryConfig(name="test", api_url="https://api.test.com")
        manager = TestModelManager(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            metadata = manager.get_cached_model_metadata(
                "microsoft/DialoGPT-medium",
                "main",
                cache_dir,
            )
            assert metadata is None

    def test_get_cached_model_metadata_found(self):
        """Test getting cached metadata when found."""
        config = RegistryConfig(name="test", api_url="https://api.test.com")
        manager = TestModelManager(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            model_cache = manager.get_model_cache_path(
                "microsoft/DialoGPT-medium",
                "main",
                cache_dir,
            )
            model_cache.mkdir(parents=True)

            # Create metadata file
            metadata_file = model_cache / "metadata.json"
            metadata_data = {
                "registry": "test",
                "model_id": "microsoft/DialoGPT-medium",
                "revision": "main",
                "format": "onnx",
                "files": {"model.onnx": "/tmp/model.onnx"},  # noqa: S108
                "size_bytes": 1024,
                "description": "Test model",
                "tags": ["test"],
                "metadata": {"test": True},
            }

            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump(metadata_data, f)

            metadata = manager.get_cached_model_metadata(
                "microsoft/DialoGPT-medium",
                "main",
                cache_dir,
            )

            assert metadata is not None
            assert metadata.registry == "test"
            assert metadata.model_id == "microsoft/DialoGPT-medium"
            assert metadata.format == "onnx"
            assert metadata.description == "Test model"

    def test_save_model_metadata(self):
        """Test saving model metadata."""
        config = RegistryConfig(name="test", api_url="https://api.test.com")
        manager = TestModelManager(config)

        metadata = ModelMetadata(
            registry="test",
            model_id="microsoft/DialoGPT-medium",
            revision="main",
            format="onnx",
            files={"model.onnx": Path("/tmp/model.onnx")},  # noqa: S108
            size_bytes=1024,
            description="Test model",
            tags=["test"],
            metadata={"test": True},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            manager.save_model_metadata(metadata, cache_dir)

            model_cache = manager.get_model_cache_path(
                "microsoft/DialoGPT-medium",
                "main",
                cache_dir,
            )
            metadata_file = model_cache / "metadata.json"

            assert metadata_file.exists()

            with metadata_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            assert data["registry"] == "test"
            assert data["model_id"] == "microsoft/DialoGPT-medium"
            assert data["format"] == "onnx"
            assert data["files"]["model.onnx"] == "/tmp/model.onnx"  # noqa: S108


class TestExceptions:
    """Test custom exceptions."""

    def test_registry_error(self):
        """Test RegistryError exception."""
        error = RegistryError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_model_not_found_error(self):
        """Test ModelNotFoundError exception."""
        error = ModelNotFoundError("Model not found")
        assert str(error) == "Model not found"
        assert isinstance(error, RegistryError)

    def test_conversion_error(self):
        """Test ConversionError exception."""
        error = ConversionError("Conversion failed")
        assert str(error) == "Conversion failed"
        assert isinstance(error, RegistryError)

    def test_authentication_error(self):
        """Test AuthenticationError exception."""
        error = AuthenticationError("Auth failed")
        assert str(error) == "Auth failed"
        assert isinstance(error, RegistryError)

    def test_network_error(self):
        """Test NetworkError exception."""
        error = NetworkError("Network failed")
        assert str(error) == "Network failed"
        assert isinstance(error, RegistryError)
