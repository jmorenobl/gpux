"""Base model manager interface for registry integrations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RegistryConfig:
    """Configuration for a model registry.

    Attributes:
        name: Registry name (e.g., "huggingface", "onnx-model-zoo")
        api_url: Base API URL for the registry
        auth_token: Authentication token (if required)
        cache_dir: Directory to cache downloaded models
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
    """

    name: str
    api_url: str
    auth_token: str | None = None
    cache_dir: Path | None = None
    timeout: int = 30
    max_retries: int = 3


@dataclass
class ModelMetadata:
    """Metadata for a model from a registry.

    Attributes:
        registry: Registry name
        model_id: Model identifier (e.g., "microsoft/DialoGPT-medium")
        revision: Model revision/branch (e.g., "main", "v1.0")
        format: Model format ("pytorch", "tensorflow", "onnx", "safetensors")
        files: List of model files and their paths
        size_bytes: Total size of model files in bytes
        description: Model description from registry
        tags: Model tags/categories
        metadata: Additional metadata dictionary
    """

    registry: str
    model_id: str
    revision: str
    format: str
    files: dict[str, Path]
    size_bytes: int
    description: str | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


class ModelManager(ABC):
    """Abstract base class for model registry managers.

    This class defines the interface that all model registry managers must implement.
    It follows the strategy pattern to allow different registries to be supported
    with a consistent API.
    """

    def __init__(self, config: RegistryConfig) -> None:
        """Initialize the model manager.

        Args:
            config: Registry configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def pull_model(
        self,
        model_id: str,
        revision: str = "main",
        cache_dir: Path | None = None,
        *,
        force_download: bool = False,
    ) -> ModelMetadata:
        """Pull a model from the registry.

        Args:
            model_id: Model identifier (e.g., "microsoft/DialoGPT-medium")
            revision: Model revision/branch to pull
            cache_dir: Custom cache directory (overrides config)
            force_download: Force re-download even if model exists locally

        Returns:
            Model metadata including file paths

        Raises:
            ModelNotFoundError: If model doesn't exist
            NetworkError: If download fails
            AuthenticationError: If authentication fails
        """

    @abstractmethod
    def search_models(
        self,
        query: str,
        limit: int = 10,
        **filters: Any,
    ) -> list[ModelMetadata]:
        """Search for models in the registry.

        Args:
            query: Search query string
            limit: Maximum number of results to return
            **filters: Additional search filters

        Returns:
            List of matching model metadata

        Raises:
            NetworkError: If search fails
            AuthenticationError: If authentication fails
        """

    @abstractmethod
    def get_model_info(
        self,
        model_id: str,
        revision: str = "main",
    ) -> ModelMetadata:
        """Get metadata for a model without downloading.

        Args:
            model_id: Model identifier
            revision: Model revision/branch

        Returns:
            Model metadata

        Raises:
            ModelNotFoundError: If model doesn't exist
            NetworkError: If request fails
            AuthenticationError: If authentication fails
        """

    @abstractmethod
    def list_model_files(
        self,
        model_id: str,
        revision: str = "main",
    ) -> list[str]:
        """List files available for a model.

        Args:
            model_id: Model identifier
            revision: Model revision/branch

        Returns:
            List of file names/paths

        Raises:
            ModelNotFoundError: If model doesn't exist
            NetworkError: If request fails
            AuthenticationError: If authentication fails
        """

    def get_cache_dir(self, cache_dir: Path | None = None) -> Path:
        """Get the cache directory for this manager.

        Args:
            cache_dir: Custom cache directory (overrides config)

        Returns:
            Cache directory path
        """
        if cache_dir is not None:
            return cache_dir
        if self.config.cache_dir is not None:
            return self.config.cache_dir
        return Path.home() / ".gpux" / "models" / self.config.name

    def ensure_cache_dir(self, cache_dir: Path | None = None) -> Path:
        """Ensure cache directory exists and return its path.

        Args:
            cache_dir: Custom cache directory (overrides config)

        Returns:
            Cache directory path
        """
        cache_path = self.get_cache_dir(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    def get_model_cache_path(
        self,
        model_id: str,
        revision: str,
        cache_dir: Path | None = None,
    ) -> Path:
        """Get the cache path for a specific model.

        Args:
            model_id: Model identifier
            revision: Model revision
            cache_dir: Custom cache directory (overrides config)

        Returns:
            Model cache directory path
        """
        base_cache = self.ensure_cache_dir(cache_dir)
        # Sanitize model_id for filesystem (replace / with --)
        safe_model_id = model_id.replace("/", "--")
        return base_cache / safe_model_id / revision

    def is_model_cached(
        self,
        model_id: str,
        revision: str,
        cache_dir: Path | None = None,
    ) -> bool:
        """Check if a model is already cached.

        Args:
            model_id: Model identifier
            revision: Model revision
            cache_dir: Custom cache directory (overrides config)

        Returns:
            True if model is cached
        """
        cache_path = self.get_model_cache_path(model_id, revision, cache_dir)
        return cache_path.exists() and any(cache_path.iterdir())

    def get_cached_model_metadata(
        self,
        model_id: str,
        revision: str,
        cache_dir: Path | None = None,
    ) -> ModelMetadata | None:
        """Get metadata for a cached model.

        Args:
            model_id: Model identifier
            revision: Model revision
            cache_dir: Custom cache directory (overrides config)

        Returns:
            Cached model metadata or None if not found
        """
        cache_path = self.get_model_cache_path(model_id, revision, cache_dir)
        metadata_file = cache_path / "metadata.json"

        if not metadata_file.exists():
            return None

        try:
            import json

            with metadata_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert file paths back to Path objects
            files = {name: Path(path) for name, path in data["files"].items()}
            data["files"] = files

            return ModelMetadata(**data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.warning("Failed to load cached metadata: %s", e)
            return None

    def save_model_metadata(
        self,
        metadata: ModelMetadata,
        cache_dir: Path | None = None,
    ) -> None:
        """Save model metadata to cache.

        Args:
            metadata: Model metadata to save
            cache_dir: Custom cache directory (overrides config)
        """
        cache_path = self.get_model_cache_path(
            metadata.model_id,
            metadata.revision,
            cache_dir,
        )
        cache_path.mkdir(parents=True, exist_ok=True)

        metadata_file = cache_path / "metadata.json"

        # Convert Path objects to strings for JSON serialization
        data = {
            "registry": metadata.registry,
            "model_id": metadata.model_id,
            "revision": metadata.revision,
            "format": metadata.format,
            "files": {name: str(path) for name, path in metadata.files.items()},
            "size_bytes": metadata.size_bytes,
            "description": metadata.description,
            "tags": metadata.tags,
            "metadata": metadata.metadata,
        }

        import json

        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
