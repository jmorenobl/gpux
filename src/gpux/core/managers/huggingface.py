"""Hugging Face Hub integration for GPUX."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from gpux.core.managers.base import ModelManager, ModelMetadata, RegistryConfig
from gpux.core.managers.exceptions import (
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    RegistryError,
)

logger = logging.getLogger(__name__)


class HuggingFaceManager(ModelManager):
    """Model manager for Hugging Face Hub integration."""

    def __init__(self, config: RegistryConfig | None = None) -> None:
        """Initialize the Hugging Face manager.

        Args:
            config: Registry configuration (optional, will use defaults)
        """
        if config is None:
            config = RegistryConfig(
                name="huggingface",
                api_url="https://huggingface.co",
                auth_token=os.getenv("HF_TOKEN"),
            )

        super().__init__(config)
        self.api = HfApi(token=self.config.auth_token)
        self.console = Console()

    def pull_model(
        self,
        model_id: str,
        revision: str = "main",
        cache_dir: Path | None = None,
    ) -> ModelMetadata:
        """Pull a model from Hugging Face Hub.

        Args:
            model_id: Model identifier (e.g., "microsoft/DialoGPT-medium")
            revision: Model revision/branch to pull
            cache_dir: Custom cache directory (overrides config)

        Returns:
            Model metadata including file paths

        Raises:
            ModelNotFoundError: If model doesn't exist
            NetworkError: If download fails
            AuthenticationError: If authentication fails
        """
        try:
            # Check if model exists
            try:
                model_info = self.api.model_info(model_id, revision=revision)
            except RepositoryNotFoundError as e:
                msg = f"Model not found: {model_id}"
                raise ModelNotFoundError(msg) from e

            # Get cache directory
            cache_path = self.get_model_cache_path(model_id, revision, cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)

            # Download model files with progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task(f"Downloading {model_id}...", total=None)

                try:
                    downloaded_path = snapshot_download(  # nosec B615
                        repo_id=model_id,
                        revision=revision,
                        cache_dir=str(cache_path),
                        token=self.config.auth_token,
                    )
                except Exception as e:
                    if "401" in str(e) or "authentication" in str(e).lower():
                        msg = f"Authentication failed for {model_id}"
                        raise AuthenticationError(msg) from e
                    if "network" in str(e).lower() or "connection" in str(e).lower():
                        msg = f"Network error downloading {model_id}: {e}"
                        raise NetworkError(msg) from e
                    msg = f"Failed to download {model_id}: {e}"
                    raise RegistryError(msg) from e

                progress.update(task, description=f"Downloaded {model_id}")

            # Extract metadata
            metadata = self._extract_model_metadata(
                model_id, revision, Path(downloaded_path), model_info
            )

            # Save metadata to cache
            self.save_model_metadata(metadata, cache_dir)

        except (ModelNotFoundError, AuthenticationError, NetworkError, RegistryError):
            raise
        except Exception as e:
            msg = f"Unexpected error pulling model {model_id}: {e}"
            raise RegistryError(msg) from e
        else:
            return metadata

    def search_models(
        self,
        query: str,
        limit: int = 10,
        **filters: Any,
    ) -> list[ModelMetadata]:
        """Search for models in Hugging Face Hub.

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
        try:
            # Use HF API to search models
            models = self.api.list_models(
                search=query,
                limit=limit,
                **filters,
            )

            results = []
            for model in models:
                try:
                    metadata = self._create_metadata_from_model_info(model)
                    results.append(metadata)
                except Exception as e:
                    logger.warning("Failed to process model %s: %s", model.modelId, e)
                    continue
        except Exception as e:
            if "401" in str(e) or "authentication" in str(e).lower():
                msg = f"Authentication failed during search: {e}"
                raise AuthenticationError(msg) from e
            if "network" in str(e).lower() or "connection" in str(e).lower():
                msg = f"Network error during search: {e}"
                raise NetworkError(msg) from e
            msg = f"Search failed: {e}"
            raise RegistryError(msg) from e
        else:
            return results

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
        try:
            model_info = self.api.model_info(model_id, revision=revision)
            return self._create_metadata_from_model_info(model_info)

        except RepositoryNotFoundError as e:
            msg = f"Model not found: {model_id}"
            raise ModelNotFoundError(msg) from e
        except Exception as e:
            if "401" in str(e) or "authentication" in str(e).lower():
                msg = f"Authentication failed for {model_id}"
                raise AuthenticationError(msg) from e
            if "network" in str(e).lower() or "connection" in str(e).lower():
                msg = f"Network error getting info for {model_id}: {e}"
                raise NetworkError(msg) from e
            msg = f"Failed to get info for {model_id}: {e}"
            raise RegistryError(msg) from e

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
        try:
            model_info = self.api.model_info(model_id, revision=revision)
            siblings = model_info.siblings or []
            return [file.rfilename for file in siblings]

        except RepositoryNotFoundError as e:
            msg = f"Model not found: {model_id}"
            raise ModelNotFoundError(msg) from e
        except Exception as e:
            if "401" in str(e) or "authentication" in str(e).lower():
                msg = f"Authentication failed for {model_id}"
                raise AuthenticationError(msg) from e
            if "network" in str(e).lower() or "connection" in str(e).lower():
                msg = f"Network error listing files for {model_id}: {e}"
                raise NetworkError(msg) from e
            msg = f"Failed to list files for {model_id}: {e}"
            raise RegistryError(msg) from e

    def _extract_model_metadata(
        self,
        model_id: str,
        revision: str,
        model_path: Path,
        model_info: Any,
    ) -> ModelMetadata:
        """Extract metadata from downloaded model.

        Args:
            model_id: Model identifier
            revision: Model revision
            model_path: Path to downloaded model
            model_info: Hugging Face model info

        Returns:
            Model metadata
        """
        # Detect model format
        model_format = self._detect_model_format(model_path)

        # Calculate total size
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())

        # Create files mapping
        files = {}
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(model_path)
                files[str(rel_path)] = file_path

        # Extract description and tags
        description = getattr(model_info, "cardData", {}).get("description", "")
        tags = getattr(model_info, "tags", [])

        # Create metadata
        return ModelMetadata(
            registry="huggingface",
            model_id=model_id,
            revision=revision,
            format=model_format,
            files=files,
            size_bytes=total_size,
            description=description,
            tags=tags,
            metadata={
                "pipeline_tag": getattr(model_info, "pipeline_tag", None),
                "library_name": getattr(model_info, "library_name", None),
                "downloads": getattr(model_info, "downloads", 0),
                "last_modified": getattr(model_info, "last_modified", None),
            },
        )

    def _create_metadata_from_model_info(self, model_info: Any) -> ModelMetadata:
        """Create metadata from Hugging Face model info.

        Args:
            model_info: Hugging Face model info object

        Returns:
            Model metadata
        """
        # Extract basic info
        model_id = getattr(model_info, "modelId", "")
        revision = getattr(model_info, "sha", "main")

        # Detect format from files
        files = {}
        model_format = "unknown"
        if hasattr(model_info, "siblings"):
            for file_info in model_info.siblings:
                filename = file_info.rfilename
                files[filename] = Path(filename)  # Placeholder path

                # Detect format from file extensions
                if filename.endswith((".bin", ".safetensors")):
                    model_format = "pytorch"
                elif filename.endswith(".h5"):
                    model_format = "tensorflow"
                elif filename.endswith(".onnx"):
                    model_format = "onnx"

        # Extract metadata
        description = getattr(model_info, "cardData", {}).get("description", "")
        tags = getattr(model_info, "tags", [])

        return ModelMetadata(
            registry="huggingface",
            model_id=model_id,
            revision=revision,
            format=model_format,
            files=files,
            size_bytes=0,  # Size unknown without downloading
            description=description,
            tags=tags,
            metadata={
                "pipeline_tag": getattr(model_info, "pipeline_tag", None),
                "library_name": getattr(model_info, "library_name", None),
                "downloads": getattr(model_info, "downloads", 0),
                "last_modified": getattr(model_info, "last_modified", None),
            },
        )

    def _detect_model_format(self, model_path: Path) -> str:
        """Detect model format from files.

        Args:
            model_path: Path to model directory

        Returns:
            Detected model format
        """
        files = [f.name for f in model_path.iterdir() if f.is_file()]

        # Check for PyTorch models
        if any(f.endswith((".bin", ".safetensors")) for f in files):
            return "pytorch"

        # Check for TensorFlow models
        if any(f.endswith(".h5") for f in files):
            return "tensorflow"

        # Check for ONNX models
        if any(f.endswith(".onnx") for f in files):
            return "onnx"

        # Default to unknown
        return "unknown"
