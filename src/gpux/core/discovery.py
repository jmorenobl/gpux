"""Model discovery system for unified registry and local project support."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from gpux.core.managers.exceptions import ModelNotFoundError

logger = logging.getLogger(__name__)


class ModelDiscovery:
    """Centralized model discovery system for registry and local projects.

    This class provides a unified interface for finding model configurations
    across different locations: explicit paths, current directory, cache,
    and build directories. It follows Docker-like UX patterns where models
    can be referenced by name regardless of their source.
    """

    @staticmethod
    def find_model_config(model_name: str, config_file: str = "gpux.yml") -> Path:
        """Find model configuration file using priority-based search.

        Search priority:
        1. Explicit local paths (./my-model, /absolute/path)
        2. Current directory (./gpux.yml)
        3. Cache directory (~/.gpux/models/{registry}/{model-id}/)
        4. Build directory (.gpux/)

        Args:
            model_name: Name of the model or path to search for
            config_file: Configuration file name (default: "gpux.yml")

        Returns:
            Path to model directory containing the config file

        Raises:
            ModelNotFoundError: If model is not found in any location
        """
        search_locations = []

        # 1. Check explicit paths (starts with ./ or /)
        explicit_path = ModelDiscovery._check_explicit_path(model_name, config_file)
        search_locations.append(
            {
                "location": f"Explicit path: {model_name}",
                "found": explicit_path is not None,
                "path": str(explicit_path) if explicit_path else None,
            }
        )
        if explicit_path:
            return explicit_path

        # 2. Check current directory
        current_dir = ModelDiscovery._check_current_directory(config_file)
        search_locations.append(
            {
                "location": f"Current directory: ./{config_file}",
                "found": current_dir is not None,
                "path": str(current_dir) if current_dir else None,
            }
        )
        if current_dir:
            return current_dir

        # 3. Check cache directory
        cache_path = ModelDiscovery._check_cache_directory(model_name, config_file)
        search_locations.append(
            {
                "location": "Cache directory: ~/.gpux/models/",
                "found": cache_path is not None,
                "path": str(cache_path) if cache_path else None,
            }
        )
        if cache_path:
            return cache_path

        # 4. Check build directory
        build_path = ModelDiscovery._check_build_directory(model_name, config_file)
        search_locations.append(
            {
                "location": "Build directory: ./.gpux/",
                "found": build_path is not None,
                "path": str(build_path) if build_path else None,
            }
        )
        if build_path:
            return build_path

        # Model not found - raise helpful error
        suggestions = ModelDiscovery._generate_suggestions(model_name)
        raise ModelNotFoundError(
            model_name=model_name,
            search_locations=search_locations,
            suggestions=suggestions,
        )

    @staticmethod
    def _check_explicit_path(model_name: str, config_file: str) -> Path | None:
        """Check if model_name is an explicit path.

        Args:
            model_name: Model name or path
            config_file: Configuration file name

        Returns:
            Path to model directory if found, None otherwise
        """
        # Check if it looks like a path (starts with ./ or / or ../)
        if not (model_name.startswith(("./", "/", "../")) or "\\" in model_name):
            return None

        model_path = Path(model_name)

        # If it's a file, use the parent directory
        if model_path.is_file():
            model_path = model_path.parent

        # Check if config file exists in this directory
        config_path = model_path / config_file
        if config_path.exists():
            return model_path

        return None

    @staticmethod
    def _check_current_directory(config_file: str) -> Path | None:
        """Check current directory for config file.

        Args:
            config_file: Configuration file name

        Returns:
            Path to current directory if config exists, None otherwise
        """
        current_dir = Path()
        config_path = current_dir / config_file

        if config_path.exists():
            return current_dir

        return None

    @staticmethod
    def _check_cache_directory(model_name: str, config_file: str) -> Path | None:
        """Check cache directory for model.

        Args:
            model_name: Model name to search for
            config_file: Configuration file name

        Returns:
            Path to model directory if found, None otherwise
        """
        cache_dir = Path.home() / ".gpux" / "models"

        if not cache_dir.exists():
            return None

        # Search all registries in cache
        for registry_dir in cache_dir.iterdir():
            if not registry_dir.is_dir():
                continue

            # Search for model directories
            for model_dir in registry_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                # Check if this is the model we're looking for
                # First check direct model directory
                if ModelDiscovery._is_model_match(model_dir, model_name, config_file):
                    return model_dir

                # Then check revision subdirectories (e.g., main/, v1.0/, etc.)
                for revision_dir in model_dir.iterdir():
                    if not revision_dir.is_dir():
                        continue

                    if ModelDiscovery._is_model_match(
                        revision_dir, model_name, config_file
                    ):
                        return revision_dir

        return None

    @staticmethod
    def _check_build_directory(model_name: str, config_file: str) -> Path | None:  # noqa: ARG004
        """Check build directory (.gpux/) for model.

        Args:
            model_name: Model name to search for
            config_file: Configuration file name

        Returns:
            Path to model directory if found, None otherwise
        """
        gpux_dir = Path(".gpux")

        if not gpux_dir.exists():
            return None

        # Look for model info files
        for info_file in gpux_dir.glob("**/model_info.json"):
            try:
                with info_file.open() as f:
                    info = json.load(f)

                if info.get("name") == model_name:
                    # Return the directory containing the model
                    return info_file.parent.parent
            except (json.JSONDecodeError, OSError):
                continue

        return None

    @staticmethod
    def _is_model_match(model_dir: Path, model_name: str, config_file: str) -> bool:
        """Check if a model directory matches the search criteria.

        Args:
            model_dir: Path to model directory
            model_name: Model name to match
            config_file: Configuration file name

        Returns:
            True if this directory contains the requested model
        """
        config_path = model_dir / config_file

        if not config_path.exists():
            return False

        try:
            # Try to parse the config file to check the model name
            if config_file.endswith(".json"):
                with config_path.open() as f:
                    config_data = json.load(f)
            else:
                with config_path.open() as f:
                    config_data = yaml.safe_load(f)

            # Check if the model name matches
            if config_data and config_data.get("name") == model_name:
                return True

        except Exception:
            # If we can't parse the config, check if directory name matches
            if model_dir.name == model_name:
                return True

        return False

    @staticmethod
    def _generate_suggestions(model_name: str) -> list[str]:
        """Generate helpful suggestions when model is not found.

        Args:
            model_name: Model name that wasn't found

        Returns:
            List of suggestion strings
        """
        suggestions = []

        # Check if it looks like a Hugging Face model name
        if "/" in model_name and not model_name.startswith(("./", "/", "../")):
            suggestions.append(f"Pull from Hugging Face: gpux pull {model_name}")

        # General suggestions
        suggestions.extend(
            [
                "Use explicit path: gpux run ./my-model/",
                "Check model name spelling",
                "Run 'gpux pull <model-name>' to download from registry",
            ]
        )

        return suggestions

    @staticmethod
    def get_cache_directory() -> Path:
        """Get the cache directory path.

        Returns:
            Path to the cache directory
        """
        return Path.home() / ".gpux" / "models"

    @staticmethod
    def list_cached_models() -> dict[str, list[str]]:
        """List all cached models organized by registry.

        Returns:
            Dictionary mapping registry names to lists of model names
        """
        cache_dir = ModelDiscovery.get_cache_directory()
        cached_models: dict[str, list[str]] = {}

        if not cache_dir.exists():
            return cached_models

        for registry_dir in cache_dir.iterdir():
            if not registry_dir.is_dir():
                continue

            registry_name = registry_dir.name
            cached_models[registry_name] = []

            for model_dir in registry_dir.iterdir():
                if model_dir.is_dir():
                    cached_models[registry_name].append(model_dir.name)

        return cached_models
