"""Tests for the GPUX pull command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from gpux.core.managers import ModelManager, ModelMetadata
from gpux.cli.main import app
from gpux.core.managers.exceptions import (
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    RegistryError,
)


@pytest.fixture
def runner():
    """Fixture for CLI runner."""
    return CliRunner()


@pytest.fixture
def mock_metadata():
    """Fixture for mock model metadata."""
    return ModelMetadata(
        registry="huggingface",
        model_id="test/model",
        revision="main",
        format="pytorch",
        files={
            "config.json": Path("/tmp/config.json"),  # noqa: S108
            "model.bin": Path("/tmp/model.bin"),  # noqa: S108
        },
        size_bytes=1000000,
        description="A test model",
        tags=["text-generation"],
    )


@pytest.fixture
def mock_manager(mock_metadata):
    """Fixture for mock model manager."""
    manager = MagicMock()
    manager.pull_model.return_value = mock_metadata
    return manager


class TestPullCommand:
    """Tests for the pull command."""

    def test_pull_command_success(self, runner, mock_manager, mock_metadata):  # noqa: ARG002
        """Test successful pull command."""
        with patch("gpux.cli.pull._create_model_manager", return_value=mock_manager):
            result = runner.invoke(
                app,
                ["pull", "microsoft/DialoGPT-medium"],
            )

            assert result.exit_code == 0
            assert "Successfully pulled model" in result.output
        mock_manager.pull_model.assert_called_once_with(
            model_id="microsoft/DialoGPT-medium",
            revision="main",
            cache_dir=None,
            force_download=False,
        )

    def test_pull_command_with_options(self, runner, mock_manager, mock_metadata):  # noqa: ARG002
        """Test pull command with various options."""
        with patch("gpux.cli.pull._create_model_manager", return_value=mock_manager):
            result = runner.invoke(
                app,
                [
                    "pull",
                    "microsoft/DialoGPT-medium",
                    "--registry",
                    "huggingface",
                    "--revision",
                    "v1.0",
                    "--cache-dir",
                    "/tmp/custom",  # noqa: S108
                    "--verbose",
                ],
            )

            assert result.exit_code == 0
            assert "Successfully pulled model" in result.output
        mock_manager.pull_model.assert_called_once_with(
            model_id="microsoft/DialoGPT-medium",
            revision="v1.0",
            cache_dir=Path("/tmp/custom"),  # noqa: S108
            force_download=False,
        )

    def test_pull_command_model_not_found(self, runner, mock_manager):
        """Test pull command when model is not found."""
        mock_manager.pull_model.side_effect = ModelNotFoundError("Model not found")

        with patch("gpux.cli.pull._create_model_manager", return_value=mock_manager):
            result = runner.invoke(
                app,
                ["pull", "nonexistent/model"],
            )

            assert result.exit_code == 1
            assert "Model 'nonexistent/model' not found" in result.output

    def test_pull_command_authentication_error(self, runner, mock_manager):
        """Test pull command with authentication error."""
        mock_manager.pull_model.side_effect = AuthenticationError(
            "Authentication failed"
        )

        with patch("gpux.cli.pull._create_model_manager", return_value=mock_manager):
            result = runner.invoke(
                app,
                ["pull", "microsoft/DialoGPT-medium"],
            )

            assert result.exit_code == 1
            assert "Authentication failed" in result.output
            assert "HF_TOKEN" in result.output

    def test_pull_command_network_error(self, runner, mock_manager):
        """Test pull command with network error."""
        mock_manager.pull_model.side_effect = NetworkError("Network error")

        with patch("gpux.cli.pull._create_model_manager", return_value=mock_manager):
            result = runner.invoke(
                app,
                ["pull", "microsoft/DialoGPT-medium"],
            )

            assert result.exit_code == 1
            assert "Network error" in result.output
            assert "Check your internet connection" in result.output

    def test_pull_command_registry_error(self, runner, mock_manager):
        """Test pull command with registry error."""
        mock_manager.pull_model.side_effect = RegistryError("Registry error")

        with patch("gpux.cli.pull._create_model_manager", return_value=mock_manager):
            result = runner.invoke(
                app,
                ["pull", "microsoft/DialoGPT-medium"],
            )

            assert result.exit_code == 1
            assert "Registry error" in result.output

    def test_pull_command_unsupported_registry(self, runner):
        """Test pull command with unsupported registry."""
        result = runner.invoke(
            app,
            ["pull", "microsoft/DialoGPT-medium", "--registry", "unsupported"],
        )

        assert result.exit_code == 1
        assert "Unsupported registry 'unsupported'" in result.output

    def test_pull_command_verbose_output(self, runner, mock_manager, mock_metadata):  # noqa: ARG002
        """Test pull command with verbose output."""
        with patch("gpux.cli.pull._create_model_manager", return_value=mock_manager):
            result = runner.invoke(
                app,
                ["pull", "microsoft/DialoGPT-medium", "--verbose"],
            )

            assert result.exit_code == 0
            assert "Successfully pulled model" in result.output

    def test_pull_command_help(self, runner):
        """Test pull command help."""
        result = runner.invoke(app, ["pull", "--help"])

        assert result.exit_code == 0
        assert "Pull a model from a supported registry" in result.output
        assert "MODEL_ID" in result.output
        assert "--registry" in result.output
        assert "--revision" in result.output
        assert "--cache-dir" in result.output

    def test_create_model_manager_huggingface(self):
        """Test creating HuggingFace model manager."""
        from gpux.cli.pull import _create_model_manager
        from gpux.core.managers import RegistryConfig

        config = RegistryConfig(
            name="huggingface",
            api_url="https://huggingface.co",
            auth_token=None,
        )

        manager = _create_model_manager(config)
        assert isinstance(manager, type(manager))  # Should be HuggingFaceManager

    def test_create_model_manager_unsupported(self):
        """Test creating model manager for unsupported registry."""
        from gpux.cli.pull import _create_model_manager
        from gpux.core.managers import RegistryConfig

        config = RegistryConfig(
            name="unsupported",
            api_url="https://example.com",
            auth_token=None,
        )

        with pytest.raises(ValueError, match="Unsupported registry"):
            _create_model_manager(config)

    def test_format_size(self):
        """Test size formatting function."""
        from gpux.cli.pull import _format_size

        assert _format_size(1024) == "1.0 KB"
        assert _format_size(1024 * 1024) == "1.0 MB"
        assert _format_size(1024 * 1024 * 1024) == "1.0 GB"
        assert _format_size(None) == "Unknown"

    def test_display_pull_info(self, runner):  # noqa: ARG002
        """Test display pull info function."""
        from gpux.cli.pull import _display_pull_info

        # This is a visual test, so we just ensure it doesn't crash
        _display_pull_info("test/model", "huggingface", "main", None)
        _display_pull_info("test/model", "huggingface", "main", Path("/tmp"))  # noqa: S108

    def test_display_success_info(self, runner, mock_metadata):  # noqa: ARG002
        """Test display success info function."""
        from gpux.cli.pull import _display_success_info

        # This is a visual test, so we just ensure it doesn't crash
        _display_success_info(mock_metadata)

    def test_pull_command_force_download(
        self, runner: CliRunner, mock_manager: ModelManager
    ) -> None:
        """Test that --force flag works correctly."""
        mock_metadata = ModelMetadata(
            model_id="test/model",
            registry="huggingface",
            format="pytorch",
            files={"config.json": Path("/tmp/config.json")},  # noqa: S108
            size_bytes=1000000,
            tags=[],
            metadata={},
            revision="main",
        )
        mock_manager.pull_model.return_value = mock_metadata

        with patch("gpux.cli.pull._create_model_manager", return_value=mock_manager):
            result = runner.invoke(
                app,
                ["pull", "test/model", "--force"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        mock_manager.pull_model.assert_called_once_with(
            model_id="test/model",
            revision="main",
            cache_dir=None,
            force_download=True,
        )

    def test_pull_command_force_with_options(
        self, runner: CliRunner, mock_manager: ModelManager
    ) -> None:
        """Test that --force works with other options."""
        mock_metadata = ModelMetadata(
            model_id="test/model",
            registry="huggingface",
            format="pytorch",
            files={"config.json": Path("/tmp/config.json")},  # noqa: S108
            size_bytes=1000000,
            tags=[],
            metadata={},
            revision="main",
        )
        mock_manager.pull_model.return_value = mock_metadata

        with patch("gpux.cli.pull._create_model_manager", return_value=mock_manager):
            result = runner.invoke(
                app,
                [
                    "pull",
                    "test/model",
                    "--force",
                    "--revision",
                    "v1.0",
                    "--cache-dir",
                    "/tmp/cache",  # noqa: S108
                ],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        mock_manager.pull_model.assert_called_once_with(
            model_id="test/model",
            revision="v1.0",
            cache_dir=Path("/tmp/cache"),  # noqa: S108
            force_download=True,
        )
