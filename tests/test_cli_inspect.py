"""Tests for inspect CLI functionality."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from gpux.cli.inspect import (
    _display_config_info,
    _display_model_info,
    _display_runtime_info,
    _inspect_model_by_name,
    _inspect_model_file,
    _inspect_runtime,
    inspect_command,
)
from gpux.cli.main import app
from gpux.core.managers.exceptions import ModelNotFoundError
from gpux.core.models import InputSpec, OutputSpec


class TestInspectCLI:
    """Test cases for inspect CLI functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_inspect_command_creation(self) -> None:
        """Test that the inspect command is created correctly."""
        assert callable(inspect_command)

    def test_inspect_command_help(self) -> None:
        """Test inspect command help output."""
        result = self.runner.invoke(app, ["inspect", "--help"])
        assert result.exit_code == 0
        assert "Inspect models and runtime information" in result.output
        assert "--config" in result.output
        assert "--model" in result.output
        assert "--json" in result.output

    def test_inspect_model_file_success(self, simple_onnx_model: Path) -> None:
        """Test inspecting a model file directly."""
        with patch("gpux.cli.inspect.ModelInspector") as mock_inspector_class:
            mock_inspector = MagicMock()
            mock_inspector_class.return_value = mock_inspector

            mock_model_info = MagicMock()
            mock_model_info.to_dict.return_value = {
                "name": "test-model",
                "version": "1.0.0",
            }
            # Set string attributes for Rich table rendering
            mock_model_info.name = "test-model"
            mock_model_info.version = "1.0.0"
            mock_model_info.format = "onnx"
            mock_model_info.size_bytes = 1024 * 1024  # 1MB in bytes
            mock_model_info.path = str(simple_onnx_model)
            mock_model_info.inputs = []
            mock_model_info.outputs = []
            mock_model_info.metadata = {}
            mock_inspector.inspect.return_value = mock_model_info

            with patch("gpux.cli.inspect.console.print") as mock_print:
                _inspect_model_file(simple_onnx_model, json_output=False)
                mock_inspector.inspect.assert_called_once_with(simple_onnx_model)
                mock_print.assert_called()

    def test_inspect_model_file_json_output(self, simple_onnx_model: Path) -> None:
        """Test inspecting a model file with JSON output."""
        with patch("gpux.cli.inspect.ModelInspector") as mock_inspector_class:
            mock_inspector = MagicMock()
            mock_inspector_class.return_value = mock_inspector

            mock_model_info = MagicMock()
            mock_model_info.to_dict.return_value = {
                "name": "test-model",
                "version": "1.0.0",
            }
            # Set string attributes for Rich table rendering
            mock_model_info.name = "test-model"
            mock_model_info.version = "1.0.0"
            mock_model_info.format = "onnx"
            mock_model_info.size_bytes = 1024 * 1024  # 1MB in bytes
            mock_model_info.path = str(simple_onnx_model)
            mock_model_info.inputs = []
            mock_model_info.outputs = []
            mock_model_info.metadata = {}
            mock_inspector.inspect.return_value = mock_model_info

            with patch("gpux.cli.inspect.console.print") as mock_print:
                _inspect_model_file(simple_onnx_model, json_output=True)
                mock_inspector.inspect.assert_called_once_with(simple_onnx_model)
                mock_print.assert_called()

    def test_inspect_model_file_not_found(self, temp_dir: Path) -> None:
        """Test inspecting a non-existent model file."""
        nonexistent_model = temp_dir / "nonexistent.onnx"

        with pytest.raises(typer.Exit) as exc_info:
            _inspect_model_file(nonexistent_model, json_output=False)
        assert exc_info.value.exit_code == 1

    def test_inspect_model_by_name_success(self, temp_dir: Path) -> None:
        """Test inspecting a model by name."""
        with (
            patch(
                "gpux.cli.inspect.ModelDiscovery.find_model_config",
                return_value=temp_dir,
            ),
            patch("gpux.cli.inspect.GPUXConfigParser") as mock_parser_class,
        ):
            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser
            mock_config = MagicMock()
            mock_config.dict.return_value = {"name": "test-model"}
            mock_config.name = "test-model"
            mock_config.version = "1.0.0"
            mock_config.description = "Test model"

            # Mock the model object
            mock_model = MagicMock()
            mock_model.format = "onnx"
            mock_model.source = "model.onnx"
            mock_config.model = mock_model

            # Mock the runtime object
            mock_runtime = MagicMock()
            mock_gpu = MagicMock()
            mock_gpu.memory = "1GB"
            mock_gpu.backend = "auto"
            mock_runtime.gpu = mock_gpu
            mock_config.runtime = mock_runtime

            mock_parser.parse_file.return_value = mock_config
            # Mock the model file path and its existence
            mock_model_file = MagicMock()
            mock_model_file.exists.return_value = True
            mock_parser.get_model_path.return_value = mock_model_file

            with patch("gpux.cli.inspect.ModelInspector") as mock_inspector_class:
                mock_inspector = MagicMock()
                mock_inspector_class.return_value = mock_inspector
                mock_model_info = MagicMock()
                mock_model_info.to_dict.return_value = {"name": "test-model"}
                mock_model_info.name = "test-model"
                mock_model_info.version = "1.0.0"
                mock_model_info.format = "onnx"
                mock_model_info.size_bytes = 1024 * 1024  # 1MB in bytes
                mock_model_info.path = "model.onnx"
                # Mock input and output objects
                mock_input = MagicMock()
                mock_input.name = "input"
                mock_input.type = "float32"
                mock_input.shape = [1, 10]
                mock_input.required = True
                mock_input.description = "Input tensor"

                mock_output = MagicMock()
                mock_output.name = "output"
                mock_output.type = "float32"
                mock_output.shape = [1, 2]
                mock_output.required = True
                mock_output.description = "Output tensor"

                mock_model_info.inputs = [mock_input]
                mock_model_info.outputs = [mock_output]
                mock_model_info.metadata = {"author": "test"}
                mock_inspector.inspect.return_value = mock_model_info

                with patch("gpux.cli.inspect.console.print") as mock_print:
                    _inspect_model_by_name("test-model", "gpux.yml", json_output=False)

                # Verify the function was called correctly
                mock_parser.parse_file.assert_called_once_with(temp_dir / "gpux.yml")
                mock_parser.get_model_path.assert_called_once_with(temp_dir)
                mock_inspector.inspect.assert_called_once()
                mock_print.assert_called()

    def test_inspect_model_by_name_not_found(self) -> None:
        """Test inspecting a model by name when not found."""
        with (
            patch(
                "gpux.cli.inspect.ModelDiscovery.find_model_config",
                side_effect=ModelNotFoundError("test-model"),
            ),
            pytest.raises(ModelNotFoundError),
        ):
            _inspect_model_by_name("nonexistent-model", "gpux.yml", json_output=False)

    def test_inspect_runtime_json_output(self) -> None:
        """Test inspecting runtime with JSON output."""
        with patch("gpux.cli.inspect.ProviderManager") as mock_provider_manager_class:
            mock_provider_manager = MagicMock()
            mock_provider_manager_class.return_value = mock_provider_manager
            mock_provider_manager.get_available_providers.return_value = ["cpu", "cuda"]
            mock_provider_manager._provider_priority = [MagicMock(), MagicMock()]
            mock_provider_manager._provider_priority[0].value = "cpu"
            mock_provider_manager._provider_priority[1].value = "cuda"
            mock_provider_manager.get_provider_info.side_effect = [
                {"available": True, "platform": "test"},
                {"available": False, "platform": "test"},
            ]

            with patch("gpux.cli.inspect.console.print") as mock_print:
                _inspect_runtime(json_output=True)
                mock_print.assert_called()

    def test_inspect_runtime_normal_output(self) -> None:
        """Test inspecting runtime with normal output."""
        with patch("gpux.cli.inspect.ProviderManager") as mock_provider_manager_class:
            mock_provider_manager = MagicMock()
            mock_provider_manager_class.return_value = mock_provider_manager
            mock_provider_manager.get_available_providers.return_value = ["cpu"]
            mock_provider_manager._provider_priority = [MagicMock()]
            mock_provider_manager._provider_priority[0].value = "cpu"
            mock_provider_manager.get_provider_info.return_value = {
                "available": True,
                "platform": "test",
            }

            with patch("gpux.cli.inspect.console.print") as mock_print:
                _inspect_runtime(json_output=False)
                mock_print.assert_called()

    # Note: _find_model_config tests removed as function no longer exists
    # Model discovery is now tested in test_model_discovery.py

    def test_display_config_info(self) -> None:
        """Test _display_config_info function."""
        mock_config = MagicMock()
        mock_config.name = "test-model"
        mock_config.version = "1.0.0"
        mock_config.model.source = Path("model.onnx")
        mock_config.model.format = "onnx"
        mock_config.runtime.gpu.memory = "1GB"
        mock_config.runtime.gpu.backend = "auto"
        mock_config.runtime.batch_size = 1
        mock_config.runtime.timeout = 30

        with patch("gpux.cli.inspect.console.print") as mock_print:
            _display_config_info(mock_config)
            mock_print.assert_called()

    def test_display_model_info(self) -> None:
        """Test _display_model_info function."""
        mock_model_info = MagicMock()
        mock_model_info.name = "test-model"
        mock_model_info.version = "1.0.0"
        mock_model_info.format = "onnx"
        mock_model_info.size_bytes = 1024 * 1024 * 1.5  # 1.5MB in bytes
        mock_model_info.path = Path("model.onnx")
        mock_model_info.inputs = [
            InputSpec(
                name="input1",
                type="float32",
                shape=[1, 10],
                required=True,
                description="Test input",
            )
        ]
        mock_model_info.outputs = [
            OutputSpec(
                name="output1",
                type="float32",
                shape=[1, 2],
                labels=["class1", "class2"],
                description="Test output",
            )
        ]
        mock_model_info.metadata = {"author": "test", "license": "MIT"}

        with patch("gpux.cli.inspect.console.print") as mock_print:
            _display_model_info(mock_model_info)
            mock_print.assert_called()

    def test_display_model_info_empty(self) -> None:
        """Test _display_model_info function with empty inputs/outputs."""
        mock_model_info = MagicMock()
        mock_model_info.name = "test-model"
        mock_model_info.version = "1.0.0"
        mock_model_info.format = "onnx"
        mock_model_info.size_bytes = 1024 * 1024 * 1.5  # 1.5MB in bytes
        mock_model_info.path = Path("model.onnx")
        mock_model_info.inputs = []
        mock_model_info.outputs = []
        mock_model_info.metadata = {}

        with patch("gpux.cli.inspect.console.print") as mock_print:
            _display_model_info(mock_model_info)
            mock_print.assert_called()

    def test_display_runtime_info(self) -> None:
        """Test _display_runtime_info function."""
        mock_provider_manager = MagicMock()
        mock_provider_manager._provider_priority = [MagicMock(), MagicMock()]
        mock_provider_manager._provider_priority[0].value = "cpu"
        mock_provider_manager._provider_priority[1].value = "cuda"

        def provider_info_side_effect(provider):
            if provider.value == "cpu":
                return {
                    "available": True,
                    "platform": "test",
                    "description": "CPU provider",
                }
            return {
                "available": False,
                "platform": "test",
                "description": "CUDA provider",
            }

        mock_provider_manager.get_provider_info.side_effect = provider_info_side_effect

        with patch("gpux.cli.inspect.console.print") as mock_print:
            _display_runtime_info(mock_provider_manager)
            mock_print.assert_called()

    def test_inspect_command_model_file(self, simple_onnx_model: Path) -> None:
        """Test inspect command with model file."""
        with patch("gpux.cli.inspect._inspect_model_file") as mock_inspect:
            result = self.runner.invoke(
                app, ["inspect", "--model", str(simple_onnx_model)]
            )
            assert result.exit_code == 0
            mock_inspect.assert_called_once()

    def test_inspect_command_model_name(self) -> None:
        """Test inspect command with model name."""
        with patch("gpux.cli.inspect._inspect_model_by_name") as mock_inspect:
            result = self.runner.invoke(app, ["inspect", "test-model"])
            assert result.exit_code == 0
            mock_inspect.assert_called_once()

    def test_inspect_command_runtime(self) -> None:
        """Test inspect command for runtime information."""
        with patch("gpux.cli.inspect._inspect_runtime") as mock_inspect:
            result = self.runner.invoke(app, ["inspect"])
            assert result.exit_code == 0
            mock_inspect.assert_called_once()

    def test_inspect_command_verbose(self) -> None:
        """Test inspect command with verbose flag."""
        with (
            patch("gpux.cli.inspect._inspect_runtime"),
            patch("logging.getLogger") as mock_get_logger,
        ):
            mock_logger = mock_get_logger.return_value
            result = self.runner.invoke(app, ["inspect", "--verbose"])
            assert result.exit_code == 0
            mock_logger.setLevel.assert_called_once_with(logging.DEBUG)

    def test_inspect_command_exception_handling(self) -> None:
        """Test inspect command exception handling."""
        with patch(
            "gpux.cli.inspect._inspect_runtime", side_effect=ValueError("Test error")
        ):
            result = self.runner.invoke(app, ["inspect"])
            assert result.exit_code == 1
            assert "Inspect failed: Test error" in result.output

    def test_inspect_command_exception_handling_verbose(self) -> None:
        """Test inspect command exception handling with verbose flag."""
        with (
            patch(
                "gpux.cli.inspect._inspect_runtime",
                side_effect=RuntimeError("Test error"),
            ),
            patch("gpux.cli.inspect.console.print_exception") as mock_print_exception,
        ):
            result = self.runner.invoke(app, ["inspect", "--verbose"])
            assert result.exit_code == 1
            assert "Inspect failed: Test error" in result.output
            mock_print_exception.assert_called_once()

    def test_inspect_command_default_arguments(self) -> None:
        """Test inspect command with default arguments."""
        result = self.runner.invoke(app, ["inspect", "--help"])
        assert result.exit_code == 0
        # Check that default values are shown in help
        assert "Name of the model to inspect" in result.output
        assert "Configuration file name" in result.output
        assert "Direct path to model file" in result.output
        assert "Output in JSON format" in result.output
