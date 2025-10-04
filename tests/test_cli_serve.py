"""Tests for serve CLI functionality."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from gpux.cli.serve import (
    _display_server_info,
    _find_model_config,
    _start_server,
    serve_app,
)


class TestServeCLI:
    """Test cases for serve CLI functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_serve_app_creation(self) -> None:
        """Test that the serve app is created correctly."""
        assert isinstance(serve_app, typer.Typer)
        assert serve_app.info.name == "serve"
        assert "Start HTTP server for model serving" in serve_app.info.help

    def test_serve_command_help(self) -> None:
        """Test serve command help output."""
        result = self.runner.invoke(serve_app, ["--help"])
        assert result.exit_code == 0
        assert "Start HTTP server for model serving" in result.output
        assert "--port" in result.output
        assert "--host" in result.output
        assert "--config" in result.output
        assert "--provider" in result.output
        assert "--workers" in result.output

    def test_find_model_config_current_directory(
        self, temp_dir: Path, sample_gpuxfile: Path
    ) -> None:
        """Test finding model config in current directory."""
        config_path = temp_dir / "gpux.yml"
        config_path.write_text(sample_gpuxfile.read_text())

        with patch("gpux.cli.serve.Path") as mock_path:
            mock_path.return_value = temp_dir
            mock_path.return_value.exists.return_value = True
            result = _find_model_config("test-model", "gpux.yml")
            assert result == temp_dir

    def test_find_model_config_model_directory(
        self, temp_dir: Path, sample_gpuxfile: Path
    ) -> None:
        """Test finding model config in model directory."""
        model_dir = temp_dir / "test-model"
        model_dir.mkdir()
        config_path = model_dir / "gpux.yml"
        config_path.write_text(sample_gpuxfile.read_text())

        with patch("gpux.cli.serve.Path") as mock_path:
            # Mock Path() to return temp_dir for current directory check
            mock_path.return_value = temp_dir
            mock_path.return_value.exists.return_value = False

            # Mock Path(model_name) to return model_dir
            def path_side_effect(path_str):
                if path_str == "test-model":
                    mock_model_dir = MagicMock()
                    mock_model_dir.is_dir.return_value = True
                    mock_model_dir.__truediv__ = lambda self, other: model_dir / other
                    return mock_model_dir
                return temp_dir

            mock_path.side_effect = path_side_effect

            result = _find_model_config("test-model", "gpux.yml")
            assert result is not None

    def test_find_model_config_gpux_directory(self, temp_dir: Path) -> None:
        """Test finding model config in .gpux directory."""
        gpux_dir = temp_dir / ".gpux"
        gpux_dir.mkdir()

        model_info_file = gpux_dir / "model_info.json"
        model_info = {"name": "test-model", "version": "1.0.0"}
        model_info_file.write_text(json.dumps(model_info))

        with patch("gpux.cli.serve.Path") as mock_path:
            # Mock current directory check
            mock_path.return_value = temp_dir
            mock_path.return_value.exists.return_value = False

            # Mock model directory check
            def path_side_effect(path_str):
                if path_str == "test-model":
                    mock_model_dir = MagicMock()
                    mock_model_dir.is_dir.return_value = False
                    return mock_model_dir
                return temp_dir

            mock_path.side_effect = path_side_effect

            # Mock .gpux directory
            mock_gpux_dir = MagicMock()
            mock_gpux_dir.exists.return_value = True
            mock_gpux_dir.glob.return_value = [model_info_file]

            with patch("gpux.cli.serve.Path", return_value=mock_gpux_dir):
                result = _find_model_config("test-model", "gpux.yml")
                assert result is not None

    def test_find_model_config_not_found(self) -> None:
        """Test finding model config when not found."""
        with patch("gpux.cli.serve.Path") as mock_path:
            mock_path.return_value = temp_dir
            mock_path.return_value.exists.return_value = False

            def path_side_effect(path_str):
                if path_str == "nonexistent-model":
                    mock_model_dir = MagicMock()
                    mock_model_dir.is_dir.return_value = False
                    return mock_model_dir
                return temp_dir

            mock_path.side_effect = path_side_effect

            result = _find_model_config("nonexistent-model", "gpux.yml")
            assert result is None

    def test_display_server_info(self) -> None:
        """Test _display_server_info function."""
        mock_config = MagicMock()
        mock_config.version = "1.0.0"
        mock_config.inputs = [MagicMock(), MagicMock()]
        mock_config.outputs = [MagicMock()]

        with patch("gpux.cli.serve.console.print") as mock_print:
            _display_server_info(mock_config, "test-model", "0.0.0.0", 8080, 1)
            mock_print.assert_called()

    def test_start_server_success(self) -> None:
        """Test _start_server function with successful import."""
        mock_runtime = MagicMock()
        mock_config = MagicMock()
        mock_config.name = "test-model"
        mock_config.version = "1.0.0"

        with patch("gpux.cli.serve.importlib.import_module") as mock_import:
            # Mock successful imports
            mock_import.side_effect = [
                MagicMock(),  # numpy
                MagicMock(),  # uvicorn
                MagicMock(),  # fastapi
            ]

            with patch("gpux.cli.serve.uvicorn.run") as mock_uvicorn_run:
                with patch("gpux.cli.serve.console.print") as mock_print:
                    _start_server(mock_runtime, mock_config, "0.0.0.0", 8080, 1)
                    mock_uvicorn_run.assert_called_once()
                    mock_print.assert_called()

    def test_start_server_import_error(self) -> None:
        """Test _start_server function with import error."""
        mock_runtime = MagicMock()
        mock_config = MagicMock()

        with patch(
            "gpux.cli.serve.importlib.import_module",
            side_effect=ImportError("Test error"),
        ):
            with patch("gpux.cli.serve.console.print") as mock_print:
                with pytest.raises(SystemExit) as exc_info:
                    _start_server(mock_runtime, mock_config, "0.0.0.0", 8080, 1)
                assert exc_info.value.code == 1
                mock_print.assert_called()

    def test_start_server_keyboard_interrupt(self) -> None:
        """Test _start_server function with keyboard interrupt."""
        mock_runtime = MagicMock()
        mock_config = MagicMock()
        mock_config.name = "test-model"
        mock_config.version = "1.0.0"

        with patch("gpux.cli.serve.importlib.import_module") as mock_import:
            # Mock successful imports
            mock_import.side_effect = [
                MagicMock(),  # numpy
                MagicMock(),  # uvicorn
                MagicMock(),  # fastapi
            ]

            with patch("gpux.cli.serve.uvicorn.run", side_effect=KeyboardInterrupt):
                with patch("gpux.cli.serve.console.print") as mock_print:
                    _start_server(mock_runtime, mock_config, "0.0.0.0", 8080, 1)
                    mock_print.assert_called()

    @patch("gpux.cli.serve._find_model_config")
    @patch("gpux.cli.serve.GPUXConfigParser")
    @patch("gpux.cli.serve.GPUXRuntime")
    @patch("gpux.cli.serve._start_server")
    def test_serve_command_success(
        self,
        mock_start_server,
        mock_runtime_class,
        mock_parser_class,
        mock_find_config,
        temp_dir: Path,
        sample_gpuxfile: Path,
    ) -> None:
        """Test successful serve command execution."""
        # Setup mocks
        mock_find_config.return_value = temp_dir

        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_config = MagicMock()
        mock_config.runtime.dict.return_value = {}
        mock_parser.parse_file.return_value = mock_config
        mock_parser.get_model_path.return_value = sample_gpuxfile.parent / "model.onnx"

        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime

        # Create config file
        config_path = temp_dir / "gpux.yml"
        config_path.write_text(sample_gpuxfile.read_text())

        result = self.runner.invoke(serve_app, ["serve", "test-model"])
        assert result.exit_code == 0
        mock_start_server.assert_called_once()

    def test_serve_command_model_not_found(self) -> None:
        """Test serve command when model is not found."""
        with patch("gpux.cli.serve._find_model_config", return_value=None):
            result = self.runner.invoke(serve_app, ["serve", "nonexistent-model"])
            assert result.exit_code == 1
            assert "Model 'nonexistent-model' not found" in result.output

    def test_serve_command_model_file_not_found(
        self, temp_dir: Path, sample_gpuxfile: Path
    ) -> None:
        """Test serve command when model file is not found."""
        with patch("gpux.cli.serve._find_model_config", return_value=temp_dir):
            with patch("gpux.cli.serve.GPUXConfigParser") as mock_parser_class:
                mock_parser = MagicMock()
                mock_parser_class.return_value = mock_parser
                mock_config = MagicMock()
                mock_config.runtime.dict.return_value = {}
                mock_parser.parse_file.return_value = mock_config
                mock_parser.get_model_path.return_value = None

                result = self.runner.invoke(serve_app, ["serve", "test-model"])
                assert result.exit_code == 1
                assert "Model file not found" in result.output

    def test_serve_command_verbose(self) -> None:
        """Test serve command with verbose flag."""
        with patch("gpux.cli.serve._find_model_config", return_value=None):
            with patch("logging.getLogger") as mock_get_logger:
                mock_logger = mock_get_logger.return_value
                result = self.runner.invoke(
                    serve_app, ["serve", "test-model", "--verbose"]
                )
                assert result.exit_code == 1
                mock_logger.setLevel.assert_called_once_with(logging.DEBUG)

    def test_serve_command_with_custom_options(
        self, temp_dir: Path, sample_gpuxfile: Path
    ) -> None:
        """Test serve command with custom host, port, and workers."""
        with patch("gpux.cli.serve._find_model_config", return_value=temp_dir):
            with patch("gpux.cli.serve.GPUXConfigParser") as mock_parser_class:
                mock_parser = MagicMock()
                mock_parser_class.return_value = mock_parser
                mock_config = MagicMock()
                mock_config.runtime.dict.return_value = {}
                mock_parser.parse_file.return_value = mock_config
                mock_parser.get_model_path.return_value = (
                    sample_gpuxfile.parent / "model.onnx"
                )

                with patch("gpux.cli.serve.GPUXRuntime") as mock_runtime_class:
                    mock_runtime = MagicMock()
                    mock_runtime_class.return_value = mock_runtime

                    with patch("gpux.cli.serve._start_server") as mock_start_server:
                        result = self.runner.invoke(
                            serve_app,
                            [
                                "serve",
                                "test-model",
                                "--host",
                                "127.0.0.1",
                                "--port",
                                "9000",
                                "--workers",
                                "4",
                            ],
                        )
                        assert result.exit_code == 0
                        mock_start_server.assert_called_once()

    def test_serve_command_exception_handling(self) -> None:
        """Test serve command exception handling."""
        with patch(
            "gpux.cli.serve._find_model_config", side_effect=ValueError("Test error")
        ):
            result = self.runner.invoke(serve_app, ["serve", "test-model"])
            assert result.exit_code == 1
            assert "Serve failed: Test error" in result.output

    def test_serve_command_exception_handling_verbose(self) -> None:
        """Test serve command exception handling with verbose flag."""
        with patch(
            "gpux.cli.serve._find_model_config", side_effect=RuntimeError("Test error")
        ):
            with patch(
                "gpux.cli.serve.console.print_exception"
            ) as mock_print_exception:
                result = self.runner.invoke(
                    serve_app, ["serve", "test-model", "--verbose"]
                )
                assert result.exit_code == 1
                assert "Serve failed: Test error" in result.output
                mock_print_exception.assert_called_once()

    def test_serve_command_import_error(
        self, temp_dir: Path, sample_gpuxfile: Path
    ) -> None:
        """Test serve command with import error."""
        with patch("gpux.cli.serve._find_model_config", return_value=temp_dir):
            with patch("gpux.cli.serve.GPUXConfigParser") as mock_parser_class:
                mock_parser = MagicMock()
                mock_parser_class.return_value = mock_parser
                mock_config = MagicMock()
                mock_config.runtime.dict.return_value = {}
                mock_parser.parse_file.return_value = mock_config
                mock_parser.get_model_path.return_value = (
                    sample_gpuxfile.parent / "model.onnx"
                )

                with patch("gpux.cli.serve.GPUXRuntime") as mock_runtime_class:
                    mock_runtime = MagicMock()
                    mock_runtime_class.return_value = mock_runtime

                    with patch(
                        "gpux.cli.serve._start_server",
                        side_effect=ImportError("Test import error"),
                    ):
                        result = self.runner.invoke(serve_app, ["serve", "test-model"])
                        assert result.exit_code == 1
                        assert "Serve failed: Test import error" in result.output

    def test_serve_command_default_arguments(self) -> None:
        """Test serve command with default arguments."""
        result = self.runner.invoke(serve_app, ["serve", "--help"])
        assert result.exit_code == 0
        # Check that default values are shown in help
        assert "Name of the model to serve" in result.output
        assert "Port to serve on" in result.output
        assert "Host to serve on" in result.output
        assert "Configuration file name" in result.output
        assert "Number of worker processes" in result.output
