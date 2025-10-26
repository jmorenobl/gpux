"""Tests for serve CLI functionality."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from gpux.cli.main import app
from gpux.core.managers.exceptions import ModelNotFoundError
from gpux.cli.serve import (
    _display_server_info,
    _start_server,
    serve_command,
)


class TestServeCLI:
    """Test cases for serve CLI functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_serve_command_creation(self) -> None:
        """Test that the serve command is created correctly."""
        assert callable(serve_command)

    def test_serve_command_help(self) -> None:
        """Test serve command help output."""
        result = self.runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Start HTTP server for model serving" in result.output
        assert "--port" in result.output
        assert "--host" in result.output
        assert "--config" in result.output
        assert "--provider" in result.output
        assert "--workers" in result.output

    # Note: _find_model_config tests removed as function no longer exists
    # Model discovery is now tested in test_model_discovery.py

    def test_display_server_info(self) -> None:
        """Test _display_server_info function."""
        mock_config = MagicMock()
        mock_config.version = "1.0.0"
        mock_config.inputs = [MagicMock(), MagicMock()]
        mock_config.outputs = [MagicMock()]

        with patch("gpux.cli.serve.console.print") as mock_print:
            _display_server_info(mock_config, "test-model", "localhost", 8080, 1)
            mock_print.assert_called()

    def test_start_server_success(self) -> None:
        """Test _start_server function with successful import."""
        mock_runtime = MagicMock()
        mock_config = MagicMock()
        mock_config.name = "test-model"
        mock_config.version = "1.0.0"

        # Mock FastAPI and uvicorn modules
        mock_fastapi = MagicMock()
        mock_app = MagicMock()
        mock_fastapi.FastAPI.return_value = mock_app

        mock_uvicorn = MagicMock()
        mock_numpy = MagicMock()
        mock_np_array = MagicMock()
        mock_numpy.array.return_value = mock_np_array

        with (
            patch("gpux.cli.serve.console.print") as mock_print,
            patch("builtins.__import__") as mock_import,
        ):

            def import_side_effect(name, *args, **kwargs):
                if name == "numpy":
                    return mock_numpy
                if name == "uvicorn":
                    return mock_uvicorn
                if name == "fastapi":
                    return mock_fastapi
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            _start_server(mock_runtime, mock_config, "localhost", 8080, 1)

            # Verify FastAPI app was created
            mock_fastapi.FastAPI.assert_called_once()

            # Verify uvicorn.run was called
            mock_uvicorn.run.assert_called_once()

            # Verify console output
            assert mock_print.call_count >= 2  # Server start message and URL

    def test_start_server_import_error(self) -> None:
        """Test _start_server function with import error."""
        mock_runtime = MagicMock()
        mock_config = MagicMock()

        with (
            patch("gpux.cli.serve.console.print") as mock_print,
            patch("builtins.__import__", side_effect=ImportError("Test error")),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                _start_server(mock_runtime, mock_config, "localhost", 8080, 1)
            assert exc_info.value.exit_code == 1
            mock_print.assert_called()

    def test_start_server_keyboard_interrupt(self) -> None:
        """Test _start_server function with keyboard interrupt."""
        mock_runtime = MagicMock()
        mock_config = MagicMock()
        mock_config.name = "test-model"
        mock_config.version = "1.0.0"

        # Mock FastAPI and uvicorn modules
        mock_fastapi = MagicMock()
        mock_app = MagicMock()
        mock_fastapi.FastAPI.return_value = mock_app

        mock_uvicorn = MagicMock()
        mock_uvicorn.run.side_effect = KeyboardInterrupt()

        mock_numpy = MagicMock()
        mock_np_array = MagicMock()
        mock_numpy.array.return_value = mock_np_array

        with (
            patch("gpux.cli.serve.console.print") as mock_print,
            patch("builtins.__import__") as mock_import,
        ):

            def import_side_effect(name, *args, **kwargs):
                if name == "numpy":
                    return mock_numpy
                if name == "uvicorn":
                    return mock_uvicorn
                if name == "fastapi":
                    return mock_fastapi
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            _start_server(mock_runtime, mock_config, "localhost", 8080, 1)

            # Verify cleanup was called
            mock_runtime.cleanup.assert_called_once()

            # Verify keyboard interrupt message was printed
            assert any(
                "Server stopped by user" in str(call)
                for call in mock_print.call_args_list
            )

    @patch("gpux.cli.serve.ModelDiscovery.find_model_config")
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
        mock_config.version = "1.0.0"
        mock_config.inputs = [MagicMock(), MagicMock()]
        mock_config.outputs = [MagicMock()]
        mock_config.runtime.dict.return_value = {}
        mock_parser.parse_file.return_value = mock_config
        mock_parser.get_model_path.return_value = sample_gpuxfile.parent / "model.onnx"

        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime

        # Create config file
        config_path = temp_dir / "gpux.yml"
        config_path.write_text(sample_gpuxfile.read_text())

        # Create model file
        model_file = sample_gpuxfile.parent / "model.onnx"
        model_file.touch()

        result = self.runner.invoke(app, ["serve", "test-model"])
        assert result.exit_code == 0
        mock_start_server.assert_called_once()

    def test_serve_command_model_not_found(self) -> None:
        """Test serve command when model is not found."""
        with patch(
            "gpux.cli.serve.ModelDiscovery.find_model_config",
            side_effect=ModelNotFoundError("nonexistent-model"),
        ):
            result = self.runner.invoke(app, ["serve", "nonexistent-model"])
            assert result.exit_code == 1
            assert "Model 'nonexistent-model' not found" in result.output

    def test_serve_command_model_file_not_found(self, temp_dir: Path) -> None:
        """Test serve command when model file is not found."""
        with (
            patch(
                "gpux.cli.serve.ModelDiscovery.find_model_config", return_value=temp_dir
            ),
            patch("gpux.cli.serve.GPUXConfigParser") as mock_parser_class,
        ):
            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser
            mock_config = MagicMock()
            mock_config.version = "1.0.0"
            mock_config.inputs = [MagicMock(), MagicMock()]
            mock_config.outputs = [MagicMock()]
            mock_config.runtime.dict.return_value = {}
            mock_parser.parse_file.return_value = mock_config
            mock_parser.get_model_path.return_value = None

            result = self.runner.invoke(app, ["serve", "test-model"])
            assert result.exit_code == 1
            assert "Model file not found" in result.output

    def test_serve_command_verbose(self) -> None:
        """Test serve command with verbose flag."""
        with (
            patch("gpux.cli.serve.ModelDiscovery.find_model_config", return_value=None),
            patch("logging.getLogger") as mock_get_logger,
        ):
            mock_logger = mock_get_logger.return_value
            result = self.runner.invoke(app, ["serve", "test-model", "--verbose"])
            assert result.exit_code == 1
            mock_logger.setLevel.assert_called_once_with(logging.DEBUG)

    def test_serve_command_with_custom_options(
        self, temp_dir: Path, sample_gpuxfile: Path
    ) -> None:
        """Test serve command with custom host, port, and workers."""
        with (
            patch(
                "gpux.cli.serve.ModelDiscovery.find_model_config", return_value=temp_dir
            ),
            patch("gpux.cli.serve.GPUXConfigParser") as mock_parser_class,
        ):
            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser
            mock_config = MagicMock()
            mock_config.version = "1.0.0"
            mock_config.inputs = [MagicMock(), MagicMock()]
            mock_config.outputs = [MagicMock()]
            mock_config.runtime.dict.return_value = {}
            mock_parser.parse_file.return_value = mock_config
            mock_parser.get_model_path.return_value = (
                sample_gpuxfile.parent / "model.onnx"
            )

            # Create model file
            model_file = sample_gpuxfile.parent / "model.onnx"
            model_file.touch()

            with (
                patch("gpux.cli.serve.GPUXRuntime") as mock_runtime_class,
                patch("gpux.cli.serve._start_server") as mock_start_server,
            ):
                mock_runtime = MagicMock()
                mock_runtime_class.return_value = mock_runtime
                result = self.runner.invoke(
                    app,
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
            "gpux.cli.serve.ModelDiscovery.find_model_config",
            side_effect=ValueError("Test error"),
        ):
            result = self.runner.invoke(app, ["serve", "test-model"])
            assert result.exit_code == 1
            assert "Serve failed: Test error" in result.output

    def test_serve_command_exception_handling_verbose(self) -> None:
        """Test serve command exception handling with verbose flag."""
        with (
            patch(
                "gpux.cli.serve.ModelDiscovery.find_model_config",
                side_effect=RuntimeError("Test error"),
            ),
            patch("gpux.cli.serve.console.print_exception") as mock_print_exception,
        ):
            result = self.runner.invoke(app, ["serve", "test-model", "--verbose"])
            assert result.exit_code == 1
            assert "Serve failed: Test error" in result.output
            mock_print_exception.assert_called_once()

    def test_serve_command_import_error(
        self, temp_dir: Path, sample_gpuxfile: Path
    ) -> None:
        """Test serve command with import error."""
        with (
            patch(
                "gpux.cli.serve.ModelDiscovery.find_model_config", return_value=temp_dir
            ),
            patch("gpux.cli.serve.GPUXConfigParser") as mock_parser_class,
        ):
            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser
            mock_config = MagicMock()
            mock_config.version = "1.0.0"
            mock_config.inputs = [MagicMock(), MagicMock()]
            mock_config.outputs = [MagicMock()]
            mock_config.runtime.dict.return_value = {}
            mock_parser.parse_file.return_value = mock_config
            mock_parser.get_model_path.return_value = (
                sample_gpuxfile.parent / "model.onnx"
            )

            # Create model file
            model_file = sample_gpuxfile.parent / "model.onnx"
            model_file.touch()

            with (
                patch("gpux.cli.serve.GPUXRuntime") as mock_runtime_class,
                patch(
                    "gpux.cli.serve._start_server",
                    side_effect=ImportError("Test import error"),
                ),
            ):
                mock_runtime = MagicMock()
                mock_runtime_class.return_value = mock_runtime
                result = self.runner.invoke(app, ["serve", "test-model"])
                assert result.exit_code == 1
                assert "Serve failed: Test import error" in result.output

    def test_serve_command_default_arguments(self) -> None:
        """Test serve command with default arguments."""
        result = self.runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        # Check that default values are shown in help
        assert "Name of the model to serve" in result.output
        assert "Port to serve on" in result.output
        assert "Host to serve on" in result.output
        assert "Configuration file name" in result.output
        assert "Number of worker processes" in result.output


class TestFastAPIEndpoints:
    """Test cases for FastAPI endpoints functionality."""

    def test_health_endpoint(self) -> None:
        """Test health check endpoint functionality."""
        mock_runtime = MagicMock()
        mock_config = MagicMock()
        mock_config.name = "test-model"
        mock_config.version = "1.0.0"

        # Mock FastAPI app and decorators
        mock_app = MagicMock()
        mock_fastapi = MagicMock()
        mock_fastapi.FastAPI.return_value = mock_app

        # Track decorator calls
        decorator_calls = []

        def mock_get_decorator(path: str):
            def decorator(func):
                decorator_calls.append(("GET", path, func.__name__))
                return func

            return decorator

        mock_app.get = mock_get_decorator

        with (
            patch("gpux.cli.serve.console.print"),
            patch("builtins.__import__") as mock_import,
        ):

            def import_side_effect(name, *args, **kwargs):
                if name in {"numpy", "uvicorn"}:
                    return MagicMock()
                if name == "fastapi":
                    return mock_fastapi
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            _start_server(mock_runtime, mock_config, "localhost", 8080, 1)

            # Verify health endpoint was registered
            assert ("GET", "/health", "health_check") in decorator_calls

    def test_info_endpoint(self) -> None:
        """Test model info endpoint functionality."""
        mock_runtime = MagicMock()
        mock_model_info = MagicMock()
        mock_model_info.to_dict.return_value = {
            "name": "test-model",
            "version": "1.0.0",
        }
        mock_runtime.get_model_info.return_value = mock_model_info

        mock_config = MagicMock()
        mock_config.name = "test-model"
        mock_config.version = "1.0.0"

        # Mock FastAPI app
        mock_app = MagicMock()
        mock_fastapi = MagicMock()
        mock_fastapi.FastAPI.return_value = mock_app

        # Track decorator calls
        decorator_calls = []

        def mock_get_decorator(path: str):
            def decorator(func):
                decorator_calls.append(("GET", path, func.__name__))
                return func

            return decorator

        mock_app.get = mock_get_decorator

        with (
            patch("gpux.cli.serve.console.print"),
            patch("builtins.__import__") as mock_import,
        ):

            def import_side_effect(name, *args, **kwargs):
                if name in {"numpy", "uvicorn"}:
                    return MagicMock()
                if name == "fastapi":
                    return mock_fastapi
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            _start_server(mock_runtime, mock_config, "localhost", 8080, 1)

            # Verify info endpoint was registered
            assert ("GET", "/info", "model_info") in decorator_calls

    def test_metrics_endpoint(self) -> None:
        """Test metrics endpoint functionality."""
        mock_runtime = MagicMock()
        mock_runtime.get_provider_info.return_value = {
            "name": "CPUExecutionProvider",
            "available": True,
        }
        mock_runtime.get_available_providers.return_value = ["CPUExecutionProvider"]

        mock_config = MagicMock()
        mock_config.name = "test-model"
        mock_config.version = "1.0.0"

        # Mock FastAPI app
        mock_app = MagicMock()
        mock_fastapi = MagicMock()
        mock_fastapi.FastAPI.return_value = mock_app

        # Track decorator calls
        decorator_calls = []

        def mock_get_decorator(path: str):
            def decorator(func):
                decorator_calls.append(("GET", path, func.__name__))
                return func

            return decorator

        mock_app.get = mock_get_decorator

        with (
            patch("gpux.cli.serve.console.print"),
            patch("builtins.__import__") as mock_import,
        ):

            def import_side_effect(name, *args, **kwargs):
                if name in {"numpy", "uvicorn"}:
                    return MagicMock()
                if name == "fastapi":
                    return mock_fastapi
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            _start_server(mock_runtime, mock_config, "localhost", 8080, 1)

            # Verify metrics endpoint was registered
            assert ("GET", "/metrics", "metrics") in decorator_calls

    def test_predict_endpoint(self) -> None:
        """Test prediction endpoint functionality."""
        mock_runtime = MagicMock()
        mock_runtime.infer.return_value = {"output": [[1.0, 2.0]]}

        mock_config = MagicMock()
        mock_config.name = "test-model"
        mock_config.version = "1.0.0"

        # Mock FastAPI app
        mock_app = MagicMock()
        mock_fastapi = MagicMock()
        mock_fastapi.FastAPI.return_value = mock_app

        # Track decorator calls
        decorator_calls = []

        def mock_post_decorator(path: str):
            def decorator(func):
                decorator_calls.append(("POST", path, func.__name__))
                return func

            return decorator

        mock_app.post = mock_post_decorator

        with (
            patch("gpux.cli.serve.console.print"),
            patch("builtins.__import__") as mock_import,
        ):

            def import_side_effect(name, *args, **kwargs):
                if name in {"numpy", "uvicorn"}:
                    return MagicMock()
                if name == "fastapi":
                    return mock_fastapi
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            _start_server(mock_runtime, mock_config, "localhost", 8080, 1)

            # Verify predict endpoint was registered
            assert ("POST", "/predict", "predict") in decorator_calls

    def test_server_configuration_with_workers(self) -> None:
        """Test server configuration with multiple workers."""
        mock_runtime = MagicMock()
        mock_config = MagicMock()
        mock_config.name = "test-model"
        mock_config.version = "1.0.0"

        mock_uvicorn = MagicMock()
        mock_fastapi = MagicMock()
        mock_fastapi.FastAPI.return_value = MagicMock()

        with (
            patch("gpux.cli.serve.console.print"),
            patch("builtins.__import__") as mock_import,
        ):

            def import_side_effect(name, *args, **kwargs):
                if name == "numpy":
                    return MagicMock()
                if name == "uvicorn":
                    return mock_uvicorn
                if name == "fastapi":
                    return mock_fastapi
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            _start_server(mock_runtime, mock_config, "localhost", 8080, 4)

            # Verify uvicorn.run was called with correct parameters
            mock_uvicorn.run.assert_called_once()
            call_args = mock_uvicorn.run.call_args
            assert call_args[1]["host"] == "localhost"
            assert call_args[1]["port"] == 8080
            assert call_args[1]["workers"] == 4

    def test_server_configuration_single_worker(self) -> None:
        """Test server configuration with single worker (None workers)."""
        mock_runtime = MagicMock()
        mock_config = MagicMock()
        mock_config.name = "test-model"
        mock_config.version = "1.0.0"

        mock_uvicorn = MagicMock()
        mock_fastapi = MagicMock()
        mock_fastapi.FastAPI.return_value = MagicMock()

        with (
            patch("gpux.cli.serve.console.print"),
            patch("builtins.__import__") as mock_import,
        ):

            def import_side_effect(name, *args, **kwargs):
                if name == "numpy":
                    return MagicMock()
                if name == "uvicorn":
                    return mock_uvicorn
                if name == "fastapi":
                    return mock_fastapi
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            _start_server(mock_runtime, mock_config, "localhost", 8080, 1)

            # Verify uvicorn.run was called with None workers
            mock_uvicorn.run.assert_called_once()
            call_args = mock_uvicorn.run.call_args
            assert call_args[1]["workers"] is None

    def test_server_log_level_debug(self) -> None:
        """Test server log level configuration for debug mode."""
        mock_runtime = MagicMock()
        mock_config = MagicMock()
        mock_config.name = "test-model"
        mock_config.version = "1.0.0"

        mock_uvicorn = MagicMock()
        mock_fastapi = MagicMock()
        mock_fastapi.FastAPI.return_value = MagicMock()

        # Mock debug logging
        mock_logger = MagicMock()
        mock_logger.isEnabledFor.return_value = True

        with (
            patch("gpux.cli.serve.console.print"),
            patch("builtins.__import__") as mock_import,
        ):

            def import_side_effect(name, *args, **kwargs):
                if name == "numpy":
                    return MagicMock()
                if name == "uvicorn":
                    return mock_uvicorn
                if name == "fastapi":
                    return mock_fastapi
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            with patch("gpux.cli.serve.logging.getLogger", return_value=mock_logger):
                _start_server(mock_runtime, mock_config, "localhost", 8080, 1)

                # Verify uvicorn.run was called with debug log level
                mock_uvicorn.run.assert_called_once()
                call_args = mock_uvicorn.run.call_args
                assert call_args[1]["log_level"] == "debug"

    def test_server_log_level_info(self) -> None:
        """Test server log level configuration for info mode."""
        mock_runtime = MagicMock()
        mock_config = MagicMock()
        mock_config.name = "test-model"
        mock_config.version = "1.0.0"

        mock_uvicorn = MagicMock()
        mock_fastapi = MagicMock()
        mock_fastapi.FastAPI.return_value = MagicMock()

        # Mock info logging
        mock_logger = MagicMock()
        mock_logger.isEnabledFor.return_value = False

        with (
            patch("gpux.cli.serve.console.print"),
            patch("builtins.__import__") as mock_import,
        ):

            def import_side_effect(name, *args, **kwargs):
                if name == "numpy":
                    return MagicMock()
                if name == "uvicorn":
                    return mock_uvicorn
                if name == "fastapi":
                    return mock_fastapi
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            with patch("gpux.cli.serve.logging.getLogger", return_value=mock_logger):
                _start_server(mock_runtime, mock_config, "localhost", 8080, 1)

                # Verify uvicorn.run was called with info log level
                mock_uvicorn.run.assert_called_once()
                call_args = mock_uvicorn.run.call_args
                assert call_args[1]["log_level"] == "info"


class TestServeCLIErrorHandling:
    """Test cases for serve CLI error handling and edge cases."""

    # Note: _find_model_config tests removed as function no longer exists
    # Model discovery is now tested in test_model_discovery.py
