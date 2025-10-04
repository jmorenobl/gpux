"""Tests for build CLI functionality."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

from gpux.cli.build import _display_build_results, build_app
from gpux.core.models import InputSpec, OutputSpec


class TestBuildCLI:
    """Test cases for build CLI functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_build_app_creation(self) -> None:
        """Test that the build app is created correctly."""
        assert isinstance(build_app, typer.Typer)
        assert build_app.info.name == "build"
        assert "Build and optimize models" in build_app.info.help

    def test_build_command_help(self) -> None:
        """Test build command help output."""
        result = self.runner.invoke(build_app, ["--help"])
        assert result.exit_code == 0
        assert "Build and optimize models for GPU inference" in result.output
        assert "--config" in result.output
        assert "--optimize" in result.output
        assert "--provider" in result.output

    def test_build_command_config_file_not_found(self, temp_dir: Path) -> None:
        """Test build command when config file doesn't exist."""
        result = self.runner.invoke(
            build_app, ["build", str(temp_dir), "--config", "nonexistent.yml"]
        )
        assert result.exit_code == 1
        assert "Configuration file not found" in result.output

    @patch("gpux.cli.build.GPUXConfigParser")
    @patch("gpux.cli.build.ModelInspector")
    @patch("gpux.cli.build.ProviderManager")
    def test_build_command_success(
        self,
        mock_provider_manager_class,
        mock_inspector_class,
        mock_parser_class,
        temp_dir: Path,
        sample_gpuxfile: Path,
    ) -> None:
        """Test successful build command execution."""
        # Setup mocks
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.validate_model_path.return_value = True
        mock_parser.get_model_path.return_value = sample_gpuxfile.parent / "model.onnx"

        mock_inspector = MagicMock()
        mock_inspector_class.return_value = mock_inspector
        mock_model_info = MagicMock()
        mock_model_info.name = "test-model"
        mock_model_info.version = "1.0.0"
        mock_model_info.format = "onnx"
        mock_model_info.size_bytes = 1024 * 1024  # 1MB
        mock_model_info.inputs = []
        mock_model_info.outputs = []
        mock_inspector.inspect.return_value = mock_model_info

        mock_provider_manager = MagicMock()
        mock_provider_manager_class.return_value = mock_provider_manager
        mock_provider = MagicMock()
        mock_provider.value = "cpu"
        mock_provider_manager.get_best_provider.return_value = mock_provider
        mock_provider_info = {
            "platform": "test",
            "available": True,
            "description": "Test",
        }
        mock_provider_manager.get_provider_info.return_value = mock_provider_info

        # Create config file
        config_path = temp_dir / "gpux.yml"
        config_path.write_text(sample_gpuxfile.read_text())

        result = self.runner.invoke(build_app, ["build", str(temp_dir)])
        assert result.exit_code == 0
        assert "Build completed successfully" in result.output

    @patch("gpux.cli.build.GPUXConfigParser")
    def test_build_command_model_validation_fails(
        self, mock_parser_class, temp_dir: Path, sample_gpuxfile: Path
    ) -> None:
        """Test build command when model validation fails."""
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.validate_model_path.return_value = False

        config_path = temp_dir / "gpux.yml"
        config_path.write_text(sample_gpuxfile.read_text())

        result = self.runner.invoke(build_app, ["build", str(temp_dir)])
        assert result.exit_code == 1
        assert "Model file not found" in result.output

    @patch("gpux.cli.build.GPUXConfigParser")
    def test_build_command_model_path_resolution_fails(
        self, mock_parser_class, temp_dir: Path, sample_gpuxfile: Path
    ) -> None:
        """Test build command when model path resolution fails."""
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.validate_model_path.return_value = True
        mock_parser.get_model_path.return_value = None

        config_path = temp_dir / "gpux.yml"
        config_path.write_text(sample_gpuxfile.read_text())

        result = self.runner.invoke(build_app, ["build", str(temp_dir)])
        assert result.exit_code == 1
        assert "Could not resolve model path" in result.output

    @patch("gpux.cli.build.GPUXConfigParser")
    @patch("gpux.cli.build.ModelInspector")
    @patch("gpux.cli.build.ProviderManager")
    def test_build_command_with_provider(
        self,
        mock_provider_manager_class,
        mock_inspector_class,
        mock_parser_class,
        temp_dir: Path,
        sample_gpuxfile: Path,
    ) -> None:
        """Test build command with specific provider."""
        # Setup mocks
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.validate_model_path.return_value = True
        mock_parser.get_model_path.return_value = sample_gpuxfile.parent / "model.onnx"

        mock_inspector = MagicMock()
        mock_inspector_class.return_value = mock_inspector
        mock_model_info = MagicMock()
        mock_model_info.name = "test-model"
        mock_model_info.version = "1.0.0"
        mock_model_info.format = "onnx"
        mock_model_info.size_bytes = 1024 * 1024
        mock_model_info.inputs = []
        mock_model_info.outputs = []
        mock_inspector.inspect.return_value = mock_model_info

        mock_provider_manager = MagicMock()
        mock_provider_manager_class.return_value = mock_provider_manager
        mock_provider = MagicMock()
        mock_provider.value = "cuda"
        mock_provider_manager.get_best_provider.return_value = mock_provider
        mock_provider_info = {
            "platform": "test",
            "available": True,
            "description": "Test",
        }
        mock_provider_manager.get_provider_info.return_value = mock_provider_info

        config_path = temp_dir / "gpux.yml"
        config_path.write_text(sample_gpuxfile.read_text())

        result = self.runner.invoke(
            build_app, ["build", str(temp_dir), "--provider", "cuda"]
        )
        assert result.exit_code == 0
        mock_provider_manager.get_best_provider.assert_called_once_with("cuda")

    @patch("gpux.cli.build.GPUXConfigParser")
    @patch("gpux.cli.build.ModelInspector")
    @patch("gpux.cli.build.ProviderManager")
    def test_build_command_no_optimize(
        self,
        mock_provider_manager_class,
        mock_inspector_class,
        mock_parser_class,
        temp_dir: Path,
        sample_gpuxfile: Path,
    ) -> None:
        """Test build command without optimization."""
        # Setup mocks
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.validate_model_path.return_value = True
        mock_parser.get_model_path.return_value = sample_gpuxfile.parent / "model.onnx"

        mock_inspector = MagicMock()
        mock_inspector_class.return_value = mock_inspector
        mock_model_info = MagicMock()
        mock_model_info.name = "test-model"
        mock_model_info.version = "1.0.0"
        mock_model_info.format = "onnx"
        mock_model_info.size_bytes = 1024 * 1024
        mock_model_info.inputs = []
        mock_model_info.outputs = []
        mock_inspector.inspect.return_value = mock_model_info

        mock_provider_manager = MagicMock()
        mock_provider_manager_class.return_value = mock_provider_manager
        mock_provider = MagicMock()
        mock_provider.value = "cpu"
        mock_provider_manager.get_best_provider.return_value = mock_provider
        mock_provider_info = {
            "platform": "test",
            "available": True,
            "description": "Test",
        }
        mock_provider_manager.get_provider_info.return_value = mock_provider_info

        config_path = temp_dir / "gpux.yml"
        config_path.write_text(sample_gpuxfile.read_text())

        result = self.runner.invoke(
            build_app, ["build", str(temp_dir), "--no-optimize"]
        )
        assert result.exit_code == 0

    def test_build_command_verbose(self, temp_dir: Path, sample_gpuxfile: Path) -> None:
        """Test build command with verbose flag."""
        with patch("gpux.cli.build.GPUXConfigParser") as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser
            mock_parser.validate_model_path.return_value = False

            config_path = temp_dir / "gpux.yml"
            config_path.write_text(sample_gpuxfile.read_text())

            with patch("logging.getLogger") as mock_get_logger:
                mock_logger = mock_get_logger.return_value
                result = self.runner.invoke(
                    build_app, ["build", str(temp_dir), "--verbose"]
                )
                assert result.exit_code == 1
                mock_logger.setLevel.assert_called_once_with(logging.DEBUG)

    def test_display_build_results(self) -> None:
        """Test _display_build_results function."""
        # Create mock model info
        mock_model_info = MagicMock()
        mock_model_info.name = "test-model"
        mock_model_info.version = "1.0.0"
        mock_model_info.format = "onnx"
        mock_model_info.size_bytes = 1024 * 1024  # 1MB
        mock_model_info.inputs = [
            InputSpec(name="input1", type="float32", shape=[1, 10], required=True),
            InputSpec(name="input2", type="float32", shape=[1, 5], required=False),
        ]
        mock_model_info.outputs = [
            OutputSpec(name="output1", type="float32", shape=[1, 2]),
            OutputSpec(name="output2", type="float32", shape=[1, 1]),
        ]

        # Create mock provider
        mock_provider = MagicMock()
        mock_provider.value = "cpu"

        # Create mock provider info
        provider_info = {
            "platform": "test",
            "available": True,
            "description": "Test provider",
        }

        # Test that the function doesn't raise any exceptions
        _display_build_results(mock_model_info, mock_provider, provider_info)

    def test_display_build_results_empty_inputs_outputs(self) -> None:
        """Test _display_build_results with empty inputs and outputs."""
        mock_model_info = MagicMock()
        mock_model_info.name = "test-model"
        mock_model_info.version = "1.0.0"
        mock_model_info.format = "onnx"
        mock_model_info.size_bytes = 1024 * 1024
        mock_model_info.inputs = []
        mock_model_info.outputs = []

        mock_provider = MagicMock()
        mock_provider.value = "cpu"

        provider_info = {
            "platform": "test",
            "available": False,
            "description": "Test provider",
        }

        # Test that the function doesn't raise any exceptions
        _display_build_results(mock_model_info, mock_provider, provider_info)

    @patch("gpux.cli.build.GPUXConfigParser")
    def test_build_command_exception_handling(
        self, mock_parser_class, temp_dir: Path, sample_gpuxfile: Path
    ) -> None:
        """Test build command exception handling."""
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_file.side_effect = ValueError("Test error")

        config_path = temp_dir / "gpux.yml"
        config_path.write_text(sample_gpuxfile.read_text())

        result = self.runner.invoke(build_app, ["build", str(temp_dir)])
        assert result.exit_code == 1
        assert "Build failed: Test error" in result.output

    @patch("gpux.cli.build.GPUXConfigParser")
    def test_build_command_exception_handling_verbose(
        self, mock_parser_class, temp_dir: Path, sample_gpuxfile: Path
    ) -> None:
        """Test build command exception handling with verbose flag."""
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_file.side_effect = RuntimeError("Test runtime error")

        config_path = temp_dir / "gpux.yml"
        config_path.write_text(sample_gpuxfile.read_text())

        with patch("gpux.cli.build.console.print_exception") as mock_print_exception:
            result = self.runner.invoke(
                build_app, ["build", str(temp_dir), "--verbose"]
            )
            assert result.exit_code == 1
            assert "Build failed: Test runtime error" in result.output
            mock_print_exception.assert_called_once()

    def test_build_command_default_arguments(self) -> None:
        """Test build command with default arguments."""
        result = self.runner.invoke(build_app, ["build", "--help"])
        assert result.exit_code == 0
        # Check that default values are shown in help
        assert "Path to the GPUX project directory" in result.output
        assert "Configuration file name" in result.output
        assert "Enable model optimization" in result.output
