"""Tests for run CLI functionality."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from gpux.cli.main import app
from gpux.cli.run import (
    _find_model_config,
    _load_input_data,
    _run_benchmark,
    _run_inference,
    run_command,
)


class TestRunCLI:
    """Test cases for run CLI functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_run_command_creation(self) -> None:
        """Test that the run command is created correctly."""
        assert callable(run_command)

    def test_run_command_help(self) -> None:
        """Test run command help output."""
        result = self.runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run inference on a model" in result.output
        assert "--input" in result.output
        assert "--file" in result.output
        assert "--output" in result.output
        assert "--benchmark" in result.output

    def test_find_model_config_current_directory(
        self, temp_dir: Path, sample_gpuxfile: Path
    ) -> None:
        """Test finding model config in current directory."""
        config_path = temp_dir / "gpux.yml"
        config_path.write_text(sample_gpuxfile.read_text())

        with patch("gpux.cli.run.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            result = _find_model_config("test-model", "gpux.yml")
            assert result == mock_path_instance

    def test_find_model_config_model_directory(
        self, temp_dir: Path, sample_gpuxfile: Path
    ) -> None:
        """Test finding model config in model directory."""
        model_dir = temp_dir / "test-model"
        model_dir.mkdir()
        config_path = model_dir / "gpux.yml"
        config_path.write_text(sample_gpuxfile.read_text())

        with patch("gpux.cli.run.Path") as mock_path:
            # Mock Path() to return temp_dir for current directory check
            mock_current_dir = MagicMock()
            mock_current_dir.exists.return_value = False
            mock_path.return_value = mock_current_dir

            # Mock Path(model_name) to return model_dir
            def path_side_effect(*args, **kwargs):
                if args and args[0] == "test-model":
                    mock_model_dir = MagicMock()
                    mock_model_dir.is_dir.return_value = True
                    mock_model_dir.__truediv__ = lambda self, other: model_dir / other
                    return mock_model_dir
                return mock_current_dir

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

        with patch("gpux.cli.run.Path") as mock_path:
            # Mock current directory check
            mock_current_dir = MagicMock()
            mock_current_dir.exists.return_value = False
            mock_path.return_value = mock_current_dir

            # Mock model directory check
            def path_side_effect(*args, **kwargs):
                if args and args[0] == "test-model":
                    mock_model_dir = MagicMock()
                    mock_model_dir.is_dir.return_value = False
                    return mock_model_dir
                return mock_current_dir

            mock_path.side_effect = path_side_effect

            # Mock .gpux directory
            mock_gpux_dir = MagicMock()
            mock_gpux_dir.exists.return_value = True
            mock_gpux_dir.glob.return_value = [model_info_file]

            with patch("gpux.cli.run.Path", return_value=mock_gpux_dir):
                result = _find_model_config("test-model", "gpux.yml")
                assert result is not None

    def test_find_model_config_not_found(self) -> None:
        """Test finding model config when not found."""
        # Mock the current directory check to return False
        with patch("gpux.cli.run.Path") as mock_path:

            def path_side_effect(*args, **kwargs):
                if not args:  # Path() with no arguments
                    mock_current_dir = MagicMock()
                    mock_current_dir.exists.return_value = False
                    # Mock the __truediv__ method for current_dir / config_file
                    mock_config_file = MagicMock()
                    mock_config_file.exists.return_value = False
                    mock_current_dir.__truediv__ = lambda self, other: mock_config_file
                    return mock_current_dir
                if args and args[0] == "definitely-nonexistent-model-12345":
                    mock_model_dir = MagicMock()
                    mock_model_dir.is_dir.return_value = False
                    mock_model_dir.__truediv__ = lambda self, other: MagicMock()
                    return mock_model_dir
                if args and args[0] == ".gpux":
                    mock_gpux_dir = MagicMock()
                    mock_gpux_dir.exists.return_value = False
                    return mock_gpux_dir
                return MagicMock()

            mock_path.side_effect = path_side_effect

            result = _find_model_config(
                "definitely-nonexistent-model-12345", "gpux.yml"
            )
            assert result is None

    def test_load_input_data_from_file(self, temp_dir: Path) -> None:
        """Test loading input data from file."""
        input_file = temp_dir / "input.json"
        input_data = {"input1": [1, 2, 3], "input2": [4, 5, 6]}
        input_file.write_text(json.dumps(input_data))

        result = _load_input_data(None, str(input_file))
        assert result == input_data

    def test_load_input_data_from_file_with_at_prefix(self, temp_dir: Path) -> None:
        """Test loading input data from file with @ prefix."""
        input_file = temp_dir / "input.json"
        input_data = {"input1": [1, 2, 3], "input2": [4, 5, 6]}
        input_file.write_text(json.dumps(input_data))

        result = _load_input_data(f"@{input_file}", None)
        assert result == input_data

    def test_load_input_data_from_json_string(self) -> None:
        """Test loading input data from JSON string."""
        input_data = {"input1": [1, 2, 3], "input2": [4, 5, 6]}
        json_string = json.dumps(input_data)

        result = _load_input_data(json_string, None)
        assert result == input_data

    def test_load_input_data_invalid_json_string(self) -> None:
        """Test loading input data from invalid JSON string."""
        with patch("gpux.cli.run.console.print") as mock_print:
            result = _load_input_data("invalid json", None)
            assert result is None
            mock_print.assert_called_once()

    def test_load_input_data_invalid_file(self, temp_dir: Path) -> None:
        """Test loading input data from invalid file."""
        input_file = temp_dir / "invalid.json"
        input_file.write_text("invalid json")

        with patch("gpux.cli.run.console.print") as mock_print:
            result = _load_input_data(None, str(input_file))
            assert result is None
            mock_print.assert_called_once()

    def test_load_input_data_none(self) -> None:
        """Test loading input data when both parameters are None."""
        result = _load_input_data(None, None)
        assert result is None

    def test_load_input_data_non_dict_data(self, temp_dir: Path) -> None:
        """Test loading input data that is not a dictionary."""
        input_file = temp_dir / "input.json"
        input_file.write_text(json.dumps([1, 2, 3]))  # List instead of dict

        result = _load_input_data(None, str(input_file))
        assert result is None

    def test_run_inference(self, temp_dir: Path) -> None:
        """Test _run_inference function."""
        mock_runtime = MagicMock()
        mock_runtime.infer.return_value = {"output1": [0.1, 0.2], "output2": [0.3, 0.4]}

        input_data = {"input1": [1, 2, 3], "input2": [4, 5, 6]}

        with patch("gpux.cli.run.console.print") as mock_print:
            _run_inference(mock_runtime, input_data, None)
            mock_runtime.infer.assert_called_once()
            mock_print.assert_called()

    def test_run_inference_with_output_file(self, temp_dir: Path) -> None:
        """Test _run_inference function with output file."""
        mock_runtime = MagicMock()
        mock_runtime.infer.return_value = {"output1": [0.1, 0.2], "output2": [0.3, 0.4]}

        input_data = {"input1": [1, 2, 3], "input2": [4, 5, 6]}
        output_file = temp_dir / "output.json"

        with patch("gpux.cli.run.console.print") as mock_print:
            _run_inference(mock_runtime, input_data, str(output_file))
            mock_runtime.infer.assert_called_once()

            # Check that output file was created
            assert output_file.exists()
            with output_file.open() as f:
                saved_data = json.load(f)
                assert saved_data == {"output1": [0.1, 0.2], "output2": [0.3, 0.4]}

    def test_run_benchmark(self, temp_dir: Path) -> None:
        """Test _run_benchmark function."""
        mock_runtime = MagicMock()
        mock_runtime.benchmark.return_value = {
            "avg_inference_time": 1.5,
            "min_inference_time": 1.2,
            "max_inference_time": 1.8,
            "fps": 666.7,
        }

        input_data = {"input1": [1, 2, 3], "input2": [4, 5, 6]}

        with patch("gpux.cli.run.console.print") as mock_print:
            _run_benchmark(mock_runtime, input_data, 100, 10, None)
            mock_runtime.benchmark.assert_called_once()
            mock_print.assert_called()

    def test_run_benchmark_with_output_file(self, temp_dir: Path) -> None:
        """Test _run_benchmark function with output file."""
        mock_runtime = MagicMock()
        mock_runtime.benchmark.return_value = {
            "avg_inference_time": 1.5,
            "min_inference_time": 1.2,
            "max_inference_time": 1.8,
            "fps": 666.7,
        }

        input_data = {"input1": [1, 2, 3], "input2": [4, 5, 6]}
        output_file = temp_dir / "benchmark.json"

        with patch("gpux.cli.run.console.print") as mock_print:
            _run_benchmark(mock_runtime, input_data, 100, 10, str(output_file))
            mock_runtime.benchmark.assert_called_once()

            # Check that output file was created
            assert output_file.exists()
            with output_file.open() as f:
                saved_data = json.load(f)
                assert saved_data == mock_runtime.benchmark.return_value

    @patch("gpux.cli.run._find_model_config")
    @patch("gpux.cli.run.GPUXConfigParser")
    @patch("gpux.cli.run.GPUXRuntime")
    def test_run_command_success(
        self,
        mock_runtime_class,
        mock_parser_class,
        mock_find_config,
        temp_dir: Path,
        sample_gpuxfile: Path,
    ) -> None:
        """Test successful run command execution."""
        # Setup mocks
        mock_find_config.return_value = temp_dir

        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_config = MagicMock()
        mock_config.runtime.dict.return_value = {}
        mock_parser.parse_file.return_value = mock_config
        # Mock the model file path and its existence
        mock_model_file = MagicMock()
        mock_model_file.exists.return_value = True
        mock_parser.get_model_path.return_value = mock_model_file

        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime
        mock_runtime.infer.return_value = {"output": [0.1, 0.2]}

        # Create config file
        config_path = temp_dir / "gpux.yml"
        config_path.write_text(sample_gpuxfile.read_text())

        result = self.runner.invoke(
            app, ["run", "test-model", "--input", '{"input1": [1, 2, 3]}']
        )
        assert result.exit_code == 0

    def test_run_command_model_not_found(self) -> None:
        """Test run command when model is not found."""
        with patch("gpux.cli.run._find_model_config", return_value=None):
            result = self.runner.invoke(app, ["run", "nonexistent-model"])
            assert result.exit_code == 1
            assert "Model 'nonexistent-model' not found" in result.output

    def test_run_command_no_input_data(
        self, temp_dir: Path, sample_gpuxfile: Path
    ) -> None:
        """Test run command when no input data is provided."""
        with patch("gpux.cli.run._find_model_config", return_value=temp_dir):
            result = self.runner.invoke(app, ["run", "test-model"])
            assert result.exit_code == 1
            assert "No input data provided" in result.output

    def test_run_command_verbose(self) -> None:
        """Test run command with verbose flag."""
        with patch("gpux.cli.run._find_model_config", return_value=None):
            with patch("logging.getLogger") as mock_get_logger:
                mock_logger = mock_get_logger.return_value
                result = self.runner.invoke(app, ["run", "test-model", "--verbose"])
                assert result.exit_code == 1
                mock_logger.setLevel.assert_called_once_with(logging.DEBUG)

    def test_run_command_benchmark(self, temp_dir: Path, sample_gpuxfile: Path) -> None:
        """Test run command with benchmark flag."""
        with patch("gpux.cli.run._find_model_config", return_value=temp_dir):
            with patch("gpux.cli.run.GPUXConfigParser") as mock_parser_class:
                mock_parser = MagicMock()
                mock_parser_class.return_value = mock_parser
                mock_config = MagicMock()
                mock_config.runtime.dict.return_value = {}
                mock_parser.parse_file.return_value = mock_config
                # Mock the model file path and its existence
                mock_model_file = MagicMock()
                mock_model_file.exists.return_value = True
                mock_parser.get_model_path.return_value = mock_model_file

                with patch("gpux.cli.run.GPUXRuntime") as mock_runtime_class:
                    mock_runtime = MagicMock()
                    mock_runtime_class.return_value = mock_runtime
                    mock_runtime.benchmark.return_value = {"avg_inference_time": 1.5}

                    result = self.runner.invoke(
                        app,
                        [
                            "run",
                            "test-model",
                            "--input",
                            '{"input1": [1, 2, 3]}',
                            "--benchmark",
                        ],
                    )
                    assert result.exit_code == 0

    def test_run_command_exception_handling(self) -> None:
        """Test run command exception handling."""
        with patch(
            "gpux.cli.run._find_model_config", side_effect=ValueError("Test error")
        ):
            result = self.runner.invoke(app, ["run", "test-model"])
            assert result.exit_code == 1
            assert "Run failed: Test error" in result.output

    def test_run_command_exception_handling_verbose(self) -> None:
        """Test run command exception handling with verbose flag."""
        with (
            patch(
                "gpux.cli.run._find_model_config",
                side_effect=RuntimeError("Test error"),
            ),
            patch("gpux.cli.run.console.print_exception") as mock_print_exception,
        ):
            result = self.runner.invoke(app, ["run", "test-model", "--verbose"])
            assert result.exit_code == 1
            assert "Run failed: Test error" in result.output
            mock_print_exception.assert_called_once()

    def test_run_command_default_arguments(self) -> None:
        """Test run command with default arguments."""
        result = self.runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        # Check that default values are shown in help
        assert "Name of the model to run" in result.output
        assert "Input data" in result.output
        assert "Number of benchmark runs" in result.output
        assert "Number of warmup runs" in result.output
