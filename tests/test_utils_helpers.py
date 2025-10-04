"""Tests for utils helpers functionality."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gpux.utils.helpers import (
    check_dependencies,
    create_gpuxfile_template,
    ensure_directory,
    find_files,
    format_bytes,
    format_time,
    get_gpu_info,
    get_system_info,
    run_command,
    validate_file_extension,
    validate_gpuxfile,
)


class TestSystemInfo:
    """Test cases for system information functions."""

    def test_get_system_info(self) -> None:
        """Test get_system_info function."""
        info = get_system_info()

        assert isinstance(info, dict)
        assert "platform" in info
        assert "system" in info
        assert "machine" in info
        assert "processor" in info
        assert "python_version" in info
        assert "architecture" in info

        # Check that values are strings
        for key, value in info.items():
            assert isinstance(value, (str, tuple))
            if key == "architecture":
                assert isinstance(value, tuple)
            else:
                assert isinstance(value, str)

    def test_format_bytes(self) -> None:
        """Test format_bytes function."""
        assert format_bytes(0) == "0.0 B"
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1024 * 1024) == "1.0 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"
        assert format_bytes(1024 * 1024 * 1024 * 1024) == "1.0 TB"

        # Test fractional values
        assert format_bytes(1536) == "1.5 KB"
        assert format_bytes(1536 * 1024) == "1.5 MB"

    def test_format_time(self) -> None:
        """Test format_time function."""
        assert format_time(0.000001) == "1.0 μs"
        assert format_time(0.001) == "1.0 ms"
        assert format_time(1.0) == "1.00 s"
        assert format_time(0.5) == "500.0 ms"
        assert format_time(0.000000001) == "1.0 ns"


class TestFileOperations:
    """Test cases for file operation functions."""

    def test_ensure_directory(self, temp_dir: Path) -> None:
        """Test ensure_directory function."""
        new_dir = temp_dir / "new_directory"
        result = ensure_directory(new_dir)

        assert result == new_dir
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_ensure_directory_existing(self, temp_dir: Path) -> None:
        """Test ensure_directory function with existing directory."""
        result = ensure_directory(temp_dir)

        assert result == temp_dir
        assert temp_dir.exists()

    def test_find_files(self, temp_dir: Path) -> None:
        """Test find_files function."""
        # Create test files
        (temp_dir / "test1.txt").write_text("test1")
        (temp_dir / "test2.txt").write_text("test2")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "test3.txt").write_text("test3")

        # Test recursive search
        files = find_files(temp_dir, "*.txt", recursive=True)
        assert len(files) == 3
        assert all(f.name.endswith(".txt") for f in files)

        # Test non-recursive search
        files = find_files(temp_dir, "*.txt", recursive=False)
        assert len(files) == 2
        assert all(f.name.endswith(".txt") for f in files)

    def test_find_files_no_matches(self, temp_dir: Path) -> None:
        """Test find_files function with no matches."""
        files = find_files(temp_dir, "*.nonexistent")
        assert files == []

    def test_validate_file_extension(self, temp_dir: Path) -> None:
        """Test validate_file_extension function."""
        test_file = temp_dir / "test.onnx"
        test_file.write_text("test")

        assert validate_file_extension(test_file, [".onnx"]) is True
        assert validate_file_extension(test_file, [".txt"]) is False
        assert validate_file_extension(test_file, [".onnx", ".txt"]) is True
        assert validate_file_extension("test.onnx.gz", [".onnx.gz"]) is True

    def test_validate_file_extension_string_path(self) -> None:
        """Test validate_file_extension function with string path."""
        assert validate_file_extension("test.onnx", [".onnx"]) is True
        assert validate_file_extension("test.txt", [".onnx"]) is False


class TestCommandExecution:
    """Test cases for command execution functions."""

    def test_run_command_success(self) -> None:
        """Test run_command function with successful command."""
        result = run_command(["echo", "test"], capture_output=True)

        assert result.returncode == 0
        assert result.stdout.strip() == "test"

    def test_run_command_failure(self) -> None:
        """Test run_command function with failing command."""
        with pytest.raises(subprocess.CalledProcessError):
            run_command(["false"], capture_output=True)

    def test_run_command_no_check(self) -> None:
        """Test run_command function with check=False."""
        result = run_command(["false"], capture_output=True, check=False)

        assert result.returncode != 0

    def test_run_command_no_capture(self) -> None:
        """Test run_command function without capturing output."""
        result = run_command(["echo", "test"], capture_output=False)

        assert result.returncode == 0
        assert result.stdout is None

    def test_run_command_with_cwd(self, temp_dir: Path) -> None:
        """Test run_command function with working directory."""
        result = run_command(["pwd"], capture_output=True, cwd=temp_dir)

        assert result.returncode == 0
        assert str(temp_dir) in result.stdout


class TestDependencies:
    """Test cases for dependency checking functions."""

    def test_check_dependencies(self) -> None:
        """Test check_dependencies function."""
        deps = check_dependencies()

        assert isinstance(deps, dict)
        assert "onnxruntime" in deps
        assert "onnx" in deps
        assert "numpy" in deps
        assert "yaml" in deps
        assert "click" in deps
        assert "typer" in deps
        assert "rich" in deps
        assert "pydantic" in deps

        # All values should be booleans
        for value in deps.values():
            assert isinstance(value, bool)

    def test_get_gpu_info_no_onnxruntime(self) -> None:
        """Test get_gpu_info function without ONNX Runtime."""
        with patch("builtins.__import__", side_effect=ImportError):
            gpu_info = get_gpu_info()

            assert gpu_info["available"] is False
            assert gpu_info["devices"] == []
            assert gpu_info["provider"] is None

    def test_get_gpu_info_with_onnxruntime(self) -> None:
        """Test get_gpu_info function with ONNX Runtime."""
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
        ]

        with patch("builtins.__import__", return_value=mock_ort):
            gpu_info = get_gpu_info()

            assert gpu_info["available"] is True
            assert "CUDAExecutionProvider" in gpu_info["providers"]
            assert gpu_info["provider"] == "CUDAExecutionProvider"

    def test_get_gpu_info_cpu_only(self) -> None:
        """Test get_gpu_info function with CPU-only providers."""
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        with patch("builtins.__import__", return_value=mock_ort):
            gpu_info = get_gpu_info()

            assert gpu_info["available"] is False
            assert gpu_info.get("providers", []) == []


class TestGPUXFileOperations:
    """Test cases for GPUX file operations."""

    def test_create_gpuxfile_template(self, temp_dir: Path) -> None:
        """Test create_gpuxfile_template function."""
        model_path = temp_dir / "model.onnx"
        model_path.write_text("test")
        output_path = temp_dir / "gpux.yml"

        create_gpuxfile_template("test-model", model_path, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "name: test-model" in content
        assert "model.onnx" in content
        assert "format: onnx" in content

    def test_create_gpuxfile_template_default_output(self, temp_dir: Path) -> None:
        """Test create_gpuxfile_template function with default output path."""
        model_path = temp_dir / "model.onnx"
        model_path.write_text("test")

        with patch("gpux.utils.helpers.Path") as mock_path:
            mock_output_path = MagicMock()
            mock_path.return_value = mock_output_path

            create_gpuxfile_template("test-model", model_path)

            mock_output_path.open.assert_called_once()

    def test_validate_gpuxfile_valid(
        self, temp_dir: Path, sample_gpuxfile: Path
    ) -> None:
        """Test validate_gpuxfile function with valid file."""
        config_path = temp_dir / "gpux.yml"
        config_path.write_text(sample_gpuxfile.read_text())

        with patch("gpux.config.parser.GPUXConfigParser") as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser

            result = validate_gpuxfile(config_path)

            assert result is True
            mock_parser.parse_file.assert_called_once_with(config_path)

    def test_validate_gpuxfile_invalid(self, temp_dir: Path) -> None:
        """Test validate_gpuxfile function with invalid file."""
        invalid_file = temp_dir / "invalid.yml"
        invalid_file.write_text("invalid: yaml: content:")

        with patch("gpux.config.parser.GPUXConfigParser") as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser
            mock_parser.parse_file.side_effect = Exception("Invalid YAML")

            result = validate_gpuxfile(invalid_file)

            assert result is False

    def test_validate_gpuxfile_nonexistent(self, temp_dir: Path) -> None:
        """Test validate_gpuxfile function with non-existent file."""
        nonexistent_file = temp_dir / "nonexistent.yml"

        with patch("gpux.config.parser.GPUXConfigParser") as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser
            mock_parser.parse_file.side_effect = FileNotFoundError("File not found")

            result = validate_gpuxfile(nonexistent_file)

            assert result is False


class TestEdgeCases:
    """Test cases for edge cases and error conditions."""

    def test_format_bytes_very_large(self) -> None:
        """Test format_bytes function with very large values."""
        # Test petabyte range
        pb_value = 1024 * 1024 * 1024 * 1024 * 1024
        result = format_bytes(pb_value)
        assert "PB" in result

    def test_format_time_very_small(self) -> None:
        """Test format_time function with very small values."""
        result = format_time(0.0000000001)  # 0.1 ns
        assert "ns" in result

    def test_format_time_very_large(self) -> None:
        """Test format_time function with very large values."""
        result = format_time(3600.0)  # 1 hour
        assert "s" in result

    def test_run_command_empty_command(self) -> None:
        """Test run_command function with empty command."""
        with pytest.raises((subprocess.CalledProcessError, IndexError)):
            run_command([], capture_output=True)

    def test_find_files_empty_directory(self, temp_dir: Path) -> None:
        """Test find_files function with empty directory."""
        files = find_files(temp_dir, "*.txt")
        assert files == []

    def test_validate_file_extension_no_extension(self) -> None:
        """Test validate_file_extension function with file without extension."""
        assert validate_file_extension("file", [".txt"]) is False
        assert validate_file_extension("file.", [".txt"]) is False

    def test_ensure_directory_nested(self, temp_dir: Path) -> None:
        """Test ensure_directory function with nested directory."""
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        result = ensure_directory(nested_dir)

        assert result == nested_dir
        assert nested_dir.exists()
        assert nested_dir.is_dir()
