"""Integration tests for the GPUX CLI entrypoint.

These tests exercise the `gpux` CLI at a high level using Typer's CliRunner.
They avoid heavy dependencies by mocking runtime/model interactions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from gpux.cli.main import app

runner = CliRunner()


def _write_minimal_gpux_project(tmp_path: Path) -> Path:
    """Create a minimal gpux project directory with config and model file.

    Returns the project directory path.
    """
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    # Minimal config
    (project_dir / "gpux.yml").write_text(
        """
name: test-model
version: 1.0.0
model:
  source: ./model.onnx
  format: onnx
inputs:
  input:
    type: float32
    shape: [1, 2]
outputs:
  output:
    type: float32
    shape: [1, 2]
runtime: {}
        """.strip()
    )
    # Dummy model file presence is sufficient for our mocked flows
    (project_dir / "model.onnx").write_bytes(b"\x08\x03dummy")
    return project_dir


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Typer prints command names in help
    assert "build" in result.stdout
    assert "run" in result.stdout
    assert "serve" in result.stdout
    assert "inspect" in result.stdout


@pytest.mark.parametrize("benchmark", [False, True])
def test_cli_run_happy_path(tmp_path: Path, *, benchmark: bool) -> None:
    project_dir = _write_minimal_gpux_project(tmp_path)

    # Prepare minimal input data
    input_path = project_dir / "input.json"
    input_payload: dict[str, Any] = {"input": [[1.0, 2.0]]}
    input_path.write_text(json.dumps(input_payload))

    # Mock parser and runtime to avoid importing ORT / heavy ops
    with (
        patch("gpux.cli.run.GPUXConfigParser") as mock_parser_cls,
        patch("gpux.cli.run.GPUXRuntime") as mock_runtime_cls,
    ):
        mock_parser = mock_parser_cls.return_value
        # Emulate pydantic-like config: .runtime.dict() used by code
        mock_parser.parse_file.return_value = MagicMock()
        mock_parser.get_model_path.return_value = project_dir / "model.onnx"
        mock_config = MagicMock()
        mock_config.dict.return_value = {}
        mock_parser.runtime = mock_config

        mock_runtime = mock_runtime_cls.return_value
        # For normal run, infer returns a simple mapping
        mock_runtime.infer.return_value = {"output": [[0.1, 0.9]]}
        # For benchmark path, provide metrics
        mock_runtime.benchmark.return_value = {
            "avg_time_ms": 1.23,
            "p50_time_ms": 1.00,
            "fps": 800.0,
        }

        args = [
            "run",
            str(project_dir),
            "--file",
            str(input_path),
        ]
        if benchmark:
            args.append("--benchmark")

        result = runner.invoke(app, args, catch_exceptions=False)
        assert result.exit_code == 0

        # Validate interactions
        mock_parser_cls.assert_called()
        mock_runtime_cls.assert_called()
        if benchmark:
            assert mock_runtime.benchmark.called is True
        else:
            assert mock_runtime.infer.called is True


def test_cli_run_model_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Do not create the project dir; use a name that doesn't exist
    monkeypatch.chdir(tmp_path)
    missing = Path("missing")
    result = runner.invoke(app, ["run", str(missing), "--file", str(Path("in.json"))])
    # Typer exits with code 1 on our error path
    assert result.exit_code == 1
    assert "Error: Model '" in result.stdout


def test_cli_run_model_file_missing(tmp_path: Path) -> None:
    # Create project with config but no model.onnx
    project_dir = tmp_path / "proj2"
    project_dir.mkdir()
    (project_dir / "gpux.yml").write_text(
        """
name: test-model
version: 1.0.0
model:
  source: ./model.onnx
  format: onnx
inputs:
  input:
    type: float32
    shape: [1, 2]
outputs:
  output:
    type: float32
    shape: [1, 2]
runtime: {}
        """.strip()
    )

    input_path = project_dir / "input.json"
    input_path.write_text(json.dumps({"input": [[1.0, 2.0]]}))

    with patch("gpux.cli.run.GPUXConfigParser") as mock_parser_cls:
        mock_parser = mock_parser_cls.return_value
        mock_parser.parse_file.return_value = MagicMock()
        # Point to non-existent model file
        mock_parser.get_model_path.return_value = project_dir / "model.onnx"
        mock_config = MagicMock()
        mock_config.dict.return_value = {}
        mock_parser.runtime = mock_config

        result = runner.invoke(
            app, ["run", str(project_dir), "--file", str(input_path)]
        )
        assert result.exit_code == 1
        assert "Model file not found" in result.stdout


def test_cli_run_no_input_data(tmp_path: Path) -> None:
    project_dir = _write_minimal_gpux_project(tmp_path)
    # No input flags provided
    with (
        patch("gpux.cli.run.GPUXConfigParser") as mock_parser_cls,
        patch("gpux.cli.run.GPUXRuntime") as mock_runtime_cls,
    ):
        mock_parser = mock_parser_cls.return_value
        mock_parser.parse_file.return_value = MagicMock()
        mock_parser.get_model_path.return_value = project_dir / "model.onnx"
        mock_config = MagicMock()
        mock_config.dict.return_value = {}
        mock_parser.runtime = mock_config

        _ = mock_runtime_cls.return_value

        result = runner.invoke(app, ["run", str(project_dir)])
        assert result.exit_code == 1
        assert "No input data provided" in result.stdout


def test_cli_run_invalid_input_json(tmp_path: Path) -> None:
    project_dir = _write_minimal_gpux_project(tmp_path)
    bad_json_path = project_dir / "bad.json"
    bad_json_path.write_text("{")  # invalid JSON

    with (
        patch("gpux.cli.run.GPUXConfigParser") as mock_parser_cls,
        patch("gpux.cli.run.GPUXRuntime") as mock_runtime_cls,
    ):
        mock_parser = mock_parser_cls.return_value
        mock_parser.parse_file.return_value = MagicMock()
        mock_parser.get_model_path.return_value = project_dir / "model.onnx"
        mock_config = MagicMock()
        mock_config.dict.return_value = {}
        mock_parser.runtime = mock_config

        _ = mock_runtime_cls.return_value

        result = runner.invoke(
            app, ["run", str(project_dir), "--file", str(bad_json_path)]
        )
        assert result.exit_code == 1
        assert "Error loading input file" in result.stdout


def test_cli_build_happy_path(tmp_path: Path) -> None:
    project_dir = _write_minimal_gpux_project(tmp_path)

    with (
        patch("gpux.cli.build.GPUXConfigParser") as mock_parser_cls,
        patch("gpux.cli.build.ModelInspector") as mock_inspector_cls,
        patch("gpux.cli.build.ProviderManager") as mock_pm_cls,
    ):
        mock_parser = mock_parser_cls.return_value
        mock_parser.parse_file.return_value = MagicMock()
        mock_parser.validate_model_path.return_value = True
        mock_parser.get_model_path.return_value = project_dir / "model.onnx"

        mock_inspector = mock_inspector_cls.return_value
        mock_model_info = MagicMock()
        # Provide attributes used in display
        mock_model_info.name = "test-model"
        mock_model_info.version = "1.0.0"
        mock_model_info.format = "onnx"
        mock_model_info.size_bytes = 1024
        mock_model_info.inputs = []
        mock_model_info.outputs = []
        mock_inspector.inspect.return_value = mock_model_info

        mock_pm = mock_pm_cls.return_value
        mock_provider = MagicMock()
        mock_provider.value = "CPUExecutionProvider"
        mock_pm.get_best_provider.return_value = mock_provider
        mock_pm.get_provider_info.return_value = {
            "available": True,
            "platform": "cpu",
            "description": "CPU Fallback",
        }

        result = runner.invoke(app, ["build", str(project_dir)])
        assert result.exit_code == 0
        assert "Build completed successfully" in result.stdout


def test_cli_inspect_runtime_json() -> None:
    # No args -> runtime info
    with patch("gpux.cli.inspect.ProviderManager") as mock_pm_cls:
        mock_pm = mock_pm_cls.return_value
        mock_pm.get_available_providers.return_value = ["CPUExecutionProvider"]
        # iterate over _provider_priority in code: provide at least CPU
        cpu_provider = MagicMock()
        cpu_provider.value = "CPUExecutionProvider"
        mock_pm._provider_priority = [cpu_provider]
        mock_pm.get_provider_info.return_value = {
            "available": True,
            "platform": "cpu",
            "description": "CPU Fallback",
        }

        result = runner.invoke(app, ["inspect", "--json-output"])
        assert result.exit_code == 0
        assert "available_providers" in result.stdout


# Build error-paths
def test_cli_build_missing_config(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(app, ["build", str(empty)])
    assert result.exit_code == 1
    assert "Configuration file not found" in result.stdout


def test_cli_build_model_not_found(tmp_path: Path) -> None:
    project_dir = tmp_path / "proj-missing-model"
    project_dir.mkdir()
    (project_dir / "gpux.yml").write_text(
        "name: x\nversion: 0\nmodel:\n  source: ./model.onnx\n  format: onnx\n"
    )

    with patch("gpux.cli.build.GPUXConfigParser") as mock_parser_cls:
        mock_parser = mock_parser_cls.return_value
        mock_parser.parse_file.return_value = MagicMock()
        mock_parser.validate_model_path.return_value = False

        result = runner.invoke(app, ["build", str(project_dir)])
        assert result.exit_code == 1
        assert "Model file not found" in result.stdout


def test_cli_build_inspect_failure(tmp_path: Path) -> None:
    project_dir = _write_minimal_gpux_project(tmp_path)
    with (
        patch("gpux.cli.build.GPUXConfigParser") as mock_parser_cls,
        patch("gpux.cli.build.ModelInspector") as mock_inspector_cls,
    ):
        mock_parser = mock_parser_cls.return_value
        mock_parser.parse_file.return_value = MagicMock()
        mock_parser.validate_model_path.return_value = True
        mock_parser.get_model_path.return_value = project_dir / "model.onnx"

        mock_inspector = mock_inspector_cls.return_value
        mock_inspector.inspect.side_effect = RuntimeError("inspect failed")

        result = runner.invoke(app, ["build", str(project_dir)])
        assert result.exit_code == 1
        assert "Build failed" in result.stdout


# Inspect error-paths
def test_cli_inspect_model_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "nope.onnx"
    result = runner.invoke(app, ["inspect", "--model", str(missing)])
    assert result.exit_code == 1
    assert "Error: Model file not found" in result.stdout


def test_cli_inspect_model_by_name_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["inspect", "unknown-model"])
    assert result.exit_code == 1
    assert "Error: Model 'unknown-model' not found" in result.stdout


def test_cli_inspect_model_by_name_model_file_missing(tmp_path: Path) -> None:
    project_dir = tmp_path / "proj-inspect"
    project_dir.mkdir()
    (project_dir / "gpux.yml").write_text(
        "name: test\nversion: 1\nmodel:\n  source: ./model.onnx\n  format: onnx\n"
    )
    with (
        patch("gpux.cli.inspect.GPUXConfigParser") as mock_parser_cls,
        patch("gpux.cli.inspect._find_model_config", return_value=project_dir),
    ):
        mock_parser = mock_parser_cls.return_value
        mock_parser.parse_file.return_value = MagicMock()
        mock_parser.get_model_path.return_value = (
            project_dir / "model.onnx"
        )  # does not exist

        result = runner.invoke(app, ["inspect", project_dir.name])
        assert result.exit_code == 1
        assert "Model file not found" in result.stdout


def test_cli_inspect_model_file_runtime_error(tmp_path: Path) -> None:
    model_path = tmp_path / "m.onnx"
    model_path.write_bytes(b"\x00\x01")
    with patch("gpux.cli.inspect.ModelInspector") as mock_inspector_cls:
        mock_inspector_cls.return_value.inspect.side_effect = RuntimeError("boom")
        result = runner.invoke(app, ["inspect", "--model", str(model_path)])
        assert result.exit_code == 1
        assert "Inspect failed" in result.stdout
