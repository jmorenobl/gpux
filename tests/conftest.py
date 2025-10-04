"""Pytest configuration and fixtures for GPUX tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import onnx
import pytest
from gpux.config.parser import GPUXConfigParser
from gpux.core.providers import ProviderManager
from gpux.core.runtime import GPUXRuntime
from onnx import TensorProto, helper


@pytest.fixture()
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture()
def simple_onnx_model(temp_dir: Path) -> Path:
    """Create a simple ONNX model for testing."""
    # Create a simple model: input -> add -> output
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])

    # Create a simple add node
    add_node = helper.make_node(
        "Add",
        inputs=["input", "input"],  # Add input to itself
        outputs=["output"],
        name="add_node",
    )

    # Create the graph
    graph = helper.make_graph(
        nodes=[add_node],
        name="simple_model",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    # Create the model
    model = helper.make_model(graph)
    model.opset_import[0].version = 11
    # Set IR version to 11 (compatible with ONNX Runtime)
    model.ir_version = 11

    # Save the model
    model_path = temp_dir / "simple_model.onnx"
    onnx.save(model, str(model_path))

    return model_path


@pytest.fixture()
def sample_gpuxfile(temp_dir: Path, simple_onnx_model: Path) -> Path:
    """Create a sample gpux.yml for testing."""
    config_content = f"""name: test-model
version: 1.0.0
description: "Test model for GPUX"

model:
  source: {simple_onnx_model.name}
  format: onnx

inputs:
  input:
    type: float32
    shape: [1, 2]
    required: true
    description: "Input tensor"

outputs:
  output:
    type: float32
    shape: [1, 2]
    description: "Output tensor"

runtime:
  gpu:
    memory: 1GB
    backend: auto

serving:
  port: 8080
  host: 0.0.0.0
  batch_size: 1
  timeout: 5
"""

    config_path = temp_dir / "gpux.yml"
    with config_path.open("w") as f:
        f.write(config_content)

    return config_path


@pytest.fixture()
def sample_input_data() -> dict:
    """Create sample input data for testing."""
    return {"input": np.array([[1.0, 2.0]], dtype=np.float32)}


@pytest.fixture()
def expected_output_data() -> dict:
    """Create expected output data for testing."""
    return {
        "output": np.array([[2.0, 4.0]], dtype=np.float32)  # input + input
    }


@pytest.fixture()
def provider_manager() -> ProviderManager:
    """Create a provider manager for testing."""
    return ProviderManager()


@pytest.fixture()
def runtime(simple_onnx_model: Path) -> GPUXRuntime:
    """Create a GPUX runtime for testing."""
    return GPUXRuntime(model_path=simple_onnx_model)


@pytest.fixture()
def config_parser() -> GPUXConfigParser:
    """Create a configuration parser for testing."""
    return GPUXConfigParser()
