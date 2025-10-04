"""Tests for configuration parsing functionality."""

from pathlib import Path

import pytest

from gpux.config.parser import (
    GPUConfig,
    GPUXConfig,
    GPUXConfigParser,
    InputConfig,
    ModelConfig,
    OutputConfig,
)


class TestInputConfig:
    """Test cases for InputConfig class."""

    def test_input_config_creation(self):
        """Test creating input configuration."""
        config = InputConfig(
            name="input",
            type="float32",
            shape=[1, 10],
            required=True,
            description="Test input",
        )

        assert config.name == "input"
        assert config.type == "float32"
        assert config.shape == [1, 10]
        assert config.required is True
        assert config.description == "Test input"

    def test_input_config_to_dict(self):
        """Test converting input config to dictionary."""
        config = InputConfig(name="input", type="float32", shape=[1, 10])

        config_dict = config.to_dict()

        assert config_dict["name"] == "input"
        assert config_dict["type"] == "float32"
        assert config_dict["shape"] == [1, 10]
        assert config_dict["required"] is True

    def test_input_config_from_dict(self):
        """Test creating input config from dictionary."""
        config_dict = {
            "name": "input",
            "type": "float32",
            "shape": [1, 10],
            "required": False,
        }

        config = InputConfig.from_dict(config_dict)

        assert config.name == "input"
        assert config.type == "float32"
        assert config.shape == [1, 10]
        assert config.required is False


class TestOutputConfig:
    """Test cases for OutputConfig class."""

    def test_output_config_creation(self):
        """Test creating output configuration."""
        config = OutputConfig(
            name="output",
            type="float32",
            shape=[1, 2],
            labels=["negative", "positive"],
            description="Test output",
        )

        assert config.name == "output"
        assert config.type == "float32"
        assert config.shape == [1, 2]
        assert config.labels == ["negative", "positive"]
        assert config.description == "Test output"

    def test_output_config_to_dict(self):
        """Test converting output config to dictionary."""
        config = OutputConfig(
            name="output", type="float32", shape=[1, 2], labels=["negative", "positive"]
        )

        config_dict = config.to_dict()

        assert config_dict["name"] == "output"
        assert config_dict["type"] == "float32"
        assert config_dict["shape"] == [1, 2]
        assert config_dict["labels"] == ["negative", "positive"]


class TestModelConfig:
    """Test cases for ModelConfig class."""

    def test_model_config_creation(self):
        """Test creating model configuration."""
        config = ModelConfig(source="./model.onnx", format="onnx", version="1.0.0")

        assert config.source == Path("./model.onnx")
        assert config.format == "onnx"
        assert config.version == "1.0.0"

    def test_model_config_path_validation(self):
        """Test model config path validation."""
        config = ModelConfig(source="model.onnx")
        assert config.source == Path("model.onnx")

        config = ModelConfig(source=Path("model.onnx"))
        assert config.source == Path("model.onnx")


class TestGPUConfig:
    """Test cases for GPUConfig class."""

    def test_gpu_config_creation(self):
        """Test creating GPU configuration."""
        config = GPUConfig(memory="4GB", backend="cuda")

        assert config.memory == "4GB"
        assert config.backend == "cuda"

    def test_gpu_config_memory_validation(self):
        """Test GPU config memory validation."""
        # Valid memory specifications
        GPUConfig(memory="2GB")
        GPUConfig(memory="512MB")
        GPUConfig(memory="1024KB")

        # Invalid memory specifications
        with pytest.raises(ValueError, match="Invalid memory format"):
            GPUConfig(memory="2")

        with pytest.raises(ValueError, match="Invalid memory format"):
            GPUConfig(memory="2TB")


class TestGPUXConfig:
    """Test cases for GPUXConfig class."""

    def test_gpux_config_creation(self, sample_gpuxfile):
        """Test creating GPUX configuration."""
        parser = GPUXConfigParser()
        config = parser.parse_file(sample_gpuxfile)

        assert config.name == "test-model"
        assert config.version == "1.0.0"
        assert config.model.format == "onnx"
        assert len(config.inputs) == 1
        assert len(config.outputs) == 1

    def test_gpux_config_validation(self):
        """Test GPUX configuration validation."""
        # Valid config
        config = GPUXConfig(
            name="test",
            model=ModelConfig(source="model.onnx"),
            inputs=[InputConfig(name="input", type="float32", shape=[1, 10])],
            outputs=[OutputConfig(name="output", type="float32", shape=[1, 2])],
        )

        assert config.name == "test"
        assert len(config.inputs) == 1
        assert len(config.outputs) == 1

        # Invalid config - no inputs
        with pytest.raises(ValueError, match="Inputs are required"):
            GPUXConfig(
                name="test",
                model=ModelConfig(source="model.onnx"),
                inputs=[],
                outputs=[OutputConfig(name="output", type="float32", shape=[1, 2])],
            )

        # Invalid config - no outputs
        with pytest.raises(ValueError, match="Outputs are required"):
            GPUXConfig(
                name="test",
                model=ModelConfig(source="model.onnx"),
                inputs=[InputConfig(name="input", type="float32", shape=[1, 10])],
                outputs=[],
            )


class TestGPUXConfigParser:
    """Test cases for GPUXConfigParser class."""

    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = GPUXConfigParser()

        assert parser._config is None  # noqa: SLF001

    def test_parse_file(self, sample_gpuxfile):
        """Test parsing configuration file."""
        parser = GPUXConfigParser()
        config = parser.parse_file(sample_gpuxfile)

        assert config is not None
        assert config.name == "test-model"
        assert config.version == "1.0.0"
        assert parser._config == config  # noqa: SLF001

    def test_parse_file_nonexistent(self, temp_dir):
        """Test parsing non-existent file."""
        parser = GPUXConfigParser()
        nonexistent_file = temp_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            parser.parse_file(nonexistent_file)

    def test_parse_file_invalid_yaml(self, temp_dir):
        """Test parsing invalid YAML file."""
        parser = GPUXConfigParser()
        invalid_file = temp_dir / "invalid.yaml"

        with invalid_file.open("w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(ValueError, match="Invalid YAML"):
            parser.parse_file(invalid_file)

    def test_parse_string(self):
        """Test parsing configuration string."""
        config_str = """
name: test-model
version: 1.0.0
model:
  source: model.onnx
  format: onnx
inputs:
  input:
    type: float32
    shape: [1, 10]
outputs:
  output:
    type: float32
    shape: [1, 2]
"""

        parser = GPUXConfigParser()
        config = parser.parse_string(config_str)

        assert config is not None
        assert config.name == "test-model"
        assert config.version == "1.0.0"

    def test_normalize_config_data(self):
        """Test configuration data normalization."""
        parser = GPUXConfigParser()

        # Test inputs normalization
        data = {
            "name": "test",
            "model": {"source": "model.onnx", "format": "onnx"},
            "inputs": {
                "input1": {"type": "float32", "shape": [1, 10]},
                "input2": "float32",
            },
            "outputs": {
                "output1": {"type": "float32", "shape": [1, 2]},
                "output2": "float32",
            },
        }

        normalized = parser._normalize_config_data(data)  # noqa: SLF001

        assert isinstance(normalized["inputs"], list)
        assert len(normalized["inputs"]) == 2
        assert normalized["inputs"][0]["name"] == "input1"
        assert normalized["inputs"][1]["name"] == "input2"

        assert isinstance(normalized["outputs"], list)
        assert len(normalized["outputs"]) == 2
        assert normalized["outputs"][0]["name"] == "output1"
        assert normalized["outputs"][1]["name"] == "output2"

    def test_validate_model_path(self, sample_gpuxfile, simple_onnx_model, temp_dir):
        """Test model path validation."""
        parser = GPUXConfigParser()
        parser.parse_file(sample_gpuxfile)

        # Should be valid with the model file
        assert parser.validate_model_path(simple_onnx_model.parent)

        # Should be invalid with different directory
        assert not parser.validate_model_path(temp_dir)

    def test_get_model_path(self, sample_gpuxfile, simple_onnx_model):
        """Test getting model path."""
        parser = GPUXConfigParser()
        parser.parse_file(sample_gpuxfile)

        model_path = parser.get_model_path(simple_onnx_model.parent)

        assert model_path is not None
        assert model_path.name == simple_onnx_model.name

    def test_to_dict(self, sample_gpuxfile):
        """Test converting config to dictionary."""
        parser = GPUXConfigParser()
        parser.parse_file(sample_gpuxfile)

        config_dict = parser.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "test-model"
        assert config_dict["version"] == "1.0.0"

    def test_save(self, sample_gpuxfile, temp_dir):
        """Test saving configuration."""
        parser = GPUXConfigParser()
        config = parser.parse_file(sample_gpuxfile)

        output_file = temp_dir / "saved_config.yaml"
        parser.save(output_file)

        assert output_file.exists()

        # Verify the saved file can be parsed
        parser2 = GPUXConfigParser()
        config2 = parser2.parse_file(output_file)

        assert config2.name == config.name
        assert config2.version == config.version
