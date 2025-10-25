"""Tests for the ONNX conversion pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from gpux.core.conversion import (
    ConfigGenerator,
    ModelOptimizer,
    PyTorchConverter,
)
from gpux.core.conversion.optimizer import ConversionError
from gpux.core.managers.base import ModelMetadata


class TestONNXConverter:
    """Tests for the ONNXConverter base class."""

    def test_init(self):
        """Test converter initialization."""
        converter = PyTorchConverter()
        assert converter.cache_dir.name == "converted"
        assert converter.cache_dir.parent.name == "models"

    def test_init_custom_cache_dir(self):
        """Test converter initialization with custom cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "custom_cache"
            converter = PyTorchConverter(cache_dir)
            assert converter.cache_dir == cache_dir

    def test_get_conversion_info(self):
        """Test getting conversion information."""
        converter = PyTorchConverter()
        metadata = ModelMetadata(
            registry="huggingface",
            model_id="test/model",
            revision="main",
            format="pytorch",
            files={},
            size_bytes=1000000,
        )
        info = converter.get_conversion_info(metadata)
        assert info["converter"] == "PyTorchConverter"
        assert info["model_id"] == "test/model"
        assert info["registry"] == "huggingface"
        assert info["original_format"] == "pytorch"
        assert info["target_format"] == "onnx"


class TestPyTorchConverter:
    """Tests for the PyTorchConverter."""

    @pytest.fixture
    def converter(self):
        """Create a PyTorchConverter instance."""
        return PyTorchConverter()

    @pytest.fixture
    def pytorch_metadata(self):
        """Create PyTorch model metadata."""
        return ModelMetadata(
            registry="huggingface",
            model_id="test/pytorch-model",
            revision="main",
            format="pytorch",
            files={
                "config.json": Path("/tmp/config.json"),  # noqa: S108
                "pytorch_model.bin": Path("/tmp/pytorch_model.bin"),  # noqa: S108
            },
            size_bytes=1000000,
        )

    @pytest.fixture
    def safetensors_metadata(self):
        """Create SafeTensors model metadata."""
        return ModelMetadata(
            registry="huggingface",
            model_id="test/safetensors-model",
            revision="main",
            format="safetensors",
            files={
                "config.json": Path("/tmp/config.json"),  # noqa: S108
                "model.safetensors": Path("/tmp/model.safetensors"),  # noqa: S108
            },
            size_bytes=1000000,
        )

    @pytest.fixture
    def non_pytorch_metadata(self):
        """Create non-PyTorch model metadata."""
        return ModelMetadata(
            registry="huggingface",
            model_id="test/tensorflow-model",
            revision="main",
            format="tensorflow",
            files={
                "config.json": Path("/tmp/config.json"),  # noqa: S108
                "tf_model.h5": Path("/tmp/tf_model.h5"),  # noqa: S108
            },
            size_bytes=1000000,
        )

    def test_can_convert_pytorch(self, converter, pytorch_metadata):
        """Test PyTorch model conversion capability."""
        assert converter.can_convert(pytorch_metadata)

    def test_can_convert_safetensors(self, converter, safetensors_metadata):
        """Test SafeTensors model conversion capability."""
        assert converter.can_convert(safetensors_metadata)

    def test_can_convert_non_pytorch(self, converter, non_pytorch_metadata):
        """Test non-PyTorch model conversion capability."""
        assert not converter.can_convert(non_pytorch_metadata)

    def test_get_input_shapes_from_config(self, converter, pytorch_metadata):
        """Test input shape extraction from config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                '{"model_type": "bert", "max_position_embeddings": 512, "vocab_size": 30000}'  # noqa: E501
            )

            # Update metadata with real path
            pytorch_metadata.files["config.json"] = config_path

            shapes = converter.get_input_shapes(pytorch_metadata)
            assert "input_ids" in shapes
            assert "attention_mask" in shapes
            assert shapes["input_ids"] == [1, 512]

    def test_get_input_shapes_no_config(self, converter, pytorch_metadata):
        """Test input shape extraction without config."""
        # Remove config.json from files
        pytorch_metadata.files = {"pytorch_model.bin": Path("/tmp/model.bin")}  # noqa: S108

        shapes = converter.get_input_shapes(pytorch_metadata)
        assert shapes == {}

    def test_get_output_shapes(self, converter, pytorch_metadata):
        """Test output shape extraction."""
        # Mock input shapes
        with patch.object(
            converter, "get_input_shapes", return_value={"input_ids": [1, 128]}
        ):
            shapes = converter.get_output_shapes(pytorch_metadata)
            assert "last_hidden_state" in shapes
            assert "pooler_output" in shapes
            assert shapes["last_hidden_state"] == [1, 128, 768]

    def test_get_output_shapes_no_inputs(self, converter, pytorch_metadata):
        """Test output shape extraction without input shapes."""
        with patch.object(converter, "get_input_shapes", return_value={}):
            shapes = converter.get_output_shapes(pytorch_metadata)
            assert shapes == {}

    def test_extract_input_shapes_from_config(self, converter):
        """Test input shape extraction from config object."""
        config = MagicMock()
        config.max_position_embeddings = 256
        config.vocab_size = 10000
        config.type_vocab_size = 2

        shapes = converter._extract_input_shapes_from_config(config)
        assert "input_ids" in shapes
        assert "attention_mask" in shapes
        assert "token_type_ids" in shapes
        assert shapes["input_ids"] == [1, 256]

    def test_get_default_input_shapes(self, converter, pytorch_metadata):
        """Test default input shape generation."""
        shapes = converter._get_default_input_shapes(pytorch_metadata, None)
        assert "input_ids" in shapes
        assert "attention_mask" in shapes
        assert shapes["input_ids"] == [1, 128]

    def test_create_dummy_inputs(self, converter):
        """Test dummy input creation."""
        input_shapes = {
            "input_ids": [1, 128],
            "attention_mask": [1, 128],
            "token_type_ids": [1, 128],
        }

        dummy_inputs = converter._create_dummy_inputs(input_shapes, None)
        assert len(dummy_inputs) == 3
        assert all(isinstance(tensor, torch.Tensor) for tensor in dummy_inputs)
        assert dummy_inputs[0].shape == (1, 128)  # input_ids
        assert dummy_inputs[1].shape == (1, 128)  # attention_mask
        assert dummy_inputs[2].shape == (1, 128)  # token_type_ids

    def test_get_output_names(self, converter, pytorch_metadata):
        """Test output name generation."""
        names = converter._get_output_names(pytorch_metadata)
        assert "last_hidden_state" in names
        assert "pooler_output" in names

    def test_get_dynamic_axes(self, converter):
        """Test dynamic axes generation."""
        input_shapes = {
            "input_ids": [1, 128],
            "attention_mask": [1, 128],
            "single_dim": [1],
        }

        dynamic_axes = converter._get_dynamic_axes(input_shapes)
        assert "input_ids" in dynamic_axes
        assert "attention_mask" in dynamic_axes
        assert "single_dim" not in dynamic_axes
        assert dynamic_axes["input_ids"] == {0: "batch_size", 1: "sequence_length"}

    @patch("gpux.core.conversion.pytorch.AutoModel")
    @patch("gpux.core.conversion.pytorch.AutoTokenizer")
    def test_convert_with_torch_success(
        self, mock_tokenizer, mock_model, converter, pytorch_metadata
    ):
        """Test successful conversion with torch.onnx.export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup mocks
            mock_model_instance = MagicMock()
            mock_model_instance.eval.return_value = None
            mock_model.from_pretrained.return_value = mock_model_instance

            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            # Create dummy model files
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            config_path = model_dir / "config.json"
            config_path.write_text(
                '{"model_type": "bert", "max_position_embeddings": 512, "vocab_size": 30000}'  # noqa: E501
            )

            # Update metadata
            pytorch_metadata.files["config.json"] = config_path

            # Mock torch.onnx.export and model validation
            with (
                patch("torch.onnx.export") as mock_export,
                patch(
                    "gpux.core.conversion.optimizer.ModelOptimizer.validate_model"
                ) as mock_validate,
                patch(
                    "gpux.core.conversion.optimizer.ModelOptimizer.optimize_model"
                ) as mock_optimize,
            ):

                def create_dummy_onnx(*args, **kwargs):  # noqa: ARG001
                    output_path = args[2]  # Third argument is the output path
                    Path(output_path).write_text("dummy onnx content")

                mock_export.side_effect = create_dummy_onnx
                mock_validate.return_value = True
                mock_optimize.return_value = Path(tmpdir) / "model.onnx"

                output_path = converter.convert(
                    pytorch_metadata, Path(tmpdir) / "model.onnx"
                )

                mock_export.assert_called_once()
                mock_validate.assert_called_once()
                mock_optimize.assert_called_once()
                assert output_path.name == "model.onnx"

    def test_convert_cannot_convert(self, converter, non_pytorch_metadata):
        """Test conversion failure for unsupported format."""
        with pytest.raises(ConversionError, match="Cannot convert model format"):
            converter.convert(non_pytorch_metadata)

    @patch("gpux.core.conversion.pytorch.AutoModel")
    def test_convert_with_torch_model_not_found(
        self,
        mock_model,  # noqa: ARG002
        converter,
        pytorch_metadata,
    ):
        """Test conversion failure when model files not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create non-existent model path
            pytorch_metadata.files["config.json"] = (
                Path(tmpdir) / "nonexistent" / "config.json"
            )

            with pytest.raises(ConversionError, match="Model path not found"):
                converter.convert(pytorch_metadata, Path(tmpdir) / "model.onnx")


class TestModelOptimizer:
    """Tests for the ModelOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create a ModelOptimizer instance."""
        return ModelOptimizer()

    def test_init(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer is not None

    def test_optimize_model_basic(self, optimizer):
        """Test basic model optimization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple ONNX model for testing
            model_path = Path(tmpdir) / "test.onnx"
            self._create_simple_onnx_model(model_path)

            output_path = optimizer.optimize_model(
                model_path, optimization_level="basic"
            )
            assert output_path.exists()
            assert output_path.name == "test_optimized.onnx"

    def test_optimize_model_custom_output(self, optimizer):
        """Test model optimization with custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test.onnx"
            self._create_simple_onnx_model(model_path)

            custom_output = Path(tmpdir) / "custom_optimized.onnx"
            output_path = optimizer.optimize_model(model_path, custom_output)
            assert output_path == custom_output
            assert custom_output.exists()

    def test_optimize_model_invalid_path(self, optimizer):
        """Test optimization with invalid model path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "nonexistent.onnx"

            with pytest.raises(ConversionError, match="Model optimization failed"):
                optimizer.optimize_model(invalid_path)

    def test_validate_model_valid(self, optimizer):
        """Test validation of valid model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test.onnx"
            self._create_simple_onnx_model(model_path)

            result = optimizer.validate_model(model_path)
            assert result is True

    def test_validate_model_invalid(self, optimizer):
        """Test validation of invalid model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "invalid.onnx"
            invalid_path.write_text("not a valid onnx model")

            with pytest.raises(ConversionError, match="Model validation failed"):
                optimizer.validate_model(invalid_path)

    def test_get_model_info(self, optimizer):
        """Test getting model information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test.onnx"
            self._create_simple_onnx_model(model_path)

            info = optimizer.get_model_info(model_path)
            assert "ir_version" in info
            assert "opset_version" in info
            assert "producer_name" in info
            assert "input_count" in info
            assert "output_count" in info

    def test_get_model_info_invalid(self, optimizer):
        """Test getting model info for invalid model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "invalid.onnx"
            invalid_path.write_text("not a valid onnx model")

            with pytest.raises(ConversionError, match="Failed to get model info"):
                optimizer.get_model_info(invalid_path)

    def _create_simple_onnx_model(self, path: Path) -> None:
        """Create a simple ONNX model for testing."""
        import onnx
        from onnx import TensorProto, helper

        # Create a simple graph
        input_tensor = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, 3, 224, 224]
        )
        output_tensor = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 1000]
        )

        # Create a simple identity node
        identity_node = helper.make_node(
            "Identity", ["input"], ["output"], name="identity"
        )

        # Create the graph
        graph = helper.make_graph(
            [identity_node], "simple_model", [input_tensor], [output_tensor]
        )

        # Create the model
        model = helper.make_model(graph)
        model.opset_import[0].version = 11

        # Save the model
        onnx.save(model, str(path))


class TestConfigGenerator:
    """Tests for the ConfigGenerator."""

    @pytest.fixture
    def generator(self):
        """Create a ConfigGenerator instance."""
        return ConfigGenerator()

    @pytest.fixture
    def metadata(self):
        """Create model metadata."""
        return ModelMetadata(
            registry="huggingface",
            model_id="test/model",
            revision="main",
            format="pytorch",
            files={},
            size_bytes=1000000,
            description="A test model",
            tags=["text-classification", "bert"],
        )

    @pytest.fixture
    def model_info(self):
        """Create mock model info."""
        return {
            "inputs": [
                {"name": "input_ids", "shape": [1, 128], "type": "tensor(int64)"},
                {"name": "attention_mask", "shape": [1, 128], "type": "tensor(int64)"},
            ],
            "outputs": [
                {
                    "name": "last_hidden_state",
                    "shape": [1, 128, 768],
                    "type": "tensor(float32)",
                },
                {"name": "pooler_output", "shape": [1, 768], "type": "tensor(float32)"},
            ],
            "opset_version": 11,
            "ir_version": 8,
            "producer_name": "test-producer",
        }

    def test_generate_config_success(self, generator, metadata, model_info):
        """Test successful config generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            onnx_path.write_text("dummy onnx content")

            with patch(
                "gpux.core.conversion.config_generator.ModelInspector"
            ) as mock_inspector:
                mock_inspector_instance = MagicMock()
                mock_inspector_instance.inspect.return_value = model_info
                mock_inspector.return_value = mock_inspector_instance

                config_path = generator.generate_config(metadata, onnx_path)

                assert config_path.exists()
                assert config_path.name == "gpux.yml"

                # Check config content
                import yaml

                with config_path.open() as f:
                    config = yaml.safe_load(f)

                assert config["name"] == "test-model"
                assert config["model"]["format"] == "onnx"
                assert "input_ids" in config["inputs"]
                assert "attention_mask" in config["inputs"]
                assert "last_hidden_state" in config["outputs"]

    def test_generate_config_onnx_not_found(self, generator, metadata, model_info):  # noqa: ARG002
        """Test config generation with non-existent ONNX file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "nonexistent.onnx"

            with pytest.raises(ConversionError, match="ONNX model not found"):
                generator.generate_config(metadata, onnx_path)

    def test_generate_config_custom_output(self, generator, metadata, model_info):
        """Test config generation with custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            onnx_path.write_text("dummy onnx content")

            custom_output = Path(tmpdir) / "custom_config.yml"

            with patch(
                "gpux.core.conversion.config_generator.ModelInspector"
            ) as mock_inspector:
                mock_inspector_instance = MagicMock()
                mock_inspector_instance.inspect.return_value = model_info
                mock_inspector.return_value = mock_inspector_instance

                config_path = generator.generate_config(
                    metadata, onnx_path, custom_output
                )

                assert config_path == custom_output
                assert custom_output.exists()

    def test_create_config_dict(self, generator, metadata, model_info):
        """Test config dictionary creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            onnx_path.write_text("dummy onnx content")

            config = generator._create_config_dict(metadata, model_info, onnx_path)

            assert config["name"] == "test-model"
            assert config["version"] == "1.0.0"
            assert config["description"] == "A test model"
            assert config["model"]["format"] == "onnx"
            assert config["tags"] == ["text-classification", "bert"]
            assert config["metadata"]["original_model_id"] == "test/model"
            assert config["metadata"]["original_registry"] == "huggingface"

    def test_extract_inputs(self, generator, model_info):
        """Test input extraction from model info."""
        inputs = generator._extract_inputs(model_info)

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert inputs["input_ids"]["type"] == "int64"
        assert inputs["input_ids"]["shape"] == [1, 128]
        assert inputs["input_ids"]["required"] is True

    def test_extract_inputs_fallback(self, generator):
        """Test input extraction fallback."""
        mock_info = MagicMock()
        mock_info.inputs = []

        inputs = generator._extract_inputs(mock_info)

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert inputs["input_ids"]["type"] == "int64"

    def test_extract_outputs(self, generator, model_info):
        """Test output extraction from model info."""
        outputs = generator._extract_outputs(model_info)

        assert "last_hidden_state" in outputs
        assert "pooler_output" in outputs
        assert outputs["last_hidden_state"]["type"] == "float32"
        assert outputs["last_hidden_state"]["shape"] == [1, 128, 768]

    def test_extract_outputs_fallback(self, generator):
        """Test output extraction fallback."""
        mock_info = MagicMock()
        mock_info.outputs = []

        outputs = generator._extract_outputs(mock_info)

        assert "last_hidden_state" in outputs
        assert "pooler_output" in outputs

    def test_convert_onnx_type_to_numpy(self, generator):
        """Test ONNX type to NumPy type conversion."""
        assert generator._convert_onnx_type_to_numpy("tensor(float)") == "float32"
        assert generator._convert_onnx_type_to_numpy("tensor(double)") == "float64"
        assert generator._convert_onnx_type_to_numpy("tensor(int32)") == "int32"
        assert generator._convert_onnx_type_to_numpy("tensor(int64)") == "int64"
        assert generator._convert_onnx_type_to_numpy("tensor(bool)") == "bool"
        assert generator._convert_onnx_type_to_numpy("tensor(string)") == "str"
        assert generator._convert_onnx_type_to_numpy("unknown_type") == "float32"

    def test_update_existing_config(self, generator, metadata, model_info):
        """Test updating existing configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "gpux.yml"
            onnx_path = Path(tmpdir) / "model.onnx"
            onnx_path.write_text("dummy onnx content")

            # Create existing config
            existing_config = {
                "name": "existing-model",
                "version": "0.1.0",
                "model": {"source": "old_model.onnx", "format": "pytorch"},
                "metadata": {"existing_key": "existing_value"},
            }

            import yaml

            with config_path.open("w") as f:
                yaml.dump(existing_config, f)

            with patch(
                "gpux.core.conversion.config_generator.ModelInspector"
            ) as mock_inspector:
                mock_inspector_instance = MagicMock()
                mock_inspector_instance.inspect.return_value = model_info
                mock_inspector.return_value = mock_inspector_instance

                updated_path = generator.update_existing_config(
                    config_path, metadata, onnx_path
                )

                assert updated_path == config_path

                # Check updated config
                with config_path.open() as f:
                    updated_config = yaml.safe_load(f)

                assert updated_config["name"] == "existing-model"  # Preserved
                assert updated_config["model"]["format"] == "onnx"  # Updated
                assert (
                    updated_config["metadata"]["existing_key"] == "existing_value"
                )  # Preserved
                assert (
                    updated_config["metadata"]["original_model_id"] == "test/model"
                )  # Added

    def test_update_existing_config_not_exists(self, generator, metadata, model_info):
        """Test updating non-existent configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nonexistent.yml"
            onnx_path = Path(tmpdir) / "model.onnx"
            onnx_path.write_text("dummy onnx content")

            with patch(
                "gpux.core.conversion.config_generator.ModelInspector"
            ) as mock_inspector:
                mock_inspector_instance = MagicMock()
                mock_inspector_instance.inspect.return_value = model_info
                mock_inspector.return_value = mock_inspector_instance

                # Should create new config
                updated_path = generator.update_existing_config(
                    config_path, metadata, onnx_path
                )

                assert updated_path == config_path
                assert config_path.exists()
