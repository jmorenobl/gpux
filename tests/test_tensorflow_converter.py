"""Tests for TensorFlow converter."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

try:
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    TENSORFLOW_AVAILABLE = False

from gpux.core.conversion.tensorflow import TensorFlowConverter
from gpux.core.conversion.optimizer import ConversionError
from gpux.core.managers.base import ModelMetadata


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
class TestTensorFlowConverter:
    """Test TensorFlow converter functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.converter = TensorFlowConverter(cache_dir=self.temp_dir)

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_can_convert_pytorch_model(self) -> None:
        """Test that converter correctly identifies PyTorch models."""
        metadata = ModelMetadata(
            registry="test",
            model_id="test-model",
            revision="main",
            format="pytorch",
            files={"model.pth": Path(self.temp_dir / "model.pth")},
            size_bytes=1024,
        )

        assert not self.converter.can_convert(metadata)

    def test_can_convert_tensorflow_model(self) -> None:
        """Test that converter correctly identifies TensorFlow models."""
        metadata = ModelMetadata(
            registry="test",
            model_id="test-model",
            revision="main",
            format="tensorflow",
            files={"model.h5": Path(self.temp_dir / "model.h5")},
            size_bytes=1024,
        )

        assert self.converter.can_convert(metadata)

    def test_can_convert_keras_model(self) -> None:
        """Test that converter correctly identifies Keras models."""
        metadata = ModelMetadata(
            registry="test",
            model_id="test-model",
            revision="main",
            format="keras",
            files={"model.h5": Path(self.temp_dir / "model.h5")},
            size_bytes=1024,
        )

        assert self.converter.can_convert(metadata)

    def test_can_convert_h5_model(self) -> None:
        """Test that converter correctly identifies H5 models."""
        metadata = ModelMetadata(
            registry="test",
            model_id="test-model",
            revision="main",
            format="h5",
            files={"model.h5": Path(self.temp_dir / "model.h5")},
            size_bytes=1024,
        )

        assert self.converter.can_convert(metadata)

    def test_cannot_convert_unsupported_format(self) -> None:
        """Test that converter rejects unsupported formats."""
        metadata = ModelMetadata(
            registry="test",
            model_id="test-model",
            revision="main",
            format="onnx",
            files={"model.onnx": Path(self.temp_dir / "model.onnx")},
            size_bytes=1024,
        )

        assert not self.converter.can_convert(metadata)

    def test_find_model_file_h5(self) -> None:
        """Test finding H5 model files."""
        metadata = ModelMetadata(
            registry="test",
            model_id="test-model",
            revision="main",
            format="tensorflow",
            files={"model.h5": Path(self.temp_dir / "model.h5")},
            size_bytes=1024,
        )

        model_file = self.converter._find_model_file(metadata)
        assert model_file == Path(self.temp_dir / "model.h5")

    def test_find_model_file_pb(self) -> None:
        """Test finding PB model files."""
        metadata = ModelMetadata(
            registry="test",
            model_id="test-model",
            revision="main",
            format="tensorflow",
            files={"model.pb": Path(self.temp_dir / "model.pb")},
            size_bytes=1024,
        )

        model_file = self.converter._find_model_file(metadata)
        assert model_file == Path(self.temp_dir / "model.pb")

    def test_find_model_file_savedmodel(self) -> None:
        """Test finding SavedModel directories."""
        savedmodel_dir = Path(self.temp_dir / "savedmodel")
        metadata = ModelMetadata(
            registry="test",
            model_id="test-model",
            revision="main",
            format="tensorflow",
            files={"savedmodel": savedmodel_dir},
            size_bytes=1024,
        )

        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(
                Path, "__truediv__", return_value=savedmodel_dir / "saved_model.pb"
            ),
            patch.object(Path, "exists", return_value=True),
        ):
            model_file = self.converter._find_model_file(metadata)
            assert model_file == savedmodel_dir

    def test_find_model_file_not_found(self) -> None:
        """Test when no model file is found."""
        metadata = ModelMetadata(
            registry="test",
            model_id="test-model",
            revision="main",
            format="tensorflow",
            files={"config.json": Path(self.temp_dir / "config.json")},
            size_bytes=1024,
        )

        model_file = self.converter._find_model_file(metadata)
        assert model_file is None

    def test_get_input_shapes_keras_model(self) -> None:
        """Test getting input shapes from Keras model."""
        # Create a simple Keras model
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(10, input_shape=(5,)), tf.keras.layers.Dense(1)]
        )

        metadata = ModelMetadata(
            registry="test",
            model_id="test-model",
            revision="main",
            format="keras",
            files={"model.h5": Path(self.temp_dir / "model.h5")},
            size_bytes=1024,
        )

        with (
            patch.object(
                self.converter,
                "_find_model_file",
                return_value=Path(self.temp_dir / "model.h5"),
            ),
            patch("tensorflow.keras.models.load_model", return_value=model),
        ):
            input_shapes = self.converter.get_input_shapes(metadata)
            assert "input_0" in input_shapes
            assert input_shapes["input_0"] == [None, 5]

    def test_get_output_shapes_keras_model(self) -> None:
        """Test getting output shapes from Keras model."""
        # Create a simple Keras model
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(10, input_shape=(5,)), tf.keras.layers.Dense(1)]
        )

        metadata = ModelMetadata(
            registry="test",
            model_id="test-model",
            revision="main",
            format="keras",
            files={"model.h5": Path(self.temp_dir / "model.h5")},
            size_bytes=1024,
        )

        with (
            patch.object(
                self.converter,
                "_find_model_file",
                return_value=Path(self.temp_dir / "model.h5"),
            ),
            patch("tensorflow.keras.models.load_model", return_value=model),
        ):
            output_shapes = self.converter.get_output_shapes(metadata)
            assert "output_0" in output_shapes
            assert output_shapes["output_0"] == [None, 1]

    def test_get_tf_input_shapes_keras(self) -> None:
        """Test getting input shapes from Keras model."""
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(10, input_shape=(5,)), tf.keras.layers.Dense(1)]
        )

        input_shapes = self.converter._get_tf_input_shapes(model)
        assert input_shapes is not None
        assert len(input_shapes) == 1
        assert input_shapes[0] == (None, 5)

    def test_get_tf_input_shapes_savedmodel(self) -> None:
        """Test getting input shapes from SavedModel."""
        # Mock SavedModel with proper attributes
        mock_shape1 = Mock()
        mock_shape1.as_list.return_value = [None, 10]
        mock_shape1.rank = 2

        mock_shape2 = Mock()
        mock_shape2.as_list.return_value = [None, 5]
        mock_shape2.rank = 2

        mock_input1 = Mock()
        mock_input1.shape = mock_shape1

        mock_input2 = Mock()
        mock_input2.shape = mock_shape2

        mock_signature = Mock()
        mock_signature.inputs = [mock_input1, mock_input2]

        mock_model = Mock()
        mock_model.signatures = {"serving_default": mock_signature}
        # Remove input_shape attribute entirely for SavedModel test
        del mock_model.input_shape

        input_shapes = self.converter._get_tf_input_shapes(mock_model)
        assert input_shapes is not None
        assert len(input_shapes) == 2
        assert input_shapes[0] == (None, 10)
        assert input_shapes[1] == (None, 5)

    def test_get_tf_input_shapes_no_signatures(self) -> None:
        """Test getting input shapes when no signatures available."""
        mock_model = Mock()
        mock_model.signatures = {}
        # Ensure input_shape is None so it goes to signatures
        mock_model.input_shape = None

        input_shapes = self.converter._get_tf_input_shapes(mock_model)
        assert input_shapes == []

    def test_convert_with_tf2onnx_missing_dependency(self) -> None:
        """Test conversion when tf2onnx is not available."""
        metadata = ModelMetadata(
            registry="test",
            model_id="test-model",
            revision="main",
            format="tensorflow",
            files={"model.h5": Path(self.temp_dir / "model.h5")},
            size_bytes=1024,
        )

        # Mock the tf2onnx import inside the method
        with patch(
            "builtins.__import__", side_effect=ImportError("tf2onnx not available")
        ):
            with pytest.raises(ConversionError) as exc_info:
                self.converter._convert_with_tf2onnx(
                    metadata, Path(self.temp_dir / "output.onnx")
                )

            assert "tf2onnx is required" in str(exc_info.value)

    def test_convert_with_tf2onnx_no_model_file(self) -> None:
        """Test conversion when no model file is found."""
        metadata = ModelMetadata(
            registry="test",
            model_id="test-model",
            revision="main",
            format="tensorflow",
            files={"config.json": Path(self.temp_dir / "config.json")},
            size_bytes=1024,
        )

        with pytest.raises(ConversionError) as exc_info:
            self.converter._convert_with_tf2onnx(
                metadata, Path(self.temp_dir / "output.onnx")
            )

        assert "No TensorFlow model file found" in str(exc_info.value)

    def test_convert_with_manual_fallback(self) -> None:
        """Test manual conversion fallback."""
        metadata = ModelMetadata(
            registry="test",
            model_id="test-model",
            revision="main",
            format="tensorflow",
            files={"model.h5": Path(self.temp_dir / "model.h5")},
            size_bytes=1024,
        )

        # Mock tf2onnx import to fail, which should trigger the manual conversion error
        with (
            patch(
                "builtins.__import__", side_effect=ImportError("tf2onnx not available")
            ),
            pytest.raises(ConversionError) as exc_info,
        ):
            self.converter.convert(metadata, Path(self.temp_dir / "output.onnx"))

        assert "Manual TensorFlow to ONNX conversion not implemented" in str(
            exc_info.value
        )

    def test_convert_invalid_format(self) -> None:
        """Test conversion with invalid format."""
        metadata = ModelMetadata(
            registry="test",
            model_id="test-model",
            revision="main",
            format="pytorch",
            files={"model.pth": Path(self.temp_dir / "model.pth")},
            size_bytes=1024,
        )

        with pytest.raises(ConversionError) as exc_info:
            self.converter.convert(metadata, Path(self.temp_dir / "output.onnx"))

        assert "Cannot convert model format" in str(exc_info.value)

    def test_conversion_info(self) -> None:
        """Test conversion info generation."""
        metadata = ModelMetadata(
            registry="test",
            model_id="test-model",
            revision="main",
            format="tensorflow",
            files={"model.h5": Path(self.temp_dir / "model.h5")},
            size_bytes=1024,
        )

        info = self.converter.get_conversion_info(metadata)
        assert info["converter"] == "TensorFlowConverter"
        assert info["model_id"] == "test-model"
        assert info["registry"] == "test"
        assert info["original_format"] == "tensorflow"
        assert info["target_format"] == "onnx"
