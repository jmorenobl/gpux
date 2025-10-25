"""Tests for Hugging Face manager."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from gpux.core.managers.huggingface import HuggingFaceManager
from gpux.core.managers.exceptions import (
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
)


class TestHuggingFaceManager:
    """Test HuggingFaceManager implementation."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        manager = HuggingFaceManager()

        assert manager.config.name == "huggingface"
        assert manager.config.api_url == "https://huggingface.co"
        assert manager.api is not None
        assert manager.console is not None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        from gpux.core.managers.base import RegistryConfig

        config = RegistryConfig(
            name="custom-hf",
            api_url="https://custom-hf.com",
            auth_token="custom-token",  # noqa: S106
        )

        manager = HuggingFaceManager(config)

        assert manager.config == config
        assert manager.api.token == "custom-token"  # noqa: S105

    @patch("gpux.core.managers.huggingface.HfApi")
    def test_init_with_env_token(self, mock_hf_api):
        """Test initialization with HF_TOKEN environment variable."""
        with patch.dict("os.environ", {"HF_TOKEN": "env-token"}):
            manager = HuggingFaceManager()

            assert manager.config.auth_token == "env-token"  # noqa: S105
            mock_hf_api.assert_called_once_with(token="env-token")  # noqa: S106

    @patch("gpux.core.managers.huggingface.snapshot_download")
    @patch("gpux.core.managers.huggingface.HfApi")
    def test_pull_model_success(self, mock_hf_api, mock_snapshot_download):
        """Test successful model pull."""
        # Mock API response
        mock_model_info = Mock()
        mock_model_info.cardData = {"description": "Test model"}
        mock_model_info.tags = ["text-generation"]
        mock_model_info.pipeline_tag = "text-generation"
        mock_model_info.library_name = "transformers"
        mock_model_info.downloads = 1000
        mock_model_info.last_modified = "2023-01-01T00:00:00Z"

        mock_api_instance = Mock()
        mock_api_instance.model_info.return_value = mock_model_info
        mock_hf_api.return_value = mock_api_instance

        # Mock download
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()

            # Create dummy model files
            (model_path / "config.json").write_text('{"model_type": "gpt2"}')
            (model_path / "pytorch_model.bin").write_text("dummy model data")

            mock_snapshot_download.return_value = str(model_path)

            manager = HuggingFaceManager()
            metadata = manager.pull_model("microsoft/DialoGPT-medium")

            assert metadata.registry == "huggingface"
            assert metadata.model_id == "microsoft/DialoGPT-medium"
            assert metadata.format == "pytorch"
            assert metadata.description == "Test model"
            assert metadata.tags == ["text-generation"]
            assert "config.json" in metadata.files
            assert "pytorch_model.bin" in metadata.files

    @patch("gpux.core.managers.huggingface.HfApi")
    def test_pull_model_not_found(self, mock_hf_api):
        """Test model not found error."""
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_api_instance = Mock()
        mock_api_instance.model_info.side_effect = RepositoryNotFoundError("Not found")
        mock_hf_api.return_value = mock_api_instance

        manager = HuggingFaceManager()

        with pytest.raises(ModelNotFoundError, match="Model not found"):
            manager.pull_model("nonexistent/model")

    @patch("gpux.core.managers.huggingface.snapshot_download")
    @patch("gpux.core.managers.huggingface.HfApi")
    def test_pull_model_authentication_error(self, mock_hf_api, mock_snapshot_download):
        """Test authentication error during pull."""
        mock_model_info = Mock()
        mock_model_info.cardData = {}
        mock_model_info.tags = []

        mock_api_instance = Mock()
        mock_api_instance.model_info.return_value = mock_model_info
        mock_hf_api.return_value = mock_api_instance

        mock_snapshot_download.side_effect = Exception("401 Unauthorized")

        manager = HuggingFaceManager()

        with pytest.raises(AuthenticationError, match="Authentication failed"):
            manager.pull_model("private/model")

    @patch("gpux.core.managers.huggingface.snapshot_download")
    @patch("gpux.core.managers.huggingface.HfApi")
    def test_pull_model_network_error(self, mock_hf_api, mock_snapshot_download):
        """Test network error during pull."""
        mock_model_info = Mock()
        mock_model_info.cardData = {}
        mock_model_info.tags = []

        mock_api_instance = Mock()
        mock_api_instance.model_info.return_value = mock_model_info
        mock_hf_api.return_value = mock_api_instance

        mock_snapshot_download.side_effect = Exception("Network connection failed")

        manager = HuggingFaceManager()

        with pytest.raises(NetworkError, match="Network error"):
            manager.pull_model("some/model")

    @patch("gpux.core.managers.huggingface.HfApi")
    def test_search_models_success(self, mock_hf_api):
        """Test successful model search."""
        # Mock search results
        mock_model1 = Mock()
        mock_model1.modelId = "microsoft/DialoGPT-medium"
        mock_model1.sha = "main"
        mock_model1.cardData = {"description": "DialoGPT model"}
        mock_model1.tags = ["text-generation"]
        mock_model1.pipeline_tag = "text-generation"
        mock_model1.library_name = "transformers"
        mock_model1.downloads = 1000
        mock_model1.last_modified = "2023-01-01T00:00:00Z"
        mock_model1.siblings = [
            Mock(rfilename="config.json"),
            Mock(rfilename="pytorch_model.bin"),
        ]

        mock_model2 = Mock()
        mock_model2.modelId = "gpt2"
        mock_model2.sha = "main"
        mock_model2.cardData = {"description": "GPT-2 model"}
        mock_model2.tags = ["text-generation"]
        mock_model2.pipeline_tag = "text-generation"
        mock_model2.library_name = "transformers"
        mock_model2.downloads = 5000
        mock_model2.last_modified = "2023-01-02T00:00:00Z"
        mock_model2.siblings = [
            Mock(rfilename="config.json"),
            Mock(rfilename="pytorch_model.bin"),
        ]

        mock_api_instance = Mock()
        mock_api_instance.list_models.return_value = [mock_model1, mock_model2]
        mock_hf_api.return_value = mock_api_instance

        manager = HuggingFaceManager()
        results = manager.search_models("gpt", limit=2)

        assert len(results) == 2
        assert results[0].model_id == "microsoft/DialoGPT-medium"
        assert results[1].model_id == "gpt2"
        assert all(r.format == "pytorch" for r in results)

    @patch("gpux.core.managers.huggingface.HfApi")
    def test_search_models_authentication_error(self, mock_hf_api):
        """Test authentication error during search."""
        mock_api_instance = Mock()
        mock_api_instance.list_models.side_effect = Exception("401 Unauthorized")
        mock_hf_api.return_value = mock_api_instance

        manager = HuggingFaceManager()

        with pytest.raises(AuthenticationError, match="Authentication failed"):
            manager.search_models("gpt")

    @patch("gpux.core.managers.huggingface.HfApi")
    def test_get_model_info_success(self, mock_hf_api):
        """Test successful model info retrieval."""
        mock_model_info = Mock()
        mock_model_info.modelId = "microsoft/DialoGPT-medium"
        mock_model_info.sha = "main"
        mock_model_info.cardData = {"description": "DialoGPT model"}
        mock_model_info.tags = ["text-generation"]
        mock_model_info.pipeline_tag = "text-generation"
        mock_model_info.library_name = "transformers"
        mock_model_info.downloads = 1000
        mock_model_info.last_modified = "2023-01-01T00:00:00Z"
        mock_model_info.siblings = [
            Mock(rfilename="config.json"),
            Mock(rfilename="pytorch_model.bin"),
        ]

        mock_api_instance = Mock()
        mock_api_instance.model_info.return_value = mock_model_info
        mock_hf_api.return_value = mock_api_instance

        manager = HuggingFaceManager()
        metadata = manager.get_model_info("microsoft/DialoGPT-medium")

        assert metadata.registry == "huggingface"
        assert metadata.model_id == "microsoft/DialoGPT-medium"
        assert metadata.format == "pytorch"
        assert metadata.description == "DialoGPT model"
        assert metadata.tags == ["text-generation"]

    @patch("gpux.core.managers.huggingface.HfApi")
    def test_get_model_info_not_found(self, mock_hf_api):
        """Test model not found error."""
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_api_instance = Mock()
        mock_api_instance.model_info.side_effect = RepositoryNotFoundError("Not found")
        mock_hf_api.return_value = mock_api_instance

        manager = HuggingFaceManager()

        with pytest.raises(ModelNotFoundError, match="Model not found"):
            manager.get_model_info("nonexistent/model")

    @patch("gpux.core.managers.huggingface.HfApi")
    def test_list_model_files_success(self, mock_hf_api):
        """Test successful file listing."""
        mock_file1 = Mock()
        mock_file1.rfilename = "config.json"
        mock_file2 = Mock()
        mock_file2.rfilename = "pytorch_model.bin"

        mock_model_info = Mock()
        mock_model_info.siblings = [mock_file1, mock_file2]

        mock_api_instance = Mock()
        mock_api_instance.model_info.return_value = mock_model_info
        mock_hf_api.return_value = mock_api_instance

        manager = HuggingFaceManager()
        files = manager.list_model_files("microsoft/DialoGPT-medium")

        assert files == ["config.json", "pytorch_model.bin"]

    @patch("gpux.core.managers.huggingface.HfApi")
    def test_list_model_files_not_found(self, mock_hf_api):
        """Test file listing for non-existent model."""
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_api_instance = Mock()
        mock_api_instance.model_info.side_effect = RepositoryNotFoundError("Not found")
        mock_hf_api.return_value = mock_api_instance

        manager = HuggingFaceManager()

        with pytest.raises(ModelNotFoundError, match="Model not found"):
            manager.list_model_files("nonexistent/model")

    def test_detect_model_format_pytorch(self):
        """Test PyTorch model format detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            (model_path / "config.json").write_text("{}")
            (model_path / "pytorch_model.bin").write_text("dummy")

            manager = HuggingFaceManager()
            format_type = manager._detect_model_format(model_path)

            assert format_type == "pytorch"

    def test_detect_model_format_tensorflow(self):
        """Test TensorFlow model format detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            (model_path / "config.json").write_text("{}")
            (model_path / "tf_model.h5").write_text("dummy")

            manager = HuggingFaceManager()
            format_type = manager._detect_model_format(model_path)

            assert format_type == "tensorflow"

    def test_detect_model_format_onnx(self):
        """Test ONNX model format detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            (model_path / "model.onnx").write_text("dummy")

            manager = HuggingFaceManager()
            format_type = manager._detect_model_format(model_path)

            assert format_type == "onnx"

    def test_detect_model_format_unknown(self):
        """Test unknown model format detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            (model_path / "config.json").write_text("{}")
            (model_path / "other_file.txt").write_text("dummy")

            manager = HuggingFaceManager()
            format_type = manager._detect_model_format(model_path)

            assert format_type == "unknown"

    def test_extract_model_metadata(self):
        """Test metadata extraction from downloaded model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            (model_path / "config.json").write_text('{"model_type": "gpt2"}')
            (model_path / "pytorch_model.bin").write_text("dummy model data")

            mock_model_info = Mock()
            mock_model_info.cardData = {"description": "Test model"}
            mock_model_info.tags = ["text-generation"]
            mock_model_info.pipeline_tag = "text-generation"
            mock_model_info.library_name = "transformers"
            mock_model_info.downloads = 1000
            mock_model_info.last_modified = "2023-01-01T00:00:00Z"

            manager = HuggingFaceManager()
            metadata = manager._extract_model_metadata(
                "test/model", "main", model_path, mock_model_info
            )

            assert metadata.registry == "huggingface"
            assert metadata.model_id == "test/model"
            assert metadata.revision == "main"
            assert metadata.format == "pytorch"
            assert metadata.description == "Test model"
            assert metadata.tags == ["text-generation"]
            assert metadata.size_bytes > 0
            assert "config.json" in metadata.files
            assert "pytorch_model.bin" in metadata.files

    def test_create_metadata_from_model_info(self):
        """Test metadata creation from model info."""
        mock_file1 = Mock()
        mock_file1.rfilename = "config.json"
        mock_file2 = Mock()
        mock_file2.rfilename = "pytorch_model.bin"

        mock_model_info = Mock()
        mock_model_info.modelId = "test/model"
        mock_model_info.sha = "main"
        mock_model_info.cardData = {"description": "Test model"}
        mock_model_info.tags = ["text-generation"]
        mock_model_info.pipeline_tag = "text-generation"
        mock_model_info.library_name = "transformers"
        mock_model_info.downloads = 1000
        mock_model_info.last_modified = "2023-01-01T00:00:00Z"
        mock_model_info.siblings = [mock_file1, mock_file2]

        manager = HuggingFaceManager()
        metadata = manager._create_metadata_from_model_info(mock_model_info)

        assert metadata.registry == "huggingface"
        assert metadata.model_id == "test/model"
        assert metadata.revision == "main"
        assert metadata.format == "pytorch"
        assert metadata.description == "Test model"
        assert metadata.tags == ["text-generation"]
        assert metadata.size_bytes == 0  # Size unknown without downloading
        assert "config.json" in metadata.files
        assert "pytorch_model.bin" in metadata.files
