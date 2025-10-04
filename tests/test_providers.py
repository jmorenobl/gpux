"""Tests for execution provider functionality."""

from gpux.core.providers import ExecutionProvider, ProviderManager


class TestExecutionProvider:
    """Test cases for ExecutionProvider enum."""

    def test_provider_values(self):
        """Test provider enum values."""
        assert ExecutionProvider.TENSORRT.value == "TensorrtExecutionProvider"
        assert ExecutionProvider.CUDA.value == "CUDAExecutionProvider"
        assert ExecutionProvider.ROCM.value == "ROCmExecutionProvider"
        assert ExecutionProvider.COREML.value == "CoreMLExecutionProvider"
        assert ExecutionProvider.DIRECTML.value == "DirectMLExecutionProvider"
        assert ExecutionProvider.OPENVINO.value == "OpenVINOExecutionProvider"
        assert ExecutionProvider.CPU.value == "CPUExecutionProvider"


class TestProviderManager:
    """Test cases for ProviderManager class."""

    def test_provider_manager_initialization(self):
        """Test provider manager initialization."""
        manager = ProviderManager()

        assert manager._available_providers is not None
        assert len(manager._available_providers) > 0
        assert "CPUExecutionProvider" in manager._available_providers
        assert manager._provider_priority is not None
        assert len(manager._provider_priority) > 0

    def test_get_best_provider_default(self, provider_manager):
        """Test getting best provider with default settings."""
        provider = provider_manager.get_best_provider()

        assert provider is not None
        assert isinstance(provider, ExecutionProvider)
        assert provider_manager._is_provider_available(provider)

    def test_get_best_provider_preferred(self, provider_manager):
        """Test getting best provider with preferred provider."""
        # Test with CPU provider (should always be available)
        provider = provider_manager.get_best_provider("cpu")

        assert provider is not None
        assert provider == ExecutionProvider.CPU

    def test_get_best_provider_invalid_preferred(self, provider_manager):
        """Test getting best provider with invalid preferred provider."""
        # Should fall back to best available provider
        provider = provider_manager.get_best_provider("invalid_provider")

        assert provider is not None
        assert isinstance(provider, ExecutionProvider)

    def test_parse_provider_name(self, provider_manager):
        """Test parsing provider names."""
        # Test various name formats
        assert provider_manager._parse_provider_name("cuda") == ExecutionProvider.CUDA
        assert provider_manager._parse_provider_name("CUDA") == ExecutionProvider.CUDA
        assert (
            provider_manager._parse_provider_name("coreml")
            == ExecutionProvider.COREML
        )
        assert provider_manager._parse_provider_name("core") == ExecutionProvider.COREML
        assert (
            provider_manager._parse_provider_name("tensorrt")
            == ExecutionProvider.TENSORRT
        )
        assert (
            provider_manager._parse_provider_name("trt")
            == ExecutionProvider.TENSORRT
        )
        assert provider_manager._parse_provider_name("rocm") == ExecutionProvider.ROCM
        assert (
            provider_manager._parse_provider_name("directml")
            == ExecutionProvider.DIRECTML
        )
        assert (
            provider_manager._parse_provider_name("dml")
            == ExecutionProvider.DIRECTML
        )
        assert (
            provider_manager._parse_provider_name("openvino")
            == ExecutionProvider.OPENVINO
        )
        assert provider_manager._parse_provider_name("cpu") == ExecutionProvider.CPU

        # Test invalid names
        assert provider_manager._parse_provider_name("invalid") is None
        assert provider_manager._parse_provider_name("") is None

    def test_is_provider_available(self, provider_manager):
        """Test checking provider availability."""
        # CPU should always be available
        assert provider_manager._is_provider_available(ExecutionProvider.CPU) is True

        # Test other providers (may or may not be available depending on system)
        for provider in ExecutionProvider:
            if provider != ExecutionProvider.CPU:
                # We can't test the actual availability without knowing the system
                # but we can test that the method doesn't crash
                result = provider_manager._is_provider_available(provider)
                assert isinstance(result, bool)

    def test_get_provider_config(self, provider_manager):
        """Test getting provider configuration."""
        # Test CPU provider (should have minimal config)
        cpu_config = provider_manager.get_provider_config(ExecutionProvider.CPU)
        assert isinstance(cpu_config, dict)

        # Test other providers
        for provider in ExecutionProvider:
            config = provider_manager.get_provider_config(provider)
            assert isinstance(config, dict)

    def test_get_available_providers(self, provider_manager):
        """Test getting available providers list."""
        providers = provider_manager.get_available_providers()

        assert isinstance(providers, list)
        assert len(providers) > 0
        assert "CPUExecutionProvider" in providers

    def test_get_provider_info(self, provider_manager):
        """Test getting provider information."""
        for provider in ExecutionProvider:
            info = provider_manager.get_provider_info(provider)

            assert isinstance(info, dict)
            assert "name" in info
            assert "available" in info
            assert "config" in info
            assert info["name"] == provider.value
            assert isinstance(info["available"], bool)
            assert isinstance(info["config"], dict)

    def test_provider_priority_order(self, provider_manager):
        """Test that provider priority is reasonable."""
        priority = provider_manager._provider_priority

        # CPU should be last (fallback)
        assert priority[-1] == ExecutionProvider.CPU

        # TensorRT should be first (best performance)
        assert priority[0] == ExecutionProvider.TENSORRT

        # Check that all providers are in the priority list
        all_providers = set(ExecutionProvider)
        priority_providers = set(priority)
        assert all_providers == priority_providers
