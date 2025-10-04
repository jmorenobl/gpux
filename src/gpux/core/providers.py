"""Execution provider management for GPUX."""

from __future__ import annotations

import logging
import platform
from enum import Enum
from typing import Any

import onnxruntime as ort

logger = logging.getLogger(__name__)


class ExecutionProvider(Enum):
    """Available execution providers for ONNX Runtime."""

    TENSORRT = "TensorrtExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    ROCM = "ROCmExecutionProvider"
    COREML = "CoreMLExecutionProvider"
    DIRECTML = "DirectMLExecutionProvider"
    OPENVINO = "OpenVINOExecutionProvider"
    CPU = "CPUExecutionProvider"


class ProviderManager:
    """Manages execution provider selection and configuration."""

    def __init__(self) -> None:
        """Initialize the provider manager."""
        self._available_providers = self._detect_available_providers()
        self._provider_priority = self._get_provider_priority()

    def _detect_available_providers(self) -> list[str]:
        """Detect available execution providers on the current system."""
        try:
            available = ort.get_available_providers()
            logger.info("Available providers: %s", available)
            return list(available)
        except (RuntimeError, ImportError) as e:
            logger.warning("Failed to detect providers: %s", e)
            return ["CPUExecutionProvider"]

    def _get_provider_priority(self) -> list[ExecutionProvider]:
        """Get provider priority based on platform and performance."""
        system = platform.system().lower()
        arch = platform.machine().lower()

        # Base priority for all platforms
        priority = [
            ExecutionProvider.TENSORRT,  # Best performance for NVIDIA
            ExecutionProvider.CUDA,  # Good NVIDIA performance
            ExecutionProvider.ROCM,  # AMD GPUs
            ExecutionProvider.COREML,  # Apple Silicon
            ExecutionProvider.DIRECTML,  # Windows GPUs
            ExecutionProvider.OPENVINO,  # Intel GPUs
            ExecutionProvider.CPU,  # Universal fallback
        ]

        # Platform-specific optimizations
        if system == "darwin" and arch in ["arm64", "aarch64"]:
            # Apple Silicon - prioritize CoreML
            priority = [
                ExecutionProvider.COREML,
                ExecutionProvider.CPU,
            ] + [
                p
                for p in priority
                if p not in [ExecutionProvider.COREML, ExecutionProvider.CPU]
            ]

        elif system == "windows":
            # Windows - prioritize DirectML
            priority = [
                ExecutionProvider.TENSORRT,
                ExecutionProvider.CUDA,
                ExecutionProvider.DIRECTML,
                ExecutionProvider.OPENVINO,
                ExecutionProvider.CPU,
            ] + [
                p
                for p in priority
                if p
                not in [
                    ExecutionProvider.TENSORRT,
                    ExecutionProvider.CUDA,
                    ExecutionProvider.DIRECTML,
                    ExecutionProvider.OPENVINO,
                    ExecutionProvider.CPU,
                ]
            ]

        return priority

    def get_best_provider(self, preferred: str | None = None) -> ExecutionProvider:
        """Get the best available execution provider.

        Args:
            preferred: Preferred provider name (e.g., "cuda", "coreml")

        Returns:
            Best available execution provider

        Raises:
            RuntimeError: If no providers are available
        """
        if preferred:
            preferred_provider = self._parse_provider_name(preferred)
            if preferred_provider and self._is_provider_available(preferred_provider):
                logger.info("Using preferred provider: %s", preferred_provider.value)
                return preferred_provider

        for provider in self._provider_priority:
            if self._is_provider_available(provider):
                logger.info("Selected provider: %s", provider.value)
                return provider

        msg = "No execution providers available"
        raise RuntimeError(msg)

    def _parse_provider_name(self, name: str) -> ExecutionProvider | None:  # noqa: PLR0911
        """Parse provider name string to ExecutionProvider enum."""
        name_lower = name.lower()

        if "tensorrt" in name_lower or "trt" in name_lower:
            return ExecutionProvider.TENSORRT
        if "cuda" in name_lower:
            return ExecutionProvider.CUDA
        if "rocm" in name_lower:
            return ExecutionProvider.ROCM
        if "coreml" in name_lower or "core" in name_lower:
            return ExecutionProvider.COREML
        if "directml" in name_lower or "dml" in name_lower:
            return ExecutionProvider.DIRECTML
        if "openvino" in name_lower:
            return ExecutionProvider.OPENVINO
        if "cpu" in name_lower:
            return ExecutionProvider.CPU

        return None

    def _is_provider_available(self, provider: ExecutionProvider) -> bool:
        """Check if a provider is available on the current system."""
        return provider.value in self._available_providers

    def get_provider_config(self, provider: ExecutionProvider) -> dict[str, Any]:
        """Get configuration for a specific provider.

        Args:
            provider: Execution provider

        Returns:
            Provider-specific configuration
        """
        config: dict[str, Any] = {}

        if provider == ExecutionProvider.TENSORRT:
            config = {
                "trt_max_workspace_size": 1 << 30,  # 1GB
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
            }
        elif provider == ExecutionProvider.CUDA:
            config = {
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }
        elif provider == ExecutionProvider.COREML:
            config = {
                "coreml_flags": 0,  # Use default settings
            }
        elif provider == ExecutionProvider.DIRECTML:
            config = {
                "device_id": 0,
            }

        return config

    def get_available_providers(self) -> list[str]:
        """Get list of available provider names."""
        return self._available_providers.copy()

    def get_provider_info(self, provider: ExecutionProvider) -> dict[str, Any]:
        """Get information about a specific provider.

        Args:
            provider: Execution provider

        Returns:
            Provider information dictionary
        """
        info = {
            "name": provider.value,
            "available": self._is_provider_available(provider),
            "config": self.get_provider_config(provider),
        }

        # Add platform-specific info
        if provider == ExecutionProvider.COREML:
            info["platform"] = "Apple Silicon"
            info["description"] = "Optimized for Apple devices"
        elif provider == ExecutionProvider.CUDA:
            info["platform"] = "NVIDIA"
            info["description"] = "NVIDIA CUDA acceleration"
        elif provider == ExecutionProvider.ROCM:
            info["platform"] = "AMD"
            info["description"] = "AMD ROCm acceleration"
        elif provider == ExecutionProvider.DIRECTML:
            info["platform"] = "Windows"
            info["description"] = "Windows DirectML acceleration"
        elif provider == ExecutionProvider.OPENVINO:
            info["platform"] = "Intel"
            info["description"] = "Intel OpenVINO acceleration"
        elif provider == ExecutionProvider.CPU:
            info["platform"] = "Universal"
            info["description"] = "CPU fallback execution"

        return info
