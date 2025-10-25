"""Custom exceptions for model managers."""

from __future__ import annotations


class RegistryError(Exception):
    """Base exception for registry-related errors."""


class ModelNotFoundError(RegistryError):
    """Raised when a model is not found in the registry."""


class ConversionError(RegistryError):
    """Raised when model conversion fails."""


class AuthenticationError(RegistryError):
    """Raised when authentication fails."""


class NetworkError(RegistryError):
    """Raised when network operations fail."""
