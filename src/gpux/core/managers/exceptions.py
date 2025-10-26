"""Custom exceptions for model managers."""

from __future__ import annotations

from typing import Any


class RegistryError(Exception):
    """Base exception for registry-related errors."""


class ModelNotFoundError(RegistryError):
    """Raised when a model is not found in any location.

    This exception provides detailed information about where the model
    was searched and helpful suggestions for resolving the issue.
    """

    def __init__(
        self,
        model_name: str,
        search_locations: list[dict[str, Any]] | None = None,
        suggestions: list[str] | None = None,
        message: str | None = None,
    ) -> None:
        """Initialize ModelNotFoundError with helpful context.

        Args:
            model_name: Name of the model that wasn't found
            search_locations: List of locations that were searched
            suggestions: List of helpful suggestions
            message: Custom error message (optional)
        """
        self.model_name = model_name
        self.search_locations = search_locations or []
        self.suggestions = suggestions or []

        if message:
            super().__init__(message)
        else:
            super().__init__(f"Model '{model_name}' not found")

    def format_error_message(self) -> str:
        """Format a detailed error message with search locations and suggestions.

        Returns:
            Formatted error message string
        """
        lines = [f"Error: Model '{self.model_name}' not found", ""]

        if self.search_locations:
            lines.append("Searched locations:")
            for location in self.search_locations:
                status = "✓" if location["found"] else "✗"
                lines.append(f"  {status} {location['location']}")
            lines.append("")

        if self.suggestions:
            lines.append("Suggestions:")
            lines.extend(f"  • {suggestion}" for suggestion in self.suggestions)

        return "\n".join(lines)


class ConversionError(RegistryError):
    """Raised when model conversion fails."""


class AuthenticationError(RegistryError):
    """Raised when authentication fails."""


class NetworkError(RegistryError):
    """Raised when network operations fail."""
