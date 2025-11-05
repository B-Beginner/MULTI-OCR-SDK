"""
Custom exceptions for DeepSeek OCR SDK.

This module defines all custom exception types used by the SDK.
"""

from typing import Optional


class DeepSeekOCRError(Exception):
    """Base exception for all DeepSeek OCR SDK errors."""

    pass


class ConfigurationError(DeepSeekOCRError):
    """Raised when there is a configuration error."""

    pass


class APIError(DeepSeekOCRError):
    """Raised when the API returns an error response."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
    ):
        """
        Initialize APIError.

        Args:
            message: Error message.
            status_code: HTTP status code from the API.
            response_text: Raw response text from the API.
        """
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class FileProcessingError(DeepSeekOCRError):
    """Raised when there is an error processing the input file."""

    pass


class InvalidModeError(DeepSeekOCRError):
    """Raised when an invalid OCR mode is specified."""

    pass


class TimeoutError(DeepSeekOCRError):
    """Raised when an API request times out."""

    pass
