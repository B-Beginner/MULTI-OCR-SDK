"""
Basic utilities module for multi-OCR SDK.

This module provides common utilities shared across different clients:
- Configuration management
- File processing utilities
- Rate limiting and retry logic
- API request handling
"""

from .base_config import BaseConfig
from .file_processor import FileProcessor
from .rate_limiter import RateLimiter
from .api_requester import APIRequester

__all__ = [
    "BaseConfig",
    "FileProcessor",
    "RateLimiter",
    "APIRequester",
]
