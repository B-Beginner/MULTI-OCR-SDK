"""
Base configuration class for API clients.

Provides common configuration management functionality shared across
different client implementations.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..exceptions import ConfigurationError


@dataclass
class BaseConfig:
    """
    Base configuration for API clients.

    Attributes:
        api_key: API key for authentication (required).
        base_url: Base URL for the API endpoint (required).
        timeout: Request timeout in seconds (default: 60).
        max_tokens: Maximum tokens in response (default: 4000).
        temperature: Temperature for response generation (default: 0.0).
        request_delay: Delay in seconds between API requests (default: 0.0).
        enable_rate_limit_retry: Enable automatic retry on 429 errors (default: True).
        max_rate_limit_retries: Maximum number of retries for rate limit errors (default: 3).
        rate_limit_retry_delay: Initial delay in seconds before retrying (default: 5.0).
    """

    api_key: str
    base_url: str
    timeout: int = 60
    max_tokens: int = 4000
    temperature: float = 0.0
    request_delay: float = 0.0
    enable_rate_limit_retry: bool = True
    max_rate_limit_retries: int = 3
    rate_limit_retry_delay: float = 5.0

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.api_key:
            raise ConfigurationError("API key is required.")

        if not self.base_url:
            raise ConfigurationError("Base URL is required.")

        if self.timeout <= 0:
            raise ConfigurationError(f"Timeout must be positive. Got: {self.timeout}")

        if self.max_tokens <= 0:
            raise ConfigurationError(
                f"max_tokens must be positive. Got: {self.max_tokens}"
            )

        if self.request_delay < 0:
            raise ConfigurationError(
                f"request_delay must be non-negative. Got: {self.request_delay}"
            )

        if self.max_rate_limit_retries < 0:
            raise ConfigurationError(
                f"max_rate_limit_retries must be non-negative. "
                f"Got: {self.max_rate_limit_retries}"
            )

        if self.rate_limit_retry_delay < 0:
            raise ConfigurationError(
                f"rate_limit_retry_delay must be non-negative. "
                f"Got: {self.rate_limit_retry_delay}"
            )

    @staticmethod
    def _get_env(key: str, default: Any = None, type_func: Any = str) -> Any:
        """
        Helper to get environment variable with type conversion.

        Args:
            key: Environment variable name.
            default: Default value if not found.
            type_func: Type conversion function.

        Returns:
            Converted value or default.
        """
        val = os.getenv(key)
        if val is None:
            return default
        if type_func is bool:
            return val.lower() in ("true", "1", "yes", "on")
        return type_func(val)

    @staticmethod
    def _filter_none_values(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove None values from config dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            Filtered configuration with None values removed.
        """
        return {k: v for k, v in config.items() if v is not None}
