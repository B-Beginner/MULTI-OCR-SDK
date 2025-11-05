"""
Configuration management for DeepSeek OCR SDK.

This module provides configuration management with support for
environment variables and explicit parameters.
"""

import os
from dataclasses import dataclass
from typing import Optional

from .exceptions import ConfigurationError


@dataclass
class OCRConfig:
    """
    Configuration for DeepSeek OCR client.

    Configuration priority (high to low):
        1. Explicit parameters passed to methods
        2. Explicit parameters passed to __init__
        3. Environment variables
        4. Default values

    Environment Variables:
        DS_OCR_API_KEY: API key for DeepSeek OCR (required)
        DS_OCR_BASE_URL: Base URL for the API
        DS_OCR_MODEL: Model name
        DS_OCR_TIMEOUT: Request timeout in seconds
        DS_OCR_MAX_TOKENS: Maximum tokens in response
        DS_OCR_DPI: DPI for PDF to image conversion
        DS_OCR_FALLBACK_ENABLED: Enable intelligent fallback
        DS_OCR_FALLBACK_MODE: Fallback mode to use
        DS_OCR_MIN_OUTPUT_THRESHOLD: Minimum output length before fallback

    Attributes:
        api_key: API key for authentication (required).
        base_url: Base URL for the API endpoint.
        model_name: Name of the OCR model to use.
        timeout: Request timeout in seconds.
        max_tokens: Maximum number of tokens in the response.
        temperature: Temperature for response generation (0.0 = deterministic).
        dpi: DPI setting for PDF to image conversion (150, 200, or 300).
        fallback_enabled: Enable automatic fallback to better mode.
        fallback_mode: Mode to fallback to when output is insufficient.
        min_output_threshold: Minimum output length to trigger fallback.
    """

    api_key: str
    base_url: str = "https://api.siliconflow.cn/v1/chat/completions"
    model_name: str = "deepseek-ai/DeepSeek-OCR"
    timeout: int = 60
    max_tokens: int = 4000
    temperature: float = 0.0
    dpi: int = 200
    fallback_enabled: bool = True
    fallback_mode: str = "grounding"
    min_output_threshold: int = 500

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.api_key:
            raise ConfigurationError(
                "API key is required. Set DS_OCR_API_KEY environment variable "
                "or pass api_key parameter."
            )

        if self.dpi not in [150, 200, 300]:
            raise ConfigurationError(
                f"DPI must be 150, 200, or 300. Got: {self.dpi}. "
                "Recommended: 200 for optimal balance."
            )

        if self.timeout <= 0:
            raise ConfigurationError(f"Timeout must be positive. Got: {self.timeout}")

        if self.max_tokens <= 0:
            raise ConfigurationError(
                f"max_tokens must be positive. Got: {self.max_tokens}"
            )

        if self.min_output_threshold < 0:
            raise ConfigurationError(
                f"min_output_threshold must be non-negative. Got: {self.min_output_threshold}"
            )

    @classmethod
    def from_env(cls, **overrides: Optional[str]) -> "OCRConfig":
        """
        Create configuration from environment variables.

        Args:
            **overrides: Override specific configuration values.

        Returns:
            OCRConfig instance.

        Raises:
            ConfigurationError: If required configuration is missing or invalid.

        Example:
            >>> config = OCRConfig.from_env(dpi=300)
            >>> config = OCRConfig.from_env(api_key="custom_key")
        """
        api_key = overrides.get("api_key") or os.getenv("DS_OCR_API_KEY", "")
        base_url = overrides.get("base_url") or os.getenv(
            "DS_OCR_BASE_URL", "https://api.siliconflow.cn/v1/chat/completions"
        )
        model_name = overrides.get("model_name") or os.getenv(
            "DS_OCR_MODEL", "deepseek-ai/DeepSeek-OCR"
        )
        timeout = int(overrides.get("timeout") or os.getenv("DS_OCR_TIMEOUT", "60"))
        max_tokens = int(
            overrides.get("max_tokens") or os.getenv("DS_OCR_MAX_TOKENS", "4000")
        )
        temperature = float(
            overrides.get("temperature") or os.getenv("DS_OCR_TEMPERATURE", "0.0")
        )
        dpi = int(overrides.get("dpi") or os.getenv("DS_OCR_DPI", "200"))
        fallback_enabled = (
            overrides.get("fallback_enabled")
            or os.getenv("DS_OCR_FALLBACK_ENABLED", "true")
        ).lower() in ("true", "1", "yes")
        fallback_mode = overrides.get("fallback_mode") or os.getenv(
            "DS_OCR_FALLBACK_MODE", "grounding"
        )
        min_output_threshold = int(
            overrides.get("min_output_threshold")
            or os.getenv("DS_OCR_MIN_OUTPUT_THRESHOLD", "500")
        )

        return cls(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            dpi=dpi,
            fallback_enabled=fallback_enabled,
            fallback_mode=fallback_mode,
            min_output_threshold=min_output_threshold,
        )
