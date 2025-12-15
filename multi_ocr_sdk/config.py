import os
from dataclasses import dataclass
from typing import Any, Optional, cast

from .basic_utils.base_config import BaseConfig
from .exceptions import ConfigurationError


@dataclass
class OCRConfig(BaseConfig):
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
        DS_OCR_MIN_OUTPUT_THRESHOLD: Minimum output length before
            fallback
        DS_OCR_PAGE_SEPARATOR: Separator used between pages in
            multi-page results
        DS_OCR_REQUEST_DELAY: Delay in seconds between API requests
            to prevent rate limiting
        DS_OCR_ENABLE_RATE_LIMIT_RETRY: Enable automatic retry on
            429 rate limit errors
        DS_OCR_MAX_RATE_LIMIT_RETRIES: Maximum number of retries for
            rate limit errors
        DS_OCR_RATE_LIMIT_RETRY_DELAY: Initial delay in seconds before
            retrying after 429 error

    Attributes:
        api_key: API key for authentication (required).
        base_url: Base URL for the API endpoint.
        model_name: Name of the OCR model to use.
        timeout: Request timeout in seconds.
        max_tokens: Maximum number of tokens in the response.
        temperature: Temperature for response generation (0.0 =
            deterministic).
        dpi: DPI setting for PDF to image conversion (150, 200, or
            300).
        fallback_enabled: Enable automatic fallback to better mode.
        fallback_mode: Mode to fallback to when output is
            insufficient.
        min_output_threshold: Minimum output length to trigger
            fallback.
        page_separator: Separator string used between pages in
            multi-page results.
        request_delay: Delay in seconds between API requests to
            prevent rate limiting (0 = no delay).
        enable_rate_limit_retry: Enable automatic retry on 429 rate
            limit errors.
        max_rate_limit_retries: Maximum number of retries for rate
            limit errors.
        rate_limit_retry_delay: Initial delay in seconds before
            retrying after 429 error (uses exponential backoff).
    """

    api_key: str
    base_url: str
    model_name: str = "deepseek-ai/DeepSeek-OCR"
    timeout: int = 60
    max_tokens: int = 4000
    temperature: float = 0.0
    dpi: int = 200
    fallback_enabled: bool = True
    fallback_mode: str = "grounding"
    min_output_threshold: int = 500
    page_separator: str = "\n\n---\n\n"
    request_delay: float = 0.0
    enable_rate_limit_retry: bool = True
    max_rate_limit_retries: int = 3
    rate_limit_retry_delay: float = 5.0

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.api_key:
            raise ConfigurationError(
                "API key is required. Set DS_OCR_API_KEY environment variable "
                "or pass api_key parameter."
            )

        if not self.base_url:
            raise ConfigurationError(
                "Base URL is required. Set DS_OCR_BASE_URL environment "
                "variable or pass base_url parameter. "
                "Known provider:\n"
                "  - SiliconFlow: "
                "https://api.siliconflow.cn/v1/chat/completions\n"
                "Note: DeepSeek's official API does not support the "
                "DeepSeek-OCR model."
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
                f"min_output_threshold must be non-negative. "
                f"Got: {self.min_output_threshold}"
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

    @classmethod
    def from_env(cls, **overrides: Optional[str]) -> "OCRConfig":
        """
        Create configuration from environment variables.

        Args:
            **overrides: Override specific configuration values.

        Returns:
            OCRConfig instance.

        Raises:
            ConfigurationError: If required configuration is missing
                or invalid.

        Example:
            >>> config = OCRConfig.from_env(dpi=300)
            >>> config = OCRConfig.from_env(api_key="custom_key")
        """
        api_key = cast(str, overrides.get("api_key") or os.getenv("DS_OCR_API_KEY", ""))
        base_url = cast(
            str,
            overrides.get("base_url") or os.getenv("DS_OCR_BASE_URL", ""),
        )
        model_name = cast(
            str,
            overrides.get("model_name")
            or os.getenv("DS_OCR_MODEL", "deepseek-ai/DeepSeek-OCR"),
        )
        timeout_str = cast(
            str,
            overrides.get("timeout") or os.getenv("DS_OCR_TIMEOUT", "60"),
        )
        timeout = int(timeout_str)
        max_tokens_str = cast(
            str,
            overrides.get("max_tokens") or os.getenv("DS_OCR_MAX_TOKENS", "4000"),
        )
        max_tokens = int(max_tokens_str)
        temperature_str = cast(
            str,
            overrides.get("temperature") or os.getenv("DS_OCR_TEMPERATURE", "0.0"),
        )
        temperature = float(temperature_str)
        dpi_str = cast(str, overrides.get("dpi") or os.getenv("DS_OCR_DPI", "200"))
        dpi = int(dpi_str)
        fallback_enabled_str = cast(
            str,
            overrides.get("fallback_enabled")
            or os.getenv("DS_OCR_FALLBACK_ENABLED", "true"),
        )
        fallback_enabled = fallback_enabled_str.lower() in ("true", "1", "yes")
        fallback_mode = cast(
            str,
            overrides.get("fallback_mode")
            or os.getenv("DS_OCR_FALLBACK_MODE", "grounding"),
        )
        min_threshold_str = cast(
            str,
            overrides.get("min_output_threshold")
            or os.getenv("DS_OCR_MIN_OUTPUT_THRESHOLD", "500"),
        )
        min_output_threshold = int(min_threshold_str)
        page_separator = cast(
            str,
            overrides.get("page_separator")
            or os.getenv("DS_OCR_PAGE_SEPARATOR", "\n\n---\n\n"),
        )

        request_delay_str = cast(
            str,
            overrides.get("request_delay") or os.getenv("DS_OCR_REQUEST_DELAY", "0.0"),
        )
        request_delay = float(request_delay_str)

        enable_rate_limit_retry_val = overrides.get("enable_rate_limit_retry")
        if enable_rate_limit_retry_val is None:
            enable_rate_limit_retry_val = os.getenv(
                "DS_OCR_ENABLE_RATE_LIMIT_RETRY", "true"
            )
        if isinstance(enable_rate_limit_retry_val, bool):
            enable_rate_limit_retry = enable_rate_limit_retry_val
        else:
            enable_rate_limit_retry = str(enable_rate_limit_retry_val).lower() in (
                "true",
                "1",
                "yes",
            )

        max_rate_limit_retries_str = cast(
            str,
            overrides.get("max_rate_limit_retries")
            or os.getenv("DS_OCR_MAX_RATE_LIMIT_RETRIES", "3"),
        )
        max_rate_limit_retries = int(max_rate_limit_retries_str)

        rate_limit_retry_delay_str = cast(
            str,
            overrides.get("rate_limit_retry_delay")
            or os.getenv("DS_OCR_RATE_LIMIT_RETRY_DELAY", "5.0"),
        )
        rate_limit_retry_delay = float(rate_limit_retry_delay_str)

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
            page_separator=page_separator,
            request_delay=request_delay,
            enable_rate_limit_retry=enable_rate_limit_retry,
            max_rate_limit_retries=max_rate_limit_retries,
            rate_limit_retry_delay=rate_limit_retry_delay,
        )
