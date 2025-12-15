"""
Vision-Language Model (VLM) client for DeepSeek OCR SDK.

This module provides a lightweight client wrapper for calling VLM-style
chat completions (image + text). It mirrors the design patterns used by
the existing OCR client to reuse configuration and rate-limit behavior.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .exceptions import (
    APIError,
    ConfigurationError,
    FileProcessingError,
    RateLimitError,
    TimeoutError,
)
from .basic_utils import FileProcessor, RateLimiter, APIRequester

logger = logging.getLogger(__name__)


@dataclass
class VLMConfig:
    api_key: str
    base_url: str
    model_name: str = "Qwen3-VL-8B"
    timeout: int = 60
    max_tokens: int = 8000 #qwen3-vl-8b支持8k上下文，此处设置为8000
    temperature: float = 0.0
    request_delay: float = 0.0
    enable_rate_limit_retry: bool = True
    max_rate_limit_retries: int = 3
    rate_limit_retry_delay: float = 5.0

    @classmethod
    def from_env(cls, **overrides: Any) -> "VLMConfig":
        """Create configuration from environment variables with overrides."""
        # Helper to get env var with type conversion
        def get_env(key: str, default: Any = None, type_func: Any = str) -> Any:
            val = os.getenv(key)
            if val is None:
                return default
            if type_func is bool:
                return val.lower() in ("true", "1", "yes", "on")
            return type_func(val)

        # Load from environment variables
        env_config = {
            "api_key": get_env("VLM_API_KEY"),
            "base_url": get_env("VLM_BASE_URL"),
            "model_name": get_env("VLM_MODEL_NAME"),
            "timeout": get_env("VLM_TIMEOUT", type_func=int),
            "max_tokens": get_env("VLM_MAX_TOKENS", type_func=int),
            "temperature": get_env("VLM_TEMPERATURE", type_func=float),
            "request_delay": get_env("VLM_REQUEST_DELAY", type_func=float),
            "enable_rate_limit_retry": get_env("VLM_ENABLE_RATE_LIMIT_RETRY", type_func=bool),
            "max_rate_limit_retries": get_env("VLM_MAX_RATE_LIMIT_RETRIES", type_func=int),
            "rate_limit_retry_delay": get_env("VLM_RATE_LIMIT_RETRY_DELAY", type_func=float),
        }

        # Remove None values so defaults in __init__ apply
        env_config = {k: v for k, v in env_config.items() if v is not None}
        
        # Apply overrides
        env_config.update({k: v for k, v in overrides.items() if v is not None})
        
        # Handle required fields that might still be missing
        if "api_key" not in env_config:
            # Fallback to DS_OCR_API_KEY if VLM_API_KEY not set (backward compatibility/convenience)
            env_config["api_key"] = os.getenv("DS_OCR_API_KEY", "")
            
        if "base_url" not in env_config:
             # Fallback to DS_OCR_BASE_URL if VLM_BASE_URL not set
            env_config["base_url"] = os.getenv("DS_OCR_BASE_URL", "")

        return cls(**env_config)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ConfigurationError("VLM API key is required.")
        if not self.base_url:
            raise ConfigurationError("VLM base_url is required.")
        if self.timeout <= 0:
            raise ConfigurationError("timeout must be > 0")
        if self.request_delay < 0:
            raise ConfigurationError("request_delay must be >= 0")
        if self.max_rate_limit_retries < 0:
            raise ConfigurationError("max_rate_limit_retries must be >= 0")
        if self.rate_limit_retry_delay < 0:
            raise ConfigurationError("rate_limit_retry_delay must be >= 0")

        # Auto-fix base_url if it looks like a root URL (OpenAI style)
        # Many users might provide ".../v1" expecting the client to append "/chat/completions"
        if self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/chat/completions"
        elif self.base_url.endswith("/v1/"):
            self.base_url = f"{self.base_url}chat/completions"


class _CompletionsAPI:
    def __init__(self, client: "VLMClient") -> None:
        self._client = client

    def create(self, model: str, messages: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """Synchronous completion call."""
        return self._client._make_api_request_sync(model, messages, **kwargs)


class _ChatAPI:
    def __init__(self, client: "VLMClient") -> None:
        self.completions = _CompletionsAPI(client)


class VLMClient:
    """Lightweight VLM client with rate limiting and retry support.

    Usage:
        from deepseek_ocr import vlm
        client = vlm.VLMClient(api_key="xxx", base_url="https://...")
        result = client.chat.completions.create(model="Qwen3-VL-8B", messages=[...])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        request_delay: Optional[float] = None,
        enable_rate_limit_retry: Optional[bool] = None,
        max_rate_limit_retries: Optional[int] = None,
        rate_limit_retry_delay: Optional[float] = None,
        **overrides: Any,
    ) -> None:
        # Build config from provided arguments and env if set
        config_args = {
            "api_key": api_key,
            "base_url": base_url,
            "model_name": model_name,
            "timeout": timeout,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "request_delay": request_delay,
            "enable_rate_limit_retry": enable_rate_limit_retry,
            "max_rate_limit_retries": max_rate_limit_retries,
            "rate_limit_retry_delay": rate_limit_retry_delay,
        }
        config_args.update(overrides)
        
        self.config = VLMConfig.from_env(**config_args)
        
        # Initialize shared utilities
        self._rate_limiter = RateLimiter(
            request_delay=self.config.request_delay,
            max_retries=self.config.max_rate_limit_retries,
            retry_delay=self.config.rate_limit_retry_delay,
        )
        self._api_requester = APIRequester(self._rate_limiter, self.config.timeout)

        # Provide a chat API surface similar to other SDKs
        self.chat = _ChatAPI(self)

    def parse(
        self,
        file_path: Union[str, Path],
        prompt: str,
        model: Optional[str] = None,
        dpi: int = 72,
        pages: Optional[Union[int, List[int]]] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Process a local file (PDF or image) with VLM synchronously.
        
        Args:
            file_path: Path to the file.
            prompt: Text prompt for the VLM.
            model: Model name (optional).
            dpi: DPI for rendering PDF pages (default: 72).
                 Note: Higher DPI results in larger images and significantly more tokens.
                 If you hit token limits (e.g. "decoder prompt is longer than..."), try reducing DPI (e.g. to 72).
            pages: Specific pages to process (1-indexed).
            **kwargs: Additional arguments for the API call.
            
        Returns:
            Combined text response from all processed pages.
        """
        # Convert file to base64 images using shared utility
        logger.info(f"Processing {file_path} with dpi={dpi}, pages={pages} (type: {type(pages)})")
        image_b64_result = FileProcessor.file_to_base64(file_path, dpi, pages)
        
        if isinstance(image_b64_result, str):
            images = [image_b64_result]
        else:
            images = image_b64_result

        logger.info(f"Converted to {len(images)} images for processing")

        all_texts = []
        for page_idx, image_b64 in enumerate(images):
            logger.debug(f"Processing page {page_idx + 1}/{len(images)}")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            result = self.chat.completions.create(
                model=model or self.config.model_name,
                messages=messages,
                timeout=timeout,
                **kwargs
            )
            
            if "choices" in result and len(result["choices"]) > 0:
                text = str(result["choices"][0]["message"]["content"])
                all_texts.append(text)
            else:
                all_texts.append("")

        return "\n\n---\n\n".join(all_texts)

    def _apply_rate_limit_sync(self) -> None:
        self._rate_limiter.apply_rate_limit_sync()

    def _make_api_request_sync(self, model: str, messages: List[Dict[str, Any]], timeout: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or self.config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        return self._api_requester.request_sync(
            self.config.base_url,
            headers,
            payload,
            enable_rate_limit_retry=self.config.enable_rate_limit_retry,
            timeout_override=timeout,
        )


__all__ = ["VLMClient", "VLMConfig"]
