"""
DeepSeek OCR client for the multi-OCR SDK.

This module provides the main client for interacting with the DeepSeek OCR API,
using shared utilities from basic_utils for file processing, rate limiting, and API requests.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import OCRConfig
from .enums import OCRMode
from .exceptions import APIError
from .basic_utils import FileProcessor, RateLimiter, APIRequester

logger = logging.getLogger(__name__)


class DeepSeekOCR:
    """
    Client for DeepSeek OCR API.

    This client provides synchronous methods for
    document OCR processing using the DeepSeek OCR API.

    Note:
        Asynchronous support has been removed. Only synchronous methods are available in this version.
    Example:
        >>> # Synchronous usage
        >>> client = DeepSeekOCR(api_key="your_api_key")
        >>> result = client.parse("document.pdf")
        >>> print(result)

    Attributes:
        config: OCRConfig instance containing all configuration.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
        dpi: Optional[int] = None,
        request_delay: Optional[float] = None,
        enable_rate_limit_retry: Optional[bool] = None,
        max_rate_limit_retries: Optional[int] = None,
        rate_limit_retry_delay: Optional[float] = None,
        **kwargs: str,
    ):
        """
        Initialize DeepSeekOCR client.

        Args:
            api_key: API key for authentication. If not provided, will read
                     from DS_OCR_API_KEY environment variable.
            base_url: Base URL for the API endpoint.
            model_name: Name of the OCR model to use.
            timeout: Request timeout in seconds.
            max_tokens: Maximum tokens in response.
            dpi: DPI for PDF to image conversion (150, 200, or 300).
            request_delay: Delay in seconds between API requests to prevent
                          rate limiting (0 = no delay).
            enable_rate_limit_retry: Enable automatic retry on 429 rate
                                    limit errors.
            max_rate_limit_retries: Maximum number of retries for rate
                                   limit errors.
            rate_limit_retry_delay: Initial delay in seconds before retrying
                                   after 429 error (uses exponential backoff).
            **kwargs: Additional configuration parameters.

        Raises:
            ConfigurationError: If required configuration is missing
                or invalid.
        """
        # Build overrides dict from provided arguments
        overrides: Dict[str, Any] = {}
        if api_key is not None:
            overrides["api_key"] = api_key
        if base_url is not None:
            overrides["base_url"] = base_url
        if model_name is not None:
            overrides["model_name"] = model_name
        if timeout is not None:
            overrides["timeout"] = str(timeout)
        if max_tokens is not None:
            overrides["max_tokens"] = str(max_tokens)
        if dpi is not None:
            overrides["dpi"] = str(dpi)
        if request_delay is not None:
            overrides["request_delay"] = str(request_delay)
        if enable_rate_limit_retry is not None:
            overrides["enable_rate_limit_retry"] = enable_rate_limit_retry
        if max_rate_limit_retries is not None:
            overrides["max_rate_limit_retries"] = str(max_rate_limit_retries)
        if rate_limit_retry_delay is not None:
            overrides["rate_limit_retry_delay"] = str(rate_limit_retry_delay)
        overrides.update(kwargs)

        self.config = OCRConfig.from_env(**overrides)
        logger.info(
            f"Initialized DeepSeekOCR client with model: {self.config.model_name}"
        )

        # Initialize shared utilities
        self._rate_limiter = RateLimiter(
            request_delay=self.config.request_delay,
            max_retries=self.config.max_rate_limit_retries,
            retry_delay=self.config.rate_limit_retry_delay,
        )
        self._api_requester = APIRequester(self._rate_limiter, self.config.timeout)

    def _build_prompt(self, mode: OCRMode) -> str:
        """
        Build the prompt for the API request.

        Args:
            mode: OCR mode to use.

        Returns:
            Prompt string.
        """
        return mode.get_prompt()

    def _clean_output(self, text: str) -> str:
        """
        Clean the OCR output by removing special tags.

        Args:
            text: Raw OCR output text.

        Returns:
            Cleaned text.
        """
        # Remove special tags but preserve HTML tables
        text = re.sub(r"<\|ref\|>", "", text)
        text = re.sub(r"<\|det\|>", "", text)
        return text.strip()

    def parse(
        self,
        file_path: Union[str, Path],
        mode: Union[str, OCRMode] = OCRMode.FREE_OCR,
        dpi: Optional[int] = None,
        pages: Optional[Union[int, List[int]]] = None,
    ) -> str:
        """
        Parse document synchronously with per-page fallback.

        IMPORTANT - Behavior Change:
            Default behavior changed from processing only the first page
            to processing ALL pages of a PDF. This may result in:
            - Increased API costs
            - Longer processing time
            - Different output format (multi-page results separated by
              page_separator)

            To process only the first page (old behavior), use: pages=1

        Intelligent Fallback:
            Each page is individually evaluated for quality. If a page's
            OCR output is shorter than min_output_threshold (default: 500
            chars), it will automatically retry with the fallback mode
            (default: "grounding"). This per-page approach ensures optimal
            results without reprocessing all pages when only some need
            better quality.

        Args:
            file_path: Path to PDF or image file.
            mode: OCR mode ("free_ocr", "grounding", "ocr_image" or enum).
            dpi: DPI for PDF conversion. If None, uses config default.
            pages: Page(s) to process. Can be:
                   - None: Process all pages (default, BREAKING CHANGE)
                   - int: Process single page (1-indexed)
                   - list: Process specific pages (1-indexed)

        Returns:
            Extracted text in Markdown format. For multi-page documents,
            pages are separated by page_separator (default: "\\n\\n---\\n\\n").

        Raises:
            FileProcessingError: If file cannot be processed.
            APIError: If API returns an error.
            TimeoutError: If request times out.

        Example:
            >>> client = DeepSeekOCR(api_key="xxx")
            >>> # Process all pages (new default behavior)
            >>> text = client.parse("document.pdf")
            >>> # Process only first page (old default behavior)
            >>> text = client.parse("document.pdf", pages=1)
            >>> # With options
            >>> text = client.parse(
            ...     "document.pdf",
            ...     mode="grounding",
            ...     dpi=300
            ... )
            >>> # Process specific pages
            >>> text = client.parse(
            ...     "document.pdf",
            ...     pages=[1, 3, 5]
            ... )
        """
        # Convert mode string to enum if needed
        if isinstance(mode, str):
            mode = OCRMode(mode)

        # Use config DPI if not specified
        if dpi is None:
            dpi = self.config.dpi

        # Convert PDF to base64 using shared utility
        logger.info(f"Processing {file_path} with mode={mode} and dpi={dpi}")
        image_b64_result = FileProcessor.pdf_to_base64(file_path, dpi, pages)

        # Build prompt
        prompt = self._build_prompt(mode)

        # Handle single page or multiple pages
        if isinstance(image_b64_result, str):
            # Single page
            images = [image_b64_result]
        else:
            # Multiple pages
            images = image_b64_result

        # Process all pages with per-page intelligent fallback
        all_texts = []
        for page_idx, image_b64 in enumerate(images):
            logger.debug(f"Processing page {page_idx + 1}/{len(images)}")

            # Prepare request payload
            payload = {
                "model": self.config.model_name,
                "messages": [
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
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }

            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }

            # Make API request using shared requester
            result = self._api_requester.request_sync(
                self.config.base_url,
                headers,
                payload,
                enable_rate_limit_retry=self.config.enable_rate_limit_retry,
            )

            # Extract text from response
            if "choices" not in result or len(result["choices"]) == 0:
                raise APIError("Invalid API response: no choices returned")

            text: str = str(result["choices"][0]["message"]["content"])
            text = self._clean_output(text)

            # Log token usage
            usage = result.get("usage", {})
            logger.debug(
                f"Page {page_idx + 1} API usage: "
                f"prompt_tokens={usage.get('prompt_tokens')}, "
                f"completion_tokens={usage.get('completion_tokens')}, "
                f"total_tokens={usage.get('total_tokens')}"
            )

            # Per-page intelligent fallback
            # If output is too short and fallback is enabled,
            # retry with fallback mode
            if (
                self.config.fallback_enabled
                and mode == OCRMode.FREE_OCR
                and len(text) < self.config.min_output_threshold
            ):
                logger.warning(
                    f"Page {page_idx + 1} output too short "
                    f"({len(text)} chars), falling back to "
                    f"{self.config.fallback_mode}"
                )
                # Retry this page with fallback mode
                try:
                    fallback_prompt = self._build_prompt(
                        OCRMode(self.config.fallback_mode)
                    )
                    fallback_payload = {
                        "model": self.config.model_name,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                                    },
                                    {"type": "text", "text": fallback_prompt},
                                ],
                            }
                        ],
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                    }

                    fallback_result = self._api_requester.request_sync(
                        self.config.base_url,
                        headers,
                        fallback_payload,
                        enable_rate_limit_retry=self.config.enable_rate_limit_retry,
                    )

                    if (
                        "choices" in fallback_result
                        and len(fallback_result["choices"]) > 0
                    ):
                        text = str(fallback_result["choices"][0]["message"]["content"])
                        text = self._clean_output(text)
                        logger.info(
                            f"Page {page_idx + 1} fallback successful: "
                            f"{len(text)} chars"
                        )
                except Exception as e:
                    logger.error(
                        f"Page {page_idx + 1} fallback failed: {e}, "
                        f"using original result"
                    )

            all_texts.append(text)

        # Combine all pages with page separator
        combined_text = self.config.page_separator.join(all_texts)

        logger.info(
            f"Successfully processed {file_path}: "
            f"{len(images)} page(s), {len(combined_text)} chars"
        )
        return combined_text
