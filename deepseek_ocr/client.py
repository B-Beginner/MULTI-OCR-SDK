"""
Core OCR client for DeepSeek OCR SDK.

This module provides the main client for interacting with the DeepSeek OCR API.
"""

import asyncio
import base64
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Union

import aiohttp
import fitz  # PyMuPDF
import requests

from .config import OCRConfig
from .enums import OCRMode
from .exceptions import APIError, FileProcessingError, TimeoutError

logger = logging.getLogger(__name__)


class DeepSeekOCR:
    """
    Client for DeepSeek OCR API.

    This client provides both synchronous and asynchronous methods for
    document OCR processing using the DeepSeek OCR API.

    Example:
        >>> # Synchronous usage
        >>> client = DeepSeekOCR(api_key="your_api_key")
        >>> result = client.parse("document.pdf")
        >>> print(result)

        >>> # Asynchronous usage
        >>> import asyncio
        >>> async def main():
        ...     client = DeepSeekOCR(api_key="your_api_key")
        ...     result = await client.parse_async("document.pdf")
        ...     print(result)
        >>> asyncio.run(main())

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
        **kwargs: Optional[str],
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
            **kwargs: Additional configuration parameters.

        Raises:
            ConfigurationError: If required configuration is missing or invalid.
        """
        # Build overrides dict from provided arguments
        overrides = {}
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
        overrides.update(kwargs)

        self.config = OCRConfig.from_env(**overrides)
        logger.info(
            f"Initialized DeepSeekOCR client with model: {self.config.model_name}"
        )

    def _pdf_to_base64(self, file_path: Union[str, Path], dpi: int) -> str:
        """
        Convert PDF first page to base64-encoded image.

        Args:
            file_path: Path to the PDF file.
            dpi: DPI for rendering (150, 200, or 300).

        Returns:
            Base64-encoded image string.

        Raises:
            FileProcessingError: If file cannot be processed.
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileProcessingError(f"File not found: {file_path}")

            doc = fitz.open(str(file_path))
            if len(doc) == 0:
                raise FileProcessingError(f"PDF has no pages: {file_path}")

            # Process only first page
            page = doc[0]
            # Render page to image
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            # Convert to bytes
            img_bytes = pix.tobytes("png")
            doc.close()

            # Encode to base64
            b64_string = base64.b64encode(img_bytes).decode("utf-8")
            logger.debug(
                f"Converted PDF to image: {len(b64_string)} bytes at {dpi} DPI"
            )
            return b64_string

        except Exception as e:
            raise FileProcessingError(f"Failed to process PDF: {e}") from e

    def _build_prompt(self, mode: OCRMode, chinese_hint: bool) -> str:
        """
        Build the prompt for the API request.

        Args:
            mode: OCR mode to use.
            chinese_hint: Whether to add Chinese language hint.

        Returns:
            Prompt string.
        """
        prompt = mode.get_prompt()
        if chinese_hint:
            prompt += " Please extract all text in Chinese (中文) and English."
        return prompt

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

    async def _make_api_request_async(
        self, image_b64: str, prompt: str
    ) -> Dict[str, any]:
        """
        Make async API request to DeepSeek OCR.

        Args:
            image_b64: Base64-encoded image.
            prompt: Prompt for OCR processing.

        Returns:
            API response as dictionary.

        Raises:
            APIError: If API returns an error.
            TimeoutError: If request times out.
        """
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

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

        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.config.base_url, headers=headers, json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise APIError(
                            f"API request failed: {error_text}",
                            status_code=response.status,
                            response_text=error_text,
                        )

                    result = await response.json()
                    return result

        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"Request timed out after {self.config.timeout} seconds"
            ) from e

    def _make_api_request_sync(self, image_b64: str, prompt: str) -> Dict[str, any]:
        """
        Make synchronous API request to DeepSeek OCR.

        Args:
            image_b64: Base64-encoded image.
            prompt: Prompt for OCR processing.

        Returns:
            API response as dictionary.

        Raises:
            APIError: If API returns an error.
            TimeoutError: If request times out.
        """
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

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

        try:
            response = requests.post(
                self.config.base_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )

            if response.status_code != 200:
                raise APIError(
                    f"API request failed: {response.text}",
                    status_code=response.status_code,
                    response_text=response.text,
                )

            return response.json()

        except requests.Timeout as e:
            raise TimeoutError(
                f"Request timed out after {self.config.timeout} seconds"
            ) from e

    async def parse_async(
        self,
        file_path: Union[str, Path],
        mode: Union[str, OCRMode] = OCRMode.FREE_OCR,
        dpi: Optional[int] = None,
        chinese_hint: bool = False,
    ) -> str:
        """
        Parse document asynchronously.

        Args:
            file_path: Path to PDF or image file.
            mode: OCR mode to use ("free_ocr", "grounding", "ocr_image" or OCRMode enum).
            dpi: DPI for PDF conversion. If None, uses config default.
            chinese_hint: Add Chinese language hint for simple tables.

        Returns:
            Extracted text in Markdown format.

        Raises:
            FileProcessingError: If file cannot be processed.
            APIError: If API returns an error.
            TimeoutError: If request times out.

        Example:
            >>> client = DeepSeekOCR(api_key="xxx")
            >>> text = await client.parse_async("document.pdf")
            >>> # With options
            >>> text = await client.parse_async(
            ...     "document.pdf",
            ...     mode="grounding",
            ...     dpi=300,
            ...     chinese_hint=True
            ... )
        """
        # Convert mode string to enum if needed
        if isinstance(mode, str):
            mode = OCRMode(mode)

        # Use config DPI if not specified
        if dpi is None:
            dpi = self.config.dpi

        # Convert PDF to base64
        logger.info(f"Processing {file_path} with mode={mode} and dpi={dpi}")
        image_b64 = self._pdf_to_base64(file_path, dpi)

        # Build prompt
        prompt = self._build_prompt(mode, chinese_hint)

        # Make API request
        result = await self._make_api_request_async(image_b64, prompt)

        # Extract text from response
        if "choices" not in result or len(result["choices"]) == 0:
            raise APIError("Invalid API response: no choices returned")

        text = result["choices"][0]["message"]["content"]
        text = self._clean_output(text)

        # Log token usage
        usage = result.get("usage", {})
        logger.debug(
            f"API usage: prompt_tokens={usage.get('prompt_tokens')}, "
            f"completion_tokens={usage.get('completion_tokens')}, "
            f"total_tokens={usage.get('total_tokens')}"
        )

        # Intelligent fallback
        if (
            self.config.fallback_enabled
            and mode == OCRMode.FREE_OCR
            and len(text) < self.config.min_output_threshold
        ):
            logger.warning(
                f"Output too short ({len(text)} chars), falling back to {self.config.fallback_mode}"
            )
            return await self.parse_async(
                file_path,
                mode=OCRMode(self.config.fallback_mode),
                dpi=dpi,
                chinese_hint=chinese_hint,
            )

        logger.info(f"Successfully processed {file_path}: {len(text)} chars")
        return text

    def parse(
        self,
        file_path: Union[str, Path],
        mode: Union[str, OCRMode] = OCRMode.FREE_OCR,
        dpi: Optional[int] = None,
        chinese_hint: bool = False,
    ) -> str:
        """
        Parse document synchronously.

        Args:
            file_path: Path to PDF or image file.
            mode: OCR mode to use ("free_ocr", "grounding", "ocr_image" or OCRMode enum).
            dpi: DPI for PDF conversion. If None, uses config default.
            chinese_hint: Add Chinese language hint for simple tables.

        Returns:
            Extracted text in Markdown format.

        Raises:
            FileProcessingError: If file cannot be processed.
            APIError: If API returns an error.
            TimeoutError: If request times out.

        Example:
            >>> client = DeepSeekOCR(api_key="xxx")
            >>> text = client.parse("document.pdf")
            >>> # With options
            >>> text = client.parse(
            ...     "document.pdf",
            ...     mode="grounding",
            ...     dpi=300,
            ...     chinese_hint=True
            ... )
        """
        # Convert mode string to enum if needed
        if isinstance(mode, str):
            mode = OCRMode(mode)

        # Use config DPI if not specified
        if dpi is None:
            dpi = self.config.dpi

        # Convert PDF to base64
        logger.info(f"Processing {file_path} with mode={mode} and dpi={dpi}")
        image_b64 = self._pdf_to_base64(file_path, dpi)

        # Build prompt
        prompt = self._build_prompt(mode, chinese_hint)

        # Make API request
        result = self._make_api_request_sync(image_b64, prompt)

        # Extract text from response
        if "choices" not in result or len(result["choices"]) == 0:
            raise APIError("Invalid API response: no choices returned")

        text = result["choices"][0]["message"]["content"]
        text = self._clean_output(text)

        # Log token usage
        usage = result.get("usage", {})
        logger.debug(
            f"API usage: prompt_tokens={usage.get('prompt_tokens')}, "
            f"completion_tokens={usage.get('completion_tokens')}, "
            f"total_tokens={usage.get('total_tokens')}"
        )

        # Intelligent fallback
        if (
            self.config.fallback_enabled
            and mode == OCRMode.FREE_OCR
            and len(text) < self.config.min_output_threshold
        ):
            logger.warning(
                f"Output too short ({len(text)} chars), falling back to {self.config.fallback_mode}"
            )
            return self.parse(
                file_path,
                mode=OCRMode(self.config.fallback_mode),
                dpi=dpi,
                chinese_hint=chinese_hint,
            )

        logger.info(f"Successfully processed {file_path}: {len(text)} chars")
        return text
