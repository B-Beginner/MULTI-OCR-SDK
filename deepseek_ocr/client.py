"""
Core OCR client for DeepSeek OCR SDK.

This module provides the main client for interacting with the DeepSeek OCR API.
"""

import asyncio
import base64
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
            **kwargs: Additional configuration parameters.

        Raises:
            ConfigurationError: If required configuration is missing
                or invalid.
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

    def _pdf_page_to_base64(self, doc: fitz.Document, page_num: int, dpi: int) -> str:
        """
        Convert a single PDF page to base64-encoded image.

        Args:
            doc: PyMuPDF document object.
            page_num: Page number (0-indexed).
            dpi: DPI for rendering (150, 200, or 300).

        Returns:
            Base64-encoded image string.

        Raises:
            FileProcessingError: If page cannot be processed.
        """
        try:
            page = doc[page_num]
            # Render page to image
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            # Convert to bytes
            img_bytes = pix.tobytes("png")

            # Encode to base64
            b64_string = base64.b64encode(img_bytes).decode("utf-8")
            logger.debug(
                f"Converted page {page_num + 1} to image: "
                f"{len(b64_string)} bytes at {dpi} DPI"
            )
            return b64_string

        except Exception as e:
            raise FileProcessingError(
                f"Failed to process page {page_num + 1}: {e}"
            ) from e

    def _pdf_to_base64(
        self,
        file_path: Union[str, Path],
        dpi: int,
        pages: Optional[Union[int, List[int]]] = None,
    ) -> Union[str, List[str]]:
        """
        Convert PDF pages to base64-encoded image(s).

        Args:
            file_path: Path to the PDF file.
            dpi: DPI for rendering (150, 200, or 300).
            pages: Page(s) to process. Can be:
                   - None: Process all pages (default)
                   - int: Process single page (1-indexed)
                   - list: Process specific pages (1-indexed)

        Returns:
            Base64-encoded image string (single page) or
            list of strings (multiple pages).

        Raises:
            FileProcessingError: If file cannot be processed.
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileProcessingError(f"File not found: {file_path}")

            doc = fitz.open(str(file_path))
            try:
                if len(doc) == 0:
                    raise FileProcessingError(f"PDF has no pages: {file_path}")

                # Determine which pages to process
                if pages is None:
                    # Process all pages
                    page_nums = list(range(len(doc)))
                elif isinstance(pages, int):
                    # Process single page (convert 1-indexed to 0-indexed)
                    if pages < 1 or pages > len(doc):
                        raise FileProcessingError(
                            f"Page {pages} out of range. Page numbers are 1-indexed "
                            f"(valid range: 1 to {len(doc)})"
                        )
                    page_nums = [pages - 1]
                else:
                    # Process list of pages (convert 1-indexed to 0-indexed)
                    if not pages:
                        raise FileProcessingError(
                            "Pages list cannot be empty. "
                            "Use None to process all pages."
                        )

                    # Deduplicate while preserving order
                    seen = set()
                    page_nums = []
                    for p in pages:
                        if p < 1 or p > len(doc):
                            raise FileProcessingError(
                                f"Page {p} out of range. Page numbers are 1-indexed "
                                f"(valid range: 1 to {len(doc)})"
                            )
                        page_idx = p - 1
                        if page_idx not in seen:
                            seen.add(page_idx)
                            page_nums.append(page_idx)

                # Process pages
                results = []
                for page_num in page_nums:
                    b64_string = self._pdf_page_to_base64(doc, page_num, dpi)
                    results.append(b64_string)

                # Return single string if single page, list otherwise
                if len(results) == 1:
                    return results[0]
                return results

            finally:
                doc.close()

        except Exception as e:
            if isinstance(e, FileProcessingError):
                raise
            raise FileProcessingError(f"Failed to process PDF: {e}") from e

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

    async def _make_api_request_async(
        self, image_b64: str, prompt: str
    ) -> Dict[str, Any]:
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

                    result: Dict[str, Any] = await response.json()
                    return result

        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"Request timed out after {self.config.timeout} seconds"
            ) from e

    def _make_api_request_sync(self, image_b64: str, prompt: str) -> Dict[str, Any]:
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

            result: Dict[str, Any] = response.json()
            return result

        except requests.Timeout as e:
            raise TimeoutError(
                f"Request timed out after {self.config.timeout} seconds"
            ) from e

    async def parse_async(
        self,
        file_path: Union[str, Path],
        mode: Union[str, OCRMode] = OCRMode.FREE_OCR,
        dpi: Optional[int] = None,
        pages: Optional[Union[int, List[int]]] = None,
    ) -> str:
        """
        Parse document asynchronously with concurrent page processing
        and per-page fallback.

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
            >>> text = await client.parse_async("document.pdf")
            >>> # Process only first page (old default behavior)
            >>> text = await client.parse_async("document.pdf", pages=1)
            >>> # With options
            >>> text = await client.parse_async(
            ...     "document.pdf",
            ...     mode="grounding",
            ...     dpi=300
            ... )
            >>> # Process specific pages
            >>> text = await client.parse_async(
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

        # Convert PDF to base64
        logger.info(f"Processing {file_path} with mode={mode} and dpi={dpi}")
        image_b64_result = self._pdf_to_base64(file_path, dpi, pages)

        # Build prompt
        prompt = self._build_prompt(mode)

        # Handle single page or multiple pages
        if isinstance(image_b64_result, str):
            # Single page
            images = [image_b64_result]
        else:
            # Multiple pages
            images = image_b64_result

        # Helper function to process a single page with intelligent fallback
        async def process_page(page_idx: int, image_b64: str) -> str:
            logger.debug(f"Processing page {page_idx + 1}/{len(images)}")

            # Make API request
            result = await self._make_api_request_async(image_b64, prompt)

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
                    fallback_result = await self._make_api_request_async(
                        image_b64, fallback_prompt
                    )

                    if (
                        "choices" in fallback_result
                        and len(fallback_result["choices"]) > 0
                    ):
                        text = fallback_result["choices"][0]["message"]["content"]
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

            return text

        # Process all pages concurrently for better performance
        all_results = await asyncio.gather(
            *[process_page(idx, img) for idx, img in enumerate(images)],
            return_exceptions=True,
        )

        # Handle exceptions and collect texts
        all_texts: List[str] = []
        for idx, result in enumerate(all_results):
            if isinstance(result, Exception):
                logger.error(f"Error processing page {idx + 1}: {result}")
                raise result  # Re-raise to maintain existing error behavior
            # At this point, result must be str (not Exception)
            all_texts.append(str(result))

        # Combine all pages with page separator
        combined_text = self.config.page_separator.join(all_texts)

        logger.info(
            f"Successfully processed {file_path}: "
            f"{len(images)} page(s), {len(combined_text)} chars"
        )
        return combined_text

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

        # Convert PDF to base64
        logger.info(f"Processing {file_path} with mode={mode} and dpi={dpi}")
        image_b64_result = self._pdf_to_base64(file_path, dpi, pages)

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

            # Make API request
            result = self._make_api_request_sync(image_b64, prompt)

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
                    fallback_result = self._make_api_request_sync(
                        image_b64, fallback_prompt
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
