"""
Batch processing utilities for DeepSeek OCR SDK.

This module provides tools for processing multiple documents efficiently
with progress tracking and error handling.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

from tqdm import tqdm

from .client import DeepSeekOCR
from .enums import OCRMode

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """
    Result of processing a single document in batch.

    Attributes:
        file_path: Path to the processed file.
        success: Whether processing succeeded.
        text: Extracted text (if successful).
        error: Error message (if failed).
        mode_used: OCR mode that was used.
    """

    file_path: Path
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None
    mode_used: Optional[OCRMode] = None


@dataclass
class BatchSummary:
    """
    Summary of batch processing results.

    Attributes:
        total: Total number of documents processed.
        successful: Number of successful processes.
        failed: Number of failed processes.
        results: List of individual results.
    """

    total: int
    successful: int
    failed: int
    results: List[BatchResult]

    def print_summary(self) -> None:
        """Print a human-readable summary."""
        print(f"\n{'='*60}")
        print("Batch Processing Summary")
        print(f"{'='*60}")
        print(f"Total documents: {self.total}")
        success_pct = self.successful / self.total * 100
        fail_pct = self.failed / self.total * 100
        print(f"Successful: {self.successful} ({success_pct:.1f}%)")
        print(f"Failed: {self.failed} ({fail_pct:.1f}%)")

        if self.failed > 0:
            print("\nFailed documents:")
            for result in self.results:
                if not result.success:
                    print(f"  - {result.file_path.name}: {result.error}")


class BatchProcessor:
    """
    Batch processor for multiple documents.

    This processor handles multiple documents efficiently with:
    - Async processing for better performance
    - Progress tracking with tqdm
    - Automatic error handling and retry

    Example:
        >>> client = DeepSeekOCR(api_key="xxx")
        >>> processor = BatchProcessor(client)
        >>> files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        >>> summary = await processor.process_batch(files)
        >>> summary.print_summary()
    """

    def __init__(
        self,
        client: DeepSeekOCR,
        max_concurrent: int = 3,
        retry_count: int = 1,
    ):
        """
        Initialize BatchProcessor.

        Args:
            client: DeepSeekOCR client instance.
            max_concurrent: Maximum number of concurrent requests.
            retry_count: Number of retries for failed requests.
        """
        self.client = client
        self.max_concurrent = max_concurrent
        self.retry_count = retry_count

    async def _process_single(
        self,
        file_path: Union[str, Path],
        mode: Optional[OCRMode] = None,
        **kwargs: Any,
    ) -> BatchResult:
        """
        Process a single document with retry logic.

        Args:
            file_path: Path to document.
            mode: OCR mode (if None, uses FREE_OCR).
            **kwargs: Additional arguments for parse_async.

        Returns:
            BatchResult for this document.
        """
        file_path = Path(file_path)

        # Determine mode
        if mode is None:
            mode = OCRMode.FREE_OCR

        # Try processing with retries
        last_error = None
        for attempt in range(self.retry_count + 1):
            try:
                text = await self.client.parse_async(file_path, mode=mode, **kwargs)
                return BatchResult(
                    file_path=file_path,
                    success=True,
                    text=text,
                    mode_used=mode,
                )
            except Exception as e:
                last_error = str(e)
                if attempt < self.retry_count:
                    logger.warning(
                        f"Attempt {attempt + 1} failed for "
                        f"{file_path.name}, retrying: {e}"
                    )
                    await asyncio.sleep(1)  # Brief delay before retry
                else:
                    logger.error(f"All attempts failed for {file_path.name}: {e}")

        return BatchResult(
            file_path=file_path,
            success=False,
            error=last_error,
            mode_used=mode,
        )

    async def process_batch(
        self,
        file_paths: List[Union[str, Path]],
        mode: Optional[OCRMode] = None,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> BatchSummary:
        """
        Process multiple documents in batch.

        Args:
            file_paths: List of file paths to process.
            mode: OCR mode to use for all documents (default: FREE_OCR).
            show_progress: Show progress bar.
            **kwargs: Additional arguments for parse_async.

        Returns:
            BatchSummary with all results.

        Example:
            >>> processor = BatchProcessor(client)
            >>> files = list(Path("docs").glob("*.pdf"))
            >>> summary = await processor.process_batch(
            ...     files,
            ...     mode="grounding",
            ...     show_progress=True
            ... )
            >>> summary.print_summary()
        """
        if not file_paths:
            return BatchSummary(total=0, successful=0, failed=0, results=[])

        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_with_semaphore(file_path: Union[str, Path]) -> BatchResult:
            async with semaphore:
                return await self._process_single(file_path, mode=mode, **kwargs)

        # Process all files
        if show_progress:
            tasks = [process_with_semaphore(fp) for fp in file_paths]
            with tqdm(total=len(file_paths), desc="Processing documents") as pbar:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    results.append(result)
                    pbar.update(1)
                    if result.success:
                        pbar.set_postfix({"status": f"✓ {result.file_path.name}"})
                    else:
                        pbar.set_postfix({"status": f"✗ {result.file_path.name}"})
        else:
            tasks = [process_with_semaphore(fp) for fp in file_paths]
            results = await asyncio.gather(*tasks)

        # Calculate summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        return BatchSummary(
            total=len(results),
            successful=successful,
            failed=failed,
            results=results,
        )

    def process_batch_sync(
        self,
        file_paths: List[Union[str, Path]],
        mode: Optional[OCRMode] = None,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> BatchSummary:
        """
        Process multiple documents in batch (synchronous wrapper).

        This is a synchronous wrapper around process_batch() for convenience.

        Args:
            file_paths: List of file paths to process.
            mode: OCR mode to use for all documents (default: FREE_OCR).
            show_progress: Show progress bar.
            **kwargs: Additional arguments for parse_async.

        Returns:
            BatchSummary with all results.

        Example:
            >>> processor = BatchProcessor(client)
            >>> files = ["doc1.pdf", "doc2.pdf"]
            >>> summary = processor.process_batch_sync(files)
            >>> summary.print_summary()
        """
        return asyncio.run(
            self.process_batch(
                file_paths=file_paths,
                mode=mode,
                show_progress=show_progress,
                **kwargs,
            )
        )
