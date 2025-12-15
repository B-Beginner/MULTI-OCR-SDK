"""
File processing utilities for handling PDFs and images.

Provides utilities to convert PDF pages and images to base64 format
for API requests.
"""

import base64
import logging
from pathlib import Path
from typing import List, Optional, Union

import fitz  # PyMuPDF

from ..exceptions import FileProcessingError

logger = logging.getLogger(__name__)


class FileProcessor:
    """Utilities for processing PDF and image files."""

    @staticmethod
    def pdf_page_to_base64(doc: fitz.Document, page_num: int, dpi: int) -> str:
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

    @staticmethod
    def file_to_base64(
        file_path: Union[str, Path],
        dpi: int,
        pages: Optional[Union[int, List[int]]] = None,
    ) -> Union[str, List[str]]:
        """
        Convert file (PDF or image) pages to base64-encoded image(s).

        Args:
            file_path: Path to the file.
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

            # Support common image files directly (single-image -> single base64 string)
            image_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
            if file_path.suffix.lower() in image_exts:
                try:
                    img_bytes = file_path.read_bytes()
                    return base64.b64encode(img_bytes).decode("utf-8")
                except Exception as e:
                    raise FileProcessingError(f"Failed to read image file: {e}") from e

            # Fallback to PDF-like handling for multi-page documents (uses PyMuPDF)
            doc = fitz.open(str(file_path))
            try:
                if len(doc) == 0:
                    raise FileProcessingError(f"File has no pages: {file_path}")

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
                    b64_string = FileProcessor.pdf_page_to_base64(doc, page_num, dpi)
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
            raise FileProcessingError(f"Failed to process file: {e}") from e

    @staticmethod
    def pdf_to_base64(
        file_path: Union[str, Path],
        dpi: int,
        pages: Optional[Union[int, List[int]]] = None,
    ) -> Union[str, List[str]]:
        """
        Alias for file_to_base64 specifically for PDF files.

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
        return FileProcessor.file_to_base64(file_path, dpi, pages)
