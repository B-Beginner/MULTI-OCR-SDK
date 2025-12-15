"""
Tests for the DeepSeekOCR client.
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepseek_ocr import DeepSeekOCR
from deepseek_ocr.exceptions import FileProcessingError


@pytest.fixture
def mock_pdf():
    """Create a mock PDF document with multiple pages."""
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 3  # 3 pages

    # Mock pages
    mock_pages = []
    for i in range(3):
        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.tobytes.return_value = f"page{i}_image_bytes".encode()
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_pages.append(mock_page)

    mock_doc.__getitem__.side_effect = lambda idx: mock_pages[idx]
    return mock_doc


def test_pdf_to_base64_all_pages(mock_pdf):
    """Test processing all pages of a PDF."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    with patch("fitz.open", return_value=mock_pdf):
        with patch("pathlib.Path.exists", return_value=True):
            # Process all pages (default)
            result = client._pdf_to_base64(Path("test.pdf"), dpi=200, pages=None)

            # Should return a list of 3 base64 strings
            assert isinstance(result, list)
            assert len(result) == 3


def test_pdf_to_base64_single_page(mock_pdf):
    """Test processing a single page of a PDF."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    with patch("fitz.open", return_value=mock_pdf):
        with patch("pathlib.Path.exists", return_value=True):
            # Process single page
            result = client._pdf_to_base64(Path("test.pdf"), dpi=200, pages=1)

            # Should return a single base64 string
            assert isinstance(result, str)


def test_pdf_to_base64_specific_pages(mock_pdf):
    """Test processing specific pages of a PDF."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    with patch("fitz.open", return_value=mock_pdf):
        with patch("pathlib.Path.exists", return_value=True):
            # Process pages 1 and 3
            result = client._pdf_to_base64(Path("test.pdf"), dpi=200, pages=[1, 3])

            # Should return a list of 2 base64 strings
            assert isinstance(result, list)
            assert len(result) == 2


def test_pdf_to_base64_page_out_of_range(mock_pdf):
    """Test error handling for page out of range."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    with patch("fitz.open", return_value=mock_pdf):
        with patch("pathlib.Path.exists", return_value=True):
            # Try to access page 5 (out of range)
            with pytest.raises(FileProcessingError) as exc_info:
                client._pdf_to_base64(Path("test.pdf"), dpi=200, pages=5)

            assert "out of range" in str(exc_info.value).lower()


def test_pdf_to_base64_file_not_found():
    """Test error handling for non-existent file."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileProcessingError) as exc_info:
            client._pdf_to_base64(Path("nonexistent.pdf"), dpi=200)

        assert "not found" in str(exc_info.value).lower()


def test_pdf_to_base64_empty_pdf():
    """Test error handling for PDF with no pages."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 0

    with patch("fitz.open", return_value=mock_doc):
        with patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(FileProcessingError) as exc_info:
                client._pdf_to_base64(Path("empty.pdf"), dpi=200)

            assert "no pages" in str(exc_info.value).lower()


def test_pdf_to_base64_empty_pages_list(mock_pdf):
    """Test error handling for empty pages list."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    with patch("fitz.open", return_value=mock_pdf):
        with patch("pathlib.Path.exists", return_value=True):
            # Try to process with empty list
            with pytest.raises(FileProcessingError) as exc_info:
                client._pdf_to_base64(Path("test.pdf"), dpi=200, pages=[])

            assert "cannot be empty" in str(exc_info.value).lower()


def test_pdf_to_base64_negative_page_number(mock_pdf):
    """Test error handling for negative page numbers."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    with patch("fitz.open", return_value=mock_pdf):
        with patch("pathlib.Path.exists", return_value=True):
            # Try to access page -1 (negative)
            with pytest.raises(FileProcessingError) as exc_info:
                client._pdf_to_base64(Path("test.pdf"), dpi=200, pages=-1)

            assert "out of range" in str(exc_info.value).lower()
            assert "1-indexed" in str(exc_info.value).lower()


def test_pdf_to_base64_zero_page_number(mock_pdf):
    """Test error handling for zero page number."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    with patch("fitz.open", return_value=mock_pdf):
        with patch("pathlib.Path.exists", return_value=True):
            # Try to access page 0 (invalid in 1-indexed)
            with pytest.raises(FileProcessingError) as exc_info:
                client._pdf_to_base64(Path("test.pdf"), dpi=200, pages=0)

            assert "out of range" in str(exc_info.value).lower()
            assert "1-indexed" in str(exc_info.value).lower()


def test_pdf_to_base64_single_page_as_list(mock_pdf):
    """Test processing single page specified as list."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    with patch("fitz.open", return_value=mock_pdf):
        with patch("pathlib.Path.exists", return_value=True):
            result = client._pdf_to_base64(Path("test.pdf"), dpi=200, pages=[1])
            # Should return string (single page), not list
            assert isinstance(result, str)


def test_pdf_to_base64_duplicate_pages(mock_pdf):
    """Test deduplication of duplicate page numbers."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    with patch("fitz.open", return_value=mock_pdf):
        with patch("pathlib.Path.exists", return_value=True):
            # Process pages with duplicates: [1, 2, 1, 3, 2]
            result = client._pdf_to_base64(
                Path("test.pdf"), dpi=200, pages=[1, 2, 1, 3, 2]
            )

            # Should return only 3 unique pages in order: [1, 2, 3]
            assert isinstance(result, list)
            assert len(result) == 3


# Integration tests for parse/parse_async methods


def test_parse_multipage_integration(mock_pdf):
    """Test parse method with multi-page PDF."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    # Disable fallback for this test
    client.config.fallback_enabled = False

    # Mock API response
    mock_response = {
        "choices": [{"message": {"content": "Page content"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }

    with patch("fitz.open", return_value=mock_pdf):
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(
                client, "_make_api_request_sync", return_value=mock_response
            ) as mock_api:
                result = client.parse(Path("test.pdf"), pages=[1, 2])

                # Should call API twice (once per page)
                assert mock_api.call_count == 2
                # Should combine results with page separator
                assert "---" in result
                # Should have content from both pages
                assert result.count("Page content") == 2


@pytest.mark.asyncio
async def test_parse_async_multipage_integration(mock_pdf):
    """Test parse_async method with multi-page PDF."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    # Disable fallback for this test
    client.config.fallback_enabled = False

    # Mock API response
    mock_response = {
        "choices": [{"message": {"content": "Page content"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }

    with patch("fitz.open", return_value=mock_pdf):
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(
                client, "_make_api_request_async", return_value=mock_response
            ) as mock_api:
                result = await client.parse_async(Path("test.pdf"), pages=[1, 2])

                # Should call API twice (once per page)
                assert mock_api.call_count == 2
                # Should combine results with page separator
                assert "---" in result
                # Should have content from both pages
                assert result.count("Page content") == 2


@pytest.mark.asyncio
async def test_parse_async_with_fallback(mock_pdf):
    """Test parse_async with fallback when output is short."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    # Configure fallback
    client.config.fallback_enabled = True
    client.config.min_output_threshold = 500
    client.config.fallback_mode = "grounding"

    # First response is short (triggers fallback), second is long
    short_response = {
        "choices": [{"message": {"content": "Short"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    long_response = {
        "choices": [{"message": {"content": "A" * 600}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }

    with patch("fitz.open", return_value=mock_pdf):
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(
                client,
                "_make_api_request_async",
                side_effect=[short_response, long_response],
            ) as mock_api:
                result = await client.parse_async(Path("test.pdf"), pages=1)

                # Should call API twice (initial + fallback)
                assert mock_api.call_count == 2
                # Should return long fallback result
                assert len(result) > 500


def test_parse_with_fallback(mock_pdf):
    """Test parse with fallback when output is short."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    # Configure fallback
    client.config.fallback_enabled = True
    client.config.min_output_threshold = 500
    client.config.fallback_mode = "grounding"

    # First response is short (triggers fallback), second is long
    short_response = {
        "choices": [{"message": {"content": "Short"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    long_response = {
        "choices": [{"message": {"content": "A" * 600}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }

    with patch("fitz.open", return_value=mock_pdf):
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(
                client,
                "_make_api_request_sync",
                side_effect=[short_response, long_response],
            ) as mock_api:
                result = client.parse(Path("test.pdf"), pages=1)

                # Should call API twice (initial + fallback)
                assert mock_api.call_count == 2
                # Should return long fallback result
                assert len(result) > 500


# Tests for save functionality


def test_save_output():
    """Test saving OCR output to markdown file."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    test_text = "# Test Document\n\nThis is a test."
    test_file = Path("/tmp/test_document.pdf")

    with patch("pathlib.Path.cwd") as mock_cwd:
        mock_cwd.return_value = Path("/tmp")

        # Call save method
        output_path = client._save_output(test_text, test_file)

        # Check output path is correct
        assert output_path == Path("/tmp/ocr-output/test_document.md")
        assert output_path.exists()

        # Check content is saved correctly
        saved_content = output_path.read_text(encoding="utf-8")
        assert saved_content == test_text

        # Cleanup
        if output_path.parent.exists():
            shutil.rmtree(output_path.parent)


def test_parse_with_save(mock_pdf):
    """Test parse method with save parameter."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    # Disable fallback for this test
    client.config.fallback_enabled = False

    # Mock API response
    mock_response = {
        "choices": [{"message": {"content": "Test content"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }

    with patch("fitz.open", return_value=mock_pdf):
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(
                client, "_make_api_request_sync", return_value=mock_response
            ):
                with patch("pathlib.Path.cwd") as mock_cwd:
                    mock_cwd.return_value = Path("/tmp")

                    # Parse with save=True
                    result = client.parse(Path("test.pdf"), pages=1, save=True)

                    # Check result is returned
                    assert result == "Test content"

                    # Check file is saved
                    output_path = Path("/tmp/ocr-output/test.md")
                    assert output_path.exists()

                    # Check content
                    saved_content = output_path.read_text(encoding="utf-8")
                    assert saved_content == "Test content"

                    # Cleanup
                    if output_path.parent.exists():
                        shutil.rmtree(output_path.parent)


@pytest.mark.asyncio
async def test_parse_async_with_save(mock_pdf):
    """Test parse_async method with save parameter."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    # Disable fallback for this test
    client.config.fallback_enabled = False

    # Mock API response
    mock_response = {
        "choices": [{"message": {"content": "Test async content"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }

    with patch("fitz.open", return_value=mock_pdf):
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(
                client, "_make_api_request_async", return_value=mock_response
            ):
                with patch("pathlib.Path.cwd") as mock_cwd:
                    mock_cwd.return_value = Path("/tmp")

                    # Parse with save=True
                    result = await client.parse_async(
                        Path("test.pdf"), pages=1, save=True
                    )

                    # Check result is returned
                    assert result == "Test async content"

                    # Check file is saved
                    output_path = Path("/tmp/ocr-output/test.md")
                    assert output_path.exists()

                    # Check content
                    saved_content = output_path.read_text(encoding="utf-8")
                    assert saved_content == "Test async content"

                    # Cleanup
                    if output_path.parent.exists():
                        shutil.rmtree(output_path.parent)


def test_parse_without_save(mock_pdf):
    """Test parse method with save=False (default)."""
    client = DeepSeekOCR(api_key="test_key", base_url="http://test.com")

    # Disable fallback for this test
    client.config.fallback_enabled = False

    # Mock API response
    mock_response = {
        "choices": [{"message": {"content": "Test content"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }

    with patch("fitz.open", return_value=mock_pdf):
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(
                client, "_make_api_request_sync", return_value=mock_response
            ):
                with patch.object(client, "_save_output") as mock_save:
                    # Parse with save=False (default)
                    result = client.parse(Path("test.pdf"), pages=1)

                    # Check result is returned
                    assert result == "Test content"

                    # Check _save_output was NOT called
                    mock_save.assert_not_called()
