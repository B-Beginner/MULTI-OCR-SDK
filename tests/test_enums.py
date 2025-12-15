"""Tests for enums module."""

from multi_ocr_sdk import OCRMode


def test_ocr_mode_values():
    """Test OCR mode enum values."""
    assert OCRMode.FREE_OCR.value == "free_ocr"
    assert OCRMode.GROUNDING.value == "grounding"
    assert OCRMode.OCR_IMAGE.value == "ocr_image"


def test_ocr_mode_prompts():
    """Test OCR mode prompt generation."""
    assert OCRMode.FREE_OCR.get_prompt() == "Free OCR."
    assert "grounding" in OCRMode.GROUNDING.get_prompt().lower()
    assert "grounding" in OCRMode.OCR_IMAGE.get_prompt().lower()


def test_ocr_mode_string_representation():
    """Test OCR mode string representation."""
    assert str(OCRMode.FREE_OCR) == "free_ocr"
    assert str(OCRMode.GROUNDING) == "grounding"
    assert str(OCRMode.OCR_IMAGE) == "ocr_image"
