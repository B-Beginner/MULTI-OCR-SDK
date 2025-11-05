"""
Enumerations for DeepSeek OCR SDK.

This module defines the enumeration types used throughout the SDK.
"""

from enum import Enum


class OCRMode(Enum):
    """
    OCR processing modes for DeepSeek OCR API.

    Attributes:
        FREE_OCR: Fast mode that returns pure Markdown output.
                  Best for 80% of document processing scenarios.
                  Speed: 3.95-10.95s per page.

        GROUNDING: Advanced mode with HTML output and bounding boxes.
                   Optimal for complex tables (≥20 rows).
                   Speed: 5.18-8.31s per page.

        OCR_IMAGE: Detailed mode with word-level bounding boxes.
                   Slower and less stable, use only for edge cases.
                   Speed: 19-26s per page.

    Performance Guidelines:
        - Simple documents: Use FREE_OCR
        - Complex tables (≥20 rows): Use GROUNDING
        - Simple tables (<10 rows): Use FREE_OCR (not GROUNDING)
        - Chinese documents: Use FREE_OCR with chinese_hint=True
    """

    FREE_OCR = "free_ocr"
    GROUNDING = "grounding"
    OCR_IMAGE = "ocr_image"

    def get_prompt(self) -> str:
        """
        Get the API prompt string for this mode.

        Returns:
            The prompt string to send to the DeepSeek OCR API.
        """
        prompts = {
            OCRMode.FREE_OCR: "Free OCR.",
            OCRMode.GROUNDING: "<|grounding|>Convert the document to markdown.",
            OCRMode.OCR_IMAGE: "<|grounding|>OCR this image.",
        }
        return prompts[self]

    def __str__(self) -> str:
        """String representation of the mode."""
        return self.value
