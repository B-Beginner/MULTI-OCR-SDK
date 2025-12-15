"""Pytest configuration and fixtures."""

import pytest

from multi_ocr_sdk import DeepSeekOCR, OCRConfig


@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing."""
    return "test_api_key_12345"


@pytest.fixture
def test_config(mock_api_key):
    """Provide a test configuration."""
    return OCRConfig(
        api_key=mock_api_key,
        base_url="https://api.test.com/v1/chat/completions",
        timeout=30,
        dpi=200,
    )


@pytest.fixture
def test_client(mock_api_key):
    """Provide a test client instance."""
    return DeepSeekOCR(
        api_key=mock_api_key,
        base_url="https://api.test.com/v1/chat/completions",
        timeout=30,
    )
