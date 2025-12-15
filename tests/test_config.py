"""Tests for configuration module."""

import os

import pytest

from multi_ocr_sdk import ConfigurationError, OCRConfig


def test_config_from_env_with_api_key():
    """Test creating config from environment with API key."""
    os.environ["DS_OCR_API_KEY"] = "test_key"
    os.environ["DS_OCR_BASE_URL"] = "https://api.test.com/v1/chat/completions"
    config = OCRConfig.from_env()
    assert config.api_key == "test_key"
    assert config.base_url == "https://api.test.com/v1/chat/completions"
    assert config.dpi == 200
    del os.environ["DS_OCR_API_KEY"]
    del os.environ["DS_OCR_BASE_URL"]


def test_config_with_overrides():
    """Test creating config with overrides."""
    config = OCRConfig.from_env(
        api_key="override_key",
        base_url="https://api.override.com/v1/chat/completions",
        dpi="300",
    )
    assert config.api_key == "override_key"
    assert config.base_url == "https://api.override.com/v1/chat/completions"
    assert config.dpi == 300


def test_config_invalid_dpi():
    """Test that invalid DPI raises error."""
    with pytest.raises(ConfigurationError):
        OCRConfig(
            api_key="test",
            base_url="https://api.test.com/v1/chat/completions",
            dpi=100,
        )


def test_config_missing_api_key():
    """Test that missing API key raises error."""
    with pytest.raises(ConfigurationError):
        OCRConfig.from_env()


def test_config_defaults():
    """Test default configuration values."""
    config = OCRConfig(
        api_key="test_key", base_url="https://api.test.com/v1/chat/completions"
    )
    assert config.base_url == "https://api.test.com/v1/chat/completions"
    assert config.model_name == "deepseek-ai/DeepSeek-OCR"
    assert config.timeout == 60
    assert config.max_tokens == 4000
    assert config.dpi == 200
    assert config.fallback_enabled is True
