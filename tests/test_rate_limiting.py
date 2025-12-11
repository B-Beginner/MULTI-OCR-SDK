"""
Tests for rate limiting functionality.
"""

import time
from typing import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepseek_ocr import DeepSeekOCR
from deepseek_ocr.exceptions import RateLimitError


@pytest.fixture
def client():
    """Create a test client with rate limiting enabled."""
    return DeepSeekOCR(
        api_key="test_key",
        base_url="http://test.com",
        request_delay=1.0,
        enable_rate_limit_retry=True,
        max_rate_limit_retries=2,
        rate_limit_retry_delay=0.5,
    )


@pytest.fixture
def client_no_retry():
    """Create a test client with rate limiting retry disabled."""
    return DeepSeekOCR(
        api_key="test_key",
        base_url="http://test.com",
        request_delay=0.5,
        enable_rate_limit_retry=False,
    )


def create_mock_session(response_factory: Callable[[], Awaitable]):
    """
    Create a mock aiohttp.ClientSession for testing.

    Args:
        response_factory: Async function that creates mock responses.

    Returns:
        Mock session class.
    """

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args, **kwargs):
            pass

        def post(self, *args, **kwargs):
            class PostContextManager:
                async def __aenter__(self):
                    return await response_factory()

                async def __aexit__(self, *args, **kwargs):
                    pass

            return PostContextManager()

    return MockSession


def test_rate_limit_config_validation():
    """Test that rate limiting configuration is validated."""
    # Valid configuration
    client = DeepSeekOCR(
        api_key="test_key",
        base_url="http://test.com",
        request_delay=1.0,
        max_rate_limit_retries=3,
        rate_limit_retry_delay=5.0,
    )
    assert client.config.request_delay == 1.0
    assert client.config.max_rate_limit_retries == 3
    assert client.config.rate_limit_retry_delay == 5.0


def test_request_delay_sync(client):
    """Test that request delay is applied in synchronous requests."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "test content"}}]
    }

    with patch("requests.post", return_value=mock_response):
        start_time = time.time()

        # First request (no delay expected)
        client._make_api_request_sync("image_b64", "prompt")

        # Second request (should have delay)
        client._make_api_request_sync("image_b64", "prompt")

        elapsed = time.time() - start_time

        # Should have at least 1 second delay between requests
        # Allow some tolerance for execution time
        assert elapsed >= 0.9


@pytest.mark.asyncio
async def test_request_delay_async(client):
    """Test that request delay is applied in asynchronous requests."""

    async def create_mock_response():
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={"choices": [{"message": {"content": "test content"}}]}
        )
        return mock_resp

    MockSession = create_mock_session(create_mock_response)

    with patch("aiohttp.ClientSession", MockSession):
        start_time = time.time()

        # First request (no delay expected)
        await client._make_api_request_async("image_b64", "prompt")

        # Second request (should have delay)
        await client._make_api_request_async("image_b64", "prompt")

        elapsed = time.time() - start_time

        # Should have at least 1 second delay between requests
        # Allow some tolerance for execution time
        assert elapsed >= 0.9


@pytest.mark.asyncio
async def test_rate_limit_retry_async(client):
    """Test that 429 errors trigger retry with exponential backoff."""
    call_count = 0

    async def create_mock_response():
        nonlocal call_count
        call_count += 1

        mock_resp = AsyncMock()
        # First two calls return 429, third returns 200
        if call_count <= 2:
            mock_resp.status = 429
            mock_resp.text = AsyncMock(return_value="Rate limit exceeded")
        else:
            mock_resp.status = 200
            mock_resp.json = AsyncMock(
                return_value={"choices": [{"message": {"content": "success"}}]}
            )
        return mock_resp

    MockSession = create_mock_session(create_mock_response)

    with patch("aiohttp.ClientSession", MockSession):
        start_time = time.time()
        result = await client._make_api_request_async("image_b64", "prompt")
        elapsed = time.time() - start_time

        # Should have succeeded after retries
        assert result["choices"][0]["message"]["content"] == "success"
        # Should have waited for backoff delays (0.5s + 1.0s = 1.5s minimum)
        assert elapsed >= 1.4  # Allow some tolerance
        # Should have made 3 calls
        assert call_count == 3


def test_rate_limit_retry_sync(client):
    """Test that 429 errors trigger retry in synchronous mode."""
    call_count = 0

    def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        mock_response = MagicMock()
        # First two calls return 429, third returns 200
        if call_count <= 2:
            mock_response.status_code = 429
            mock_response.text = "Rate limit exceeded"
        else:
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "success"}}]
            }
        return mock_response

    with patch("requests.post", side_effect=mock_post):
        start_time = time.time()
        result = client._make_api_request_sync("image_b64", "prompt")
        elapsed = time.time() - start_time

        # Should have succeeded after retries
        assert result["choices"][0]["message"]["content"] == "success"
        # Should have waited for backoff delays (0.5s + 1.0s = 1.5s minimum)
        assert elapsed >= 1.4  # Allow some tolerance
        # Should have made 3 calls
        assert call_count == 3


@pytest.mark.asyncio
async def test_rate_limit_no_retry_async(client_no_retry):
    """Test that 429 errors are raised immediately when retry is disabled."""

    async def create_mock_response():
        mock_resp = AsyncMock()
        mock_resp.status = 429
        mock_resp.text = AsyncMock(return_value="Rate limit exceeded")
        return mock_resp

    MockSession = create_mock_session(create_mock_response)

    with patch("aiohttp.ClientSession", MockSession):
        with pytest.raises(RateLimitError) as exc_info:
            await client_no_retry._make_api_request_async("image_b64", "prompt")

        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in str(exc_info.value)


def test_rate_limit_no_retry_sync(client_no_retry):
    """Test that 429 errors are raised immediately when retry is disabled."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = "Rate limit exceeded"

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(RateLimitError) as exc_info:
            client_no_retry._make_api_request_sync("image_b64", "prompt")

        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_rate_limit_exhausted_retries_async(client):
    """Test that RateLimitError is raised when retries are exhausted."""

    async def create_mock_response():
        mock_resp = AsyncMock()
        mock_resp.status = 429
        mock_resp.text = AsyncMock(return_value="Rate limit exceeded")
        return mock_resp

    MockSession = create_mock_session(create_mock_response)

    with patch("aiohttp.ClientSession", MockSession):
        with pytest.raises(RateLimitError) as exc_info:
            await client._make_api_request_async("image_b64", "prompt")

        assert exc_info.value.status_code == 429


def test_rate_limit_exhausted_retries_sync(client):
    """Test that RateLimitError is raised when retries are exhausted."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = "Rate limit exceeded"

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(RateLimitError) as exc_info:
            client._make_api_request_sync("image_b64", "prompt")

        assert exc_info.value.status_code == 429
