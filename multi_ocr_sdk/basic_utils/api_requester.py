"""
API request handling utilities.

Provides common functionality for making API requests with rate limiting,
retry logic, and error handling.
"""

import logging
from typing import Any, Dict, List, Optional

import requests

from ..exceptions import APIError, RateLimitError, TimeoutError
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class APIRequester:
    """Handles API requests with rate limiting and retry logic."""

    def __init__(self, rate_limiter: RateLimiter, timeout: int):
        """
        Initialize API requester.

        Args:
            rate_limiter: RateLimiter instance for managing delays and retries.
            timeout: Request timeout in seconds.
        """
        self.rate_limiter = rate_limiter
        self.timeout = timeout

    def request_sync(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        enable_rate_limit_retry: bool = True,
        timeout_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make synchronous API request with rate limiting and retry.

        Args:
            url: API endpoint URL.
            headers: Request headers (should include Authorization).
            payload: Request payload.
            enable_rate_limit_retry: Enable automatic retry on 429 errors.

        Returns:
            API response as dictionary.

        Raises:
            APIError: If API returns an error.
            RateLimitError: If rate limit is exceeded and retries exhausted.
            TimeoutError: If request times out.
        """

        for attempt in range(self.rate_limiter.max_retries + 1):
            try:
                # Apply rate limiting delay (updates _last_request_time atomically)
                self.rate_limiter.apply_rate_limit_sync()

                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=timeout_override or self.timeout,
                )

                # Handle rate limiting (429)
                if response.status_code == 429:
                    if (
                        not enable_rate_limit_retry
                        or attempt >= self.rate_limiter.max_retries
                    ):
                        raise RateLimitError(
                            f"Rate limit exceeded: {response.text}",
                            status_code=429,
                            response_text=response.text,
                        )

                    # Exponential backoff: delay * (2 ^ attempt)
                    retry_delay = self.rate_limiter.get_retry_delay(attempt)
                    logger.warning(
                        f"Rate limit hit (429), retrying in {retry_delay:.1f}s "
                        f"(attempt {attempt + 1}/{self.rate_limiter.max_retries})"
                    )
                    import time
                    time.sleep(retry_delay)
                    continue

                if response.status_code != 200:
                    raise APIError(
                        f"API request failed: {response.text}",
                        status_code=response.status_code,
                        response_text=response.text,
                    )

                result: Dict[str, Any] = response.json()
                return result

            except requests.Timeout as e:
                actual_timeout = timeout_override if timeout_override is not None else self.timeout
                raise TimeoutError(
                    f"Request timed out after {actual_timeout} seconds"
                ) from e

        # Should not reach here, but just in case
        raise RateLimitError("Rate limit retries exhausted", status_code=429)
