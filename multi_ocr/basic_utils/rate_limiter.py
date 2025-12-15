"""
Rate limiting utilities for API requests.

Provides rate limiting and retry logic with support for both
synchronous and asynchronous operations.
"""

import asyncio
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """Manages rate limiting for API requests."""

    def __init__(
        self,
        request_delay: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        """
        Initialize rate limiter.

        Args:
            request_delay: Delay in seconds between API requests.
            max_retries: Maximum number of retries for rate limit errors.
            retry_delay: Initial delay in seconds before retrying (exponential backoff).
        """
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Track last request time for rate limiting
        self._last_request_time: Optional[float] = None
        # Locks to ensure thread-safe and async-safe rate limiting
        self._async_lock: Optional[asyncio.Lock] = None
        self._sync_lock = threading.Lock()

    def _get_async_lock(self) -> asyncio.Lock:
        """
        Get or create the async lock for rate limiting.

        Lazy initialization to avoid issues when client is created
        outside an event loop.

        Returns:
            asyncio.Lock instance.
        """
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    async def apply_rate_limit_async(self) -> None:
        """
        Apply rate limiting delay before making a request (async).

        If request_delay is configured, ensures minimum time between requests.
        Uses asyncio.Lock to ensure thread-safe access to _last_request_time
        in concurrent async operations.
        """
        if self.request_delay > 0:
            async with self._get_async_lock():
                if self._last_request_time is not None:
                    elapsed = time.time() - self._last_request_time
                    if elapsed < self.request_delay:
                        delay = self.request_delay - elapsed
                        logger.debug(
                            f"Rate limiting: waiting {delay:.2f}s before next request"
                        )
                        await asyncio.sleep(delay)
                # Update last request time inside the lock
                self._last_request_time = time.time()

    def apply_rate_limit_sync(self) -> None:
        """
        Apply rate limiting delay before making a request (sync).

        If request_delay is configured, ensures minimum time between requests.
        Uses threading.Lock to ensure thread-safe access to _last_request_time
        in concurrent sync operations.
        """
        if self.request_delay > 0:
            with self._sync_lock:
                if self._last_request_time is not None:
                    elapsed = time.time() - self._last_request_time
                    if elapsed < self.request_delay:
                        delay = self.request_delay - elapsed
                        logger.debug(
                            f"Rate limiting: waiting {delay:.2f}s before next request"
                        )
                        time.sleep(delay)
                # Update last request time inside the lock
                self._last_request_time = time.time()

    def get_retry_delay(self, attempt: int) -> float:
        """
        Calculate retry delay with exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Delay in seconds: retry_delay * (2 ^ attempt).
        """
        return self.retry_delay * (2**attempt)

    def should_retry(self, attempt: int, enable_retry: bool) -> bool:
        """
        Determine if should retry based on attempt count and settings.

        Args:
            attempt: Current attempt number (0-indexed).
            enable_retry: Whether retry is enabled.

        Returns:
            True if should retry, False otherwise.
        """
        return enable_retry and attempt < self.max_retries
