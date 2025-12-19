#20251219开始人工审阅
"""
这是一个用于异步处理的函数，但是现在把异步砍掉了，这个函数好像没用了？等看完全部代码再说
"""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    # 这个类用于管理API请求的速率限制和重试逻辑

    def __init__(
        self,
        request_delay: float = 0.0, 
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        """
        初始化rate limiter。
        参数说明：
            request_delay: API请求之间的延迟时间（秒）。
            max_retries: 速率限制错误的最大重试次数。
            retry_delay: 重试前的初始延迟时间（秒）（指数退避
        """
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 记录上一次请求的时间，用于速率限制
        self._last_request_time: Optional[float] = None
        # 确保线程安全的速率限制
        self._sync_lock = threading.Lock()

    def apply_rate_limit_sync(self) -> None:
        """
        在发起请求之前应用速率限制延迟
        如果 request_delay 被配置，确保请求之间的最小时间间隔。
        使用 threading.Lock 确保在并发同步操作中对 _last_request_time 的线程安全访问。
        """
        if self.request_delay > 0:
            # 仅在配置了 request_delay 时才启用等待逻辑
            # 使用锁以保证多线程环境下对 _last_request_time 的访问是线程安全的
            with self._sync_lock:
                # 如果存在上一次请求时间，则计算自上次请求以来经过的时间
                if self._last_request_time is not None:
                    elapsed = time.time() - self._last_request_time
                    # 如果距离上次请求时间小于配置的最小间隔，则需要等待剩余时间
                    if elapsed < self.request_delay:
                        delay = self.request_delay - elapsed
                        logger.debug(
                            f"Rate limiting: waiting {delay:.2f}s before next request"
                        )
                        # 阻塞当前线程等待，以确保两个请求之间至少间隔 request_delay 秒
                        time.sleep(delay)
                # 在锁内更新上一次请求时间，避免竞争条件
                self._last_request_time = time.time()

    def get_retry_delay(self, attempt: int) -> float:
        """
        计算重试延迟时间，第一次是2的1次方，第二次是2的2次方，以此类推。
        """
        return self.retry_delay * (2**attempt)

    def should_retry(self, attempt: int, enable_retry: bool) -> bool:
        """
        如果启用了重试并且尝试次数小于最大重试次数，则返回True。
        用于决定在遇到速率限制错误时是否应进行重试。
        """
        return enable_retry and attempt < self.max_retries

# 20251219结束人工审阅