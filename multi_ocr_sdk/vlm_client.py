"""
该模块提供一个SDK，用于调用 VLM 完成 OCR。
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from .exceptions import (
    ConfigurationError,
)
from .basic_utils import FileProcessor, RateLimiter, APIRequester, BaseConfig
from .basic_utils.basic_logger import setup_file_logger

logger = logging.getLogger(__name__)

# 20251219开始人工审阅
"""
kw_only=True 是一个 python3.10 及以上版本 dataclass 的参数
启用了这个参数以后，有默认值的字段可省略（使用默认值），无默认值的字段必须填写
"""
@dataclass(kw_only=True)
class VLMConfig(BaseConfig):
    # VLM 特有的字段；常见字段（如 api_key、base_url、timeout 等）由 BaseConfig 提供并校验
    model: str
    timeout: int = 60
    temperature: float = 0.0 # 温度越小，幻觉越少，OCR场景的温度设置为0
    request_delay: float = 0.0 # 两次请求之间的间隔，如果达到了访问上限，这个值可以调高一些
    enable_rate_limit_retry: bool = True # 如果遇到429报错（限流）是否重试
    max_rate_limit_retries: int = 3 # 最大重试次数
    rate_limit_retry_delay: float = 5.0 # 重试的间隔
    enable_log: bool = False  # 如果为 True，则在运行目录创建 multi-ocr-sdk-logs 并写入日志文件

    @classmethod
    def from_env(cls, **overrides: Any) -> "VLMConfig":
        get_env = BaseConfig._get_env

        env_config = {
            "api_key": get_env("VLM_API_KEY"),
            "base_url": get_env("VLM_BASE_URL"),
            "model": get_env("VLM_MODEL"),
            "timeout": get_env("VLM_TIMEOUT", type_func=int),
            "max_tokens": get_env("VLM_MAX_TOKENS", type_func=int),
            "temperature": get_env("VLM_TEMPERATURE", type_func=float),
            "request_delay": get_env("VLM_REQUEST_DELAY", type_func=float),
            "enable_rate_limit_retry": get_env("VLM_ENABLE_RATE_LIMIT_RETRY", type_func=bool),
            "max_rate_limit_retries": get_env("VLM_MAX_RATE_LIMIT_RETRIES", type_func=int),
            "enable_log": get_env("VLM_ENABLE_LOG", type_func=bool),
            "rate_limit_retry_delay": get_env("VLM_RATE_LIMIT_RETRY_DELAY", type_func=float),
        }

        # 删除环境变量里没有指定的项，使用前面设置好的默认值
        env_config = {k: v for k, v in env_config.items() if v is not None}
        
        # 将加载好的环境变量覆盖传入的参数（如果没有覆盖，还用的是前面设置好的默认值）
        env_config.update({k: v for k, v in overrides.items() if v is not None})

        return cls(**env_config)

    def __post_init__(self) -> None:
        # 使用 BaseConfig 提供的通用校验
        super().__post_init__()

        # VLM 特有的校验
        if not self.model:
            raise ConfigurationError("VLM model is required.")

        # 规范化 base_url，确保它指向正确的端点 xxx.com/v1/chat/completions
        if self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/chat/completions"
        elif self.base_url.endswith("/v1/"):
            self.base_url = f"{self.base_url}chat/completions"


class _CompletionsAPI:
    """
    这是干嘛的？没看懂 #question
    """ 
    def __init__(self, client: "VLMClient") -> None:
        self._client = client

    def create(self, model: str, messages: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """Synchronous completion call."""
        return self._client._make_api_request_sync(model, messages, **kwargs)


class _ChatAPI:
    """
    这是干嘛的？没看懂  #question
    """
    def __init__(self, client: "VLMClient") -> None:
        self.completions = _CompletionsAPI(client)


class VLMClient:
    """
    构建一个VLM client，用于OCR
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        request_delay: Optional[float] = None,
        enable_log: Optional[bool] = None,
        **overrides: Any,
    ) -> None:
        # 设置client要使用的变量
        config_args = {
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "timeout": timeout,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "request_delay": request_delay,
            "enable_log": enable_log,
            # "enable_rate_limit_retry": enable_rate_limit_retry,
            # "max_rate_limit_retries": max_rate_limit_retries,
            # "rate_limit_retry_delay": rate_limit_retry_delay,
        }
        config_args.update(overrides)
        
        self.config = VLMConfig.from_env(**config_args)

        # 如果启用了日志功能，委托 basic_logger.setup_file_logger 去创建目录和文件处理器
        if getattr(self.config, "enable_log", False):
            log_file = setup_file_logger()
            logger.info(f"Logging enabled. Writing logs to {log_file}")
        
        # 初始化 RateLimiter 和 APIRequester
        self._rate_limiter = RateLimiter(
            request_delay=self.config.request_delay,
            max_retries=self.config.max_rate_limit_retries,
            retry_delay=self.config.rate_limit_retry_delay,
        )
        self._api_requester = APIRequester(self._rate_limiter, self.config.timeout)

        # 初始化 Chat API（用于发送消息和接收回复）
        self.chat = _ChatAPI(self)

    def _process_single_page(
        self,
        image_b64: str,
        prompt: str,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        # 构建消息内容，包含图像和prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # 调用VLM的chat completions接口，获取OCR结果
        # 将 max_tokens 显式地传递给 API 调用（如果提供）
        call_kwargs = dict(kwargs)
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens

        result = self.chat.completions.create(
            model=model or self.config.model,
            messages=messages,
            timeout=timeout,
            **call_kwargs
        )
        
        # 如果返回结果中包含文本，提取出来并添加到结果列表中
        if "choices" in result and len(result["choices"]) > 0:
            return str(result["choices"][0]["message"]["content"])
        else:
            return ""

    def parse(
        self,
        file_path: Union[str, Path],
        prompt: str,
        model: Optional[str] = None,
        dpi: int = 72, # 默认72dpi，如果过高，token会超限，如果过低图片会变模糊，OCR效果会变差
        pages: Optional[Union[int, List[int]]] = None,
        timeout: Optional[int] = None,
        concurrency_num: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        这个函数直接使用VLMclient对文件进行OCR识别。
        """
        # 将文件转换为base64编码的图像
        logger.info(f"Processing {file_path} with dpi={dpi}, pages={pages} (type: {type(pages)})")
        image_b64_result = FileProcessor.file_to_base64(file_path, dpi, pages)

        # 处理单页和多页的情况
        if isinstance(image_b64_result, str):
            images = [image_b64_result]
        else:
            images = image_b64_result

        logger.info(f"Converted to {len(images)} images for processing")

        # 确定并发数（仅在 parse 层面生效，client 级别不再保存默认并发）
        actual_concurrency = concurrency_num if concurrency_num is not None else 1
        if actual_concurrency < 1:
            actual_concurrency = 1

        # 处理每页的 max_tokens（优先级：parse 参数 > client config）
        actual_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        # 新建一个空列表，用于存储每一页的OCR结果
        all_texts = [""] * len(images)

        # 如果多页，且设置并发大于1，才启动并发
        if actual_concurrency > 1 and len(images) > 1: 
            logger.info(f"Processing {len(images)} pages with concurrency={actual_concurrency}")

            # 创建一个thread池，并提交任务
            with ThreadPoolExecutor(max_workers=actual_concurrency) as executor:
                
                # 创建一个字典，存储每个pdf页面对应的图像base64编码、prompt、model、timeout等参数
                future_to_idx = {
                    executor.submit(
                        self._process_single_page, 
                        image_b64, 
                        prompt, 
                        model, 
                        timeout,
                        actual_max_tokens,
                        **kwargs
                    ): idx 
                    for idx, image_b64 in enumerate(images)
                }
                
                # 遍历完成的任务，获取每个页面的OCR结果，并存储到all_texts列表中
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        logger.debug(f"Processing page {idx + 1}/{len(images)}")
                        all_texts[idx] = future.result()
                    except Exception as e:
                        logger.error(f"Error processing page {idx + 1}: {e}")
                        all_texts[idx] = "" # 或者抛出异常
        else:

            # 如果不用并发，顺序处理每一页
            for page_idx, image_b64 in enumerate(images):
                logger.debug(f"Processing page {page_idx + 1}/{len(images)}")
                all_texts[page_idx] = self._process_single_page(
                    image_b64, prompt, model, timeout, actual_max_tokens, **kwargs
                )

        return "\n\n---\n\n".join(all_texts)



    # 设置限流功能（这是个异步并发时才用到的功能吗？） #question
    def _apply_rate_limit_sync(self) -> None:
        self._rate_limiter.apply_rate_limit_sync()


    def _make_api_request_sync(self, model: str, messages: List[Dict[str, Any]], timeout: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
        # 构建请求头（包含apikey）和请求体（包含模型名称、消息内容等）
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        return self._api_requester.request_sync(
            self.config.base_url,
            headers,
            payload,
            enable_rate_limit_retry=self.config.enable_rate_limit_retry,
            timeout_override=timeout,
        )

"""
__all__ 是模块级的导出列表，表示该模块的“公共 API”。
把 ["VLMClient", "VLMConfig"] 放在 __all__ 中意味着：当别人写 from multi_ocr_sdk.vlm_client import * 时，只会导入 VLMClient 和 VLMConfig。
它也作为 API 意图的声明（告诉用户和文档/IDE 哪些名字是公开的）。
"""
__all__ = ["VLMClient", "VLMConfig"]
# 20251219人工审阅结束