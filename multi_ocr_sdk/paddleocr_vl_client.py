"""
该模块提供一个 SDK，用于调用 PaddleOCR-VL 服务化部署的 REST API 完成文档解析（OCR）。

服务化部署参考：https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html

工作流程：
  1. 读取本地文件（PDF 或图片），使用 Base64 编码后发送到 /layout-parsing 接口
  2. 将解析结果发送到 /restructure-pages 接口，合并多页结果并返回 Markdown 文本
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .exceptions import ConfigurationError, FileProcessingError
from .basic_utils import RateLimiter, APIRequester, BaseConfig
from .basic_utils.basic_logger import setup_file_logger

logger = logging.getLogger(__name__)

# 支持的图片后缀
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
# fileType 常量（PaddleOCR 服务 API 约定：0=PDF，1=图像）
_FILE_TYPE_PDF = 0
_FILE_TYPE_IMAGE = 1


@dataclass(kw_only=True)
class PaddleOCRVLConfig(BaseConfig):
    """
    PaddleOCR-VL 服务客户端配置。继承 BaseConfig，复用 api_key / base_url /
    timeout / 限流参数的校验逻辑。

    额外属性：
        enable_log: 是否在运行目录创建日志文件（multi-ocr-sdk-logs/）
    """

    # PaddleOCR 处理大文件耗时较长，覆盖 BaseConfig 的 60s 默认值
    timeout: int = 120
    enable_log: bool = False

    def __post_init__(self) -> None:
        # 复用 BaseConfig 的通用校验（api_key、base_url、timeout、限流参数）
        super().__post_init__()
        # 去除 base_url 末尾斜杠，统一格式
        self.base_url = self.base_url.rstrip("/")


class PaddleOCRVLClient:
    """
    PaddleOCR-VL 服务客户端。

    调用 PaddleOCR-VL 服务化部署的两个接口：
      - POST /layout-parsing   ：版面解析，将文件（图片或 PDF）转换为结构化结果
      - POST /restructure-pages：合并多页解析结果，输出统一 Markdown 文本

    使用示例：
        client = PaddleOCRVLClient(api_key="your_key", base_url="http://localhost:8080")
        markdown_text = client.parse("./document.pdf")
        print(markdown_text)
    """

    def __init__(
        self,
        api_key: str,        # 必填，本地部署也可传任意非空字符串
        base_url: str,       # 必填，例如 "http://localhost:8080"
        timeout: int = 120,
        request_delay: float = 0.0,
        enable_rate_limit_retry: bool = True,
        max_rate_limit_retries: int = 3,
        rate_limit_retry_delay: float = 5.0,
        enable_log: bool = False,
    ) -> None:
        self.config = PaddleOCRVLConfig(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            request_delay=request_delay,
            enable_rate_limit_retry=enable_rate_limit_retry,
            max_rate_limit_retries=max_rate_limit_retries,
            rate_limit_retry_delay=rate_limit_retry_delay,
            enable_log=enable_log,
        )

        if self.config.enable_log:
            log_file = setup_file_logger()
            logger.info(f"Logging enabled. Writing logs to {log_file}")

        self._rate_limiter = RateLimiter(
            request_delay=self.config.request_delay,
            max_retries=self.config.max_rate_limit_retries,
            retry_delay=self.config.rate_limit_retry_delay,
        )
        self._api_requester = APIRequester(self._rate_limiter, self.config.timeout)

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def parse(
        self,
        file_path: Union[str, Path],
        concatenate_pages: bool = True,
    ) -> str:
        """
        对本地文件（图片或 PDF）进行 OCR 解析，返回 Markdown 格式的文本。

        Args:
            file_path:        本地文件路径，支持 PDF 及常见图片格式（jpg/png 等）
            concatenate_pages: 多页 PDF 是否合并为单个 Markdown 文档。
                               True（默认）：多页合并为一个文档；
                               False：各页独立，用分隔符拼接

        Returns:
            Markdown 格式的 OCR 结果字符串
        """
        file_path = Path(file_path)
        logger.info(f"Parsing file: {file_path}, concatenate_pages={concatenate_pages}")

        # Step 1: 调用版面解析接口
        layout_result = self._call_layout_parsing(file_path)

        # 从响应中提取每页的解析结果
        pages = self._extract_pages_from_layout_result(layout_result)
        logger.info(f"layout-parsing returned {len(pages)} page(s)")

        # Step 2: 调用页面重组接口，合并多页并返回 Markdown
        markdown_text = self._call_restructure_pages(pages, concatenate_pages)
        logger.info(f"Parsing complete. Markdown length: {len(markdown_text)} chars")
        return markdown_text

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _build_headers(self) -> Dict[str, str]:
        """构建请求头。本地部署无需鉴权，云端部署需要 api_key。"""
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    @staticmethod
    def _read_file_as_base64(file_path: Path) -> str:
        """读取本地文件并返回 Base64 编码字符串（ASCII）。"""
        try:
            with open(file_path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
        except OSError as e:
            raise FileProcessingError(f"Failed to read file '{file_path}': {e}") from e

    @staticmethod
    def _detect_file_type(file_path: Path) -> int:
        """
        根据文件后缀判断 fileType。
        返回 1（图片）或 2（PDF）。
        """
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return _FILE_TYPE_PDF
        if ext in _IMAGE_EXTENSIONS:
            return _FILE_TYPE_IMAGE
        # 未知格式默认作为图片处理
        logger.warning(
            f"Unknown file extension '{ext}', treating as image (fileType=1). "
            "Supported: .pdf and common image formats."
        )
        return _FILE_TYPE_IMAGE

    def _call_layout_parsing(self, file_path: Path) -> Dict[str, Any]:
        """
        基于paddleocr-vl后端官方api
        调用 /layout-parsing 接口，返回原始响应 JSON。

        请求体格式：
            {
                "file":     "<Base64 编码的文件内容>",
                "fileType": 1  # 0=PDF, 1=图片
            }
        """
        file_data = self._read_file_as_base64(file_path)
        file_type = self._detect_file_type(file_path)
        payload: Dict[str, Any] = {
            "file": file_data,
            "fileType": file_type,
        }
        url = f"{self.config.base_url}/layout-parsing"
        logger.debug(f"POST {url} (fileType={file_type})")
        return self._api_requester.request_sync(
            url=url,
            headers=self._build_headers(),
            payload=payload,
            enable_rate_limit_retry=self.config.enable_rate_limit_retry,
        )

    @staticmethod
    def _extract_pages_from_layout_result(
        layout_response: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        从 /layout-parsing 响应中提取每页数据，用于后续 /restructure-pages 调用。

        仅保留 prunedResult 和 markdown.images（图片数据），舍弃 outputImages。
        """
        layout_parsing_results: List[Dict[str, Any]] = (
            layout_response.get("result", {}).get("layoutParsingResults", [])
        )
        pages = []
        for res in layout_parsing_results:
            pages.append(
                {
                    "prunedResult": res.get("prunedResult"),
                    "markdownImages": res.get("markdown", {}).get("images"),
                }
            )
        return pages

    def _call_restructure_pages(
        self,
        pages: List[Dict[str, Any]],
        concatenate_pages: bool,
    ) -> str:
        """
        调用 /restructure-pages 接口，将多页结果合并为 Markdown 文本。

        请求体格式：
            {
                "pages":            [ { "prunedResult": ..., "markdownImages": ... }, ... ],
                "concatenatePages": true
            }

        Returns:
            Markdown 格式字符串
        """
        payload: Dict[str, Any] = {
            "pages": pages,
            "concatenatePages": concatenate_pages,
        }
        url = f"{self.config.base_url}/restructure-pages"
        logger.debug(f"POST {url} (concatenatePages={concatenate_pages})")
        response = self._api_requester.request_sync(
            url=url,
            headers=self._build_headers(),
            payload=payload,
            enable_rate_limit_retry=self.config.enable_rate_limit_retry,
        )

        # 提取 Markdown 文本
        layout_parsing_results: List[Dict[str, Any]] = (
            response.get("result", {}).get("layoutParsingResults", [])
        )
        if not layout_parsing_results:
            logger.warning("restructure-pages returned empty layoutParsingResults")
            return ""

        # concatenatePages=True 时只有一个条目；False 时有多个条目
        markdown_parts = []
        for res in layout_parsing_results:
            text = res.get("markdown", {}).get("text", "")
            if text:
                markdown_parts.append(text)

        return "\n\n---\n\n".join(markdown_parts)


__all__ = ["PaddleOCRVLClient", "PaddleOCRVLConfig"]
