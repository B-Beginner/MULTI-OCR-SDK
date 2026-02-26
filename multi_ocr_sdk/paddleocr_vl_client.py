"""
该模块提供一个 SDK，用于调用 PaddleOCR-VL 服务化部署的 REST API 完成文档解析（OCR）。

服务化部署参考：https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html

工作流程：
  1. 读取本地文件（PDF 或图片），使用 Base64 编码后发送到 /layout-parsing 接口
  2. 将解析结果发送到 /restructure-pages 接口，合并多页结果并返回 Markdown 文本

输出模式：
  - 默认模式（return_layout_info=False）：parse() 返回纯 Markdown 字符串，不含任何图片 Base64
  - 富结果模式（return_layout_info=True）：parse() 返回 PaddleOCRVLResult，
    包含 Markdown 文本与每页版面定位信息（prunedResult 边界框数据），均不含 Base64
  - to_dict() 可将富结果序列化为 JSON 友好的字典
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# 富结果数据类（当 return_layout_info=True 时使用）
# ---------------------------------------------------------------------------

@dataclass
class PageLayoutInfo:
    """单页版面解析信息，包含边界框坐标等结构化数据。

    Attributes:
        pruned_result:  prunedResult 字段的原始内容（dict），包含本页所有检测到的
                        版面区块及其边界框坐标、类别、置信度等结构化信息。
    """

    pruned_result: Dict[str, Any]


@dataclass
class PaddleOCRVLResult:
    """PaddleOCR-VL 富解析结果，同时包含 Markdown 文本与版面定位信息。

    Attributes:
        markdown:      Markdown 格式的 OCR 文本（与 parse() 默认返回的字符串内容
                       完全一致）。
        pages_layout:  按页顺序排列的版面解析信息列表，每个元素为一个
                       :class:`PageLayoutInfo`，提供该页的边界框数据（不含 Base64
                       图片数据）。
    """

    markdown: str
    pages_layout: List[PageLayoutInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """将结果序列化为字典（可直接用于 json.dumps）。

        Returns:
            包含 ``markdown`` 文本与每页 ``pruned_result`` 版面信息的字典。
        """
        return {
            "markdown": self.markdown,
            "pages_layout": [
                {"pruned_result": page.pruned_result}
                for page in self.pages_layout
            ],
        }


@dataclass(kw_only=True)
class PaddleOCRVLConfig(BaseConfig):
    """
    PaddleOCR-VL 服务客户端配置。继承 BaseConfig，复用 api_key / base_url /
    timeout / 限流参数的校验逻辑。

    额外属性：
        enable_log:          是否在运行目录创建日志文件（multi-ocr-sdk-logs/）
        return_layout_info:  True 时 parse() 返回 PaddleOCRVLResult（包含 Markdown
                             文本及每页的版面定位信息 prunedResult）；
                             False（默认）时 parse() 直接返回 Markdown 字符串
                             （向后兼容原有行为）。
        visualize:           传递给 /layout-parsing 接口的 visualize 参数。
                             默认为 False，服务端不返回可视化图像，输出结果中不含
                             Base64 图片数据。设为 True 时服务端会执行可视化渲染
                             （但渲染结果不会出现在 SDK 返回值中）。
    """

    # PaddleOCR 处理大文件耗时较长，覆盖 BaseConfig 的 60s 默认值
    timeout: int = 120
    enable_log: bool = False
    return_layout_info: bool = False
    visualize: Optional[bool] = False

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

    使用示例（默认模式，仅返回 Markdown 文本）：
        client = PaddleOCRVLClient(api_key="your_key", base_url="http://localhost:8080")
        markdown_text = client.parse("./document.pdf")
        print(markdown_text)

    使用示例（富结果模式，同时返回版面定位信息）：
        client = PaddleOCRVLClient(
            api_key="your_key",
            base_url="http://localhost:8080",
            return_layout_info=True,
            visualize=True,   # 可选：要求服务端返回标注边界框的可视化图像
        )
        result = client.parse("./document.pdf")
        print(result.markdown)           # Markdown 文本（不含 Base64 图片）
        for i, page in enumerate(result.pages_layout):
            print(f"Page {i}: bboxes =", page.pruned_result)
        # 输出完整 JSON（包含文本及边界框，不含 Base64）
        import json; print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
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
        return_layout_info: bool = False,
        visualize: Optional[bool] = False,
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
            return_layout_info=return_layout_info,
            visualize=visualize,
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
    ) -> Union[str, "PaddleOCRVLResult"]:
        """
        对本地文件（图片或 PDF）进行 OCR 解析。

        Args:
            file_path:         本地文件路径，支持 PDF 及常见图片格式（jpg/png 等）
            concatenate_pages: 多页 PDF 是否合并为单个 Markdown 文档。
                               True（默认）：多页合并为一个文档；
                               False：各页独立，用分隔符拼接

        Returns:
            - 当 ``return_layout_info=False``（默认）时：返回纯 Markdown 格式字符串
              （不含 Base64 图片数据）。
            - 当 ``return_layout_info=True`` 时：返回 :class:`PaddleOCRVLResult`，
              其中 ``markdown`` 字段包含 Markdown 文本，``pages_layout`` 字段包含
              每页的版面定位信息（边界框坐标等，不含 Base64 图片数据）。
              可调用 ``result.to_dict()`` 获得可直接序列化为 JSON 的字典。
        """
        file_path = Path(file_path)
        logger.info(
            f"Parsing file: {file_path}, concatenate_pages={concatenate_pages}, "
            f"return_layout_info={self.config.return_layout_info}, "
            f"visualize={self.config.visualize}"
        )

        # Step 1: 调用版面解析接口
        layout_result = self._call_layout_parsing(file_path)

        # 从响应中提取每页的解析结果（同时保留 outputImages，供富结果模式使用）
        pages, pages_layout_info = self._extract_pages_from_layout_result(layout_result)
        logger.info(f"layout-parsing returned {len(pages)} page(s)")

        # Step 2: 调用页面重组接口，合并多页并返回 Markdown
        markdown_text = self._call_restructure_pages(pages, concatenate_pages)
        logger.info(f"Parsing complete. Markdown length: {len(markdown_text)} chars")

        if self.config.return_layout_info:
            return PaddleOCRVLResult(
                markdown=markdown_text,
                pages_layout=pages_layout_info,
            )
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
                "fileType": 1,          # 0=PDF, 1=图片
                "visualize": true       # 可选，true 时响应中包含 outputImages
            }
        """
        file_data = self._read_file_as_base64(file_path)
        file_type = self._detect_file_type(file_path)
        payload: Dict[str, Any] = {
            "file": file_data,
            "fileType": file_type,
        }
        # 仅当用户显式设置时才传递 visualize，避免覆盖服务端默认行为
        if self.config.visualize is not None:
            payload["visualize"] = self.config.visualize
        url = f"{self.config.base_url}/layout-parsing"
        logger.debug(f"POST {url} (fileType={file_type}, visualize={self.config.visualize})")
        return self._api_requester.request_sync(
            url=url,
            headers=self._build_headers(),
            payload=payload,
            enable_rate_limit_retry=self.config.enable_rate_limit_retry,
        )

    @staticmethod
    def _extract_pages_from_layout_result(
        layout_response: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], List["PageLayoutInfo"]]:
        """
        从 /layout-parsing 响应中提取数据，同时返回两类信息：

        1. ``pages``：供 /restructure-pages 接口使用的页面列表，每项包含
           ``prunedResult`` 和 ``markdownImages``（图片 Base64 数据）。
        2. ``pages_layout_info``：:class:`PageLayoutInfo` 列表，每项包含
           ``pruned_result``（边界框等结构化数据）和 ``output_image``
           （当 visualize=True 时由服务端返回的标注可视化图像，否则为 None）。

        注：markdownImages 仍传给 /restructure-pages（服务端构建 Markdown 所需），
        但最终 Markdown 文本中的 Base64 data-URI 图片会在返回前被剥除。
        """
        layout_parsing_results: List[Dict[str, Any]] = (
            layout_response.get("result", {}).get("layoutParsingResults", [])
        )
        pages: List[Dict[str, Any]] = []
        pages_layout_info: List[PageLayoutInfo] = []
        for res in layout_parsing_results:
            pruned_result = res.get("prunedResult", {})
            markdown_images = res.get("markdown", {}).get("images")
            pages.append(
                {
                    "prunedResult": pruned_result,
                    "markdownImages": markdown_images,
                }
            )
            pages_layout_info.append(
                PageLayoutInfo(
                    pruned_result=pruned_result,
                )
            )
        return pages, pages_layout_info

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

        markdown = "\n\n---\n\n".join(markdown_parts)
        # 剥除 Markdown 中嵌入的 Base64 data-URI 图片，替换为空的占位符
        markdown = re.sub(
            r'!\[([^\]]*)\]\(data:[^)]+\)',
            r'![\1]()',
            markdown,
        )
        return markdown


__all__ = ["PaddleOCRVLClient", "PaddleOCRVLConfig", "PaddleOCRVLResult", "PageLayoutInfo"]
