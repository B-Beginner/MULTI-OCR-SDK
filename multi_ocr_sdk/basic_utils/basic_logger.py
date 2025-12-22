"""
用于记录程序日志的模块。
"""

# 20251222开始人工审阅
import os
import logging
from datetime import datetime


def setup_file_logger(
    log_dir_name: str = "multi-ocr-sdk-logs",
    logger_name: str = "multi_ocr_sdk",
    level: int = logging.INFO,
) -> str:
    """
    创建日志目录并将文件处理程序附加到指定的记录器。
    Args：
		log_dir_name: 日志目录名称。
		logger_name: 日志文件名
        level: 日志记录级别。

    Returns:
        path：创建的日志文件的路径。
    """
    # 创建日志目录
    log_dir = os.path.join(os.getcwd(), log_dir_name)
    os.makedirs(log_dir, exist_ok=True)
    
	# 创建日志，基于当前时间戳命名
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{logger_name}_{ts}.log")
    
	# 配置日志记录器
    sdk_logger = logging.getLogger(logger_name)
    sdk_logger.setLevel(level)
    
	# 确保只添加一个文件处理程序，如果已经添加过文件处理程序，则跳过
    # 找出并删除所有名为 "multi_ocr_sdk_file_handler" 的既往的handler，防止旧的handler把日志写入新的log文件
    to_remove = [h for h in list(sdk_logger.handlers) if getattr(h, "name", "") == "multi_ocr_sdk_file_handler"]

    for existing_handler in to_remove:
        try:
            existing_handler.close()
        except Exception:
            # If closing fails, continue removing the handler regardless
            pass
        sdk_logger.removeHandler(existing_handler)
    
	# 再接下里新创建一个multi_ocr_file_handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    fh.name = "multi_ocr_sdk_file_handler"
    sdk_logger.addHandler(fh)

    return log_file
# 20251222结束人工审阅