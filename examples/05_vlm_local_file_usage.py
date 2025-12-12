"""
VLM usage example with local file (PDF or Image) using the high-level parse method.

Replace `API_KEY` and `BASE_URL` with your values or set
`DS_OCR_API_KEY` and `DS_OCR_BASE_URL` environment variables.
"""

import os
from pprint import pprint

from multi_ocr import vlm_client

API_KEY = os.getenv("DS_OCR_API_KEY", "your_api_key_here")
BASE_URL = os.getenv("DS_OCR_BASE_URL", "http://localhost:8000/v1/chat/completions")

client = vlm_client.VLM(api_key=API_KEY, base_url=BASE_URL)

# Path to your local file (PDF or Image)
file_path = "./example_document.pdf"  # Use a generic placeholder path

# Check if file exists
if not os.path.exists(file_path):
    print(f"File not found at {file_path}. Please provide a valid path.")
    # Create a dummy file for demonstration if it doesn't exist
    # exit(1)
else:
    print(f"Processing file: {file_path}")
    
    try:
        # Use the high-level parse method
        # This handles PDF conversion, multi-page processing, and merging results
        # Note: Reducing DPI to 72 to avoid exceeding token limits for large PDFs
        result = client.parse(
            file_path=file_path,
            prompt="你是一个ocr机器人，识别输入的文件内容，输出为markdown格式，尽可能保留图标等格式信息，你不需要评论概括文件内容，只需要输出就行",
            model="Qwen3-VL-8B",
            dpi=65  # Lower DPI to reduce token count
        )

        print("\n--- Result ---")
        pprint(result)
        
    except Exception as e:
        print(f"Error: {e}")
