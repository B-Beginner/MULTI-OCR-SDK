"""
VLM usage example with local file (PDF or Image) using the high-level parse method.

Replace `API_KEY` and `BASE_URL` with your values or set
`DS_OCR_API_KEY` and `DS_OCR_BASE_URL` environment variables.
"""

import os
from pprint import pprint
from multi_ocr import VLMClient


API_KEY = os.getenv("DS_OCR_API_KEY", "your_api_key_here")
BASE_URL = os.getenv("DS_OCR_BASE_URL", "http://localhost:8000/v1/chat/completions")

# Initialize the VLM client (uses installed package)
client = VLMClient(api_key=API_KEY, base_url=BASE_URL)

# Path to your local file (PDF or Image)
# You can place a PDF or image file in this directory to test
file_path = "./example_files/dog_and_girl.jpeg" 

# Check if file exists
if not os.path.exists(file_path):
    print(f"File not found at {file_path}. Please provide a valid path to a PDF or image file.")
    print("Example: You can copy a PDF file to this directory and name it 'example_document.pdf'")
else:
    print(f"Processing file: {file_path}")
    
    try:
        # Use the high-level parse method
        # This handles PDF conversion, multi-page processing, and merging results
        # Note: Reducing DPI to 72 (or lower) to avoid exceeding token limits for large PDFs
        result = client.parse(
            file_path=file_path,
            prompt="你是一个ocr机器人，识别输入的文件内容，输出为markdown格式，尽可能保留图标等格式信息，你不需要评论概括文件内容，只需要输出就行",
            model="Qwen3-VL-8B",
            dpi=72  # Standard screen DPI is usually sufficient for VLM
        )

        print("\n--- Result ---")
        print(result)
        
    except Exception as e:
        print(f"Error: {e}")
