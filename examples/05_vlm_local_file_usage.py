"""
VLM usage example with local file (PDF or Image) using the high-level parse method.

Replace `API_KEY` and `BASE_URL` with your values or set
`DS_OCR_API_KEY` and `DS_OCR_BASE_URL` environment variables.
"""

import os
from multi_ocr_sdk import VLMClient



file_path = "./examples/example_files/DeepSeek_OCR_paper_mini.pdf" 



# Check if file exists
if not os.path.exists(file_path):
    print(f"File not found at {file_path}. Please provide a valid path to a PDF or image file.")
    print("Example: You can copy a PDF file to this directory and name it 'example_document.pdf'")
else:
    print(f"Processing file: {file_path}")
    
    try:
        client = VLMClient(
            api_key='test_api_key',
            base_url='http://10.131.102.25:8000/v1',
            model='Qwen3-VL-8B',
        )
        result = client.parse(
            file_path=file_path,
            prompt="你是一个ocr机器人，识别输入的文件内容，输出为markdown格式，尽可能保留图表等格式信息，你不需要评论概括文件内容，只需要输出就行",
            timeout=100,
            dpi=60,  # dpi建议60-80。dpi越高识别效果越好，但也越容易超出模型的token限制
            concurrency_num=5,  # 并发处理一个pdf里的5个页面
            max_tokens=8000  #每个模型支持的最大token数不同，不要设为模型上限，因为 prompt 也占 token；如果模型上限是8192，建议max_token略小一些，留出冗余给文字prompt，比如此处设为8000
        )

        print("\n--- Result ---")
        print(result)
        
    except Exception as e:
        print(f"Error: {e}")
