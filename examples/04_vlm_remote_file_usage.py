"""
Minimal VLM usage example — synchronous only.

Replace `API_KEY` and `BASE_URL` with your values or set
`DS_OCR_API_KEY` and `DS_OCR_BASE_URL` environment variables.
"""

import os
from pprint import pprint

from multi_ocr import vlm_client

API_KEY = os.getenv("VLM_API_KEY", "123454")
BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:8000/v1")

client = vlm_client.VLM(api_key=API_KEY, base_url=BASE_URL)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                },
            },
            {"type": "text", "text": "用中文描述一下这个图片内容，不要换行，一行全部输出"},
        ],
    }
]

# Synchronous call — model is optional if VLMConfig default is acceptable
result = client.chat.completions.create(model="Qwen3-VL-8B", messages=messages)

# Print returned message content
content = result.get("choices", [])[0].get("message", {}).get("content")

pprint(content)
