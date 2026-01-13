
import os
from pprint import pprint
from multi_ocr_sdk import DeepSeekOCR



API_KEY = "your_api_key_here"
BASE_URL = "http://10.131.101.39:8004//v1/chat/completions"


client = DeepSeekOCR(
    api_key='test', # 必填
    base_url=BASE_URL, # 必填，一般以/v1，或者/v1/completions结尾
	model_name='deepseek-ocr',
	model='GROUNDING'
)

text = client.parse("examples/example_files/DeepSeek_OCR_paper.pdf")

print(text)