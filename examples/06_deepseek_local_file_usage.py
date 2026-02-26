
import os
from pprint import pprint
from multi_ocr_sdk import DeepSeekOCR


API_KEY = "test"
BASE_URL = "http://10.131.101.39:8004/v1/chat/completions"
MODEL_NAME='deepseek-ocr'

# client = DeepSeekOCR(
#     api_key=API_KEY, # 必填
#     base_url=BASE_URL, # 必填，一般以/v1，或者/v1/completions结尾
# 	model_name=MODEL_NAME,

#     # model='GROUNDING',
# 	dpi=300,
# 	max_tokens=2048,
# 	prompt='<image>\n<|grounding|>Convert the document to markdown.'

# 	# model='OCR_IMAGE'
# )




text = client.parse(r"examples/example_files/DeepSeek_OCR_paper.pdf")

print(text)