import time
from multi_ocr import VLMClient
from multi_ocr.basic_utils.file_processor import FileProcessor

c = VLMClient(api_key="...", base_url="http://10.131.102.25:8000/v1/chat/completions", timeout=300)
b64 = FileProcessor.file_to_base64("examples/example_files/DeepSeek_OCR_paper.pdf", dpi=30, pages=1)
msg = [{"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}},{"type":"text","text":"请逐字转写图片中的所有文字，只输出原文"}]}]

t0 = time.perf_counter()
res = c.chat.completions.create(model="Qwen3-VL-8B", messages=msg, max_tokens=500)
print("elapsed:", time.perf_counter()-t0)
print(res.get("choices",[{}])[0].get("message",{}).get("content"))