import json

from multi_ocr_sdk import PaddleOCRVLClient


base_url = "http://10.131.101.39:8010"
api_key = "test"

# 默认模式：直接返回纯 Markdown 字符串，不含任何 Base64 图片数据
client = PaddleOCRVLClient(base_url=base_url, api_key=api_key)
markdown_text = client.parse(r"examples/example_files/DeepSeek_OCR_paper_page1.jpg")
print(markdown_text)

# 富结果模式：返回 Markdown + 每页版面定位信息（边界框坐标），均不含 Base64
rich_client = PaddleOCRVLClient(
    base_url=base_url,
    api_key=api_key,
    return_layout_info=True,
)
result = rich_client.parse(r"examples/example_files/DeepSeek_OCR_paper_page1.jpg")

# 输出 Markdown 文本
# print(result.markdown)

# 输出 JSON（包含文本内容及版面边界框，不含 Base64）
result_dict = result.to_dict()
print(json.dumps(result_dict, ensure_ascii=False, indent=2))

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(result_dict, f, ensure_ascii=False, indent=2)
print("结果已保存到 output.json")

# # 单独访问每页的版面定位信息（边界框数据）
# for i, page in enumerate(result.pages_layout):
#     print(f"\n--- Page {i} layout info ---")
#     print(page.pruned_result)