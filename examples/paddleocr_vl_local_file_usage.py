from multi_ocr_sdk import PaddleOCRVLClient


base_url = "http://10.131.101.39:8010"
api_key = "test"

client = PaddleOCRVLClient(base_url=base_url,api_key=api_key)
markdown = client.parse(r"examples/example_files/DeepSeek_OCR_paper_page1.jpg")
print(markdown)