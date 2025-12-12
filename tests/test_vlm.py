"""
不太懂pytest怎么用，先让ai写了个，大佬完善一下
"""

import pytest

from multi_ocr import VLM, vlm_client


def test_vlm_import_and_structure():
    # Ensure vlm module is importable and symbols exist
    assert hasattr(vlm_client, "VLM") or hasattr(vlm_client, "VLMClient")

    # Construct a VLM client instance with dummy values (no network calls)
    client = VLM(api_key="test", base_url="http://localhost:8000/v1")
    assert hasattr(client, "chat")
    assert hasattr(client.chat, "completions")
    assert hasattr(client.chat.completions, "create")


