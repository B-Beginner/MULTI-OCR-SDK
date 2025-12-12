"""
Example 1: Basic Usage

This example demonstrates the basic usage of DeepSeek OCR SDK.
"""

import os
from multi_ocr import DeepSeekOCR

# Set your API key (or use DS_OCR_API_KEY environment variable)
API_KEY = os.getenv("DS_OCR_API_KEY", "your_api_key_here")


def main():
    """Basic OCR example."""
    # Initialize client
    client = DeepSeekOCR(api_key=API_KEY)

    # Example 1: Simple document (use FREE_OCR mode - fastest)
    print("Example 1: Processing simple document with FREE_OCR...")
    try:
        text = client.parse(
            "sample_docs/simple_document.pdf",
            mode="free_ocr",
        )
        print(f"Extracted text ({len(text)} chars):")
        print(text[:500])  # Print first 500 chars
        print("\n" + "=" * 60 + "\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Example 2: Document with complex tables (use GROUNDING mode)
    print("Example 2: Processing document with complex tables...")
    try:
        text = client.parse(
            "sample_docs/complex_table.pdf",
            mode="grounding",
        )
        print(f"Extracted text ({len(text)} chars):")
        print(text[:500])
        print("\n" + "=" * 60 + "\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Example 3: Custom DPI for better quality
    print("Example 3: Processing with custom DPI...")
    try:
        text = client.parse(
            "sample_docs/simple_document.pdf",
            mode="free_ocr",
            dpi=300,  # Higher DPI for better quality (slower, larger)
        )
        print(f"Extracted text ({len(text)} chars):")
        print(text[:500])
        print("\n" + "=" * 60 + "\n")
    except Exception as e:
        print(f"Error: {e}\n")


async def async_example():
    """Async OCR example."""
    client = DeepSeekOCR(api_key=API_KEY)

    print("Async Example: Processing document asynchronously...")
    try:
        text = await client.parse_async(
            "sample_docs/simple_document.pdf",
            mode="free_ocr",
        )
        print(f"Extracted text ({len(text)} chars):")
        print(text[:500])
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run synchronous examples
    main()

    # Uncomment to run async example
    # import asyncio
    # asyncio.run(async_example())
