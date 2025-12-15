"""
Example 4: Save OCR Output to File

This example demonstrates how to save OCR output to markdown files
using the save parameter.
"""

import os
from pathlib import Path
from deepseek_ocr import DeepSeekOCR

# Set your API key (or use DS_OCR_API_KEY environment variable)
API_KEY = os.getenv("DS_OCR_API_KEY", "your_api_key_here")


def main():
    """Save OCR output example."""
    # Initialize client
    client = DeepSeekOCR(api_key=API_KEY)

    # Example 1: Process and save to file
    print("Example 1: Processing document and saving to file...")
    try:
        text = client.parse(
            "sample_docs/simple_document.pdf",
            mode="free_ocr",
            save=True,  # Enable saving to file
        )
        print(f"Extracted text ({len(text)} chars):")
        print(text[:200])  # Print first 200 chars
        print("\n✓ Output saved to: ocr-output/simple_document.md")
        print("\n" + "=" * 60 + "\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Example 2: Process without saving (default behavior)
    print("Example 2: Processing without saving...")
    try:
        text = client.parse(
            "sample_docs/simple_document.pdf",
            mode="free_ocr",
            save=False,  # Explicit False (same as default)
        )
        print(f"Extracted text ({len(text)} chars):")
        print(text[:200])
        print("\n✓ Output not saved (returned only)")
        print("\n" + "=" * 60 + "\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Example 3: Batch processing with save
    print("Example 3: Processing multiple documents and saving...")
    try:
        doc_files = [
            "sample_docs/doc1.pdf",
            "sample_docs/doc2.pdf",
            "sample_docs/doc3.pdf",
        ]

        processed_count = 0
        for doc_file in doc_files:
            if Path(doc_file).exists():
                text = client.parse(doc_file, save=True)
                filename = Path(doc_file).stem
                print(
                    f"✓ Processed and saved: ocr-output/{filename}.md ({len(text)} chars)"
                )
                processed_count += 1
            else:
                print(f"⚠ Skipping {doc_file} (file not found)")

        if processed_count == 0:
            print("No files were processed (sample files not found)")

        print("\n" + "=" * 60 + "\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Example 4: Check output directory
    print("Example 4: Checking saved files...")
    output_dir = Path("ocr-output")
    if output_dir.exists():
        saved_files = list(output_dir.glob("*.md"))
        print(f"Found {len(saved_files)} saved markdown files:")
        for file in saved_files:
            size = file.stat().st_size
            print(f"  - {file.name} ({size:,} bytes)")
    else:
        print("No output directory found (no files saved)")


async def async_example():
    """Async OCR with save example."""
    client = DeepSeekOCR(api_key=API_KEY)

    print("\nAsync Example: Processing document asynchronously with save...")
    try:
        text = await client.parse_async(
            "sample_docs/simple_document.pdf",
            mode="free_ocr",
            save=True,
        )
        print(f"Extracted text ({len(text)} chars):")
        print(text[:200])
        print("\n✓ Output saved to: ocr-output/simple_document.md")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run synchronous examples
    print("=" * 60)
    print("Save Output Examples")
    print("=" * 60 + "\n")

    main()

    # Uncomment to run async example
    # import asyncio
    # asyncio.run(async_example())
