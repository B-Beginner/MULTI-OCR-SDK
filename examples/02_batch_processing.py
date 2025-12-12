"""
Example 2: Batch Processing

This example demonstrates how to process multiple documents efficiently
using the BatchProcessor.
"""

import asyncio
import os
from pathlib import Path

from multi_ocr import DeepSeekOCR, BatchProcessor

# Set your API key
API_KEY = os.getenv("DS_OCR_API_KEY", "your_api_key_here")


async def batch_example_basic():
    """Basic batch processing example."""
    print("Example 1: Basic Batch Processing")
    print("=" * 60)

    # Initialize client and processor
    client = DeepSeekOCR(api_key=API_KEY)
    processor = BatchProcessor(client, max_concurrent=3)

    # Get list of files to process
    files = [
        "sample_docs/doc1.pdf",
        "sample_docs/doc2.pdf",
        "sample_docs/doc3.pdf",
    ]

    # Process all files with FREE_OCR mode
    summary = await processor.process_batch(
        files,
        mode="free_ocr",
        show_progress=True,
    )

    # Print summary
    summary.print_summary()

    # Access individual results
    print("\n\nIndividual Results:")
    for result in summary.results:
        if result.success:
            print(f"✓ {result.file_path.name}: {len(result.text)} chars")
        else:
            print(f"✗ {result.file_path.name}: {result.error}")


async def batch_example_with_modes():
    """Batch processing with different modes."""
    print("\n\nExample 2: Batch Processing with GROUNDING Mode")
    print("=" * 60)

    client = DeepSeekOCR(api_key=API_KEY)
    processor = BatchProcessor(client, max_concurrent=2)

    # Get all PDF files in directory
    doc_dir = Path("sample_docs")
    files = list(doc_dir.glob("*.pdf"))
    print(f"Found {len(files)} PDF files")

    # Process with GROUNDING mode for complex tables
    summary = await processor.process_batch(
        files,
        mode="grounding",  # Use GROUNDING for better table handling
        show_progress=True,
    )

    # Print summary
    summary.print_summary()

    # Save successful results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for result in summary.results:
        if result.success:
            output_file = output_dir / f"{result.file_path.stem}.md"
            output_file.write_text(result.text, encoding="utf-8")
            print(f"Saved: {output_file}")


def batch_example_sync():
    """Synchronous batch processing example."""
    print("\n\nExample 3: Synchronous Batch Processing")
    print("=" * 60)

    client = DeepSeekOCR(api_key=API_KEY)
    processor = BatchProcessor(client, max_concurrent=3)

    files = ["sample_docs/doc1.pdf", "sample_docs/doc2.pdf"]

    # Synchronous wrapper
    summary = processor.process_batch_sync(
        files,
        mode="free_ocr",
        show_progress=True,
    )

    summary.print_summary()


async def batch_example_advanced():
    """Advanced batch processing with custom settings."""
    print("\n\nExample 4: Advanced Batch Processing")
    print("=" * 60)

    client = DeepSeekOCR(api_key=API_KEY)
    processor = BatchProcessor(
        client,
        max_concurrent=5,  # Process up to 5 files concurrently
        retry_count=2,  # Retry failed requests twice
    )

    files = list(Path("sample_docs").glob("*.pdf"))

    summary = await processor.process_batch(
        files,
        mode="free_ocr",
        dpi=200,  # Custom DPI
        show_progress=True,
    )

    summary.print_summary()

    # Generate report
    print("\n\nDetailed Report:")
    print("=" * 60)
    mode_counts = {}
    total_chars = 0

    for result in summary.results:
        if result.success:
            mode = result.mode_used.value
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            total_chars += len(result.text)

    print(f"Total characters extracted: {total_chars:,}")
    print("\nMode distribution:")
    for mode, count in mode_counts.items():
        print(f"  {mode}: {count} documents")


if __name__ == "__main__":
    # Run examples
    print("Running batch processing examples...\n")

    # Async examples
    asyncio.run(batch_example_basic())
    asyncio.run(batch_example_with_modes())

    # Sync example
    batch_example_sync()

    # Advanced example
    asyncio.run(batch_example_advanced())
