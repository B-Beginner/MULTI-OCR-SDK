"""
Example 3: Rate Limiting and 429 Error Handling

This example demonstrates how to use the rate limiting features
to prevent hitting API limits and handle 429 errors gracefully.

Note: BatchProcessor and async features have been removed in this version.
"""

import os
from multi_ocr_sdk import DeepSeekOCR, RateLimitError

# Set your API key
API_KEY = os.getenv("DS_OCR_API_KEY", "your_api_key_here")


def example_basic_rate_limiting():
    """Basic rate limiting with request delay."""
    print("Example 1: Basic Rate Limiting with Request Delay")
    print("=" * 60)

    # Create client with 2-second delay between requests
    # This ensures you won't exceed rate limits
    client = DeepSeekOCR(
        api_key=API_KEY,
        request_delay=2.0,  # Wait 2 seconds between requests
    )

    print("Processing 3 documents with 2-second delay between requests...")
    files = ["sample_docs/doc1.pdf", "sample_docs/doc2.pdf", "sample_docs/doc3.pdf"]

    for i, file in enumerate(files, 1):
        print(f"\nProcessing file {i}/{len(files)}: {file}")
        try:
            text = client.parse(file)
            print(f"✓ Success: {len(text)} characters extracted")
        except Exception as e:
            print(f"✗ Error: {e}")

    print("\nAll files processed with rate limiting!")


def example_429_retry():
    """Automatic retry on 429 errors with exponential backoff."""
    print("\n\nExample 2: Automatic 429 Retry with Exponential Backoff")
    print("=" * 60)

    # Create client with automatic 429 retry enabled
    client = DeepSeekOCR(
        api_key=API_KEY,
        enable_rate_limit_retry=True,  # Enable auto-retry (default: True)
        max_rate_limit_retries=3,  # Try up to 3 times (default: 3)
        rate_limit_retry_delay=5.0,  # Initial delay: 5 seconds (default: 5.0)
    )

    print(
        "If a 429 error occurs, the SDK will automatically retry with exponential backoff:"
    )
    print("  - 1st retry: wait 5 seconds")
    print("  - 2nd retry: wait 10 seconds")
    print("  - 3rd retry: wait 20 seconds")

    try:
        text = client.parse("sample_docs/doc1.pdf")
        print(f"\n✓ Success: {len(text)} characters extracted")
    except RateLimitError as e:
        print(f"\n✗ Rate limit exceeded after all retries: {e}")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def example_combined_rate_limiting():
    """Combine request delay with 429 retry for best results."""
    print("\n\nExample 3: Combined Rate Limiting (Recommended)")
    print("=" * 60)

    # For best results, combine both approaches:
    # 1. Request delay prevents hitting limits
    # 2. Auto-retry handles occasional 429 errors
    client = DeepSeekOCR(
        api_key=API_KEY,
        request_delay=1.0,  # Wait 1 second between requests
        enable_rate_limit_retry=True,  # Auto-retry on 429
        max_rate_limit_retries=3,
        rate_limit_retry_delay=5.0,
    )

    print("Using both request delay (1s) and auto-retry for optimal safety")

    files = ["sample_docs/doc1.pdf", "sample_docs/doc2.pdf"]

    for i, file in enumerate(files, 1):
        print(f"\nProcessing file {i}/{len(files)}: {file}")
        try:
            text = client.parse(file)
            print(f"✓ Success: {len(text)} characters extracted")
        except RateLimitError as e:
            print(f"✗ Rate limit error: {e}")
        except Exception as e:
            print(f"✗ Error: {e}")


def example_batch_with_rate_limiting():
    """Simple sequential processing with rate limiting."""
    print("\n\nExample 4: Sequential Processing with Rate Limiting")
    print("=" * 60)

    # For L0 tier (TPM: 80,000, RPM: 1,000):
    # - Average 1000 tokens per request → ~80 requests/minute max
    # - Safe rate: 60 requests/minute → 1 second delay
    # - With request_delay=2.0, max rate is 0.5 requests/second = 30 requests/minute

    client = DeepSeekOCR(
        api_key=API_KEY,
        request_delay=2.0,  # 2-second delay between requests (global)
        enable_rate_limit_retry=True,
    )

    print("Processing files sequentially with rate limiting")
    print("Request delay: 2 seconds between requests")
    
    # Note: BatchProcessor has been removed. Users should implement their own
    # concurrent processing if needed.
    
    files = ["sample_docs/doc1.pdf", "sample_docs/doc2.pdf", "sample_docs/doc3.pdf"]
    
    for i, file in enumerate(files, 1):
        print(f"\nProcessing file {i}/{len(files)}: {file}")
        try:
            text = client.parse(file)
            print(f"✓ Success: {len(text)} characters extracted")
        except Exception as e:
            print(f"✗ Error: {e}")


def example_tpm_rpm_calculation():
    """Calculate appropriate rate limits for your API tier."""
    print("\n\nExample 5: TPM/RPM Calculation Guide")
    print("=" * 60)

    print(
        """
Rate Limit Tiers (from DeepSeek API documentation - verify current values):
NOTE: These values are examples from the DeepSeek documentation at the time
of implementation. Please verify the current rate limits for your API tier
at https://platform.deepseek.com or your API provider's documentation.

- L0: TPM=80,000, RPM=1,000
- L1: TPM=120,000, RPM=1,200
- L2: TPM=160,000, RPM=2,000
- L3: TPM=320,000, RPM=4,000
- L4: TPM=1,000,000, RPM=8,000
- L5: TPM=5,000,000, RPM=10,000

Calculation Examples:

For L0 tier (TPM: 80,000, RPM: 1,000):
  Assume average 1000 tokens per request:
  - Max requests: 80,000 / 1,000 = 80 requests/minute
  - But RPM limit: 1,000 requests/minute
  - Bottleneck: TPM (80 requests/minute)
  - Safe rate: 60 requests/minute (75% of limit)
  - request_delay = 60s / 60 = 1.0 second

For L0 with sequential processing:
  - To stay under 60 requests/minute: request_delay = 1.0 second
  - For more conservative rate: request_delay = 2.0 seconds (30 req/min)

For L3 tier (TPM: 320,000, RPM: 4,000):
  Assume average 1000 tokens per request:
  - Max requests: 320,000 / 1,000 = 320 requests/minute
  - Safe rate: 240 requests/minute (75% of limit)
  - request_delay = 60s / 240 = 0.25 seconds
"""
    )

    # Example configuration for different tiers
    configs = {
        "L0 (conservative)": {
            "request_delay": 1.0,
            "requests_per_minute": 60,
        },
        "L0 (moderate)": {
            "request_delay": 2.0,
            "requests_per_minute": 30,
        },
        "L3 (conservative)": {
            "request_delay": 0.25,
            "requests_per_minute": 240,
        },
    }

    print("\nRecommended configurations:")
    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  client = DeepSeekOCR(request_delay={config['request_delay']})")
        print(f"  Effective rate: ~{config['requests_per_minute']} requests/minute")


def example_disable_retry():
    """Disable automatic retry if you want to handle 429 yourself."""
    print("\n\nExample 6: Disable Automatic Retry")
    print("=" * 60)

    # Disable retry if you want to implement your own retry logic
    client = DeepSeekOCR(
        api_key=API_KEY,
        enable_rate_limit_retry=False,  # Disable auto-retry
        request_delay=1.0,  # Still use request delay
    )

    print("Auto-retry disabled - will raise RateLimitError immediately on 429")

    try:
        text = client.parse("sample_docs/doc1.pdf")
        print(f"✓ Success: {len(text)} characters extracted")
    except RateLimitError as e:
        print(f"✗ Rate limit error (no retry): {e}")
        print("You can implement your own retry logic here")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    # Run synchronous examples
    print("Running rate limiting examples...\n")

    example_basic_rate_limiting()
    example_429_retry()
    example_combined_rate_limiting()

    # Run sequential processing example
    example_batch_with_rate_limiting()

    # Display calculation guide
    example_tpm_rpm_calculation()

    # Disable retry example
    example_disable_retry()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print(
        """
Summary of best practices:
1. Set request_delay based on your API tier and token usage
2. Enable auto-retry to handle occasional 429 errors
3. For concurrent processing, implement your own async wrapper
4. Monitor your usage and adjust settings as needed
"""
    )
