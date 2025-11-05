# API Reference

Complete API documentation for DeepSeek-OCR-SDK.

## Table of Contents

- [DeepSeekOCR](#deepseekocr)
- [OCRMode](#ocrmode)
- [OCRConfig](#ocrconfig)
- [BatchProcessor](#batchprocessor)
- [BatchResult](#batchresult)
- [BatchSummary](#batchsummary)
- [Exceptions](#exceptions)

---

## DeepSeekOCR

Main client for interacting with the DeepSeek OCR API.

### Constructor

```python
DeepSeekOCR(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout: Optional[int] = None,
    max_tokens: Optional[int] = None,
    dpi: Optional[int] = None,
    **kwargs
)
```

**Parameters:**

- `api_key` (str, optional): API key for authentication. Defaults to `DS_OCR_API_KEY` environment variable. **Required** if not set in environment.
- `base_url` (str, optional): Base URL for the API endpoint. Defaults to `DS_OCR_BASE_URL` environment variable. **Required** if not set in environment. Known provider:
  - SiliconFlow: `https://api.siliconflow.cn/v1/chat/completions` (Verified ✅)
  - Note: DeepSeek's official API does not support the DeepSeek-OCR model
- `model_name` (str, optional): Model name. Defaults to `deepseek-ai/DeepSeek-OCR`.
- `timeout` (int, optional): Request timeout in seconds. Defaults to 60.
- `max_tokens` (int, optional): Maximum tokens in response. Defaults to 4000.
- `dpi` (int, optional): DPI for PDF to image conversion (150, 200, or 300). Defaults to 200.
- `**kwargs`: Additional configuration parameters.

**Raises:**

- `ConfigurationError`: If required configuration is missing or invalid.

### Methods

#### parse

```python
def parse(
    file_path: Union[str, Path],
    mode: Union[str, OCRMode] = OCRMode.FREE_OCR,
    dpi: Optional[int] = None,
) -> str
```

Parse document synchronously.

**Parameters:**

- `file_path` (str | Path): Path to PDF or image file.
- `mode` (str | OCRMode): OCR mode ("free_ocr", "grounding", "ocr_image"). Defaults to FREE_OCR.
- `dpi` (int, optional): DPI for PDF conversion. If None, uses config default.

**Returns:**

- `str`: Extracted text in Markdown format.

**Raises:**

- `FileProcessingError`: If file cannot be processed.
- `APIError`: If API returns an error.
- `TimeoutError`: If request times out.

**Example:**

```python
client = DeepSeekOCR(api_key="xxx")
text = client.parse("document.pdf")
```

#### parse_async

```python
async def parse_async(
    file_path: Union[str, Path],
    mode: Union[str, OCRMode] = OCRMode.FREE_OCR,
    dpi: Optional[int] = None
) -> str
```

Parse document asynchronously.

**Parameters:** Same as `parse()`.

**Returns:** Same as `parse()`.

**Raises:** Same as `parse()`.

**Example:**

```python
client = DeepSeekOCR(api_key="xxx")
text = await client.parse_async("document.pdf")
```

---

## OCRMode

Enumeration of OCR processing modes.

### Values

#### FREE_OCR

```python
OCRMode.FREE_OCR
```

Fast mode that returns pure Markdown output. Best for 80% of document processing scenarios.

- **Speed:** 3.95-10.95s per page
- **Use case:** Simple documents, invoices, letters
- **Output:** Pure Markdown

#### GROUNDING

```python
OCRMode.GROUNDING
```

Advanced mode with HTML output and bounding boxes. Optimal for complex tables (≥20 rows).

- **Speed:** 5.18-8.31s per page
- **Use case:** Complex tables, mixed content
- **Output:** HTML + bounding boxes

#### OCR_IMAGE

```python
OCRMode.OCR_IMAGE
```

Detailed mode with word-level bounding boxes. Slower and less stable.

- **Speed:** 19-26s per page
- **Use case:** Edge cases requiring detailed extraction
- **Output:** Word-level bounding boxes

### Methods

#### get_prompt

```python
def get_prompt() -> str
```

Get the API prompt string for this mode.

**Returns:**

- `str`: The prompt string to send to the DeepSeek OCR API.

---

## OCRConfig

Configuration dataclass for OCR client.

### Constructor

```python
OCRConfig(
    api_key: str,
    base_url: str,
    model_name: str = "deepseek-ai/DeepSeek-OCR",
    timeout: int = 60,
    max_tokens: int = 4000,
    temperature: float = 0.0,
    dpi: int = 200,
    fallback_enabled: bool = True,
    fallback_mode: str = "grounding",
    min_output_threshold: int = 500
)
```

**Attributes:**

- `api_key` (str): API key (required).
- `base_url` (str): API endpoint URL (required). Known provider:
  - SiliconFlow: `https://api.siliconflow.cn/v1/chat/completions` (Verified ✅)
  - Note: DeepSeek's official API does not support the DeepSeek-OCR model
- `model_name` (str): Model name.
- `timeout` (int): Request timeout in seconds.
- `max_tokens` (int): Maximum tokens in response.
- `temperature` (float): Temperature for response generation (0.0 = deterministic).
- `dpi` (int): DPI for PDF conversion (150, 200, or 300).
- `fallback_enabled` (bool): Enable automatic fallback.
- `fallback_mode` (str): Mode to fallback to.
- `min_output_threshold` (int): Minimum output length to trigger fallback.

### Class Methods

#### from_env

```python
@classmethod
def from_env(cls, **overrides) -> OCRConfig
```

Create configuration from environment variables.

**Parameters:**

- `**overrides`: Override specific configuration values.

**Returns:**

- `OCRConfig`: Configuration instance.

**Example:**

```python
config = OCRConfig.from_env(dpi=300)
```

---

## BatchProcessor

Batch processor for multiple documents.

### Constructor

```python
BatchProcessor(
    client: DeepSeekOCR,
    max_concurrent: int = 3,
    retry_count: int = 1
)
```

**Parameters:**

- `client` (DeepSeekOCR): DeepSeekOCR client instance.
- `max_concurrent` (int): Maximum number of concurrent requests. Defaults to 3.
- `retry_count` (int): Number of retries for failed requests. Defaults to 1.

### Methods

#### process_batch

```python
async def process_batch(
    file_paths: List[Union[str, Path]],
    mode: Optional[OCRMode] = None,
    show_progress: bool = True,
    **kwargs
) -> BatchSummary
```

Process multiple documents in batch asynchronously.

**Parameters:**

- `file_paths` (List[str | Path]): List of file paths to process.
- `mode` (OCRMode, optional): OCR mode for all documents. Defaults to FREE_OCR.
- `show_progress` (bool): Show progress bar. Defaults to True.
- `**kwargs`: Additional arguments for `parse_async()`.

**Returns:**

- `BatchSummary`: Summary with all results.

**Example:**

```python
processor = BatchProcessor(client)
files = ["doc1.pdf", "doc2.pdf"]
summary = await processor.process_batch(files)
```

#### process_batch_sync

```python
def process_batch_sync(
    file_paths: List[Union[str, Path]],
    mode: Optional[OCRMode] = None,
    show_progress: bool = True,
    **kwargs
) -> BatchSummary
```

Process multiple documents in batch synchronously.

**Parameters:** Same as `process_batch()`.

**Returns:** Same as `process_batch()`.

**Example:**

```python
processor = BatchProcessor(client)
summary = processor.process_batch_sync(["doc1.pdf", "doc2.pdf"])
```

---

## BatchResult

Result of processing a single document in batch.

### Attributes

- `file_path` (Path): Path to the processed file.
- `success` (bool): Whether processing succeeded.
- `text` (str, optional): Extracted text (if successful).
- `error` (str, optional): Error message (if failed).
- `mode_used` (OCRMode, optional): OCR mode that was used.

---

## BatchSummary

Summary of batch processing results.

### Attributes

- `total` (int): Total number of documents processed.
- `successful` (int): Number of successful processes.
- `failed` (int): Number of failed processes.
- `results` (List[BatchResult]): List of individual results.

### Methods

#### print_summary

```python
def print_summary() -> None
```

Print a human-readable summary of batch processing results.

**Example:**

```python
summary.print_summary()
```

Output:

```
============================================================
Batch Processing Summary
============================================================
Total documents: 10
Successful: 9 (90.0%)
Failed: 1 (10.0%)

Failed documents:
  - corrupted_file.pdf: Failed to process PDF: Invalid PDF structure
```

---

## Exceptions

### DeepSeekOCRError

Base exception for all SDK errors.

```python
class DeepSeekOCRError(Exception)
```

### ConfigurationError

Raised when there is a configuration error.

```python
class ConfigurationError(DeepSeekOCRError)
```

**Example:**

```python
# Missing API key
DeepSeekOCR()  # Raises ConfigurationError

# Invalid DPI
OCRConfig(api_key="xxx", dpi=100)  # Raises ConfigurationError
```

### APIError

Raised when the API returns an error response.

```python
class APIError(DeepSeekOCRError)
```

**Attributes:**

- `status_code` (int, optional): HTTP status code from the API.
- `response_text` (str, optional): Raw response text from the API.

**Example:**

```python
try:
    text = client.parse("document.pdf")
except APIError as e:
    print(f"API error: {e.status_code}")
    print(f"Response: {e.response_text}")
```

### FileProcessingError

Raised when there is an error processing the input file.

```python
class FileProcessingError(DeepSeekOCRError)
```

**Example:**

```python
try:
    text = client.parse("invalid.pdf")
except FileProcessingError as e:
    print(f"File error: {e}")
```

### TimeoutError

Raised when an API request times out.

```python
class TimeoutError(DeepSeekOCRError)
```

**Example:**

```python
try:
    client = DeepSeekOCR(api_key="xxx", timeout=5)
    text = client.parse("large_document.pdf")
except TimeoutError as e:
    print(f"Request timed out: {e}")
```

### InvalidModeError

Raised when an invalid OCR mode is specified.

```python
class InvalidModeError(DeepSeekOCRError)
```

---

## Environment Variables

The SDK supports the following environment variables for configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `DS_OCR_API_KEY` | API key | **Required** |
| `DS_OCR_BASE_URL` | API endpoint URL | **Required** |
| `DS_OCR_MODEL` | Model name | `deepseek-ai/DeepSeek-OCR` |
| `DS_OCR_TIMEOUT` | Request timeout (seconds) | `60` |
| `DS_OCR_MAX_TOKENS` | Maximum tokens | `4000` |
| `DS_OCR_DPI` | PDF conversion DPI | `200` |
| `DS_OCR_FALLBACK_ENABLED` | Enable fallback | `true` |
| `DS_OCR_FALLBACK_MODE` | Fallback mode | `grounding` |
| `DS_OCR_MIN_OUTPUT_THRESHOLD` | Min output length for fallback | `500` |

**Note**: `DS_OCR_BASE_URL` must be set to your chosen API provider's endpoint. Known provider:
- SiliconFlow: `https://api.siliconflow.cn/v1/chat/completions` (Verified ✅)
- Note: DeepSeek's official API does not support the DeepSeek-OCR model

---

## Type Hints

All public APIs include comprehensive type hints for better IDE support and type checking.

```python
from typing import Union, Optional, List
from pathlib import Path

# Example with full type hints
def process_documents(
    client: DeepSeekOCR,
    files: List[Union[str, Path]],
    mode: Optional[OCRMode] = None
) -> List[str]:
    results: List[str] = []
    for file in files:
        text: str = client.parse(file, mode=mode or OCRMode.FREE_OCR)
        results.append(text)
    return results
```

---

## Best Practices

### 1. Reuse Client Instances

```python
# Good: Reuse client
client = DeepSeekOCR(api_key="xxx")
for file in files:
    text = client.parse(file)

# Bad: Create new client each time
for file in files:
    client = DeepSeekOCR(api_key="xxx")  # Wasteful
    text = client.parse(file)
```

### 2. Use Batch Processing for Multiple Files

```python
# Good: Batch processing
processor = BatchProcessor(client, max_concurrent=5)
summary = await processor.process_batch(files)

# Less efficient: Sequential processing
for file in files:
    text = await client.parse_async(file)
```

### 3. Handle Errors Appropriately

```python
# Good: Specific error handling
try:
    text = client.parse("document.pdf")
except FileProcessingError:
    # Handle file errors
    pass
except APIError as e:
    if e.status_code == 429:
        # Handle rate limiting
        pass
except TimeoutError:
    # Handle timeout
    pass

# Bad: Catch all exceptions
try:
    text = client.parse("document.pdf")
except Exception:
    pass  # Don't know what went wrong
```

### 4. Use Appropriate Modes

```python
# Good: Mode selection based on content
if is_complex_table(file):
    text = client.parse(file, mode="grounding")
else:
    text = client.parse(file, mode="free_ocr")

# Bad: Always using slowest mode
text = client.parse(file, mode="ocr_image")  # Slow for simple docs
```
