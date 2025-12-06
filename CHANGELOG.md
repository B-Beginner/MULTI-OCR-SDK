# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-12-07

### Breaking Changes

- **PDF processing now handles all pages by default** instead of only the first page
  - Previous behavior: `parse("doc.pdf")` processed only page 1
  - New behavior: `parse("doc.pdf")` processes all pages and concatenates results
  - **Migration guide**: To maintain old behavior, use `parse("doc.pdf", pages=1)`
  - **Cost impact**: Processing all pages will increase API usage and costs proportionally

### Added

- Multi-page PDF processing with flexible page selection
  - `pages=None`: Process all pages (new default)
  - `pages=1`: Process single page (old behavior)
  - `pages=[1, 3, 5]`: Process specific pages
- Per-page intelligent fallback mechanism
  - Each page evaluated independently
  - Short output triggers automatic fallback to "grounding" mode
  - Graceful error handling if fallback fails
- Concurrent processing for async methods using `asyncio.gather`
- Configurable page separator via `DS_OCR_PAGE_SEPARATOR` environment variable (default: `\n\n---\n\n`)
- Comprehensive test coverage (28 tests including 19 new tests)
- Enhanced error handling with clear validation messages

### Changed

- Improved error messages for page validation
- Better resource management with proper cleanup in try-finally blocks

### Fixed

- PDF processing no longer limited to first page only

## [0.1.1] - 2025-12-06

### Added

- Initial release with basic OCR functionality
- Support for PDF and image file processing
- Synchronous and asynchronous API methods
- Three OCR modes: FREE_OCR, GROUNDING, and MULTIMODAL
- Environment-based configuration
- Progress bar support via tqdm
