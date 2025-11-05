# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2025-01-05

### Changed
- **BREAKING**: Upgraded minimum Python version from 3.8+ to 3.10+
- Updated Python version badge in README to 3.10+
- Reduced CI test matrix from 5 to 3 Python versions (3.10, 3.11, 3.12)
- Updated black target version to py310

### Fixed
- Fixed GitHub Actions Documentation workflow Mermaid validation error
- Added twine to dev dependencies for PyPI publishing
- Fixed Python 3.8 CI test failures due to twine 6.2.0 incompatibility

### Removed
- Dropped support for Python 3.8 and 3.9 (EOL or near-EOL)

## [0.1.0] - 2025-01-05

### Added
- Initial release of DeepSeek-OCR-SDK
- Core OCR client with sync and async support
- Three OCR modes: FREE_OCR, GROUNDING, OCR_IMAGE
- Intelligent fallback mechanism
- Batch processing with progress tracking
- Comprehensive error handling
- Full type hints support
- Environment variable configuration
- Examples for basic usage and batch processing
- Complete documentation (README, API Reference, Benchmarks)
- GitHub Actions CI/CD workflows
- MIT License

### Features
- Simple and clean API
- Support for PDF and image files
- Configurable DPI for PDF conversion
- Chinese language hint support
- Automatic output cleaning
- Token usage tracking
- Concurrent batch processing with semaphore control
- Progress bar for batch operations
- Retry mechanism for failed requests

### Documentation
- Bilingual README (English + Chinese)
- API reference with examples
- Performance benchmarks vs MinerU and Docling
- Usage examples for common scenarios
- Mermaid flowcharts for architecture

### Development
- uv for dependency management
- pytest for testing
- black, isort, flake8, mypy for code quality
- GitHub Actions for CI/CD
- Type checking with mypy

[Unreleased]: https://github.com/BukeLy/DeepSeek-OCR-SDK/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/BukeLy/DeepSeek-OCR-SDK/releases/tag/v0.1.1
[0.1.0]: https://github.com/BukeLy/DeepSeek-OCR-SDK/releases/tag/v0.1.0
