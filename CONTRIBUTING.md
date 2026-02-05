# Contributing to PIE

Thanks for your interest in contributing!

## Quick Start

1. Fork the repo
2. Clone your fork
3. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and fill in your API keys
6. Run tests: `python tests/test_date_extraction.py`

## Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run the full test suite
python -m pytest tests/

# Run a small extraction
python run.py extract --limit 10
```

## Code Style

- Python 3.9+
- Type hints on all public functions
- Docstrings for classes and complex functions
- No hardcoded API keys or secrets

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run tests
4. Push and open a PR

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for system design.

## Areas for Contribution

- **Benchmarks**: Add new benchmark datasets
- **Adapters**: Add support for new data sources (Notion, Roam, etc.)
- **Visualization**: Improve the graph explorer
- **Entity Resolution**: Better alias detection and merging
- **Procedural Memory**: Pattern extraction from state transitions
