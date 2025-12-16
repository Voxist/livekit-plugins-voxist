# Contributing to livekit-plugins-voxist

Thank you for your interest in contributing to the Voxist LiveKit plugin!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/Voxist/livekit-plugins-voxist.git
cd livekit-plugins-voxist
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=livekit.plugins.voxist --cov-report=term-missing

# Run specific test file
pytest tests/test_stt.py -v
```

## Code Quality

Before submitting a PR, ensure your code passes:

```bash
# Type checking
mypy livekit/plugins/voxist

# Linting
ruff check livekit/plugins/voxist

# Formatting
ruff format livekit/plugins/voxist
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Reporting Issues

Please use GitHub Issues to report bugs or request features. Include:
- Python version
- livekit-agents version
- Steps to reproduce
- Expected vs actual behavior

## Code of Conduct

Be respectful and inclusive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
