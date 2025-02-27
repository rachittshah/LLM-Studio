# Contributing to LLM Eval Studio

We love your input! We want to make contributing to LLM Eval Studio as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Pull Request Process

1. Update the README.md with details of changes to the interface, if applicable.
2. Update the docs/ with any new documentation.
3. The PR will be merged once you have the sign-off of at least one other developer.

## Any contributions you make will be under the Apache License 2.0

In short, when you submit code changes, your submissions are understood to be under the same [Apache License 2.0](http://choosealicense.com/licenses/apache-2.0/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/llm-eval-studio/llm-eval-studio/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/llm-eval-studio/llm-eval-studio/issues/new).

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/llm-eval-studio/llm-eval-studio.git
cd llm_eval_studio
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

We use several tools to maintain code quality:

- [Black](https://black.readthedocs.io/) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [mypy](http://mypy-lang.org/) for static type checking
- [pylint](https://www.pylint.org/) for code analysis

Before submitting a PR, please run:
```bash
# Format code
black .
isort .

# Run type checking
mypy .

# Run linting
pylint src tests

# Run tests
pytest
```

## Adding New Metrics

1. Create a new file in `src/llm_eval_studio/metrics/`
2. Implement the `BaseMetric` interface
3. Add tests in `tests/metrics/`
4. Update documentation
5. Add example usage

Example:
```python
from llm_eval_studio.core.types import BaseMetric, MetricScore, MetricType

class MyNewMetric(BaseMetric):
    async def compute(self, input_text, output_text, reference_text=None, **kwargs):
        # Implement your metric logic here
        return MetricScore(
            metric_name="my_metric",
            metric_type=MetricType.CUSTOM,
            score=0.5,
            explanation="Explanation of the score"
        )
```

## Adding New Model Providers

1. Create a new file in `src/llm_eval_studio/models/`
2. Implement the provider interface
3. Add tests in `tests/models/`
4. Update documentation
5. Add example usage

## Documentation

We use [Sphinx](https://www.sphinx-doc.org/) for documentation. To build the docs:

```bash
cd docs
make html
```

## License

By contributing, you agree that your contributions will be licensed under its Apache License 2.0. 