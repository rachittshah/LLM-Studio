# LLM Eval Studio

A comprehensive open-source toolkit for evaluating and monitoring Large Language Models (LLMs).

## Features

- 🎯 **Comprehensive Evaluation Metrics**: Assess correctness, coherence, relevance, bias, safety, and more
- 🤖 **LLM-as-Judge Capabilities**: Leverage powerful language models to evaluate outputs
- 🔄 **Prompt Optimization**: Integrated with DSPy for systematic prompt evaluation and improvement
- 🔌 **Broad Integration Support**: Works with major LLM providers and frameworks
- 📊 **Interactive Dashboard**: Web UI for visualization and analysis
- 🛠️ **Flexible Deployment**: Run locally, on-premise, or in the cloud
- 🤝 **Framework Agnostic**: Compatible with LangChain, LlamaIndex, and more

## Installation

```bash
# Clone the repository
git clone https://github.com/llm-eval-studio/llm-eval-studio.git
cd llm-eval-studio

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

## Quick Start

### Command Line Interface

1. Create a configuration file (see `examples/config.yaml`):
```yaml
model:
  name: "gpt-3.5-turbo"
  api_key: "${OPENAI_API_KEY}"

metrics:
  - type: "correctness"
    judge_model: "gpt-4"
```

2. Prepare your input data (see `examples/input.json`):
```json
{
    "inputs": ["What is the capital of France?"],
    "references": ["Paris is the capital of France."]
}
```

3. Run the evaluation:
```bash
llm-eval evaluate config.yaml input.json --output results.json
```

### Python API

```python
from llm_eval_studio import LLMEvaluator
from llm_eval_studio.core.types import ModelProvider
from llm_eval_studio.metrics.correctness import CorrectnessMetric

# Configure the evaluator
evaluator = LLMEvaluator(
    model_provider=ModelProvider(
        name="gpt-3.5-turbo",
        api_key="your-api-key"
    ),
    metrics={
        "correctness": CorrectnessMetric(
            judge_model=ModelProvider(
                name="gpt-4",
                api_key="your-api-key"
            )
        )
    }
)

# Run evaluation
results = await evaluator.evaluate_batch(
    inputs=["What is the capital of France?"],
    references=["Paris is the capital of France."]
)

# Get summary
summary = evaluator.summarize_results(results)
print(f"Average correctness: {summary.metrics_summary['correctness']['mean']:.2f}")
```

### Web UI

1. Start the server:
```bash
llm-eval serve --config server_config.yaml
```

2. Visit `http://localhost:8000` in your browser

## Architecture

LLM Eval Studio follows a modular architecture:

- **Core**: Base types, interfaces, and evaluation logic
- **Metrics**: Implementations of various evaluation metrics
- **Models**: Model provider implementations (OpenAI, Anthropic, etc.)
- **API**: FastAPI server for the web interface
- **UI**: React-based dashboard (coming soon)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black .
isort .
mypy .
```

## Documentation

Full documentation is available at [https://llmevalstudio.readthedocs.io/](https://llmevalstudio.readthedocs.io/)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use LLM Eval Studio in your research, please cite:

```bibtex
@software{llm_eval_studio,
  title = {LLM Eval Studio: A Comprehensive Evaluation Toolkit for Large Language Models},
  author = {LLM Eval Studio Team},
  year = {2024},
  url = {https://github.com/llm-eval-studio/llm-eval-studio}
}
``` 