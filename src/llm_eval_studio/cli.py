"""
Command-line interface for LLM Eval Studio.
"""
import asyncio
import json
from pathlib import Path
from typing import List, Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from .core.evaluator import EvaluationConfig, LLMEvaluator
from .core.types import ModelProvider
from .metrics.correctness import CorrectnessMetric
from .models.openai_provider import OpenAIProvider

app = typer.Typer(
    name="llm-eval",
    help="LLM Eval Studio - A toolkit for evaluating LLM performance"
)
console = Console()


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML/JSON file."""
    if config_path.suffix in ['.yaml', '.yml']:
        return yaml.safe_load(config_path.read_text())
    return json.loads(config_path.read_text())


def create_evaluator(config: dict) -> LLMEvaluator:
    """Create an evaluator instance from configuration."""
    # Create model provider
    provider = OpenAIProvider(
        config=ModelProvider(
            name=config['model']['name'],
            api_key=config['model'].get('api_key'),
            api_base=config['model'].get('api_base'),
            model_kwargs=config['model'].get('kwargs', {})
        )
    )
    
    # Create metrics
    metrics = {}
    for metric_config in config.get('metrics', []):
        if metric_config['type'] == 'correctness':
            metrics['correctness'] = CorrectnessMetric(
                judge_model=ModelProvider(
                    name=metric_config['judge_model'],
                    api_key=metric_config.get('api_key'),
                    api_base=metric_config.get('api_base')
                )
            )
    
    # Create evaluator
    return LLMEvaluator(
        model_provider=provider,
        metrics=metrics,
        config=EvaluationConfig(**config.get('evaluation', {}))
    )


@app.command()
def evaluate(
    config_path: Path = typer.Argument(
        ...,
        help="Path to evaluation configuration file (YAML/JSON)",
        exists=True
    ),
    input_file: Path = typer.Argument(
        ...,
        help="Path to input data file (JSON)",
        exists=True
    ),
    output_file: Path = typer.Option(
        None,
        help="Path to save evaluation results (JSON)"
    ),
    format: str = typer.Option(
        "table",
        help="Output format (table/json)"
    )
):
    """Evaluate LLM performance using specified configuration."""
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Load input data
        input_data = json.loads(input_file.read_text())
        inputs = input_data.get('inputs', [])
        outputs = input_data.get('outputs')
        references = input_data.get('references')
        
        # Create evaluator
        evaluator = create_evaluator(config)
        
        # Run evaluation
        results = asyncio.run(
            evaluator.evaluate_batch(
                inputs=inputs,
                outputs=outputs,
                references=references
            )
        )
        
        # Generate summary
        summary = evaluator.summarize_results(
            results=results,
            dataset_name=input_file.stem
        )
        
        # Output results
        if format == 'json':
            output = {
                'results': [result.dict() for result in results],
                'summary': summary.dict()
            }
            if output_file:
                output_file.write_text(json.dumps(output, indent=2))
            else:
                console.print_json(json.dumps(output))
        else:
            # Create summary table
            table = Table(title="Evaluation Summary")
            table.add_column("Metric")
            table.add_column("Mean")
            table.add_column("Std Dev")
            table.add_column("Min")
            table.add_column("Max")
            
            for metric, stats in summary.metrics_summary.items():
                table.add_row(
                    metric,
                    f"{stats['mean']:.3f}",
                    f"{stats['std']:.3f}",
                    f"{stats['min']:.3f}",
                    f"{stats['max']:.3f}"
                )
            
            console.print(table)
            
            if output_file:
                console.save_text(str(output_file))
                
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
    config_path: Optional[Path] = typer.Option(
        None,
        help="Path to server configuration file"
    )
):
    """Start the evaluation server."""
    try:
        import uvicorn
        from .api.server import app as api_app
        
        # Load configuration if provided
        if config_path:
            config = load_config(config_path)
            # Configure API app with loaded config
            # (implementation depends on API design)
        
        # Start server
        console.print(f"Starting server on {host}:{port}")
        uvicorn.run(api_app, host=host, port=port)
        
    except Exception as e:
        console.print(f"[red]Error starting server: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app() 