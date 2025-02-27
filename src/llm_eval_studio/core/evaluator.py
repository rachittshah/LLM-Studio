"""
Core evaluator class for LLM Eval Studio.
"""
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel

from .types import (
    BaseMetric,
    EvaluationResult,
    EvaluationSummary,
    MetricScore,
    ModelOutput,
    ModelProvider,
)


class EvaluationConfig(BaseModel):
    """Configuration for an evaluation run."""
    model_name: str
    metrics: List[str]
    batch_size: int = 32
    max_concurrent_requests: int = 10
    timeout_seconds: float = 30.0
    metadata: Dict[str, Any] = {}


class LLMEvaluator:
    """Main evaluator class that orchestrates the evaluation process."""

    def __init__(
        self,
        model_provider: ModelProvider,
        metrics: Dict[str, BaseMetric],
        config: Optional[EvaluationConfig] = None,
    ):
        """Initialize the evaluator.
        
        Args:
            model_provider: Configuration for the LLM provider
            metrics: Dictionary mapping metric names to metric instances
            config: Optional evaluation configuration
        """
        self.model_provider = model_provider
        self.metrics = metrics
        self.config = config or EvaluationConfig(
            model_name=model_provider.name,
            metrics=list(metrics.keys())
        )
        
        # Initialize semaphore for concurrent request limiting
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

    async def evaluate_single(
        self,
        input_text: str,
        output_text: Optional[str] = None,
        reference_text: Optional[str] = None,
        metric_names: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Evaluate a single example.
        
        Args:
            input_text: The input prompt/query
            output_text: Optional model output (if None, will generate using model_provider)
            reference_text: Optional reference/ground truth text
            metric_names: Optional list of metrics to compute (if None, use all)
            
        Returns:
            EvaluationResult containing scores for all metrics
        """
        # Generate output if not provided
        if output_text is None:
            output = await self._generate_text(input_text)
            output_text = output.text
        
        # Select metrics to compute
        metrics_to_run = (
            [self.metrics[name] for name in metric_names]
            if metric_names
            else list(self.metrics.values())
        )
        
        # Compute all metrics concurrently
        async with self._semaphore:
            scores = await asyncio.gather(*[
                metric.compute(
                    input_text=input_text,
                    output_text=output_text,
                    reference_text=reference_text
                )
                for metric in metrics_to_run
            ])
            
        return EvaluationResult(
            input_text=input_text,
            output_text=output_text,
            reference_text=reference_text,
            scores=scores
        )

    async def evaluate_batch(
        self,
        inputs: List[str],
        outputs: Optional[List[str]] = None,
        references: Optional[List[str]] = None,
        metric_names: Optional[List[str]] = None,
    ) -> List[EvaluationResult]:
        """Evaluate a batch of examples.
        
        Args:
            inputs: List of input prompts/queries
            outputs: Optional list of model outputs
            references: Optional list of reference/ground truth texts
            metric_names: Optional list of metrics to compute
            
        Returns:
            List of EvaluationResults
        """
        if outputs is None:
            # Generate outputs in batches
            outputs = []
            for i in range(0, len(inputs), self.config.batch_size):
                batch = inputs[i:i + self.config.batch_size]
                batch_outputs = await asyncio.gather(*[
                    self._generate_text(text) for text in batch
                ])
                outputs.extend([out.text for out in batch_outputs])
                
        # Evaluate in batches
        results = []
        for i in range(0, len(inputs), self.config.batch_size):
            batch_inputs = inputs[i:i + self.config.batch_size]
            batch_outputs = outputs[i:i + self.config.batch_size]
            batch_refs = None if references is None else references[i:i + self.config.batch_size]
            
            # Run evaluation for batch
            batch_results = await asyncio.gather(*[
                self.evaluate_single(
                    input_text=inp,
                    output_text=out,
                    reference_text=ref,
                    metric_names=metric_names
                )
                for inp, out, ref in zip(batch_inputs, batch_outputs, batch_refs or [None] * len(batch_inputs))
            ])
            results.extend(batch_results)
            
        return results

    def summarize_results(
        self,
        results: List[EvaluationResult],
        dataset_name: str = "unnamed_dataset"
    ) -> EvaluationSummary:
        """Generate summary statistics for a set of evaluation results.
        
        Args:
            results: List of evaluation results
            dataset_name: Name of the dataset being evaluated
            
        Returns:
            EvaluationSummary with aggregated statistics
        """
        metrics_summary = {}
        
        # Group scores by metric
        for metric_name in self.metrics:
            scores = []
            for result in results:
                for score in result.scores:
                    if score.metric_name == metric_name:
                        scores.append(score.score)
            
            if scores:
                metrics_summary[metric_name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores))
                }
        
        return EvaluationSummary(
            model_name=self.config.model_name,
            dataset_name=dataset_name,
            num_examples=len(results),
            metrics_summary=metrics_summary,
            timestamp=datetime.utcnow().isoformat()
        )

    async def _generate_text(self, input_text: str) -> ModelOutput:
        """Generate text using the configured model provider.
        
        This is a placeholder - actual implementation would integrate with
        specific model providers (OpenAI, Anthropic, etc.)
        """
        raise NotImplementedError(
            "Text generation not implemented - subclass or provide model_provider implementation"
        ) 