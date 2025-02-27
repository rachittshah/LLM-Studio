"""
Tests for the core evaluator functionality.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from llm_eval_studio.core.evaluator import EvaluationConfig, LLMEvaluator
from llm_eval_studio.core.types import (
    BaseMetric,
    EvaluationResult,
    MetricScore,
    MetricType,
    ModelOutput,
    ModelProvider
)


class MockMetric(BaseMetric):
    """Mock metric for testing."""
    
    async def compute(self, input_text, output_text, reference_text=None, **kwargs):
        return MetricScore(
            metric_name="mock_metric",
            metric_type=MetricType.CORRECTNESS,
            score=1.0,
            explanation="Mock explanation"
        )
    
    def batch_compute(self, inputs, outputs, references=None, **kwargs):
        return [
            MetricScore(
                metric_name="mock_metric",
                metric_type=MetricType.CORRECTNESS,
                score=1.0,
                explanation="Mock explanation"
            )
            for _ in inputs
        ]


class MockModelProvider:
    """Mock model provider for testing."""
    
    async def generate(self, prompt, **kwargs):
        return ModelOutput(
            text="Mock response",
            tokens_used=10,
            latency_ms=100.0
        )
    
    async def generate_batch(self, prompts, **kwargs):
        return [
            ModelOutput(
                text="Mock response",
                tokens_used=10,
                latency_ms=100.0
            )
            for _ in prompts
        ]


@pytest.fixture
def evaluator():
    """Create a test evaluator instance."""
    config = EvaluationConfig(
        model_name="test-model",
        metrics=["mock_metric"]
    )
    
    return LLMEvaluator(
        model_provider=ModelProvider(name="test-model"),
        metrics={"mock_metric": MockMetric()},
        config=config
    )


@pytest.mark.asyncio
async def test_evaluate_single(evaluator):
    """Test single example evaluation."""
    result = await evaluator.evaluate_single(
        input_text="test input",
        output_text="test output"
    )
    
    assert isinstance(result, EvaluationResult)
    assert len(result.scores) == 1
    assert result.scores[0].metric_name == "mock_metric"
    assert result.scores[0].score == 1.0


@pytest.mark.asyncio
async def test_evaluate_batch(evaluator):
    """Test batch evaluation."""
    results = await evaluator.evaluate_batch(
        inputs=["test1", "test2"],
        outputs=["output1", "output2"]
    )
    
    assert len(results) == 2
    assert all(isinstance(r, EvaluationResult) for r in results)
    assert all(len(r.scores) == 1 for r in results)
    assert all(r.scores[0].metric_name == "mock_metric" for r in results)


def test_summarize_results(evaluator):
    """Test results summarization."""
    results = [
        EvaluationResult(
            input_text="test1",
            output_text="output1",
            scores=[
                MetricScore(
                    metric_name="mock_metric",
                    metric_type=MetricType.CORRECTNESS,
                    score=1.0
                )
            ]
        ),
        EvaluationResult(
            input_text="test2",
            output_text="output2",
            scores=[
                MetricScore(
                    metric_name="mock_metric",
                    metric_type=MetricType.CORRECTNESS,
                    score=0.5
                )
            ]
        )
    ]
    
    summary = evaluator.summarize_results(results, "test_dataset")
    
    assert summary.model_name == "test-model"
    assert summary.dataset_name == "test_dataset"
    assert summary.num_examples == 2
    assert "mock_metric" in summary.metrics_summary
    assert summary.metrics_summary["mock_metric"]["mean"] == 0.75
    assert summary.metrics_summary["mock_metric"]["min"] == 0.5
    assert summary.metrics_summary["mock_metric"]["max"] == 1.0 