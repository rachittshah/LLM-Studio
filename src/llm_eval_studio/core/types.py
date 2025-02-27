"""
Core types and interfaces for LLM Eval Studio.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class MetricType(str, Enum):
    """Types of evaluation metrics."""
    CORRECTNESS = "correctness"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    BIAS = "bias"
    TOXICITY = "toxicity"
    SAFETY = "safety"
    ROBUSTNESS = "robustness"
    EFFICIENCY = "efficiency"
    CUSTOM = "custom"


class MetricScore(BaseModel):
    """A score from a single metric evaluation."""
    metric_name: str
    metric_type: MetricType
    score: float
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = {}


class EvaluationResult(BaseModel):
    """Result of evaluating a single example."""
    input_text: str
    output_text: str
    reference_text: Optional[str] = None
    scores: List[MetricScore]
    metadata: Dict[str, Any] = {}


class EvaluationSummary(BaseModel):
    """Summary statistics for a complete evaluation run."""
    model_name: str
    dataset_name: str
    num_examples: int
    metrics_summary: Dict[str, Dict[str, float]]  # metric -> {mean, std, min, max}
    timestamp: str
    metadata: Dict[str, Any] = {}


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    async def compute(
        self,
        input_text: str,
        output_text: str,
        reference_text: Optional[str] = None,
        **kwargs: Any
    ) -> MetricScore:
        """Compute the metric score for a single example.
        
        Args:
            input_text: The input prompt/query
            output_text: The model's output to evaluate
            reference_text: Optional reference/ground truth
            **kwargs: Additional metric-specific parameters
            
        Returns:
            MetricScore containing the evaluation result
        """
        pass

    @abstractmethod
    def batch_compute(
        self,
        inputs: List[str],
        outputs: List[str],
        references: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[MetricScore]:
        """Compute metric scores for a batch of examples.
        
        Default implementation calls compute() for each example.
        Override for more efficient batch processing.
        """
        pass


@dataclass
class ModelProvider:
    """Configuration for an LLM provider."""
    name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model_kwargs: Dict[str, Any] = {}


class ModelOutput(BaseModel):
    """Output from an LLM, including metadata."""
    text: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = {} 