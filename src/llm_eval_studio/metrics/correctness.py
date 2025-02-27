"""
Correctness metric implementation using LLM-as-judge.
"""
from typing import Any, Dict, List, Optional

from ..core.types import BaseMetric, MetricScore, MetricType, ModelProvider


class CorrectnessMetric(BaseMetric):
    """Evaluates factual correctness using an LLM judge."""

    DEFAULT_PROMPT_TEMPLATE = """You are an expert evaluator assessing the factual correctness of an AI model's response.

Context:
- Input/Question: {input_text}
- Model's Answer: {output_text}
- Reference Answer (if provided): {reference_text}

Please evaluate the factual correctness of the model's answer. Consider:
1. Are all stated facts accurate?
2. Are there any contradictions?
3. Is the answer complete and addresses the question?
4. If a reference is provided, does the answer align with it?

Score the answer from 0.0 to 1.0 where:
- 1.0: Completely correct and complete
- 0.7-0.9: Mostly correct with minor issues
- 0.4-0.6: Partially correct with significant issues
- 0.0-0.3: Mostly incorrect or irrelevant

Provide your score and a brief explanation in the following format:
SCORE: <number between 0 and 1>
EXPLANATION: <your reasoning>
"""

    def __init__(
        self,
        judge_model: ModelProvider,
        prompt_template: Optional[str] = None,
        **kwargs: Any
    ):
        """Initialize the correctness metric.
        
        Args:
            judge_model: Model provider configuration for the judge
            prompt_template: Optional custom prompt template
            **kwargs: Additional configuration
        """
        self.judge_model = judge_model
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.kwargs = kwargs

    async def compute(
        self,
        input_text: str,
        output_text: str,
        reference_text: Optional[str] = None,
        **kwargs: Any
    ) -> MetricScore:
        """Compute correctness score for a single example.
        
        Args:
            input_text: The input prompt/query
            output_text: The model's output to evaluate
            reference_text: Optional reference/ground truth
            **kwargs: Additional parameters
            
        Returns:
            MetricScore with correctness evaluation
        """
        # Format the evaluation prompt
        eval_prompt = self.prompt_template.format(
            input_text=input_text,
            output_text=output_text,
            reference_text=reference_text or "Not provided"
        )
        
        # Get evaluation from judge model
        # Note: This is a placeholder - actual implementation would call the judge model
        judge_output = await self._call_judge(eval_prompt)
        
        # Parse score and explanation
        try:
            score_line = next(line for line in judge_output.split('\n') if line.startswith('SCORE:'))
            explanation_line = next(line for line in judge_output.split('\n') if line.startswith('EXPLANATION:'))
            
            score = float(score_line.split(':')[1].strip())
            explanation = explanation_line.split(':')[1].strip()
            
            return MetricScore(
                metric_name="correctness",
                metric_type=MetricType.CORRECTNESS,
                score=score,
                explanation=explanation,
                metadata={
                    "judge_model": self.judge_model.name,
                    "reference_provided": reference_text is not None
                }
            )
        except Exception as e:
            # Handle parsing errors
            return MetricScore(
                metric_name="correctness",
                metric_type=MetricType.CORRECTNESS,
                score=0.0,
                explanation=f"Error parsing judge output: {str(e)}",
                metadata={"error": True}
            )

    async def batch_compute(
        self,
        inputs: List[str],
        outputs: List[str],
        references: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[MetricScore]:
        """Compute correctness scores for a batch of examples.
        
        For simplicity, this implementation just calls compute() for each example.
        A more optimized implementation could batch the judge model calls.
        """
        if references is None:
            references = [None] * len(inputs)
            
        return [
            await self.compute(input_text, output_text, reference_text, **kwargs)
            for input_text, output_text, reference_text 
            in zip(inputs, outputs, references)
        ]

    async def _call_judge(self, prompt: str) -> str:
        """Call the judge model to evaluate correctness.
        
        This is a placeholder - actual implementation would integrate with
        specific model providers.
        """
        raise NotImplementedError(
            "Judge model calling not implemented - provide concrete implementation"
        ) 