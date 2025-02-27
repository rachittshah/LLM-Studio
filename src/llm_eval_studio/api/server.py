"""
FastAPI server implementation for LLM Eval Studio.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from ..core.evaluator import EvaluationConfig, LLMEvaluator
from ..core.types import EvaluationResult, EvaluationSummary, ModelProvider
from ..metrics.correctness import CorrectnessMetric
from ..models.openai_provider import OpenAIProvider

# Initialize FastAPI app
app = FastAPI(
    title="LLM Eval Studio API",
    description="API for evaluating and monitoring LLM performance",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key security
api_key_header = APIKeyHeader(name="X-API-Key")


# Request/Response Models
class EvaluationRequest(BaseModel):
    """Request body for evaluation endpoints."""
    inputs: List[str]
    outputs: Optional[List[str]] = None
    references: Optional[List[str]] = None
    model_name: str
    metrics: List[str]
    config: Optional[Dict[str, Any]] = None


class EvaluationResponse(BaseModel):
    """Response body for evaluation endpoints."""
    results: List[EvaluationResult]
    summary: EvaluationSummary


# Global state (in practice, use proper dependency injection)
evaluators: Dict[str, LLMEvaluator] = {}


def get_evaluator(model_name: str) -> LLMEvaluator:
    """Get or create an evaluator for the specified model."""
    if model_name not in evaluators:
        # Create model provider (example with OpenAI)
        provider = OpenAIProvider(
            config=ModelProvider(
                name=model_name,
                api_key="YOUR_API_KEY"  # In practice, load from config/env
            )
        )
        
        # Create metrics
        metrics = {
            "correctness": CorrectnessMetric(
                judge_model=ModelProvider(
                    name="gpt-4",
                    api_key="YOUR_API_KEY"
                )
            )
            # Add other metrics here
        }
        
        # Create evaluator
        evaluators[model_name] = LLMEvaluator(
            model_provider=provider,
            metrics=metrics
        )
    
    return evaluators[model_name]


@app.post("/api/v1/evaluate", response_model=EvaluationResponse)
async def evaluate(
    request: EvaluationRequest,
    api_key: str = Security(api_key_header)
) -> EvaluationResponse:
    """Evaluate a batch of examples.
    
    Args:
        request: Evaluation request containing inputs and configuration
        api_key: API key for authentication
        
    Returns:
        Evaluation results and summary
    """
    try:
        # Get or create evaluator
        evaluator = get_evaluator(request.model_name)
        
        # Update config if provided
        if request.config:
            evaluator.config = EvaluationConfig(**request.config)
        
        # Run evaluation
        results = await evaluator.evaluate_batch(
            inputs=request.inputs,
            outputs=request.outputs,
            references=request.references,
            metric_names=request.metrics
        )
        
        # Generate summary
        summary = evaluator.summarize_results(
            results=results,
            dataset_name=f"api_evaluation_{datetime.utcnow().isoformat()}"
        )
        
        return EvaluationResponse(
            results=results,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


@app.get("/api/v1/metrics")
async def list_metrics(
    api_key: str = Security(api_key_header)
) -> Dict[str, Any]:
    """List available evaluation metrics."""
    # In practice, this would be dynamic based on registered metrics
    return {
        "metrics": [
            {
                "name": "correctness",
                "type": "correctness",
                "description": "Evaluates factual correctness using LLM-as-judge"
            }
            # Add other metrics
        ]
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"} 