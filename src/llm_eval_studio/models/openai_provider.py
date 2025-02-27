"""
OpenAI model provider implementation.
"""
import asyncio
import time
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.types import ModelOutput, ModelProvider


class OpenAIProvider:
    """Implementation of model provider interface for OpenAI's models."""

    def __init__(
        self,
        config: ModelProvider,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """Initialize the OpenAI provider.
        
        Args:
            config: ModelProvider configuration with API key etc.
            timeout: Timeout in seconds for API calls
            max_retries: Maximum number of retries for failed calls
        """
        self.config = config
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
            timeout=timeout
        )
        
        # Default model parameters
        self.default_params = {
            "temperature": 0.0,  # Use deterministic outputs for evaluation
            "max_tokens": 1000,
            **config.model_kwargs
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate(
        self,
        prompt: str,
        **kwargs: Any
    ) -> ModelOutput:
        """Generate text using the OpenAI model.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            ModelOutput containing the generated text and metadata
        """
        # Merge default params with any overrides
        params = {**self.default_params, **kwargs}
        
        # Record start time for latency tracking
        start_time = time.time()
        
        try:
            # Call the OpenAI API
            response = await self.client.chat.completions.create(
                model=self.config.name,
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract the generated text
            text = response.choices[0].message.content
            
            return ModelOutput(
                text=text,
                tokens_used=response.usage.total_tokens,
                latency_ms=latency_ms,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "model": response.model,
                }
            )
            
        except Exception as e:
            # Log error and re-raise
            print(f"Error calling OpenAI API: {str(e)}")
            raise

    async def generate_batch(
        self,
        prompts: List[str],
        batch_size: int = 32,
        **kwargs: Any
    ) -> List[ModelOutput]:
        """Generate text for multiple prompts in parallel.
        
        Args:
            prompts: List of input prompts
            batch_size: Maximum number of concurrent requests
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            List of ModelOutputs
        """
        # Process prompts in batches
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            
            # Generate outputs concurrently
            batch_results = await asyncio.gather(*[
                self.generate(prompt, **kwargs)
                for prompt in batch
            ])
            
            results.extend(batch_results)
            
        return results 