import os
import time
from datetime import datetime
from typing import Dict, Any
import logging
from openai import OpenAI
from diskcache import Cache
from models.base_model import BaseModel

logger = logging.getLogger(__name__)

class HuggingFaceModel(BaseModel):
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        super().__init__(model_name, config)
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is not set.")
            
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token
        )
        self.cache = Cache(".cache/hf_responses")
        
        self.temperature = self.config.get("temperature", 0.2)
        self.max_tokens = self.config.get("max_new_tokens", 512)
        
    def generate(self, prompt: str) -> Dict[str, Any]:
        cache_key = f"{self.model_name}_{self.temperature}_{self.max_tokens}_{prompt}"
        
        force_rerun = getattr(self, 'force_rerun', False)
        if cache_key in self.cache and not force_rerun:
            logger.debug(f"Cache hit for prompt: {prompt[:30]}...")
            cached_data = self.cache[cache_key]
            cached_data["cached"] = True
            return cached_data
            
        logger.info(f"Generating response from {self.model_name}...")
        start_time = time.time()
        
        from tenacity import Retrying, wait_exponential, stop_after_attempt
        import openai
        
        retryer = Retrying(
            wait=wait_exponential(multiplier=1, min=2, max=10),
            stop=stop_after_attempt(4)
        )
        
        retry_count = 0
        content = ""
        token_estimate = 0
        generation_failed = False
        failure_reason = None
        
        try:
            for attempt in retryer:
                with attempt:
                    retry_count = attempt.retry_state.attempt_number - 1
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    content = response.choices[0].message.content
                    token_estimate = response.usage.completion_tokens if getattr(response, "usage", None) else len(content.split())
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            generation_failed = True
            if isinstance(e, openai.RateLimitError) or (hasattr(e, 'status_code') and e.status_code == 429) or "429" in str(e):
                failure_reason = "rate_limited"
            else:
                failure_reason = str(e)
            
        latency_ms = (time.time() - start_time) * 1000
        result = {
            "response": content,
            "latency_ms": latency_ms,
            "token_estimate": token_estimate,
            "cached": False,
            "timestamp": datetime.utcnow().isoformat(),
            "generation_failed": generation_failed,
            "failure_reason": failure_reason,
            "retry_count": retry_count
        }
        
        if not generation_failed:
            self.cache[cache_key] = result
            
        return result
