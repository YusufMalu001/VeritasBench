import os
import time
from datetime import datetime
from typing import Dict, Any
import logging
from groq import Groq
from diskcache import Cache
from models.base_model import BaseModel

logger = logging.getLogger(__name__)

class GroqModel(BaseModel):
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        super().__init__(model_name, config)
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
            
        self.client = Groq(api_key=groq_api_key)
        self.cache = Cache(".cache/groq_responses")
        
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
        import groq
        
        retryer = Retrying(
            wait=wait_exponential(multiplier=1, min=10, max=60),
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
            if isinstance(e, groq.RateLimitError) or (hasattr(e, 'status_code') and e.status_code == 429) or "429" in str(e):
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
