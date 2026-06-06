import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any
from openai import OpenAI
from diskcache import Cache
from models.base_model import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCapabilityError(Exception):
    pass

class HuggingFaceModel(BaseModel):
    def __init__(self, model_name: str, config: Dict[str, Any] = None, capabilities: Dict[str, bool] = None):
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
        self.capabilities = capabilities or {"supports_chat": True, "supports_text_generation": True}
        
    def generate(self, prompt: str) -> Dict[str, Any]:
        cache_key = f"{self.model_name}_{self.temperature}_{self.max_tokens}_{prompt}"
        
        if cache_key in self.cache:
            logger.debug(f"Cache hit for prompt: {prompt[:30]}...")
            cached_data = self.cache[cache_key]
            cached_data["cached"] = True
            return cached_data
            
        logger.info(f"Generating response from {self.model_name}...")
        
        start_time = time.time()
        
        try:
            if self.capabilities.get("supports_chat"):
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
            elif self.capabilities.get("supports_text_generation"):
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].text
            else:
                raise ModelCapabilityError(f"Model {self.model_name} does not support any configured generation mode.")
                
            # Extremely rough token estimate
            token_estimate = response.usage.completion_tokens if getattr(response, "usage", None) else len(content.split())
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            content = ""
            token_estimate = 0
            
        latency_ms = (time.time() - start_time) * 1000
        
        result = {
            "response": content,
            "latency_ms": latency_ms,
            "token_estimate": token_estimate,
            "cached": False,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Only cache if successful
        if content:
            self.cache[cache_key] = result
            
        return result
