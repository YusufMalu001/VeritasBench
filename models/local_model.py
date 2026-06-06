import time
import logging
from datetime import datetime
from typing import Dict, Any
from models.base_model import BaseModel

logger = logging.getLogger(__name__)

class LocalModel(BaseModel):
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        super().__init__(model_name, config)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self.device = "cpu"
            logger.info(f"Loading local model {model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            
            self.temperature = self.config.get("temperature", 0.2)
            self.max_tokens = self.config.get("max_new_tokens", 512)
        except ImportError:
            logger.error("transformers or torch not installed. Cannot use LocalModel.")
            raise
            
    def generate(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0
            )
            response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            token_estimate = len(outputs[0]) - inputs.input_ids.shape[-1]
        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            response_text = ""
            token_estimate = 0
            
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "response": response_text,
            "latency_ms": latency_ms,
            "token_estimate": token_estimate,
            "cached": False,
            "timestamp": datetime.utcnow().isoformat()
        }
