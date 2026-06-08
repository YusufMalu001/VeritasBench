import os
import json
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

class ChatModelCompatibilityError(Exception):
    pass

class CompatibilityChecker:
    def __init__(self, cache_file: str = "results/router_compatibility_report.json"):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()
        
    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
        
    def _save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)

    def validate_models(self, models: list[str]) -> None:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set.")
            
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        
        updated = False
        
        for model in models:
            if model in self.cache and self.cache[model].get("chat_supported") is True:
                logger.info(f"Compatibility cache hit: {model} supports chat.")
                continue
                
            logger.info(f"Validating chat compatibility for {model}...")
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                self.cache[model] = {
                    "chat_supported": True,
                    "last_checked": datetime.utcnow().isoformat() + "Z"
                }
                updated = True
                logger.info(f"Validation successful: {model}")
            except Exception as e:
                self.cache[model] = {
                    "chat_supported": False,
                    "last_checked": datetime.utcnow().isoformat() + "Z"
                }
                updated = True
                self._save_cache()
                raise ChatModelCompatibilityError(f"Model {model} failed chat validation: {str(e)}")
                
        if updated:
            self._save_cache()
