import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

logger = logging.getLogger(__name__)

class PromptFileNotFoundError(Exception):
    pass

class PromptValidationError(Exception):
    pass

class PromptBuilder:
    def __init__(self, prompts_files: List[str] = None):
        if prompts_files is None:
            prompts_files = ["benchmark/prompts.json"]
        self.prompts_files = [Path(f) for f in prompts_files]
        self.prompts: List[Dict[str, Any]] = []
        self._load_prompts()
        self._log_startup_stats()
        
    def _load_prompts(self) -> None:
        for pf in self.prompts_files:
            logger.info(f"Loading prompts from file path: {pf.absolute()}")
            
            if not pf.exists():
                raise PromptFileNotFoundError(f"Prompt file not found at {pf.absolute()}")
                
            try:
                with open(pf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                if isinstance(data, dict):
                    file_prompts = data.get("prompts", [])
                    tier = data.get("tier", "easy")
                    self.benchmark_version = data.get("benchmark_version", "unknown")
                else:
                    file_prompts = data
                    tier = "easy"
                    self.benchmark_version = "v1.0"
                    
                for p in file_prompts:
                    p["tier"] = tier
                    self.prompts.append(p)
                    
            except json.JSONDecodeError as e:
                raise PromptValidationError(f"Malformed JSON schema in {pf}: {e}")
                
        if not self.prompts:
            raise PromptValidationError("Prompt files contain zero prompts. Must have at least one prompt.")
            
        # Validate schema for all items
        for idx, prompt in enumerate(self.prompts):
            if not all(k in prompt for k in ["id", "category", "prompt"]):
                raise PromptValidationError(f"Prompt at index {idx} is missing required fields (id, category, prompt).")
                
    def _log_startup_stats(self) -> None:
        logger.info(f"Prompt count loaded: {len(self.prompts)}")
        categories = Counter(p.get("category", "unknown") for p in self.prompts)
        logger.info("Category distribution:")
        for cat, count in categories.items():
            logger.info(f"  - {cat}: {count}")
            
    def get_all_prompts(self) -> List[Dict[str, Any]]:
        return self.prompts
        
    def get_prompts_by_category(self, category: str) -> List[Dict[str, Any]]:
        return [p for p in self.prompts if p.get("category") == category]
        
    def get_prompt_by_id(self, prompt_id: str) -> Dict[str, Any]:
        for p in self.prompts:
            if p.get("id") == prompt_id:
                return p
        raise ValueError(f"Prompt with id {prompt_id} not found.")
