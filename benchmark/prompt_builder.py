import json
from pathlib import Path
from typing import List, Dict, Any

class PromptBuilder:
    def __init__(self, prompts_file: str = "benchmark/prompts.json"):
        self.prompts_file = Path(prompts_file)
        self.prompts: List[Dict[str, Any]] = []
        self._load_prompts()
        
    def _load_prompts(self) -> None:
        if self.prompts_file.exists():
            with open(self.prompts_file, "r", encoding="utf-8") as f:
                self.prompts = json.load(f)
        else:
            self.prompts = []
            
    def get_all_prompts(self) -> List[Dict[str, Any]]:
        return self.prompts
        
    def get_prompts_by_category(self, category: str) -> List[Dict[str, Any]]:
        return [p for p in self.prompts if p.get("category") == category]
        
    def get_prompt_by_id(self, prompt_id: str) -> Dict[str, Any]:
        for p in self.prompts:
            if p.get("id") == prompt_id:
                return p
        raise ValueError(f"Prompt with id {prompt_id} not found.")
