from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseModel(ABC):
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        self.model_name = model_name
        self.config = config or {}

    @abstractmethod
    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a response for a given prompt.
        
        Returns:
            Dict containing:
            - response: str
            - latency_ms: float
            - token_estimate: int
            - cached: bool
            - timestamp: str
        """
        pass
