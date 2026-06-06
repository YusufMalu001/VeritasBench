from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List

class EvaluationConfigurationError(Exception):
    pass

class EvalDimension(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def score(self, prompt_data: Dict[str, Any], response_text: str) -> Tuple[float, str, List[str]]:
        """
        Evaluate the response.
        
        Returns:
            Tuple containing:
            - score (float): 0.0 to 1.0
            - explanation (str): Reason for the score
            - failure_tags (List[str]): List of tags like 'hallucination', 'formatting_error', etc.
        """
        pass
