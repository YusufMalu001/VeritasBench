from typing import Dict, Any, Tuple, List
from dimensions.base_dimension import EvalDimension

class Verbosity(EvalDimension):
    @property
    def name(self) -> str:
        return "verbosity"

    def score(self, prompt_data: Dict[str, Any], response_text: str) -> Tuple[float, str, List[str]]:
        meta = prompt_data.get("metadata", {})
        expected_length = meta.get("expected_length", "medium")
        
        words = response_text.split()
        num_words = len(words)
        
        failure_tags = []
        score = 0.0
        explanation = ""
        
        if expected_length == "short":
            if num_words < 50:
                score = 1.0
                explanation = f"Response is appropriately short ({num_words} words)."
            else:
                score = 0.0
                explanation = f"Response is too long for 'short' expected length ({num_words} words)."
                failure_tags.append("verbosity_mismatch")
                
        elif expected_length == "medium":
            if 50 <= num_words <= 200:
                score = 1.0
                explanation = f"Response is appropriately medium ({num_words} words)."
            elif num_words < 50:
                score = 0.5
                explanation = f"Response is slightly too short for 'medium' expected length ({num_words} words)."
                failure_tags.append("verbosity_mismatch")
            else:
                score = 0.0
                explanation = f"Response is too long for 'medium' expected length ({num_words} words)."
                failure_tags.append("verbosity_mismatch")
                
        elif expected_length == "long":
            if num_words > 200:
                score = 1.0
                explanation = f"Response is appropriately long ({num_words} words)."
            else:
                score = 0.0
                explanation = f"Response is too short for 'long' expected length ({num_words} words)."
                failure_tags.append("verbosity_mismatch")
                
        else:
            return 1.0, "No strict verbosity expectation.", []
            
        return score, explanation, failure_tags
