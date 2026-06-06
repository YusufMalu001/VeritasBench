from typing import Dict, Any, Tuple, List
from dimensions.base_dimension import EvalDimension

class Verbosity(EvalDimension):
    @property
    def name(self) -> str:
        return "verbosity"

    def score(self, prompt_data: Dict[str, Any], response_text: str) -> Tuple[float, str, List[str]]:
        from dimensions.base_dimension import EvaluationConfigurationError
        meta = prompt_data.get("metadata", {})
        
        if "target_word_count" not in meta:
            raise EvaluationConfigurationError("Missing required metadata: 'target_word_count'")
            
        try:
            target_words = int(meta["target_word_count"])
        except ValueError:
            raise EvaluationConfigurationError("'target_word_count' must be an integer.")
            
        words = response_text.split()
        num_words = len(words)
        
        failure_tags = []
        score = 0.0
        explanation = ""
        
        # Give 1.0 if within 10% margin or exactly equal if small
        margin = max(3, int(target_words * 0.10))
        
        if abs(num_words - target_words) <= margin:
            score = 1.0
            explanation = f"Response length ({num_words} words) is within acceptable margin of target ({target_words})."
        else:
            score = max(0.0, 1.0 - (abs(num_words - target_words) / target_words))
            explanation = f"Response length ({num_words} words) missed target ({target_words})."
            failure_tags.append("verbosity_mismatch")
            
        return score, explanation, failure_tags
