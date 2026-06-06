from typing import Dict, Any, Tuple, List
from dimensions.base_dimension import EvalDimension

class Factuality(EvalDimension):
    @property
    def name(self) -> str:
        return "factuality"

    def score(self, prompt_data: Dict[str, Any], response_text: str) -> Tuple[float, str, List[str]]:
        meta = prompt_data.get("metadata", {})
        expected_answer = meta.get("answer", "").lower()
        
        failure_tags = []
        score = 0.0
        explanation = ""
        
        if not expected_answer:
            return 1.0, "No strict answer to verify.", []
            
        if expected_answer in response_text.lower():
            score = 1.0
            explanation = f"Response correctly contains '{expected_answer}'."
        else:
            score = 0.0
            explanation = f"Response does not contain expected answer '{expected_answer}'."
            failure_tags.append("hallucination")
            
        return score, explanation, failure_tags
