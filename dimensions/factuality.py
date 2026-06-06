from typing import Dict, Any, Tuple, List
from dimensions.base_dimension import EvalDimension

class Factuality(EvalDimension):
    @property
    def name(self) -> str:
        return "factuality"

    def score(self, prompt_data: Dict[str, Any], response_text: str) -> Tuple[float, str, List[str]]:
        from dimensions.base_dimension import EvaluationConfigurationError
        meta = prompt_data.get("metadata", {})
        
        if "reference_answer" not in meta:
            raise EvaluationConfigurationError("Missing required metadata: 'reference_answer'")
        if "evaluation_type" not in meta:
            raise EvaluationConfigurationError("Missing required metadata: 'evaluation_type'")
            
        expected_answer = meta["reference_answer"].lower()
        eval_type = meta["evaluation_type"]
        
        failure_tags = []
        score = 0.0
        explanation = ""
        
        # Simplified semantic match heuristic for now
        # The true implementation could use LLM-as-a-judge if injected
        response_lower = response_text.lower()
        
        if expected_answer in response_lower:
            score = 1.0
            explanation = f"Response correctly contains '{expected_answer}'."
        else:
            # Fallback simple semantic overlap heuristic
            overlap = set(expected_answer.split()) & set(response_lower.split())
            if len(overlap) / max(1, len(set(expected_answer.split()))) > 0.5:
                score = 0.5
                explanation = f"Response partially matches '{expected_answer}'."
            else:
                score = 0.0
                explanation = f"Response does not contain expected answer '{expected_answer}'."
                failure_tags.append("hallucination")
            
        return score, explanation, failure_tags
