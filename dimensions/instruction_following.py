import re
from typing import Dict, Any, Tuple, List
from dimensions.base_dimension import EvalDimension

class InstructionFollowing(EvalDimension):
    @property
    def name(self) -> str:
        return "instruction_following"

    def score(self, prompt_data: Dict[str, Any], response_text: str) -> Tuple[float, str, List[str]]:
        from dimensions.base_dimension import EvaluationConfigurationError
        meta = prompt_data.get("metadata", {})
        
        if "required_steps" not in meta:
            raise EvaluationConfigurationError("Missing required metadata: 'required_steps'")
            
        steps = meta["required_steps"]
        if not isinstance(steps, list):
            raise EvaluationConfigurationError("'required_steps' must be a list of strings.")
            
        failure_tags = []
        score = 0.0
        explanation = ""
        
        if not steps:
            raise EvaluationConfigurationError("'required_steps' cannot be empty.")
            
        response_lower = response_text.lower()
        met_steps = 0
        
        for step in steps:
            # A very simple heuristic string match to simulate checking steps
            # A real LLM judge would do this natively. We just approximate it for now.
            if any(word in response_lower for word in step.lower().split() if len(word) > 4):
                met_steps += 1
                
        score = met_steps / len(steps)
        explanation = f"Matched {met_steps} out of {len(steps)} required steps heuristically."
        
        if score < 1.0:
            failure_tags.append("instruction_violation")
            
        return score, explanation, failure_tags
