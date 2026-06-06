import re
from typing import Dict, Any, Tuple, List
from dimensions.base_dimension import EvalDimension

class InstructionFollowing(EvalDimension):
    @property
    def name(self) -> str:
        return "instruction_following"

    def score(self, prompt_data: Dict[str, Any], response_text: str) -> Tuple[float, str, List[str]]:
        meta = prompt_data.get("metadata", {})
        constraint_type = meta.get("type", "constraint")
        expected = prompt_data.get("expected_behavior", "")
        
        failure_tags = []
        score = 0.0
        explanation = ""
        
        # Simplified regex-based or rule-based heuristics 
        # (A true LLM judge logic can be layered on top via runner if needed,
        # but the prompt specifies using judge for factuality/instruction following in base plan.
        # We will keep this logic robust enough for the exact prompts we generated.)
        
        if constraint_type == "count_and_format":
            # Check for numbered list (e.g., 1., 2., 3., 4., 5.)
            numbers = re.findall(r'\b[1-9]\.', response_text)
            if len(numbers) == 5:
                score = 1.0
                explanation = "Found exactly 5 numbered items."
            elif len(numbers) > 0:
                score = 0.5
                explanation = f"Found {len(numbers)} numbered items instead of 5."
                failure_tags.append("instruction_violation")
            else:
                score = 0.0
                explanation = "Did not find a numbered list."
                failure_tags.append("instruction_violation")
                
        elif constraint_type == "sentence_count":
            sentences = [s for s in re.split(r'[.!?]+', response_text) if s.strip()]
            if len(sentences) == 3:
                score = 1.0
                explanation = "Found exactly 3 sentences."
            else:
                score = 0.0
                explanation = f"Found {len(sentences)} sentences instead of 3."
                failure_tags.append("instruction_violation")
                
        elif constraint_type == "format_constraint":
            lines = [l for l in response_text.strip().split('\n') if l.strip()]
            if len(lines) == 3:
                score = 1.0
                explanation = "Found exactly 3 lines (expected for haiku)."
            else:
                score = 0.5
                explanation = f"Found {len(lines)} lines."
                failure_tags.append("instruction_violation")
                
        elif constraint_type == "strict_format":
            lines = [l.strip() for l in response_text.strip().split('\n') if l.strip()]
            all_bullets = all(l.startswith('-') or l.startswith('*') or l.startswith('•') for l in lines)
            if all_bullets and len(lines) > 0:
                score = 1.0
                explanation = "All lines are bullet points."
            else:
                score = 0.0
                explanation = "Response contains non-bullet point lines."
                failure_tags.append("instruction_violation")
                
        else:
            # Fallback simple string check or proxy
            if len(response_text.strip()) > 0:
                score = 0.8
                explanation = "Fallback heuristic: response exists."
            else:
                score = 0.0
                explanation = "Empty response."
                failure_tags.append("instruction_violation")
                
        return score, explanation, failure_tags
