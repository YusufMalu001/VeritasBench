import json
import re
from typing import Dict, Any, Tuple, List
from dimensions.base_dimension import EvalDimension

class FormatAdherence(EvalDimension):
    @property
    def name(self) -> str:
        return "format_adherence"

    def score(self, prompt_data: Dict[str, Any], response_text: str) -> Tuple[float, str, List[str]]:
        meta = prompt_data.get("metadata", {})
        fmt = meta.get("format", "")
        
        failure_tags = []
        score = 0.0
        explanation = ""
        
        if fmt == "json":
            try:
                # find json boundaries if padded
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start != -1 and end != 0:
                    json.loads(response_text[start:end])
                    score = 1.0
                    explanation = "Valid JSON found."
                else:
                    raise ValueError("No JSON boundaries found.")
            except Exception:
                score = 0.0
                explanation = "Failed to parse JSON."
                failure_tags.append("formatting_error")
                
        elif fmt == "markdown_table":
            if "|" in response_text and "-|-" in response_text.replace(" ", ""):
                score = 1.0
                explanation = "Markdown table structure detected."
            else:
                score = 0.0
                explanation = "No valid markdown table detected."
                failure_tags.append("formatting_error")
                
        elif fmt == "numbered_bold":
            if re.search(r'\d+\.\s+\*\*', response_text):
                score = 1.0
                explanation = "Numbered bold items detected."
            else:
                score = 0.0
                explanation = "Numbered bold items missing."
                failure_tags.append("formatting_error")
                
        elif fmt == "custom_markers":
            if "PROS:" in response_text and "CONS:" in response_text:
                score = 1.0
                explanation = "Custom markers PROS: and CONS: detected."
            else:
                score = 0.0
                explanation = "Custom markers missing."
                failure_tags.append("formatting_error")
                
        elif fmt == "code_block":
            if "```python" in response_text and "```" in response_text.replace("```python", ""):
                score = 1.0
                explanation = "Python code block detected."
            else:
                score = 0.0
                explanation = "Code block missing."
                failure_tags.append("formatting_error")
                
        else:
            return 1.0, "No strict format requirement.", []
            
        return score, explanation, failure_tags
