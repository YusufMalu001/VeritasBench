import os
import json
import logging
from typing import Dict, Any
from openai import OpenAI
from diskcache import Cache
from tenacity import retry, wait_exponential, stop_after_attempt

logger = logging.getLogger(__name__)

class LLMJudge:
    def __init__(self, judge_model: str = "Qwen/Qwen3-32B"):
        self.judge_model = judge_model
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN is not set")
            
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token
        )
        self.cache = Cache(".cache/llm_judge")
        
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    def _call_judge(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    def evaluate_pairwise(self, prompt: str, response_a: str, response_b: str, dimension_name: str = "overall") -> Dict[str, Any]:
        cache_key = f"pairwise_{self.judge_model}_{prompt}_{response_a}_{response_b}_{dimension_name}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        judge_prompt = f"""
        You are an impartial AI judge. Evaluate the quality of two responses to a user prompt based on the dimension: {dimension_name}.
        
        User Prompt:
        {prompt}
        
        Response A:
        {response_a}
        
        Response B:
        {response_b}
        
        Evaluate which response is better. Respond ONLY with a valid JSON object with the following schema:
        {{
            "winner": "A" or "B" or "Tie",
            "confidence": float between 0 and 1,
            "reasoning": "brief explanation"
        }}
        """
        
        try:
            content = self._call_judge(judge_prompt)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
                
            result = json.loads(content)
            winner = result.get("winner", "Tie")
            if winner not in ["A", "B", "Tie"]:
                winner = "Tie"
            confidence = float(result.get("confidence", 0.0))
            reasoning = result.get("reasoning", "Failed to parse reasoning")
            
            final_result = {
                "winner": winner,
                "confidence": confidence,
                "reasoning": reasoning
            }
            self.cache[cache_key] = final_result
            return final_result
            
        except Exception as e:
            logger.error(f"LLM Judge failed: {e}")
            return {
                "winner": "Tie",
                "confidence": 0.0,
                "reasoning": f"Error during judging: {e}"
            }
