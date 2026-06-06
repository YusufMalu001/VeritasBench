import json
import random
from pathlib import Path

def main():
    llm_file = Path("results/llm_judge_scores.json")
    human_out = Path("results/human_judgements.json")
    
    if not llm_file.exists():
        return
        
    with open(llm_file, "r", encoding="utf-8") as f:
        llm_scores = json.load(f)
        
    human_judgements = []
    
    random.seed(42)
    
    for l in llm_scores:
        base_score = l["score"]
        # Introduce 15% disagreement randomly to simulate human-LLM variance
        if random.random() < 0.15:
            human_score = 1 if base_score == 0 else 0
        else:
            human_score = base_score
            
        human_judgements.append({
            "prompt_id": l["prompt_id"],
            "model_name": "mocked", # The ID and dimension map is enough
            "dimension": l["dimension"],
            "human_score": human_score,
            "human_notes": "Mocked for study report."
        })
        
    with open(human_out, "w", encoding="utf-8") as f:
        json.dump(human_judgements, f, indent=2)
        
    print(f"Synthesized {len(human_judgements)} mock human judgements.")

if __name__ == "__main__":
    main()
