import json
from pathlib import Path

def main():
    sample_file = Path("results/human_eval_sample.json")
    raw_scores_file = Path("results/raw_scores.json")
    output_file = Path("results/llm_judge_scores.json")
    
    if not sample_file.exists() or not raw_scores_file.exists():
        print("Required files not found.")
        return
        
    with open(sample_file, "r", encoding="utf-8") as f:
        sample = json.load(f)
        
    with open(raw_scores_file, "r", encoding="utf-8") as f:
        raw_scores = json.load(f)
        
    # Create lookup map
    raw_map = {r["prompt_id"]: r for r in raw_scores}
    
    llm_scores = []
    
    for s in sample:
        pid = s["prompt_id"]
        cat = s["category"]
        
        raw_entry = raw_map.get(pid)
        if raw_entry and cat in raw_entry.get("dimension_scores", {}):
            dim_data = raw_entry["dimension_scores"][cat]
            score_float = dim_data["score"]
            reasoning = dim_data["explanation"]
            
            # Threshold to binary to match human (0 or 1)
            binary_score = 1 if score_float >= 0.5 else 0
            
            llm_scores.append({
                "prompt_id": pid,
                "dimension": cat,
                "score": binary_score,
                "reasoning": reasoning,
                "raw_float_score": score_float
            })
            
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(llm_scores, f, indent=2)
        
    print(f"Exported {len(llm_scores)} LLM judgements to {output_file}")

if __name__ == "__main__":
    main()
