import json
import random
from pathlib import Path
from collections import defaultdict

def main():
    raw_scores_path = Path("results/raw_scores.json")
    output_path = Path("results/human_eval_sample.json")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    if not raw_scores_path.exists():
        print(f"Error: {raw_scores_path} not found.")
        return
        
    with open(raw_scores_path, "r", encoding="utf-8") as f:
        raw_scores = json.load(f)
        
    # Filter valid records with a response and a score in its category
    valid_records = []
    for r in raw_scores:
        if not r.get("generation_failed") and r.get("response"):
            cat = r.get("category")
            if cat and cat in r.get("dimension_scores", {}):
                valid_records.append(r)
            
    # Group by category
    by_category = defaultdict(list)
    for r in valid_records:
        by_category[r["category"]].append(r)
        
    # We want exactly 50 samples balanced across 5 categories, so 10 per category.
    # If a category has fewer than 10, we take all of them.
    sample = []
    categories = ['instruction_following', 'factuality', 'format_adherence', 'refusal_calibration', 'verbosity']
    target_per_cat = 50 // len(categories)
    
    random.seed(42) # For reproducibility
    
    for cat in categories:
        records = by_category.get(cat, [])
        if not records:
            continue
        n_sample = min(target_per_cat, len(records))
        sampled_records = random.sample(records, n_sample)
        sample.extend(sampled_records)
        
    # If we still need more to reach 50, grab randomly from the remaining
    remaining = [r for r in valid_records if r not in sample]
    needed = 50 - len(sample)
    if needed > 0 and remaining:
        extra = random.sample(remaining, min(needed, len(remaining)))
        sample.extend(extra)
        
    random.shuffle(sample) # Shuffle final dataset
    
    # Extract only necessary fields for human evaluation
    human_sample = []
    for r in sample:
        human_sample.append({
            "prompt_id": r["prompt_id"],
            "model_name": r["model_name"],
            "category": r["category"],
            "prompt": r["prompt"],
            "response": r["response"]
        })
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(human_sample, f, indent=2)
        
    print(f"Sampled {len(human_sample)} records for human evaluation to {output_path}")

if __name__ == "__main__":
    main()
