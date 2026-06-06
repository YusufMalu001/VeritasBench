import json
import os
from collections import Counter

def main():
    try:
        with open("results/aggregated_scores.json", "r") as f:
            agg_data = json.load(f)
        with open("results/raw_scores.json", "r") as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model_agg = agg_data.get(model_name, {})

    if not model_agg:
        print(f"Model {model_name} not found in aggregated scores.")
        return

    by_dim = model_agg.get("by_dimension", {})
    overall_score = model_agg.get("overall", {}).get("mean", 0.0)

    best_dim = None
    worst_dim = None
    best_score = -1.0
    worst_score = 2.0

    dimension_breakdown = {}
    for dim, stats in by_dim.items():
        mean = stats.get("mean", 0.0)
        dimension_breakdown[dim] = mean
        if mean > best_score:
            best_score = mean
            best_dim = dim
        if mean < worst_score:
            worst_score = mean
            worst_dim = dim

    # Collect failure tags
    all_tags = []
    for r in raw_data:
        if r.get("model_name") == model_name:
            all_tags.extend(r.get("failure_tags", []))
            
    top_tags = [tag for tag, count in Counter(all_tags).most_common(5)]

    eval_summary = {
        "overall_score": overall_score,
        "best_dimension": best_dim,
        "worst_dimension": worst_dim,
        "top_failure_tags": top_tags,
        "dimension_breakdown": dimension_breakdown
    }

    os.makedirs("results", exist_ok=True)
    with open("results/evaluation_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    print(json.dumps(eval_summary, indent=2))

if __name__ == "__main__":
    main()
