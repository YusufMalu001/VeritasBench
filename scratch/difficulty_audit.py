import json
import os
from typing import Dict, Any, List

def main():
    try:
        with open("benchmark/prompts.json", "r") as f:
            prompts_data = json.load(f)
            if isinstance(prompts_data, dict) and "prompts" in prompts_data:
                prompts_list = prompts_data["prompts"]
            else:
                prompts_list = prompts_data
                
        with open("results/raw_scores.json", "r") as f:
            raw_scores = json.load(f)
            
        with open("results/aggregated_scores.json", "r") as f:
            agg_scores = json.load(f)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    report = {
        "dimensions": {},
        "prompt_classification": {
            "Easy": [],
            "Medium": [],
            "Hard": []
        },
        "saturated_dimensions": []
    }

    # Extract score per prompt per dimension
    prompt_scores = {}
    for entry in raw_scores:
        if entry.get("evaluation_failed", False):
            continue
            
        pid = entry["prompt_id"]
        cat = entry["category"]
        
        # We only care about the dimension score that matches the category 
        # (or all dimension scores that were valid). In our v2 pipeline, 
        # a prompt generally only has valid metadata for its own category dimension.
        for dim, dim_data in entry.get("dimension_scores", {}).items():
            if dim == cat:
                if pid not in prompt_scores:
                    prompt_scores[pid] = []
                prompt_scores[pid].append(dim_data["score"])

    # Calculate average score per prompt
    avg_prompt_score = {}
    for pid, scores in prompt_scores.items():
        if scores:
            avg_prompt_score[pid] = sum(scores) / len(scores)

    # Classify Prompts
    for pid, score in avg_prompt_score.items():
        if score >= 0.8:
            report["prompt_classification"]["Easy"].append({"prompt_id": pid, "score": score})
        elif score <= 0.2:
            report["prompt_classification"]["Hard"].append({"prompt_id": pid, "score": score})
        else:
            report["prompt_classification"]["Medium"].append({"prompt_id": pid, "score": score})

    # Group by dimension
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    if model_name in agg_scores:
        by_dim = agg_scores[model_name].get("by_dimension", {})
        
        for dim, stats in by_dim.items():
            mean = stats.get("mean", 0.0)
            std = stats.get("std", 0.0)
            
            if mean == 1.0 and std == 0.0:
                report["saturated_dimensions"].append(dim)
                
            # Find prompts for this dimension
            dim_prompts = [p for p in prompts_list if p.get("category") == dim]
            
            dim_pids = [p["id"] for p in dim_prompts]
            
            # Pass rate is mean since scores are bounded 0.0-1.0
            pass_rate = mean
            
            perfect_prompts = []
            zero_prompts = []
            
            for pid in dim_pids:
                if pid in avg_prompt_score:
                    if avg_prompt_score[pid] == 1.0:
                        perfect_prompts.append(pid)
                    elif avg_prompt_score[pid] == 0.0:
                        zero_prompts.append(pid)
                        
            report["dimensions"][dim] = {
                "pass_rate": pass_rate,
                "mean": mean,
                "std": std,
                "perfect_success_prompts": perfect_prompts,
                "zero_success_prompts": zero_prompts
            }

    # Deduplicate saturated dimensions
    report["saturated_dimensions"] = list(set(report["saturated_dimensions"]))

    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_difficulty_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Difficulty report generated.")

if __name__ == "__main__":
    main()
