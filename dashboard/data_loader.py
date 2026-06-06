import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

def load_json(filepath: str) -> Any:
    path = Path(filepath)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_dashboard_data(results_dir: str = "results", benchmark_dir: str = "benchmark") -> Tuple[pd.DataFrame, Dict, Dict, Dict, pd.DataFrame, pd.DataFrame]:
    # Load Prompts (to get Tiers)
    prompts_data = []
    for prompt_file in Path(benchmark_dir).glob("*.json"):
        data = load_json(prompt_file)
        if data:
            if isinstance(data, dict):
                tier = data.get("tier", "easy")
                version = data.get("benchmark_version", "v1.0")
                for p in data.get("prompts", []):
                    p["tier"] = tier
                    p["benchmark_version"] = version
                    prompts_data.append(p)
            else:
                for p in data:
                    p["tier"] = "easy"
                    p["benchmark_version"] = "v1.0"
                    prompts_data.append(p)
                    
    prompts_df = pd.DataFrame(prompts_data)
    
    # Load Results
    raw_scores = load_json(f"{results_dir}/raw_scores.json") or []
    agg_scores = load_json(f"{results_dir}/aggregated_scores.json") or {}
    difficulty = load_json(f"{results_dir}/benchmark_difficulty_report.json") or {}
    eval_errors = load_json(f"{results_dir}/evaluation_config_errors.json") or []
    
    # Process Raw Scores into DataFrame
    processed_records = []
    for entry in raw_scores:
        record = {
            "prompt_id": entry.get("prompt_id"),
            "model_name": entry.get("model_name"),
            "category": entry.get("category"),
            "latency_ms": entry.get("latency_ms", 0),
            "token_estimate": entry.get("token_estimate", 0),
            "generation_failed": entry.get("generation_failed", False),
            "failure_tags": ", ".join(entry.get("failure_tags", [])),
            "evaluation_failed": entry.get("evaluation_failed", False),
            "failure_reason": entry.get("failure_reason", ""),
            "response": entry.get("response", ""),
            "prompt": entry.get("prompt", ""),
            "score": None,
            "explanation": ""
        }
        
        # Get the score for the dimension that matches the category
        dim_scores = entry.get("dimension_scores", {})
        cat = record["category"]
        if cat in dim_scores:
            record["score"] = dim_scores[cat].get("score")
            record["explanation"] = dim_scores[cat].get("explanation")
            
        processed_records.append(record)
        
    results_df = pd.DataFrame(processed_records)
    
    # Merge tier into results
    if not results_df.empty and not prompts_df.empty:
        results_df = results_df.merge(prompts_df[["id", "tier", "benchmark_version"]], left_on="prompt_id", right_on="id", how="left")
        
    errors_df = pd.DataFrame(eval_errors)
        
    return results_df, agg_scores, difficulty, eval_errors, prompts_df, errors_df
