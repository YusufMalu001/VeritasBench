import os
import json
import time
import yaml
import logging
from pathlib import Path
from datetime import datetime

# Import internal pipeline modules
from pipeline.runner import PipelineRunner
from pipeline.aggregator import PipelineAggregator
from analysis.bias_detector import BiasDetector
from analysis.difficulty_calibrator import DifficultyCalibrator
from analysis.comparative_analysis import ComparativeAnalyzer
from judge.agreement import AgreementAnalyzer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def update_readme_table(results_file: Path):
    if not results_file.exists():
        return
        
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    readme_path = Path("README.md")
    if not readme_path.exists():
        return
        
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    table = "\n## Benchmark Results\n\n"
    table += "| Model | Overall Score | Easy Tier | Hard Tier |\n"
    table += "|-------|---------------|-----------|-----------|\n"
    
    for model_name, res in data.get("results", {}).items():
        overall = res.get("overall", 0.0)
        easy = res.get("easy_tier", 0.0)
        hard = res.get("hard_tier", 0.0)
        table += f"| `{model_name}` | {overall:.2%} | {easy:.2%} | {hard:.2%} |\n"
        
    table += "\n"
    
    # Simple replacement logic: If "## Benchmark Results" exists, replace up to the next "## "
    if "## Benchmark Results" in content:
        start = content.find("## Benchmark Results")
        end = content.find("## ", start + 1)
        if end == -1:
            content = content[:start] + table
        else:
            content = content[:start] + table + content[end:]
    else:
        content += table
        
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    logger.info("Updated README.md with actual benchmark results.")

def main():
    start_time = time.time()
    
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        logger.error("Config not found.")
        return
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    from openai import OpenAI
    
    candidate_models = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen3-32B",
        "google/gemma-3-27b-it",
        "Qwen/Qwen2-0.5B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct"
    ]
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not found in environment. Validation may fail.")
        
    client = OpenAI(
        base_url="https://router.huggingface.co/v1/",
        api_key=hf_token
    )
    
    compatibility_report = {}
    compatible_models = []
    rejected_models = []
    
    logger.info("Validating candidate models via Hugging Face Router...")
    
    for model in candidate_models:
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            compatibility_report[model] = True
            compatible_models.append(model)
        except Exception as e:
            compatibility_report[model] = False
            rejected_models.append(model)
            
    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/model_compatibility_report.json", "w", encoding="utf-8") as f:
        json.dump(compatibility_report, f, indent=2)
        
    logger.info(f"Compatible Models: {compatible_models}")
    logger.info(f"Rejected Models: {rejected_models}")
    
    if len(compatible_models) < 2:
        logger.warning(f"Only {len(compatible_models)} compatible model(s) available. Proceeding with available models.")
        
    if not compatible_models:
        logger.error("No compatible models found. Aborting.")
        return
        
    config["models_to_evaluate"] = [
        {"name": m, "provider": "huggingface"} for m in compatible_models
    ]
    
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    logger.info("Injected Compatible Models into config.yaml")
    
    # Cost Estimation
    total_prompts = 100
    models_count = len(config["models_to_evaluate"])
    total_calls = total_prompts * models_count * 2 # Gen + Judge
    cost_per_1k = 0.0005 # Rough estimate
    avg_tokens_per_call = 400
    estimated_cost = (total_calls * avg_tokens_per_call / 1000) * cost_per_1k
    
    logger.info(f"=== VERITASBENCH EXECUTION PLAN ===")
    logger.info(f"Target Models: {', '.join(compatible_models)}")
    logger.info(f"Total API Calls expected: {total_calls}")
    logger.info(f"Estimated Cost: ${estimated_cost:.4f}")
    logger.info(f"Estimated Runtime: ~{total_calls * 2 / 60:.1f} minutes")
    logger.info("===================================")
    
    # 1. Pipeline Execution
    logger.info("Starting PipelineRunner...")
    runner = PipelineRunner(force_rerun=True)
    runner.run()
    
    # 2. Aggregation
    logger.info("Starting PipelineAggregator...")
    aggregator = PipelineAggregator()
    aggregator.aggregate()
    
    # 3. Analytics Suites
    logger.info("Executing Analysis Modules...")
    calibrator = DifficultyCalibrator()
    calibrator.run_calibration()
    
    bias = BiasDetector()
    bias.generate_bias_report()
    
    comp = ComparativeAnalyzer()
    comp.run_full_comparison()
    
    agreement = AgreementAnalyzer()
    agr_result = agreement.analyze() or {}
    
    # 4. Compile Final Report Schema
    logger.info("Compiling benchmark_results.json...")
    
    with open("results/aggregated_scores.json", "r", encoding="utf-8") as f:
        agg_scores = json.load(f)
        
    with open("results/raw_scores.json", "r", encoding="utf-8") as f:
        raw_scores = json.load(f)
        
    # Get tiers
    tier_map = {}
    for r in raw_scores:
        pid = r.get("prompt_id")
        if pid not in tier_map:
            # Simple heuristic, usually id contains 'easy' or 'hard'
            tier_map[pid] = "hard" if "hard" in pid else "easy"
            
    final_results = {
        "metadata": {
            "run_date": datetime.utcnow().isoformat() + "Z",
            "models_evaluated": compatible_models,
            "total_prompts": total_prompts,
            "dimensions": ["instruction_following", "factuality", "format_adherence", "refusal_calibration", "verbosity"],
            "judge_model": config.get("judge_model", "Qwen/Qwen3-32B")
        },
        "results": {},
        "agreement": {
            "cohens_kappa": agr_result.get("overall", {}).get("cohens_kappa", 0.0),
            "raw_agreement_pct": agr_result.get("overall", {}).get("raw_agreement_percent", 0.0),
            "by_dimension": {k: v["cohens_kappa"] for k, v in agr_result.get("per_dimension", {}).items()}
        }
    }
    
    for m in compatible_models:
        if m not in agg_scores:
            continue
            
        m_agg = agg_scores[m]
        
        # Calculate tier scores
        m_raw = [r for r in raw_scores if r["model_name"] == m]
        easy_scores = []
        hard_scores = []
        
        failure_count = 0
        failure_tags = {}
        
        for r in m_raw:
            cat = r["category"]
            score = r.get("dimension_scores", {}).get(cat, {}).get("score", 0.0)
            
            if r.get("generation_failed"):
                failure_count += 1
                
            tags = r.get("failure_tags", [])
            for t in tags:
                failure_tags[t] = failure_tags.get(t, 0) + 1
                
            tier = tier_map.get(r["prompt_id"], "easy")
            if tier == "easy":
                easy_scores.append(score)
            else:
                hard_scores.append(score)
                
        easy_avg = sum(easy_scores) / len(easy_scores) if easy_scores else 0.0
        hard_avg = sum(hard_scores) / len(hard_scores) if hard_scores else 0.0
        
        final_results["results"][m] = {
            "overall": m_agg.get("overall", {}).get("mean", 0.0),
            "easy_tier": easy_avg,
            "hard_tier": hard_avg,
            "dimensions": {k: v.get("mean", 0.0) for k, v in m_agg.get("by_category", {}).items()},
            "failure_count": failure_count,
            "failure_tags": failure_tags
        }
        
    out_file = Path("results/benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
        
    logger.info(f"Final benchmark scores saved to {out_file}")
    
    update_readme_table(out_file)
    
    runtime = time.time() - start_time
    logger.info(f"Total Execution Time: {runtime / 60:.2f} minutes")

if __name__ == "__main__":
    main()
