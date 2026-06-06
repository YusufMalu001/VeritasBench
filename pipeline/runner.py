import os
import json
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from benchmark.prompt_builder import PromptBuilder
from models.model_registry import ModelRegistry
from models.compatibility import CompatibilityChecker, ChatModelCompatibilityError
from dimensions.instruction_following import InstructionFollowing
from dimensions.factuality import Factuality
from dimensions.refusal_calibration import RefusalCalibration
from dimensions.format_adherence import FormatAdherence
from dimensions.verbosity import Verbosity

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(self, config_path: str = "configs/config.yaml", force_rerun: bool = False):
        self.force_rerun = force_rerun
        
        # 1. HF_TOKEN exists
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token or hf_token == "your_huggingface_token_here":
            raise ValueError("HF_TOKEN must be set in the environment.")
            
        # 2. Config file loads
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # Extract models to check
        model_names = [m["name"] for m in self.config.get("models_to_evaluate", [])]
        judge_model = self.config.get("judge_model")
        if judge_model:
            model_names.append(judge_model)
            
        # 3. Compatibility report check/validation
        checker = CompatibilityChecker()
        checker.validate_models(model_names)
        
        # 4. Load Models
        self.results_dir = Path(self.config.get("results_dir", "./results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.raw_scores_file = self.results_dir / "raw_scores.json"
        
        benchmark_files = self.config.get("benchmark_files", ["benchmark/prompts.json"])
        self.prompt_builder = PromptBuilder(prompts_files=benchmark_files)
        self.prompts = self.prompt_builder.get_all_prompts()
        
        self.models = []
        for model_cfg in self.config.get("models_to_evaluate", []):
            try:
                model = ModelRegistry.get_model(model_cfg, self.config)
                model.force_rerun = self.force_rerun
                self.models.append(model)
            except Exception as e:
                logger.error(f"Failed to load model {model_cfg['name']}: {e}")
                
        logger.info(f"Loaded models count: {len(self.models)}")
                
        self.dimensions = [
            InstructionFollowing(),
            Factuality(),
            RefusalCalibration(),
            FormatAdherence(),
            Verbosity()
        ]
        
    def _estimate_cost(self, num_prompts: int):
        num_models = len(self.models)
        logger.info(f"Estimating cost for {num_models} models x {num_prompts} prompts...")
        logger.info(f"Total generations: {num_models * num_prompts}")
        logger.info("Assuming avg 150 input tokens, 200 output tokens. Adjust per model pricing.")
        
    def run(self):
        total_before_filtering = len(self.prompts)
        logger.info(f"Total prompts before filtering: {total_before_filtering}")
        
        # In the future, we might filter by category. For now, all are used.
        total_after_filtering = len(self.prompts)
        logger.info(f"Total prompts after filtering: {total_after_filtering}")
        
        existing_results = []
        if self.raw_scores_file.exists():
            with open(self.raw_scores_file, "r") as f:
                existing_results = json.load(f)
                
        logger.info(f"Existing raw_scores entries count: {len(existing_results)}")
                
        processed_keys = set()
        if not self.force_rerun:
            processed_keys = {f"{r['model_name']}_{r['prompt_id']}" for r in existing_results}
            logger.info(f"Loaded completed prompt IDs count in raw_scores: {len(processed_keys)}")
        else:
            logger.info("--force-rerun flag detected. Ignoring raw_scores.json and diskcache.")
            existing_results = []
            
        tasks = []
        for model in self.models:
            # We assume diskcache path is .cache/hf_responses and accessible via model.cache
            if hasattr(model, 'cache') and not self.force_rerun:
                logger.info(f"Loaded diskcache entries count for {model.model_name}: {len(model.cache)}")
            elif hasattr(model, 'cache') and self.force_rerun:
                model.cache.clear() # Clear diskcache if forced rerun to ensure no stale data is used
            
            for prompt in self.prompts:
                key = f"{model.model_name}_{prompt['id']}"
                cache_hit = key in processed_keys
                
                # Debug logging per prompt
                logger.debug(f"Checking cache for prompt_id: {prompt['id']} | model_name: {model.model_name} | cache_key: {key} | cache_hit: {cache_hit}")
                
                if not cache_hit:
                    tasks.append((model, prompt))
                    
        total_after_cache = len(tasks)
        logger.info(f"Total prompts after cache exclusion: {total_after_cache}")
        
        self._estimate_cost(total_after_cache)
                    
        if not tasks:
            logger.info("No new tasks to run. Everything is cached.")
            return existing_results
            
        logger.info(f"Starting evaluation of {len(tasks)} tasks.")
        
        import time
        new_results = []
        for model, prompt in tqdm(tasks, desc="Evaluating"):
            try:
                gen_data = model.generate(prompt["prompt"])
                
                if gen_data.get("generation_failed", False):
                    logger.warning(f"Generation failed for {prompt['id']}: {gen_data.get('failure_reason')}")
                    time.sleep(0.5)
                    continue
                
                result_entry = {
                    "model_name": model.model_name,
                    "prompt_id": prompt["id"],
                    "category": prompt["category"],
                    "prompt": prompt["prompt"],
                    "response": gen_data["response"],
                    "latency_ms": gen_data["latency_ms"],
                    "token_estimate": gen_data["token_estimate"],
                    "cached": gen_data["cached"],
                    "timestamp": gen_data["timestamp"],
                    "generation_failed": gen_data.get("generation_failed", False),
                    "retry_count": gen_data.get("retry_count", 0),
                    "dimension_scores": {},
                    "failure_tags": []
                }
                
                for dim in self.dimensions:
                    try:
                        score, explanation, tags = dim.score(prompt, gen_data["response"])
                        result_entry["dimension_scores"][dim.name] = {
                            "score": score,
                            "explanation": explanation
                        }
                        result_entry["failure_tags"].extend(tags)
                    except Exception as e:
                        from dimensions.base_dimension import EvaluationConfigurationError
                        if isinstance(e, EvaluationConfigurationError):
                            logger.error(f"Configuration error for {dim.name} on {prompt['id']}: {e}")
                            result_entry["evaluation_failed"] = True
                            result_entry["failure_reason"] = str(e)
                            
                            # Log to results/evaluation_config_errors.json
                            err_log = self.results_dir / "evaluation_config_errors.json"
                            errs = []
                            if err_log.exists():
                                with open(err_log, "r") as ef:
                                    errs = json.load(ef)
                            errs.append({
                                "prompt_id": prompt["id"],
                                "dimension": dim.name,
                                "failure_reason": str(e)
                            })
                            with open(err_log, "w") as ef:
                                json.dump(errs, ef, indent=2)
                        else:
                            raise e
                            
                result_entry["failure_tags"] = list(set(result_entry["failure_tags"]))
                new_results.append(result_entry)
                
            except Exception as e:
                logger.error(f"Error evaluating {model.model_name} on {prompt['id']}: {e}")
                
            time.sleep(0.5)
                
        all_results = existing_results + new_results
        
        temp_file = self.raw_scores_file.with_suffix('.tmp')
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        temp_file.replace(self.raw_scores_file)
        
        logger.info(f"Saved {len(all_results)} total results to {self.raw_scores_file}")
        return all_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rerun", action="store_true", help="Ignore cache and evaluate all prompts")
    args = parser.parse_args()
    
    runner = PipelineRunner(force_rerun=args.force_rerun)
    runner.run()
