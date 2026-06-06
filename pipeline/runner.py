import os
import json
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

from benchmark.prompt_builder import PromptBuilder
from models.model_registry import ModelRegistry
from dimensions.instruction_following import InstructionFollowing
from dimensions.factuality import Factuality
from dimensions.refusal_calibration import RefusalCalibration
from dimensions.format_adherence import FormatAdherence
from dimensions.verbosity import Verbosity

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.results_dir = Path(self.config.get("results_dir", "./results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.raw_scores_file = self.results_dir / "raw_scores.json"
        
        self.prompt_builder = PromptBuilder()
        self.prompts = self.prompt_builder.get_all_prompts()
        
        self.models = []
        for model_cfg in self.config.get("models_to_evaluate", []):
            try:
                self.models.append(ModelRegistry.get_model(model_cfg, self.config))
            except Exception as e:
                logger.error(f"Failed to load model {model_cfg['name']}: {e}")
                
        self.dimensions = [
            InstructionFollowing(),
            Factuality(),
            RefusalCalibration(),
            FormatAdherence(),
            Verbosity()
        ]
        
    def _estimate_cost(self):
        # A very rough heuristic for cost
        num_prompts = len(self.prompts)
        num_models = len(self.models)
        logger.info(f"Estimating cost for {num_models} models x {num_prompts} prompts...")
        logger.info(f"Total generations: {num_models * num_prompts}")
        logger.info("Assuming avg 150 input tokens, 200 output tokens. Adjust per model pricing.")
        
    def run(self):
        self._estimate_cost()
        
        # Load existing
        existing_results = []
        if self.raw_scores_file.exists():
            with open(self.raw_scores_file, "r") as f:
                existing_results = json.load(f)
                
        processed_keys = {f"{r['model_name']}_{r['prompt_id']}" for r in existing_results}
        
        tasks = []
        for model in self.models:
            for prompt in self.prompts:
                key = f"{model.model_name}_{prompt['id']}"
                if key not in processed_keys:
                    tasks.append((model, prompt))
                    
        if not tasks:
            logger.info("No new tasks to run. Everything is cached.")
            return existing_results
            
        logger.info(f"Starting evaluation of {len(tasks)} tasks.")
        
        new_results = []
        
        # We process sequentially for API models to avoid harsh rate limits,
        # but one could use ThreadPoolExecutor here if comfortable.
        for model, prompt in tqdm(tasks, desc="Evaluating"):
            try:
                gen_data = model.generate(prompt["prompt"])
                
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
                    "dimension_scores": {},
                    "failure_tags": []
                }
                
                for dim in self.dimensions:
                    score, explanation, tags = dim.score(prompt, gen_data["response"])
                    result_entry["dimension_scores"][dim.name] = {
                        "score": score,
                        "explanation": explanation
                    }
                    result_entry["failure_tags"].extend(tags)
                    
                result_entry["failure_tags"] = list(set(result_entry["failure_tags"]))
                
                new_results.append(result_entry)
                
            except Exception as e:
                logger.error(f"Error evaluating {model.model_name} on {prompt['id']}: {e}")
                
        all_results = existing_results + new_results
        
        # Atomic write
        temp_file = self.raw_scores_file.with_suffix('.tmp')
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        temp_file.replace(self.raw_scores_file)
        
        logger.info(f"Saved {len(all_results)} total results to {self.raw_scores_file}")
        return all_results

if __name__ == "__main__":
    runner = PipelineRunner()
    runner.run()
