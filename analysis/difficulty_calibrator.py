import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class DifficultyCalibrator:
    def __init__(self, raw_scores_file: str = "results/raw_scores.json", benchmark_files: List[str] = None):
        if benchmark_files is None:
            benchmark_files = ["benchmark/prompts.json", "benchmark/prompts_v2_hard.json"]
        self.raw_scores_file = Path(raw_scores_file)
        self.benchmark_files = [Path(p) for p in benchmark_files]
        self.results_dir = Path("results")
        self.benchmark_dir = Path("benchmark")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        self.item_difficulties = {}
        self.dimension_difficulties = {}
        self.mislabeled = []

    def __repr__(self) -> str:
        return f"<DifficultyCalibrator (prompts analyzed: {len(self.item_difficulties)})>"

    def _load_data(self) -> pd.DataFrame:
        if not self.raw_scores_file.exists():
            return pd.DataFrame()
            
        with open(self.raw_scores_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        records = []
        for r in raw_data:
            cat = r.get("category")
            dim_data = r.get("dimension_scores", {}).get(cat, {})
            score = dim_data.get("score")
            
            if score is not None:
                records.append({
                    "prompt_id": r.get("prompt_id"),
                    "model_name": r.get("model_name"),
                    "category": cat,
                    "score": float(score)
                })
                
        return pd.DataFrame(records)

    def _load_benchmark_metadata(self) -> Dict[str, str]:
        prompt_tiers = {}
        for b_file in self.benchmark_files:
            if b_file.exists():
                with open(b_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Support v2 schema with dict containing "prompts" or flat list
                    if isinstance(data, dict):
                        tier = data.get("tier", "easy")
                        prompts = data.get("prompts", [])
                    else:
                        tier = "easy"
                        prompts = data
                        
                    for p in prompts:
                        prompt_tiers[p["id"]] = tier
        return prompt_tiers

    def compute_item_difficulty(self, scores_df: pd.DataFrame):
        """
        Computes IRT-inspired difficulty for each prompt: 1.0 - mean_score_across_models.
        """
        if scores_df.empty:
            return
            
        mean_scores = scores_df.groupby("prompt_id")["score"].mean()
        self.item_difficulties = (1.0 - mean_scores).to_dict()

    def compute_dimension_difficulty(self, scores_df: pd.DataFrame):
        """
        Ranks all dimensions by average difficulty. Identifies hardest per model.
        """
        if scores_df.empty:
            return
            
        # Overall dimension difficulty
        dim_means = scores_df.groupby("category")["score"].mean()
        self.dimension_difficulties["overall_ranking"] = (1.0 - dim_means).sort_values(ascending=False).to_dict()
        
        # Hardest per model
        hardest_per_model = {}
        for model in scores_df["model_name"].unique():
            model_df = scores_df[scores_df["model_name"] == model]
            model_dim_means = model_df.groupby("category")["score"].mean()
            if not model_dim_means.empty:
                hardest_dim = model_dim_means.idxmin() # Lowest score = hardest
                hardest_per_model[model] = hardest_dim
                
        self.dimension_difficulties["hardest_per_model"] = hardest_per_model

    def flag_mislabeled_tiers(self):
        """
        Finds Easy prompts with difficulty > 0.7 and Hard prompts with difficulty < 0.3.
        """
        tiers = self._load_benchmark_metadata()
        
        for pid, diff in self.item_difficulties.items():
            current_tier = tiers.get(pid, "easy").lower()
            
            if current_tier == "easy" and diff > 0.7:
                self.mislabeled.append({
                    "prompt_id": pid,
                    "current_tier": "easy",
                    "empirical_difficulty": diff,
                    "recommendation": "hard"
                })
            elif current_tier == "hard" and diff < 0.3:
                self.mislabeled.append({
                    "prompt_id": pid,
                    "current_tier": "hard",
                    "empirical_difficulty": diff,
                    "recommendation": "easy"
                })
                
        logger.info(f"Flagged {len(self.mislabeled)} mislabeled prompts.")

    def recalibrate_benchmark(self):
        """
        Reassigns tier labels based on empirical difficulty and saves to prompts_v3_calibrated.json.
        """
        recalibrations = {m["prompt_id"]: m["recommendation"] for m in self.mislabeled}
        
        calibrated_prompts = []
        
        for b_file in self.benchmark_files:
            if not b_file.exists():
                continue
                
            with open(b_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            prompts = data.get("prompts", []) if isinstance(data, dict) else data
            
            for p in prompts:
                pid = p["id"]
                if pid in recalibrations:
                    p["tier"] = recalibrations[pid]
                elif "tier" not in p:
                    # Maintain existing default if not mislabeled
                    p["tier"] = data.get("tier", "easy") if isinstance(data, dict) else "easy"
                calibrated_prompts.append(p)
                
        if not calibrated_prompts:
            logger.warning("No prompts found to recalibrate.")
            return
            
        output_file = self.benchmark_dir / "prompts_v3_calibrated.json"
        
        v3_schema = {
            "benchmark_version": "v3.0",
            "description": "Empirically recalibrated benchmark dataset.",
            "prompts": calibrated_prompts
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(v3_schema, f, indent=2)
            
        logger.info(f"Recalibrated benchmark saved to {output_file}. {len(recalibrations)} prompts changed tiers.")

    def run_calibration(self):
        df = self._load_data()
        if df.empty:
            logger.error("No score data available for calibration.")
            return
            
        self.compute_item_difficulty(df)
        self.compute_dimension_difficulty(df)
        self.flag_mislabeled_tiers()
        
        # Save analysis
        analysis_out = self.results_dir / "difficulty_analysis.json"
        with open(analysis_out, "w", encoding="utf-8") as f:
            json.dump({
                "item_difficulties": self.item_difficulties,
                "dimension_difficulties": self.dimension_difficulties,
                "mislabeled_tiers": self.mislabeled
            }, f, indent=2)
            
        self.recalibrate_benchmark()

if __name__ == "__main__":
    calibrator = DifficultyCalibrator()
    calibrator.run_calibration()
