import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import scipy.stats as stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class PipelineAggregator:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.raw_scores_file = self.results_dir / "raw_scores.json"
        self.aggregated_file = self.results_dir / "aggregated_scores.json"
        
    def _compute_stats(self, scores: List[float]) -> Dict[str, float]:
        if not scores:
            return {"mean": 0.0, "std": 0.0, "variance": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
            
        data = np.array(scores)
        mean = float(np.mean(data))
        std = float(np.std(data))
        var = float(np.var(data))
        
        if len(data) > 1 and std > 0:
            ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=stats.sem(data))
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        else:
            ci_lower, ci_upper = mean, mean
            
        return {
            "mean": mean,
            "std": std,
            "variance": var,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }

    def aggregate(self):
        if not self.raw_scores_file.exists():
            logger.error(f"{self.raw_scores_file} not found. Run runner.py first.")
            return
            
        with open(self.raw_scores_file, "r") as f:
            raw_scores = json.load(f)
            
        models_data = {}
        
        for entry in raw_scores:
            model = entry["model_name"]
            cat = entry["category"]
            
            if model not in models_data:
                models_data[model] = {
                    "overall_scores": [],
                    "dimensions": {},
                    "categories": {}
                }
                
            if cat not in models_data[model]["categories"]:
                models_data[model]["categories"][cat] = []
                
            entry_scores = []
            for dim, data in entry["dimension_scores"].items():
                score = data["score"]
                entry_scores.append(score)
                
                if dim not in models_data[model]["dimensions"]:
                    models_data[model]["dimensions"][dim] = []
                models_data[model]["dimensions"][dim].append(score)
                
            if entry_scores:
                avg_entry_score = sum(entry_scores) / len(entry_scores)
                models_data[model]["overall_scores"].append(avg_entry_score)
                models_data[model]["categories"][cat].append(avg_entry_score)
                
        final_aggregation = {}
        for model, data in models_data.items():
            final_aggregation[model] = {
                "overall": self._compute_stats(data["overall_scores"]),
                "by_dimension": {},
                "by_category": {}
            }
            
            for dim, scores in data["dimensions"].items():
                final_aggregation[model]["by_dimension"][dim] = self._compute_stats(scores)
                
            for cat, scores in data["categories"].items():
                final_aggregation[model]["by_category"][cat] = self._compute_stats(scores)
                
        # Atomic save
        temp_file = self.aggregated_file.with_suffix('.tmp')
        with open(temp_file, "w") as f:
            json.dump(final_aggregation, f, indent=2)
        temp_file.replace(self.aggregated_file)
        
        logger.info(f"Aggregation complete. Saved to {self.aggregated_file}")

if __name__ == "__main__":
    aggregator = PipelineAggregator()
    aggregator.aggregate()
