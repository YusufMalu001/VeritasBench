import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class ComparativeAnalyzer:
    def __init__(self, raw_scores_file: str = "results/raw_scores.json", benchmark_files: List[str] = None):
        if benchmark_files is None:
            benchmark_files = ["benchmark/prompts.json", "benchmark/prompts_v2_hard.json"]
        self.raw_scores_file = Path(raw_scores_file)
        self.benchmark_files = [Path(p) for p in benchmark_files]
        self.results_dir = Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.scores_df = pd.DataFrame()
        self._load_data()

    def __repr__(self) -> str:
        models = self.scores_df["model_name"].unique().tolist() if not self.scores_df.empty else []
        return f"<ComparativeAnalyzer (models loaded: {models})>"

    def _load_data(self):
        if not self.raw_scores_file.exists():
            return
            
        with open(self.raw_scores_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        # Get tiers
        prompt_tiers = {}
        for b_file in self.benchmark_files:
            if b_file.exists():
                with open(b_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        tier = data.get("tier", "easy")
                        prompts = data.get("prompts", [])
                    else:
                        tier = "easy"
                        prompts = data
                    for p in prompts:
                        prompt_tiers[p["id"]] = tier
                        
        records = []
        for r in raw_data:
            cat = r.get("category")
            dim_data = r.get("dimension_scores", {}).get(cat, {})
            score = dim_data.get("score")
            
            if score is not None:
                pid = r.get("prompt_id")
                records.append({
                    "prompt_id": pid,
                    "model_name": r.get("model_name"),
                    "category": cat,
                    "score": float(score),
                    "prompt": r.get("prompt", ""),
                    "tier": prompt_tiers.get(pid, "easy")
                })
                
        self.scores_df = pd.DataFrame(records)

    def head_to_head_breakdown(self, model_a: str, model_b: str):
        """
        Computes win rates, score deltas, and extracts top 10 most contested prompts.
        """
        if self.scores_df.empty:
            return
            
        df_a = self.scores_df[self.scores_df["model_name"] == model_a].set_index("prompt_id")
        df_b = self.scores_df[self.scores_df["model_name"] == model_b].set_index("prompt_id")
        
        common_prompts = df_a.index.intersection(df_b.index)
        if common_prompts.empty:
            logger.warning("No common prompts evaluated by both models.")
            return
            
        df_a = df_a.loc[common_prompts]
        df_b = df_b.loc[common_prompts]
        
        deltas = df_a["score"] - df_b["score"]
        
        # Win rate per dimension
        df_combined = pd.DataFrame({
            "category": df_a["category"],
            "score_a": df_a["score"],
            "score_b": df_b["score"],
            "delta": deltas,
            "prompt": df_a["prompt"]
        })
        
        win_rates = {}
        for cat in df_combined["category"].unique():
            cat_df = df_combined[df_combined["category"] == cat]
            a_wins = len(cat_df[cat_df["delta"] > 0])
            b_wins = len(cat_df[cat_df["delta"] < 0])
            ties = len(cat_df[cat_df["delta"] == 0])
            total = len(cat_df)
            
            win_rates[cat] = {
                model_a: a_wins / total if total > 0 else 0,
                model_b: b_wins / total if total > 0 else 0,
                "ties": ties / total if total > 0 else 0
            }
            
        # Top 10 most disagreed
        df_combined["abs_delta"] = df_combined["delta"].abs()
        top_disagreed = df_combined.sort_values(by="abs_delta", ascending=False).head(10)
        
        disagreement_list = []
        for pid, row in top_disagreed.iterrows():
            winner = model_a if row["delta"] > 0 else model_b if row["delta"] < 0 else "Tie"
            disagreement_list.append({
                "prompt_id": pid,
                "prompt": row["prompt"],
                "category": row["category"],
                f"{model_a}_score": row["score_a"],
                f"{model_b}_score": row["score_b"],
                "delta": row["abs_delta"],
                "winner": winner
            })
            
        out_file = self.results_dir / "disagreement_analysis.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "win_rates": win_rates,
                "top_disagreements": disagreement_list
            }, f, indent=2)
            
        logger.info(f"Head-to-head breakdown saved to {out_file}")

    def strength_weakness_profile(self, model_name: str) -> Dict[str, Any]:
        """
        Identifies top 3 strongest and weakest categories, and performance variance.
        """
        if self.scores_df.empty:
            return {}
            
        model_df = self.scores_df[self.scores_df["model_name"] == model_name]
        if model_df.empty:
            return {}
            
        means = model_df.groupby("category")["score"].mean()
        variances = model_df.groupby("category")["score"].var()
        
        sorted_means = means.sort_values(ascending=False)
        
        return {
            "model": model_name,
            "strongest_dimensions": sorted_means.head(3).to_dict(),
            "weakest_dimensions": sorted_means.tail(3).to_dict(),
            "performance_variance": variances.to_dict()
        }

    def compute_capability_gaps(self):
        """
        Calculates degradation (Easy -> Hard delta) per model per dimension.
        """
        if self.scores_df.empty or "tier" not in self.scores_df.columns:
            return
            
        gaps = {}
        for model in self.scores_df["model_name"].unique():
            model_df = self.scores_df[self.scores_df["model_name"] == model]
            
            easy_means = model_df[model_df["tier"] == "easy"].groupby("category")["score"].mean()
            hard_means = model_df[model_df["tier"] == "hard"].groupby("category")["score"].mean()
            
            # Reindex to ensure alignment
            all_cats = model_df["category"].unique()
            easy_means = easy_means.reindex(all_cats, fill_value=0)
            hard_means = hard_means.reindex(all_cats, fill_value=0)
            
            degradation = easy_means - hard_means
            
            gaps[model] = {
                "average_degradation": float(degradation.mean()),
                "degradation_by_dimension": degradation.to_dict()
            }
            
        out_file = self.results_dir / "capability_gaps.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(gaps, f, indent=2)
            
        logger.info(f"Capability gaps saved to {out_file}")

    def run_full_comparison(self):
        models = self.scores_df["model_name"].unique().tolist()
        if len(models) >= 2:
            # Compare first two models
            self.head_to_head_breakdown(models[0], models[1])
            
        profiles = {}
        for m in models:
            profiles[m] = self.strength_weakness_profile(m)
            
        profile_file = self.results_dir / "model_profiles.json"
        with open(profile_file, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2)
            
        self.compute_capability_gaps()

if __name__ == "__main__":
    analyzer = ComparativeAnalyzer()
    analyzer.run_full_comparison()
