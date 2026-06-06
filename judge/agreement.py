import json
import logging
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgreementAnalyzer:
    def __init__(self, human_file: str = "results/human_judgements.json", llm_file: str = "results/llm_judge_scores.json"):
        self.human_file = Path(human_file)
        self.llm_file = Path(llm_file)
        self.assets_dir = Path("assets")
        self.assets_dir.mkdir(exist_ok=True)
        
    def analyze(self):
        if not self.human_file.exists() or not self.llm_file.exists():
            logger.error("Missing judgements files.")
            return None
            
        with open(self.human_file, "r") as f:
            human_data = json.load(f)
            
        with open(self.llm_file, "r") as f:
            llm_data = json.load(f)
            
        human_df = pd.DataFrame(human_data)
        llm_df = pd.DataFrame(llm_data)
        
        if human_df.empty or llm_df.empty:
            logger.error("One or both judgement files are empty.")
            return None
            
        # Merge on prompt_id
        merged = pd.merge(human_df, llm_df, on="prompt_id", suffixes=("_human", "_llm"))
        
        y_human = merged["human_score"].tolist()
        y_llm = merged["score"].tolist()
        dimensions = merged["dimension_human"].tolist()
        
        # Overall metrics
        raw_agreement = sum(1 for h, l in zip(y_human, y_llm) if h == l) / len(y_human)
        kappa = cohen_kappa_score(y_human, y_llm)
        precision, recall, f1, _ = precision_recall_fscore_support(y_human, y_llm, average="macro", zero_division=0)
        
        overall = {
            "num_comparisons": len(y_human),
            "raw_agreement_percent": raw_agreement * 100,
            "cohens_kappa": kappa,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        # Per-dimension metrics
        per_dimension = {}
        for dim in set(dimensions):
            dim_mask = merged["dimension_human"] == dim
            dim_human = merged[dim_mask]["human_score"].tolist()
            dim_llm = merged[dim_mask]["score"].tolist()
            
            if not dim_human:
                continue
                
            dim_raw = sum(1 for h, l in zip(dim_human, dim_llm) if h == l) / len(dim_human)
            # Kappa can be undefined if all human scores or all llm scores are identical
            try:
                dim_kappa = cohen_kappa_score(dim_human, dim_llm)
            except:
                dim_kappa = 0.0
                
            per_dimension[dim] = {
                "num_comparisons": len(dim_human),
                "raw_agreement_percent": dim_raw * 100,
                "cohens_kappa": dim_kappa if not pd.isna(dim_kappa) else 0.0
            }
            
        result = {
            "overall": overall,
            "per_dimension": per_dimension
        }
        
        out_file = Path("results/agreement_report.json")
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
            
        logger.info(f"Agreement analysis saved to {out_file}")
        
        self.generate_visualizations(y_human, y_llm, per_dimension)
        return result

    def generate_visualizations(self, y_human, y_llm, per_dimension):
        # 1. Confusion Matrix
        cm = confusion_matrix(y_human, y_llm, labels=[0, 1])
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fail (0)", "Pass (1)"], yticklabels=["Fail (0)", "Pass (1)"])
        plt.xlabel("LLM Judge Score")
        plt.ylabel("Human Score")
        plt.title("Human vs LLM Confusion Matrix")
        plt.tight_layout()
        plt.savefig(self.assets_dir / "confusion_matrix.png")
        plt.close()
        
        # 2. Agreement Heatmap (Kappa)
        kappa_data = [{"Dimension": k, "Kappa": v["cohens_kappa"]} for k, v in per_dimension.items()]
        kappa_df = pd.DataFrame(kappa_data)
        plt.figure(figsize=(8,4))
        sns.heatmap(kappa_df.set_index("Dimension").T, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
        plt.title("Cohen's Kappa by Dimension")
        plt.tight_layout()
        plt.savefig(self.assets_dir / "agreement_heatmap.png")
        plt.close()
        
        # 3. Bar Chart (Raw Agreement)
        raw_data = [{"Dimension": k, "Raw Agreement %": v["raw_agreement_percent"]} for k, v in per_dimension.items()]
        raw_df = pd.DataFrame(raw_data)
        plt.figure(figsize=(8,5))
        sns.barplot(data=raw_df, x="Raw Agreement %", y="Dimension", palette="viridis")
        plt.title("Raw Agreement % by Dimension")
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.savefig(self.assets_dir / "agreement_breakdown.png")
        plt.close()

if __name__ == "__main__":
    analyzer = AgreementAnalyzer()
    analyzer.analyze()
