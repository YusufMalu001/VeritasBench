import json
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
import logging

logger = logging.getLogger(__name__)

class AgreementAnalyzer:
    def __init__(self, human_file: str = "results/human_judgements.json", llm_file: str = "results/llm_judgements.json"):
        self.human_file = Path(human_file)
        self.llm_file = Path(llm_file)
        
    def analyze(self):
        if not self.human_file.exists() or not self.llm_file.exists():
            logger.error("Missing judgements files.")
            return None
            
        with open(self.human_file, "r") as f:
            human_data = json.load(f)
            
        with open(self.llm_file, "r") as f:
            llm_data = json.load(f)
            
        human_map = {f"{d['prompt_id']}_{d['model_a']}_{d['model_b']}": d["human_winner"] for d in human_data}
        llm_map = {f"{d['prompt_id']}_{d['model_a']}_{d['model_b']}": d["llm_winner"] for d in llm_data}
        
        common_keys = set(human_map.keys()).intersection(set(llm_map.keys()))
        
        if not common_keys:
            logger.warning("No common judgements found between human and LLM.")
            return None
            
        y_human = [human_map[k] for k in common_keys]
        y_llm = [llm_map[k] for k in common_keys]
        
        raw_agreement = sum(1 for h, l in zip(y_human, y_llm) if h == l) / len(common_keys)
        
        labels = ["A", "B", "Tie"]
        kappa = cohen_kappa_score(y_human, y_llm, labels=labels)
        
        result = {
            "num_comparisons": len(common_keys),
            "raw_agreement_percent": raw_agreement * 100,
            "cohens_kappa": kappa
        }
        
        out_file = Path("results/agreement_report.json")
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
            
        logger.info(f"Agreement analysis saved to {out_file}")
        return result
