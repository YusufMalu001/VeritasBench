import json
import random
from pathlib import Path
from typing import Dict, Any

class HumanJudge:
    def __init__(self, raw_scores_file: str = "results/raw_scores.json", output_file: str = "results/human_judgements.json"):
        self.raw_scores_file = Path(raw_scores_file)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
    def run_cli(self, sample_size: int = 5):
        if not self.raw_scores_file.exists():
            print(f"Error: {self.raw_scores_file} not found. Run pipeline first.")
            return
            
        with open(self.raw_scores_file, "r", encoding="utf-8") as f:
            raw_scores = json.load(f)
            
        # For A/B testing we need pairwise data. Since we might only have 1 model initially,
        # we will simulate pairwise by pairing model against itself or requiring 2+ models.
        # Here we just implement the blind A/B framework.
        
        prompts_dict = {}
        for entry in raw_scores:
            pid = entry["prompt_id"]
            if pid not in prompts_dict:
                prompts_dict[pid] = {"prompt": entry["prompt"], "responses": {}}
            prompts_dict[pid]["responses"][entry["model_name"]] = entry["response"]
            
        pairs = []
        for pid, data in prompts_dict.items():
            models = list(data["responses"].keys())
            if len(models) >= 2:
                pairs.append({
                    "prompt_id": pid,
                    "prompt": data["prompt"],
                    "model_a": models[0],
                    "model_b": models[1],
                    "response_a": data["responses"][models[0]],
                    "response_b": data["responses"][models[1]],
                })
                
        if not pairs:
            print("Not enough models evaluated to do pairwise human judgement. Run pipeline with 2+ models.")
            return
            
        sample = random.sample(pairs, min(sample_size, len(pairs)))
        judgements = []
        
        print("\n--- Starting Human Evaluation ---")
        for i, pair in enumerate(sample, 1):
            print(f"\n[{i}/{len(sample)}] Prompt:")
            print(pair["prompt"])
            print("\nResponse A:")
            print(pair["response_a"])
            print("\nResponse B:")
            print(pair["response_b"])
            
            while True:
                choice = input("\nWhich is better? (A/B/T for Tie): ").strip().upper()
                if choice in ["A", "B", "T"]:
                    break
                print("Invalid choice. Enter A, B, or T.")
                
            winner = "Tie" if choice == "T" else choice
            
            judgements.append({
                "prompt_id": pair["prompt_id"],
                "model_a": pair["model_a"],
                "model_b": pair["model_b"],
                "human_winner": winner
            })
            
        # Save atomically
        temp_file = self.output_file.with_suffix('.tmp')
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(judgements, f, indent=2)
        temp_file.replace(self.output_file)
        print(f"\nSaved {len(judgements)} judgements to {self.output_file}")

if __name__ == "__main__":
    judge = HumanJudge()
    judge.run_cli(sample_size=5)
