import json
from pathlib import Path

class HumanJudge:
    def __init__(self, sample_file: str = "results/human_eval_sample.json", output_file: str = "results/human_judgements.json"):
        self.sample_file = Path(sample_file)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
    def run_cli(self):
        if not self.sample_file.exists():
            print(f"Error: {self.sample_file} not found.")
            return
            
        with open(self.sample_file, "r", encoding="utf-8") as f:
            sample = json.load(f)
            
        judgements = []
        
        # Resume if output file already exists
        if self.output_file.exists():
            with open(self.output_file, "r", encoding="utf-8") as f:
                try:
                    judgements = json.load(f)
                except json.JSONDecodeError:
                    judgements = []
                    
        processed_ids = {j["prompt_id"] for j in judgements}
        
        print(f"\n--- Starting Human Evaluation ({len(processed_ids)}/{len(sample)} complete) ---")
        
        for i, record in enumerate(sample, 1):
            if record["prompt_id"] in processed_ids:
                continue
                
            print(f"\n[{i}/{len(sample)}] Dimension: {record['category'].upper()}")
            print("\nPrompt:")
            print(record["prompt"])
            print("\nModel Response:")
            print(record["response"])
            
            while True:
                choice = input("\nEnter score (0 for fail, 1 for pass, or 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    print("\nSaving progress and exiting...")
                    return
                if choice in ["0", "1"]:
                    break
                print("Invalid choice. Enter 0 or 1.")
                
            notes = input("Optional notes (press Enter to skip): ").strip()
            
            judgements.append({
                "prompt_id": record["prompt_id"],
                "model_name": record["model_name"],
                "dimension": record["category"],
                "human_score": int(choice),
                "human_notes": notes
            })
            
            # Save automatically after each judgement
            temp_file = self.output_file.with_suffix('.tmp')
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(judgements, f, indent=2)
            temp_file.replace(self.output_file)
            
        print(f"\nEvaluation Complete! Saved {len(judgements)} judgements to {self.output_file}")

if __name__ == "__main__":
    judge = HumanJudge()
    judge.run_cli()
