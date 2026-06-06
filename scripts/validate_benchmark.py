import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main() -> None:
    prompts_file = Path("benchmark/prompts.json")
    if not prompts_file.exists():
        logging.error(f"Prompts file not found at {prompts_file}")
        return

    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # 1. Schema validation
    required_keys = {"id", "category", "prompt", "expected_behavior", "scoring_criteria"}
    for i, p in enumerate(prompts):
        missing = required_keys - set(p.keys())
        if missing:
            logging.error(f"Prompt index {i} missing keys: {missing}")
            return
            
    # 2. Duplicate prompts
    ids = [p["id"] for p in prompts]
    if len(ids) != len(set(ids)):
        logging.error("Duplicate prompt IDs found!")
        return
        
    # 3. Category balance
    categories = {}
    for p in prompts:
        cat = p["category"]
        categories[cat] = categories.get(cat, 0) + 1
        
    logging.info("Category breakdown:")
    for cat, count in categories.items():
        logging.info(f"  {cat}: {count}")
        
    # 4. Total count check
    total = len(prompts)
    logging.info(f"Total prompts: {total}")
    if total != 25:
        logging.warning(f"Expected 25 prompts, but found {total}.")
        
    logging.info("Validation passed successfully.")

if __name__ == "__main__":
    main()
