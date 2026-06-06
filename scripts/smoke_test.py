import os
import logging
import yaml
from dotenv import load_dotenv
from models.compatibility import CompatibilityChecker, ChatModelCompatibilityError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main() -> None:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token or hf_token == "your_huggingface_token_here":
        logging.error("HF_TOKEN is not set or is using the default placeholder in .env")
        return
        
    logging.info("HF_TOKEN found. Router connectivity check initialized...")
    
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    models = config.get("models_to_evaluate", [])
    if not models:
        logging.error("No models configured.")
        return
        
    model_names = [m["name"] for m in models]
    judge_model = config.get("judge_model")
    if judge_model:
        model_names.append(judge_model)
        
    checker = CompatibilityChecker()
    
    try:
        logging.info(f"Validating chat endpoint compatibility for models: {model_names}")
        checker.validate_models(model_names)
        logging.info("All configured models are chat-compatible.")
        logging.info("Smoke test passed.")
        
    except ChatModelCompatibilityError as e:
        logging.error(f"Smoke test failed due to model incompatibility: {e}")
    except Exception as e:
        logging.error(f"Smoke test failed: {e}")

if __name__ == "__main__":
    main()
