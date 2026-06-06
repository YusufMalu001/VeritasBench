import os
import logging
import yaml
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class ModelCapabilityError(Exception):
    pass

def main() -> None:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token or hf_token == "your_huggingface_token_here":
        logging.error("HF_TOKEN is not set or is using the default placeholder in .env")
        return
        
    logging.info("HF_TOKEN found. Initializing OpenAI client for Hugging Face Router...")
    
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    models = config.get("models_to_evaluate", [])
    if not models:
        logging.error("No models configured.")
        return
        
    model_cfg = models[0]
    model_name = model_cfg.get("name")
    capabilities = model_cfg.get("capabilities", {"supports_chat": True, "supports_text_generation": True})
    
    if not capabilities.get("supports_chat") and not capabilities.get("supports_text_generation"):
        raise ModelCapabilityError(f"Model {model_name} must support at least one generation type.")
    
    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token
        )
        
        logging.info(f"Testing basic completion with {model_name}...")
        
        if capabilities.get("supports_chat"):
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Say 'hello world'"}],
                max_tokens=10
            )
            content = response.choices[0].message.content
        else:
            response = client.completions.create(
                model=model_name,
                prompt="Say 'hello world'",
                max_tokens=10
            )
            content = response.choices[0].text
            
        logging.info(f"Success! Model responded: {content.strip()}")
        logging.info("Smoke test passed.")
        
    except Exception as e:
        logging.error(f"Smoke test failed: {e}")

if __name__ == "__main__":
    main()
