import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main() -> None:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token or hf_token == "your_huggingface_token_here":
        logging.error("HF_TOKEN is not set or is using the default placeholder in .env")
        return
        
    logging.info("HF_TOKEN found. Initializing OpenAI client for Hugging Face Router...")
    
    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token
        )
        
        # Test a lightweight model call
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        logging.info(f"Testing basic completion with {model_name}...")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Say 'hello world'"}
            ],
            max_tokens=10
        )
        
        content = response.choices[0].message.content
        logging.info(f"Success! Model responded: {content.strip()}")
        logging.info("Smoke test passed.")
        
    except Exception as e:
        logging.error(f"Smoke test failed: {e}")

if __name__ == "__main__":
    main()
