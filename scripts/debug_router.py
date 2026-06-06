import os
import json
import logging
from typing import Dict, Any
from dotenv import load_dotenv

import openai
from openai import OpenAI
import huggingface_hub

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def test_endpoint(client: OpenAI, model: str, endpoint_type: str) -> Dict[str, Any]:
    url = f"https://router.huggingface.co/v1/{'chat/completions' if endpoint_type == 'chat' else 'completions'}"
    logging.info(f"Testing {endpoint_type} endpoint for {model} at {url}")
    
    result = {
        "supported": False,
        "status_code": None,
        "error": None,
        "endpoint_url": url
    }
    
    try:
        if endpoint_type == "chat":
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            result["supported"] = True
            result["status_code"] = 200
            logging.info(f"  -> SUCCESS (chat)")
            
        else:
            response = client.completions.create(
                model=model,
                prompt="Hello",
                max_tokens=5
            )
            result["supported"] = True
            result["status_code"] = 200
            logging.info(f"  -> SUCCESS (completion)")
            
    except openai.APIStatusError as e:
        result["status_code"] = e.status_code
        try:
            result["error"] = e.response.json()
        except Exception:
            result["error"] = e.response.text
        logging.error(f"  -> FAILED ({e.status_code}): {result['error']}")
        
    except Exception as e:
        result["error"] = str(e)
        logging.error(f"  -> FAILED (Exception): {str(e)}")
        
    return result

def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token or hf_token == "your_huggingface_token_here":
        logging.error("Valid HF_TOKEN is required in .env")
        return

    print("=== Diagnostics Information ===")
    print(f"OpenAI SDK Version: {openai.__version__}")
    print(f"Hugging Face Hub SDK Version: {huggingface_hub.__version__}")
    print("Router URL: https://router.huggingface.co/v1")
    print("===============================\n")

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token
    )

    models_to_test = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen3-32B",
        "google/gemma-3-27b-it"
    ]
    
    report = []

    for model in models_to_test:
        print(f"\n--- Testing Model: {model} ---")
        
        chat_result = test_endpoint(client, model, "chat")
        comp_result = test_endpoint(client, model, "completion")
        
        report.append({
            "model": model,
            "chat_supported": chat_result["supported"],
            "chat_details": {
                "status_code": chat_result["status_code"],
                "error": chat_result["error"]
            },
            "completion_supported": comp_result["supported"],
            "completion_details": {
                "status_code": comp_result["status_code"],
                "error": comp_result["error"]
            }
        })

    os.makedirs("results", exist_ok=True)
    report_path = "results/router_diagnostics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nDiagnostics complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
