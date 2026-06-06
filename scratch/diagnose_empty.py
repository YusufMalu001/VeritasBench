import json
import os
from openai import OpenAI
from dotenv import load_dotenv

def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token
    )

    with open("results/raw_scores.json", "r") as f:
        scores = json.load(f)

    empty_responses = [s for s in scores if not s.get("response", "").strip()]
    
    analysis = []
    
    print(f"Found {len(empty_responses)} empty responses. Testing first 3 live...")
    
    for i, rec in enumerate(empty_responses):
        prompt = rec["prompt"]
        prompt_id = rec["prompt_id"]
        category = rec["category"]
        
        entry = {
            "prompt_id": prompt_id,
            "category": category,
            "prompt": prompt,
            "latency_ms": rec["latency_ms"],
            "original_token_estimate": rec["token_estimate"]
        }
        
        # Test live for the first 3 to get raw API data
        if i < 3:
            print(f"--- Live Testing {prompt_id} ---")
            try:
                response = client.chat.completions.create(
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=512
                )
                raw_dict = response.model_dump()
                entry["live_test"] = raw_dict
                print(json.dumps(raw_dict, indent=2))
            except Exception as e:
                entry["live_test_error"] = str(e)
                print(f"Error: {e}")
                
        analysis.append(entry)

    os.makedirs("results", exist_ok=True)
    with open("results/empty_response_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print("\nCategories affected:")
    cats = {}
    for a in analysis:
        c = a["category"]
        cats[c] = cats.get(c, 0) + 1
    for k, v in cats.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    main()
