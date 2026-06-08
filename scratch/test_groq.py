import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

models = ["gemma2-9b-it", "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]

for m in models:
    try:
        res = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1
        )
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: FAILED - {e}")
