# VeritasBench

VeritasBench is a production-grade LLM evaluation framework designed for rigorous behavioral benchmarking, human-LLM agreement analysis, and extensible ELO arena comparisons.

## Architecture

* **Models**: Scalable inference via Hugging Face Router (OpenAI-compatible client).
* **Evaluation Dimensions**:
  * Instruction Following
  * Factuality
  * Refusal Calibration
  * Format Adherence
  * Verbosity
* **Judging**: LLM-as-a-judge (`Qwen/Qwen3-32B` default) and CLI-based human evaluation.
* **Analysis**: Streamlit Dashboard and automated Markdown report generator.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Copy `.env.example` to `.env` and set your `HF_TOKEN`. 
**Note:** Your token MUST have:
- Read access to public gated repositories
- Make calls to Inference Providers

## Development Workflow

1. Verify token and router access:
```bash
python -m scripts.smoke_test
```

2. Generate baseline prompts (25 prompts):
```bash
python -m scripts.generate_prompts
python -m scripts.validate_benchmark
```

3. Run Evaluation Pipeline:
```bash
python -m pipeline.runner
```

4. Aggregate Scores:
```bash
python -m pipeline.aggregator
```

5. Launch Dashboard:
```bash
streamlit run dashboard/app.py
```
