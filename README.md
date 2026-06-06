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

## Human vs LLM Judge Validation

VeritasBench includes a rigorous Human-vs-Judge Agreement Study to validate the reliability of its automated LLM Judge (`Qwen/Qwen3-32B`).

### Methodology
- **Sampling**: Responses are systematically sampled from the `results/raw_scores.json` using `scripts/sample_for_human_eval.py`, balancing across all 5 evaluation dimensions.
- **Human Evaluation**: A human evaluator blindly reviews the prompt, model response, and evaluation dimension using the `judge/human_judge.py` CLI interface, assigning a binary score (`1` for Pass, `0` for Fail).
- **LLM Judge Export**: Continuous scores from the LLM judge are extracted and thresholded (`>= 0.5` becomes `1`) via `scripts/export_llm_judgements.py`.
- **Analysis**: The `judge/agreement.py` script calculates Raw Agreement %, Cohen's Kappa, and Macro F1 scores, overall and per-dimension.

### Results
The agreement study consistently demonstrates substantial alignment (Cohen's Kappa > 0.60) between human intuition and the LLM Judge across complex reasoning and multi-constraint formatting tasks.

### Agreement Visualizations
![Confusion Matrix](assets/confusion_matrix.png)
*Human vs LLM Confusion Matrix*

![Cohen's Kappa Heatmap](assets/agreement_heatmap.png)
*Agreement Heatmap by Dimension*

![Raw Agreement %](assets/agreement_breakdown.png)
*Raw Agreement % by Dimension*
