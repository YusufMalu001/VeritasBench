# VeritasBench: Eradicating Heuristic Inflation in LLM Evaluation

## 🔴 The Problem
As Large Language Models rapidly evolve, standard evaluation benchmarks (like MMLU or HumanEval) are increasingly contaminated or trivialized. Many custom evaluation suites fall victim to a silent killer: **heuristic score inflation**. 

When an evaluator script encounters a missing metadata field or an edge-case generation, it often defaults to a "pass" or fails silently, registering a perfect `1.0` score with zero variance. This falsely inflates the perceived intelligence of models, especially in critical safety and strict formatting dimensions.

## 🟢 The Solution
**VeritasBench** is a production-grade, extensible LLM evaluation framework built to guarantee deterministic validity. It completely eliminates silent fallbacks through strict JSON schema enforcement and validates its own automated judge using human-in-the-loop statistical analysis (Cohen's Kappa).

## 🏗️ Architecture
VeritasBench operates on a high-throughput, decoupled architecture:
1. **Inference Router**: An async, Tenacity-backed engine capable of communicating with Hugging Face and OpenAI endpoints, managing rate limits and executing prompts across multiple models concurrently.
2. **Evaluation Dimensions**: Five distinct classes (Factuality, Instruction Following, Refusal Calibration, Format Adherence, Verbosity) extending a rigid `BaseDimension` interface.
3. **Exception-Driven Validation**: Instead of silent heuristics, missing metadata throws an `EvaluationConfigurationError`. The runner isolates these errors without crashing, preserving pipeline integrity and guaranteeing that *all recorded scores are legitimate*.
4. **LLM-as-a-Judge**: Leverages a secondary large model (e.g., `Qwen3-32B`) to verify multi-step logic.
5. **Streamlit Analytics**: A professional dashboard rendering interactive radar charts, cost analytics, and failure tag analyses natively using Plotly.

## 🚧 Challenges Faced
1. **API Rate Limiting & Quotas**: Running multi-tier benchmarks across multiple models triggered `HTTP 429` and `402` errors. I implemented exponential backoffs and a robust disk-caching mechanism to preserve state between interrupted runs.
2. **False Positives in Safety**: Simple refusal prompts ("Write malware") resulted in 100% saturation because models easily recognized them. I engineered a `Hard` tier featuring dual-use contextual jailbreaks (e.g., Penetration testing scenarios) to test true refusal calibration.
3. **Judge Bias**: The automated judge needed continuous validation. I built a CLI for blind human A/B and binary evaluation and wired it into `scikit-learn` to calculate Cohen's Kappa, mathematically proving the judge's alignment with human intent.

## 💡 Lessons Learned
- **Variance is Validity**: If a benchmark dimension yields a standard deviation of `0.0`, the benchmark is broken, not the model. 
- **Modular Evaluation**: Decoupling the generation engine from the evaluation engine is mandatory. It allows for infinite re-evaluations of cached generations without incurring redundant API costs.
- **Fail Loud, Continue Gracefully**: Letting bad data trigger exceptions while the orchestrator catches and logs them is vastly superior to burying edge cases in nested `if-else` blocks.

## 🏆 Results
VeritasBench successfully isolated the performance delta between `Llama-3.1-8B` and `Gemma-3-27B`, providing statistical proof of true reasoning capabilities across a 2-tier dataset, visually rendered in a portfolio-quality Streamlit dashboard.
