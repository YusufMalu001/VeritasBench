# Resume Bullets: VeritasBench

## 1-Line Version
* Architected **VeritasBench**, a scalable LLM evaluation framework featuring exception-driven schema validation, a Streamlit analytics dashboard, and statistical LLM-as-a-Judge validation via Cohen's Kappa.

## 2-Line Version
* Engineered **VeritasBench**, a high-throughput LLM evaluation pipeline that eliminated heuristic score inflation via strict schema enforcement and multi-tier reasoning datasets.
* Validated automated LLM judges against human intuition using `scikit-learn` for Cohen's Kappa analysis, visualizing performance deltas via a production-grade Streamlit/Plotly dashboard.

## 4-Line Version
* **Architected VeritasBench**, a deterministic LLM evaluation framework capable of high-throughput benchmarking across Hugging Face and OpenAI inference endpoints, utilizing advanced caching and rate-limit mitigation.
* **Eliminated heuristic score inflation** by implementing an exception-driven schema validator that segregates configuration errors from genuine model failures, ensuring 100% statistical validity in variance tracking.
* **Engineered a multi-tier evaluation dataset** focusing on complex multi-hop factuality, nested format adherence, and dual-use refusal calibration to break benchmark saturation constraints on modern 8B+ parameter models.
* **Proved LLM-as-a-Judge reliability** by designing a blind human-in-the-loop CLI evaluator, computing Cohen's Kappa and Macro F1 scores via `scikit-learn` and surfacing results in a comprehensive Streamlit dashboard.
