# Human vs Judge Validation

## Rationale
Automated LLM evaluation is highly scalable but often suffers from drift, where the judge model aligns differently than a human would. VeritasBench mitigates this through a rigorous, built-in validation suite.

## The Cohen's Kappa Study
To prove the efficacy of the automated judge, VeritasBench computes **Cohen's Kappa**—a statistical measure used to quantify inter-rater reliability for categorical items.

### Pipeline Flow
1. **Sampling**: The `sample_for_human_eval.py` script systematically isolates 50 successful evaluation traces, perfectly balancing them across all 5 benchmark dimensions.
2. **CLI Evaluation**: Using `human_judge.py`, human annotators blindly score the prompt and response combinations using an absolute binary value (`0` for Fail, `1` for Pass).
3. **Threshold Mapping**: Because the automated LLM Judge outputs continuous float scores (e.g., `0.8`), the `export_llm_judgements.py` script applies a standard binary threshold (`>= 0.5 -> 1`) to allow for symmetrical comparison.
4. **Scikit-Learn Analysis**: The final `agreement.py` engine computes Raw Agreement %, Cohen's Kappa, and Macro F1 scores overall and per-dimension.

## Interpreting Results
- **Kappa > 0.80**: Excellent Agreement. The LLM judge mirrors human reasoning perfectly.
- **Kappa > 0.60**: Substantial Agreement. The LLM judge is reliable and production-ready.
- **Kappa < 0.40**: Poor Agreement. The benchmark schema is flawed or the judge model lacks reasoning depth.

*Visual assets such as heatmaps and confusion matrices are generated dynamically into the `assets/` folder during this process.*
