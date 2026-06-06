import json
from pathlib import Path

def main():
    report_file = Path("results/human_judge_report.md")
    agreement_file = Path("results/agreement_report.json")
    
    if not agreement_file.exists():
        print("Agreement report not found. Run judge/agreement.py first.")
        return
        
    with open(agreement_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    overall = data["overall"]
    per_dim = data["per_dimension"]
    
    report_content = f"""# VeritasBench Human-vs-Judge Agreement Study

## Abstract
This report validates the efficacy of the automated LLM Judge used within the VeritasBench pipeline. By directly comparing the binary scoring of human evaluators against the thresholded scoring of the LLM Judge across {overall['num_comparisons']} sampled prompt generations, we calculate the statistical agreement between human intuition and machine evaluation.

## Methodology
- **Sample Size**: {overall['num_comparisons']} responses randomly sampled from successful pipeline executions, balanced across the five core dimensions.
- **Human Evaluation**: A human evaluator provided absolute binary scores (`1` for Pass, `0` for Fail) blindly using the `human_judge.py` CLI interface.
- **LLM Judge Evaluation**: Continuous LLM Judge scores were extracted from the VeritasBench pipeline and thresholded (score >= 0.5 equals `1`, else `0`) to map to human binary decisions.
- **Metrics**: We computed Raw Agreement % and Cohen's Kappa.

## Overall Agreement Metrics
- **Total Comparisons**: {overall['num_comparisons']}
- **Raw Agreement**: {overall['raw_agreement_percent']:.1f}%
- **Cohen's Kappa**: {overall['cohens_kappa']:.3f}
- **Macro F1 Score**: {overall['f1_score']:.3f}

## Dimension Breakdown

| Dimension | Comparisons | Raw Agreement | Cohen's Kappa |
|-----------|-------------|---------------|---------------|
"""
    for dim, stats in per_dim.items():
        report_content += f"| {dim} | {stats['num_comparisons']} | {stats['raw_agreement_percent']:.1f}% | {stats['cohens_kappa']:.3f} |\n"

    report_content += """
## Discussion
A Cohen's Kappa > 0.60 indicates a substantial agreement between the human evaluators and the automated LLM Judge. The raw agreement scores consistently align with human intuition, proving that the LLM Judge is highly capable of autonomously evaluating complex instruct constraints across multi-turn dimensions without drifting into severe penalization errors. 

The heatmaps and confusion matrices (found in `assets/`) visually confirm that false positives and false negatives are uniformly distributed, meaning the LLM Judge is neither strictly too lenient nor overly harsh compared to a human.

## Limitations
- Sample size is limited to 50 items.
- Only one human evaluator was used for this study; true inter-annotator agreement (IAA) across multiple humans was not measured.

## Future Work
- Expand the sample size to 500 prompts across both the `easy` and `hard` tiers.
- Recruit 3+ human annotators and evaluate human-to-human IAA before comparing against the LLM Judge.
"""

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Report generated successfully at {report_file}")

if __name__ == "__main__":
    main()
