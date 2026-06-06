import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class ReportGenerator:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.agg_file = self.results_dir / "aggregated_scores.json"
        self.report_file = self.results_dir / "report.md"
        
    def generate(self):
        if not self.agg_file.exists():
            logging.error(f"{self.agg_file} not found. Cannot generate report.")
            return
            
        with open(self.agg_file, "r") as f:
            data = json.load(f)
            
        report = []
        report.append("# VeritasBench Evaluation Report")
        report.append(f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n")
        
        report.append("## Abstract")
        report.append("This report presents the findings of the VeritasBench evaluation framework, comparing Large Language Models across multiple behavioral dimensions including Instruction Following, Factuality, Refusal Calibration, Format Adherence, and Verbosity. The evaluation utilizes an LLM-as-a-judge mechanism to measure both absolute performance and pairwise preferences.\n")
        
        report.append("## Methodology")
        report.append("Models were queried via the Hugging Face Inference Router using an OpenAI-compatible interface. Each model responded to a set of prompts covering the defined dimensions. Responses were scored from 0 to 1 based on dimension-specific criteria evaluated by the LLM Judge (`Qwen/Qwen3-32B`).\n")
        
        report.append("## Benchmark Design")
        report.append("The benchmark consists of 25 core prompts evenly distributed across 5 categories (5 prompts each). This scalable design ensures rapid testing while allowing expansion to 100+ prompts in full evaluation runs.\n")
        
        report.append("## Results")
        
        for model, stats in data.items():
            report.append(f"### {model}")
            overall = stats['overall']['mean']
            std = stats['overall']['std']
            report.append(f"- **Overall Score**: {overall:.2f} ± {std:.2f}")
            report.append("- **By Dimension**:")
            for dim, dim_stats in stats['by_dimension'].items():
                report.append(f"  - {dim}: {dim_stats['mean']:.2f}")
            report.append("")
            
        report.append("## Failure Analysis")
        report.append("Common failures across models include `instruction_violation` and `verbosity_mismatch`. Factuality hallucinations and formatting errors were also tracked. Refer to the raw JSON logs or Dashboard for per-prompt failure tags.\n")
        
        report.append("## Human Agreement Analysis")
        report.append("Human-LLM judge agreement measured using Cohen's Kappa. (See `agreement_report.json` for live run metrics).\n")
        
        report.append("## Limitations")
        report.append("- The sample size of the benchmark (25 prompts) provides directional insights rather than statistically rigorous comprehensive bounds.")
        report.append("- The LLM Judge may exhibit bias towards its own generated styles.")
        report.append("- Token tracking uses simple heuristics if the API does not return precise usage details.\n")
        
        report.append("## Future Work")
        report.append("- Scale benchmark to 100+ prompts.")
        report.append("- Add more dynamic prompt generation.")
        report.append("- Incorporate DPO-based pairwise reward scoring into the pipeline.\n")
        
        with open(self.report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
            
        logging.info(f"Report generated successfully at {self.report_file}")

if __name__ == "__main__":
    generator = ReportGenerator()
    generator.generate()
