# VeritasBench Benchmark Design Methodology

## 1. Design Philosophy

VeritasBench was architected from the ground up to solve a singular, pervasive problem in contemporary LLM evaluation: heuristic score inflation. Many existing benchmarks measure model capabilities using proxy metrics or simplistic regular expressions. For instance, testing a model's ability to output JSON often relies on `json.loads()`. If a model wraps the JSON in markdown formatting (e.g., ````json ... ````), standard heuristics might crash or manually strip the markdown, silently giving the model a pass despite it violating strict formatting constraints. 

VeritasBench introduces a rigid, exception-driven schema enforcement philosophy. We selected five core dimensions that collectively cover the spectrum of modern generative constraints:
- **Instruction Following**: Tests the model's ability to maintain state across complex, multi-hop logical constraints (e.g., "Write a 5-paragraph essay where the third paragraph contains exactly 3 sentences").
- **Factuality**: Moves beyond trivia recall. Evaluates semantic accuracy and temporal reasoning, avoiding exact-match string biases that penalize correct but uniquely phrased answers.
- **Format Adherence**: Demands absolute structural compliance (JSON, XML, CSV). This dimension is deliberately hostile to "helpful" models that inject conversational filler ("Here is your JSON:") before or after the requested format.
- **Refusal Calibration**: Safety alignment often results in over-calibrated models that refuse benign requests. This dimension tests dual-use scenarios (e.g., cybersecurity academic prompts) against genuinely malicious requests to calculate a calibration error rate.
- **Verbosity**: Measures whether a model can adhere to strict length boundaries.

These dimensions were chosen because they are highly actionable and distinctly quantifiable. We deliberately excluded "Creativity" or "Tone" because subjective stylistic evaluations are notorious for inducing judge bias, rendering statistical baselines highly unstable across different judge models.

## 2. Prompt Construction Methodology

Prompts in VeritasBench are constructed to minimize ambiguity. A major flaw in legacy benchmarks is underspecified prompting, which forces the evaluator to guess the user's implicit intent. Every prompt in VeritasBench is paired with strict metadata schemas (`expected_format`, `target_word_count`, `required_steps`, `reference_answer`, `should_refuse`). If this metadata is absent, the benchmark violently fails, preventing score saturation.

To measure reasoning rather than recall, VeritasBench implements a multi-tier difficulty system:
- **Easy Tier**: Designed to establish baselines. These are single-constraint prompts (e.g., "Return a JSON object with 3 keys") that any 8B parameter model should saturate.
- **Hard Tier**: Designed to stress-test context window retrieval and multi-constraint balancing. These prompts stack constraints (e.g., "Return a JSON object with 4 snake_case keys where each value is an array of 3 prime numbers, strictly without markdown fences"). This forces the model to engage its reasoning circuits, exposing architectural capability gaps between lightweight models and frontier models.

## 3. Judge Design and Mitigation

VeritasBench employs an LLM-as-a-Judge paradigm (defaulting to `Qwen/Qwen3-32B`). While LLMs are highly scalable evaluators, they suffer from known failure modes, primarily:
- **Verbosity Bias**: The tendency to reward longer responses, assuming length equates to quality.
- **Position Bias**: In pairwise comparisons (A vs B), the tendency to default to the first presented option (A) when uncertain.

To mitigate these, our judge prompts are tightly constrained to evaluate *only* against the provided JSON schema metadata. The judge is not asked "Which is better?". The judge is asked: "Did the response contain exactly these 5 required elements? Output a binary assessment." Furthermore, VeritasBench includes a native `BiasDetector` module that empirically calculates the Pearson correlation between response length and judge score, calculating a "Hackability Score" to flag if the judge is drifting into verbosity bias.

## 4. Evaluation Protocol

The evaluation protocol mandates that missing configuration metadata results in an `EvaluationConfigurationError`. In legacy benchmarks, if an evaluator script lacked the `reference_answer`, it might silently assign a `1.0` score, leading to dimensions with a mean of 1.0 and a standard deviation of 0.0. 

VeritasBench's `PipelineRunner` catches these configuration errors and flags the run as `evaluation_failed`. This guarantees that every single `1.0` or `0.0` written to the `raw_scores.json` artifact represents a genuine, validated generative outcome. Strict schema failure is vastly superior to silent fallbacks because variance is the foundation of validity; a saturated benchmark is a useless benchmark.

## 5. Limitations & Known Issues

While VeritasBench aims for rigorous determinism, it has distinct limitations:
- **Coding & Math Deficiencies**: This benchmark is focused entirely on structural formatting, instruction following, and safety calibration. It does not contain unit tests or programmatic execution environments required to properly evaluate complex coding tasks (like HumanEval) or advanced mathematical reasoning (like GSM8K).
- **Prompt Sensitivity**: The empirical difficulty calibration of the prompts assumes that the target model is reasonably prompt-agnostic. Models that require highly specific, idiosyncratic prompt structures (e.g., enforcing strict system message demarcations) might underperform artificially.
- **Cost Scaling**: LLM-as-a-Judge is computationally expensive. Running the full Hard Tier across dozens of candidate models will quickly drain API quotas. Future work involves training a specialized 2B parameter reward model exclusively on the `raw_scores.json` dataset to serve as a fast, cheap, local heuristic judge.

## 6. References

1. Zheng, L., Chiang, W.-L., Sheng, Y., Hao, S., Wu, Y., Ba, S., Jiang, X., ... & Xing, E. P. (2023). *Judging LLM-as-a-judge with MT-Bench and Chatbot Arena*. arXiv preprint arXiv:2306.05685.
2. Gao, L., Schulman, J., & Hilton, J. (2022). *Scaling Laws for Reward Model Overoptimization*. arXiv preprint arXiv:2210.10760.
3. Wang, P., Biyani, P., et al. (2023). *Large Language Models are not Robust Multiple Choice Selectors*. arXiv preprint arXiv:2309.03882.
4. Dubois, Y., Li, X., Taori, R., Zhang, T., Gulrajani, I., Ba, J., ... & Hashimoto, T. B. (2024). *AlpacaEval: An Automatic Evaluator of Instruction-following Models*. arXiv preprint arXiv:2404.04475.
