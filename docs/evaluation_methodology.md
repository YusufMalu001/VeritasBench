# Evaluation Methodology

## LLM-as-a-Judge Paradigm
VeritasBench primarily employs the LLM-as-a-Judge technique to evaluate generated responses against complex instruction boundaries.

### How it Works
1. A **target model** (e.g., `Llama-3.1-8B`) generates a response to a given prompt.
2. An **evaluator model** (e.g., `Qwen3-32B`) receives the target's response, the original prompt, and the strict schema constraints required by the dimension.
3. The evaluator returns a structured JSON payload containing a fractional score (`0.0 - 1.0`) and an explicit string explanation.

### Avoiding "Judge Bias"
To prevent the evaluator model from showing preference to its own training style, VeritasBench focuses the evaluation entirely on *verifiable constraints* rather than subjective quality.
For example, in the **Instruction Following** dimension, the judge does not score based on "flow" or "creativity." It strictly parses whether the required elements defined in the schema's `required_steps` list are present.

### Fallback Heuristics
If the LLM judge API fails or is unavailable, dimensions fallback to programmatic parsers. For example, `FormatAdherence` will execute an actual JSON or XML parser in Python to verify structural integrity before applying a score, guaranteeing deterministic verification.
