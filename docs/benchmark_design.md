# Benchmark Design & Schema Constraints

## The Score Saturation Problem
Early iterations of VeritasBench relied on heuristic defaults. If an evaluation dimension didn't understand the prompt context, it would silently award a passing score. This led to massive score inflation and zero-variance outcomes.

## VeritasBench v2.1 Schema Enforcement
VeritasBench v2.1 solves this by enforcing strict JSON metadata boundaries.

### Example: Format Adherence
```json
{
  "id": "fmt_hard_1",
  "category": "format_adherence",
  "prompt": "Return valid JSON with: exactly 4 keys, snake_case keys only, each value is an array of exactly 3 prime numbers, and absolutely no markdown fences.",
  "metadata": {
    "expected_format": "json"
  }
}
```
If `expected_format` is missing, the dimension throws an `EvaluationConfigurationError`. 

## Easy vs Hard Tiers
To ensure the benchmark evaluates **reasoning over recall**, the dataset is split into tiers:
- **Easy Tier**: Trivial tasks ("Capital of Japan?", "Return JSON") easily saturated by 8B parameter models.
- **Hard Tier**: Contextually constrained reasoning ("Explain OpenAI CEO succession", "Jailbreak dual-use scripts in educational context").
