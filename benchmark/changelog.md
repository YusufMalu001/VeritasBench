# VeritasBench Changelog

## v2.0
- Refactored `prompts.json` schema to explicitly enforce metric boundaries.
- Replaced all heuristic-based fallback dimensions with strict metadata dependencies.
- Every dimension now explicitly requires typed parameters (e.g. `target_word_count`, `should_refuse`, `expected_format`, `reference_answer`, `evaluation_type`).
- Added robust error handling. Pipeline no longer awards artificial 1.0/0.8 scores when prompt constraints are missing; it now correctly triggers `EvaluationConfigurationError`.
- Updated prompt structure from an unstructured List to a dictionary encapsulating `benchmark_version` and `prompts` array.

## v1.0
- Initial fallback-based benchmark release.
- Featured soft-fail heuristics where dimensions defaulted to passing scores if expected metadata constraints were not rigorously defined.
- Contained rate-limiting bugs due to absent retry mechanics.
