# VeritasBench Architecture

## Overview
VeritasBench is engineered for high-throughput, deterministic LLM evaluation. It decouples generation from evaluation, allowing large-scale caching and rapid dimension iteration without incurring redundant inference costs.

## Core Components

### 1. Model Registry & Routing (`models/`)
The framework interfaces securely with external inference APIs (e.g., Hugging Face Router) via the `HuggingFaceModel` class. It features:
- **Compatibility Cache**: Pre-validates models for chat vs. completion endpoints to prevent mid-run routing failures.
- **Tenacity Retries**: Implements exponential backoff for `HTTP 429` Rate Limits and `HTTP 503` Service Unavailable errors.

### 2. Pipeline Runner (`pipeline/runner.py`)
The orchestration engine responsible for joining prompt batches with models.
- Uses **DiskCache** to store exact prompt+model combinations. 
- Gracefully intercepts and segregates `EvaluationConfigurationError` exceptions to isolate schema failures without crashing the entire benchmark suite.

### 3. Evaluator Dimensions (`dimensions/`)
Every dimension extends a `BaseDimension` interface enforcing a rigid `score(prompt, response)` contract. 
- Unlike heuristic-heavy legacy systems that default to a `1.0` pass when metadata is missing, VeritasBench uses an exception-driven architecture that demands complete metadata (e.g., `target_word_count`, `reference_answer`).

### 4. Aggregator & Dashboard
Outputs `results/raw_scores.json` which is then mathematically processed by `pipeline/aggregator.py` to yield means, standard deviations, and variances. The Streamlit dashboard ingests these structures dynamically.
