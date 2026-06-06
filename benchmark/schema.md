# VeritasBench Prompts Schema

VeritasBench v2.0 enforcing strict schema configurations. Each prompt object must conform to this schema.

## Base Prompt Object
```json
{
  "id": "string",
  "category": "string",
  "prompt": "string",
  "metadata": {}
}
```

## Category-Specific Metadata Constraints

Each category dictates strict `metadata` requirements. Failure to provide them triggers an `EvaluationConfigurationError`.

### 1. Instruction Following
Evaluates whether the model completes exact multi-step logic.
```json
"metadata": {
  "required_steps": ["list", "of", "required", "strings"]
}
```

### 2. Factuality
Evaluates factual compliance using either semantic heuristics or LLM judge overlays.
```json
"metadata": {
  "reference_answer": "string",
  "evaluation_type": "semantic_match" 
}
```

### 3. Refusal Calibration
Evaluates whether the model appropriately refuses dangerous prompts or appropriately complies with safe prompts.
```json
"metadata": {
  "should_refuse": boolean
}
```

### 4. Format Adherence
Evaluates structural adherence.
```json
"metadata": {
  "expected_format": "json" // or "markdown_table", "numbered_list", "xml_tags", "code_block"
}
```

### 5. Verbosity
Evaluates the capacity of the model to obey exact length limits.
```json
"metadata": {
  "target_word_count": integer
}
```
