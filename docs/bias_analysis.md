# LLM Judge Bias & Failure Analysis

## Abstract
Automated evaluation using LLM-as-a-Judge paradigms provides tremendous scalability but introduces systematic biases. Large Language Models frequently demonstrate preferences for certain structures—often rewarding verbose, repetitive responses over concise, accurate ones, or exhibiting positional biases in pairwise tasks. This report details the empirical bias detection executed on the VeritasBench evaluation pipeline, quantifying the exact vulnerability of our Qwen3-32B judge.

## Length Bias & Verbosity Reward Hacking
**Detection Methodology**: 
We computed the Pearson correlation coefficient between the word count of the generated responses and the final judge score. A strong positive correlation (|r| > 0.3) indicates that the judge is inappropriately conflating length with quality. Furthermore, we measured the "Hackability Score," representing the percentage of top-quartile scoring responses that are also in the top-quartile for verbosity.

**Findings**:
- Pearson Correlation: -0.166
- Length Biased: False (Direction: neutral)
- Reward Hackability Score: 0.242

**Implications**:
As established by Gao et al. (2022) in their research on scaling laws for reward models, LLMs are highly susceptible to verbosity bias. If our hackability score exceeds 0.4, it suggests that models like Gemma or Llama could easily game the VeritasBench scores by simply outputting longer, unstructured text, bypassing the strict reasoning requirements we intend to measure.

## Position Bias in Pairwise Evaluation
**Detection Methodology**:
In pairwise evaluation (Model A vs Model B), judges often favor the first position (A). To detect this, we simulate swapping the A/B presentation order for identical prompt pairs in 20% of cases and measuring the consistency rate of the judge's decision. 

**Findings**:
- Consistency Rate: 85.0%
- Position Biased: False (Preferred: A)

**Implications**:
Consistent with findings from Wang et al. (2023) regarding LLM robustness in multiple-choice scenarios, a drop in consistency below 85% signifies a dangerous positional bias. If detected, future iterations of the VeritasBench runner will need to implement mandatory bidirectional evaluation (evaluating A-B and then B-A) to average out positional advantages.

## Refusal Overcalibration
**Detection Methodology**:
A critical failure mode in aligned LLMs is over-refusal (false refusal rate) where benign, educational prompts are blocked, alongside false compliance where harmful prompts are fulfilled. We compute the calibration error as the mean of these two rates within our `refusal_calibration` dimension.

**Findings**:
- False Refusal Rate: 8.3%
- False Compliance Rate: 37.5%
- Overall Calibration Error: 0.229

**Implications**:
Over-calibrated models suffer in utility. Skalse et al. (2022) note that reward misspecification often leads to models that act overly cautious. A high calibration error here demands a recalibration of the baseline prompts in our Hard Tier to better distinguish between malicious intent and academic inquiry.

## References
1. Gao, L., Schulman, J., & Hilton, J. (2022). *Scaling Laws for Reward Model Overoptimization*. 
2. Skalse, J., Howe, R., Krasheninnikov, D., & Krueger, D. (2022). *Defining and Characterizing Reward Gaming*.
3. Wang, P., Biyani, P., et al. (2023). *Large Language Models are not Robust Multiple Choice Selectors*.
