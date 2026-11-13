# Experiment 4 prediction (written 2026-11-13, before any prediction query was run)

Question: can a model predict its own forced choice on a Moral Machine dilemma better than other models can
predict it (self-advantage, after Binder et al. 2024, "Looking Inward")?

What I had seen before writing this: the existing forced-choice answers in `results/raw/revealed__*.jsonl` (from
Experiment 1). On the first 100 scenarios × 2 orders, the share of "Case 1" answers is gemma3 99%, llama3.2 28%,
qwen3 3%, phi4-mini 73%, mistral 79%. The number of scenarios whose answer changes with the case order (the
content-driven subset) is gemma3 1, llama3.2 56, qwen3 6, phi4-mini 53, mistral 38. No self- or cross-prediction
had been queried.

## Expectations

1. **Self-advantage ≈ 0.** For every model the self-advantage (accuracy of A predicting A minus the mean accuracy
   of the other four predicting A) lies within ±10 points, and its 95% CI includes 0 for at least 4 of the 5
   models. Mean across models within ±5 points.
2. **Predictions are mostly a position habit.** Each model's predictions are dominated by one case number in the
   prediction prompt (≥ 80% the same answer for at least 3 of 5 models), so accuracy is set by whether the
   predictor's habit matches the target's habit, not by self-knowledge.
3. **No content signal after position.** On the content-driven subset (llama3.2, phi4-mini, mistral; gemma3 and
   qwen3 have too few), self-prediction accuracy is within 40–60%, i.e. not distinguishable from chance.
4. **Not better than a position-only baseline.** Self-prediction accuracy does not beat the target's best
   position-only predictor (always answer its majority case) for any model.

If (1) fails with a positive, significant self-advantage that survives (3), that would be evidence of privileged
access in 3–7B models, which I think is unlikely.
