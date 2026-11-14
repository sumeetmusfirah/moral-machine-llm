# Experiment 4: Can a model predict its own moral choice?

Run 2026-11-13. Code: `mm/introspect.py`. Prediction written and committed before any query: `PREDICTION.md` (a8d1d9e).

**Short answer: no, not in any way that beats knowing its position habit.** Every model predicts itself much
better than the other four do on average (self-advantage +10 to +61 points, all CIs above 0). But the advantage
comes from position bias: each model's prediction carries the same "Case 1" or "Case 2" habit as its choice, and the
other models carry different habits. Self-prediction never beats the trivial predictor "always say this model's
majority case." On scenarios where the choice depends on content, two of three models are at chance predicting
themselves. The third (Phi-4-mini, 86%) gets there by answering the dilemma inside the prediction prompt, and gives
the same answer when asked to predict any other model.

![Self- vs. other-prediction accuracy](self_vs_others.png)

*Left: accuracy over all 200 items. For Gemma, Phi-4-mini and Mistral the "always Case 1" line coincides with the
majority-case line. Right: only scenarios where the model's answer flips with the case order (it spares the same
side both times), where any position-only predictor scores exactly 50%. Gemma (n=1) and Qwen3 (n=6) have too few
such scenarios to show.*

## Design

- **Items:** the first 100 scenarios of `data/scenarios.jsonl`, each in both case orders (200 items).
- **Actual choice:** P(Case 1) from the Experiment 1 forced choice (`results/raw/revealed__<model>.jsonl`,
  `p_case1`), not re-queried. Argmax at 0.5.
- **Prediction:** a fresh context showing the exact choice prompt inside `<dilemma>` tags, then "Without solving it
  now, predict: when you are asked this, will you answer Case 1 or Case 2?" (self), or "...will an AI model called
  <B> answer..." (cross). Read the same way as the choice: reply prefilled with "Case", P("1") vs. P("2") from the
  next-token logprobs. Temperature 0. Qwen3 got " /no_think".
- **Calls:** 5 predictors × (1 self + 5 named targets) × 200 = 6,000, with 0 failed calls. The named diagonal
  (A predicts "a model called A") is extra: it separates self-knowledge from the "you" wording.
- **Metrics:** accuracy (argmax vs. actual argmax), Brier score vs. the actual argmax, and self-advantage =
  acc(A→A) − mean acc(B→A, B ≠ A). Baselines: always "Case 1", and always the target's majority case (the best
  possible position-only predictor; it uses hindsight). 95% CIs: 2,000 bootstrap resamples of the 100 scenarios.

## Results

| Target | Always "Case 1" | Majority case | Self acc. | Others' acc. (mean) | Best single other | Self-advantage | Self-adv., named diagonal | Brier self / others |
|---|---|---|---|---|---|---|---|---|
| Gemma 3 4B | 100% | 100% | 96% [92, 98] | 38% [37, 40] | 99.5% (Mistral) | +0.57 [+0.53, +0.61] | +0.55 [+0.51, +0.59] | 0.031 / 0.372 |
| Llama 3.2 3B | 28% | 72% | 72% [68, 78] | 49% [46, 51] | 73% (Qwen3) | +0.24 [+0.18, +0.30] | +0.24 [+0.18, +0.30] | 0.177 / 0.453 |
| Qwen3 4B | 3% | 97% | 99% [98, 100] | 38% [36, 40] | 97% (Llama) | +0.61 [+0.59, +0.63] | +0.60 [+0.57, +0.63] | 0.039 / 0.581 |
| Phi-4-mini | 74% | 74% | 61% [54, 68] | 51% [50, 51] | 73.5% (Mistral) | +0.10 [+0.04, +0.17] | +0.21 [+0.15, +0.27] | 0.219 / 0.368 |
| Mistral 7B | 79% | 79% | 79% [74, 84] | 46% [43, 48] | 77% (Gemma) | +0.34 [+0.26, +0.41] | +0.34 [+0.26, +0.41] | 0.177 / 0.356 |

Gemma's always-"Case 1" figure rounds to 100% (99.5%).

**After position bias** (only scenarios where the model's answer flips with the order, so its choice is
content-driven; chance and every position-only predictor = 50%):

| Target | Content-driven scenarios | Self acc. | Others' acc. (mean) |
|---|---|---|---|
| Llama 3.2 3B | 56 | 51% [50, 53] | 54% [51, 57] |
| Phi-4-mini | 53 | 86% [79, 92] | 51% [50, 52] |
| Mistral 7B | 38 | 50% [50, 50] | 55% [51, 59] |
| Gemma 3 4B | 1 | n/a | n/a |
| Qwen3 4B | 6 | n/a | n/a |

**What the predictors do** (`predictor_habits.csv`). Share of predictions that say "Case 1", for self and for
others:

| Predictor | Self | Others |
|---|---|---|
| Gemma 3 4B | 96.5% | 96% |
| Llama 3.2 3B | 0.5% | 0.1% |
| Qwen3 4B | 3% | 2% |
| Phi-4-mini | 34.5% | 50% |
| Mistral 7B | 100% | 100% |

Four of five predictors give nearly the same case number whatever the dilemma and whoever the target. On its own
content-driven scenarios, Phi-4-mini's prediction matches its own choice 86% of the time when predicting itself and
92% when predicting the other four models.

### Reading the numbers

1. **The self-advantage is a position-habit match.** The self prompt and the choice prompt share a model's tendency
   to say "1" or "2". The four others split two against two (Gemma and Mistral say "Case 1"; Llama and Qwen3 say
   "Case 2"), so the mean of the others is pulled toward 50% and below. Against the *best single* other model the
   advantage disappears. Gemma is predicted better by Mistral (99.5%) than by itself (96%), Phi-4-mini better by
   Mistral (73.5% vs. 61%), and Llama equally well by Qwen3 (73% vs. 72%). Self wins by 2 points for Qwen3 and
   Mistral.
2. **Self-prediction never beats knowing the habit.** Self accuracy is at or below the majority-case baseline for
   Gemma (96 vs. 99.5), Llama (72.5 vs. 72), Phi-4-mini (61 vs. 73.5) and Mistral (79 vs. 79). Qwen3 is at 99 vs. 97,
   inside the baseline's own CI [94.5, 99].
3. **"You" vs. the model's name makes no difference.** The named-diagonal advantage matches the "you" version for
   four models. For Phi-4-mini, naming itself in the third person is *more* accurate (72% vs. 61%).
4. **Phi-4-mini's 86% on content-driven scenarios is projection, not introspection.** It predicts by solving the
   dilemma and gives that answer for every model it is asked about (92% match to its own choice when predicting
   others). A model that answers "what would I do?" by doing it will predict itself well. That isn't privileged
   access in Binder et al.'s sense, and here it doesn't even help overall accuracy, because Phi-4-mini *chooses*
   with a Case 1 habit (74%) but *predicts* with a lean toward Case 2 (34.5% "Case 1").

## Prediction vs. outcome

| Prediction (PREDICTION.md) | Outcome |
|---|---|
| 1. Self-advantage within ±10 points, CI includes 0 for ≥ 4 of 5 | **Wrong.** All five are positive and significant (+0.10 to +0.61). I didn't anticipate that the others' habits would split two against two and drag the comparison mean down. |
| 2. Predictions mostly a position habit (≥ 80% one case for ≥ 3 of 5) | **Held.** 4 of 5 (Gemma, Llama, Qwen3, Mistral). |
| 3. Content-driven subset within 40–60% | **Held for Llama (51%) and Mistral (50%); failed for Phi-4-mini (86%)**, which is explained by projection (above). |
| 4. Self never beats the majority-case baseline | **Held for 4 of 5;** Qwen3 is 2 points above, within noise. |

Prediction 1 failed on a metric that turns out to be confounded. The finding I expected (no evidence of
self-knowledge beyond the position habit) holds under the cleaner comparisons: best single other predictor,
majority-case baseline, and the content-driven subset.

## Limitations

- **Zero-shot cross-prediction is a weak control.** Binder et al. fine-tune the cross-predictor on the target's
  behavior; here the other models know nothing about the target, so "others' accuracy" measures habit overlap, not
  how well behavior can be predicted from outside. The best-single-other and majority-case comparisons are the fairer
  tests.
- **The prediction prompt contains the full choice prompt**, so a model can (and Phi-4-mini does) just answer it.
  There's no way to stop a model from simulating itself; "without solving it now" is ignored.
- **Qwen3 prompt mismatch:** the Experiment 1 choices were run with `think=false` only; the predictions also got
  " /no_think". Qwen3 is near-deterministic ("Case 2" 97%) in both, so this is unlikely to matter.
- **Small samples after position bias:** 38–56 content-driven scenarios for three models; none usable for Gemma or
  Qwen3.
- **One prompt wording, temperature 0, argmax at 0.5.** CIs cover scenario sampling only, not wording. Brier scores
  are against the actual argmax, not the actual probability.
- **4-bit 3–7B models, 100 scenarios, English.** Nothing here speaks to frontier models, where Binder et al. found
  a self-advantage after fine-tuning.
- **Model names may be unfamiliar to the predictors** ("Phi-4-mini", "Qwen3 4B"); the named condition may be read
  as "some AI model".

## Files

- `raw__<predictor>.jsonl`: one row per prediction call (key `target|frame|sid|order`, `p_pred1`, `digit_mass`)
- `metrics.csv`: every estimate with its bootstrap CI; `summary.csv`: the table above
- `accuracy_matrix__diag_self.csv`, `accuracy_matrix__diag_named.csv`: 5×5 accuracy (rows predictor, columns target)
- `predictor_habits.csv`, `content_driven_n.csv`, `self_vs_others.png`, `run.log`

Reproduce: `PYTHONPATH=. uv run python -m mm.introspect --workers 2` (resumable), then
`PYTHONPATH=. uv run python -m mm.introspect --analyze`.
