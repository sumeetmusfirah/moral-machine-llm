# Do language models know their own morals?

A Moral Machine study on five small open models, run locally for $0. Started 2026-11-09.

## Question

Awad et al. (2018) asked millions of people to choose who a self-driving car with failed brakes should kill, and
estimated how much nine factors (age, number of lives, species, legality, and so on) move those choices.
Several papers have since run the same dilemmas on LLMs (below). This study asks three things they don't
answer together:

1. **Revealed preferences with position bias removed.** Small models often pick "Case 1" regardless of content. We
   ask every dilemma in both case orders and read the token probabilities, which separates position bias from moral
   preference exactly (see Method).
2. **Self-knowledge.** We also ask each model, in the abstract, how much each factor *should* matter and how much it
   *would* matter to its own choices. Does what it says match what it does? This is the stated-vs.-revealed gap
   (moral psychology) and a test of introspection (Binder et al. 2024), on the same Delta P scale as the human data.
3. **Refusal vs. preference.** Some models refuse to choose. The forced-choice probabilities show whether a
   refusing model still has a consistent preference underneath.

## Prior work (checked 2026-11-09)

| Paper | What it did | How this differs |
|---|---|---|
| Takemoto 2024, *R. Soc. Open Sci.* | Moral Machine on GPT-3.5, GPT-4, PaLM 2, Llama 2 | Free-text answers; no stated-vs.-revealed arm |
| Ahmad & Takemoto 2025, *PLoS One* 20:e0322776 ([arXiv 2411.06790](https://arxiv.org/abs/2411.06790)) | 52 LLMs; models over 10B closer to humans | Same |
| Jin et al. 2025, ICLR, MultiTP ([arXiv 2407.02273](https://arxiv.org/abs/2407.02273), [code](https://github.com/causalNLP/moralmachine)) | 107 languages, 19 models | The language twist is done; we don't repeat it |
| Vida et al. 2024, AIES ([arXiv 2407.15184](https://arxiv.org/abs/2407.15184)) | Multilingual moral preferences | Same |
| Stated vs. revealed in LLMs (e.g. arXiv 2504.06324, TrolleyBench, NeurIPS 2025) | General moral dilemmas | Not on the Moral Machine Delta P scale with a human anchor |

Before posting, search again: this area moves fast.

## Datasets

### 1. Moral Machine (Awad et al. 2018), the human anchor

- **Paper:** Awad, E., Dsouza, S., Kim, R., Schulz, J., Henrich, J., Shariff, A., Bonnefon, J.-F., & Rahwan, I. (2018).
  The Moral Machine experiment. *Nature*, 563(7729), 59–64. https://doi.org/10.1038/s41586-018-0637-6
- **Data:** OSF project [osf.io/3hvt2](https://osf.io/3hvt2/), created 2018-07-22. **License: CC BY 4.0** (cite the
  paper).
- **Scale:** about 40 million decisions, 10 languages, 233 countries and territories (2.3 million people).
- **Design:** sessions of 13 dilemmas; 9 attributes. 3 structural ones vary in every scenario (intervention: stay or
  swerve; passengers vs. pedestrians; legality: green or red light). 6 character attributes (gender, age, fitness,
  social status, number of characters, species) appear twice per session, plus one random scenario. 20 characters
  (man, woman, pregnant woman, baby in stroller, elderly man and woman, boy, girl, homeless person, large man and
  woman, criminal, male and female executive, athlete and doctor, dog, cat); 1 to 5 per side.
- **Effect size:** AMCE (average marginal component effect), a change in the probability of sparing a character
  when one attribute flips, e.g. elderly to young.

| File (OSF) | Size | What it is | Used here |
|---|---|---|---|
| `MMdata_ReadMe.txt` | 21 KB | Column codebook | yes, `data/raw/` |
| `CountriesChangePr.csv` | 22 KB (gz) | AMCE + clustered SE for 9 attributes × 130 countries | yes |
| `plotdatamain.rdata` | <1 KB | Global AMCEs from Fig. 2 | yes |
| `country_cluster_map.csv` | 2 KB | Countries → Western / Eastern / Southern clusters | yes |
| `cultures.csv` | 1 KB | Country → cultural zone (Islamic, Orthodox, LatinAmerica…) | yes |
| `moral_distance.csv` | 3 KB | Each country's moral distance from the USA | yes |
| `dendrogram_Culture.csv`, `countryInfo.csv` | 3 KB, 17 KB | Clustering and country metadata | yes |
| `SharedResponses.csv` | 3.2 GB (gz) | Every response, 41 columns, 2 rows per scenario | no (not needed) |
| `SharedResponsesFullFirstSessions.csv` | 1.6 GB (gz) | First completed session per user | no |
| `SharedResponsesSurvey.csv` | 297 MB (gz) | Responses + demographics (age, education, gender, income, political and religious sliders) | no; useful later |

Global human AMCEs (Fig. 2, from `plotdatamain.rdata`): Species 0.58, No. Characters 0.51, Age 0.49,
Law 0.35, Social Status 0.35, Fitness 0.16, Gender 0.12, Relation to AV 0.10, Intervention 0.06.

Note: the country file labels intervention "Omission -> Commission", but its values (mean 0.05) match Fig. 2's
"preference for inaction" (0.06), so it's used in that direction.

### 2. Our generated dilemmas

`data/scenarios.jsonl`: 50 sessions × 13 = 650 scenarios, seed 20261109, generated by `mm/scenarios.py` following
the paper's design. The main run uses the first 325 (25 sessions). Simplification: character sets are drawn from
matched pairs (e.g. boy ↔ man ↔ elderly man) rather than the website's exact sampler. Wording follows the website's
"show description" text.

## Models

All local through Ollama 0.40.11 on an Apple M4 (16 GB). All 4-bit (Q4_K_M) GGUF quantizations, so results describe
these quantized builds, not the full-precision releases.

| Model | Maker | Params | Context | Digest | Size | License | Notes |
|---|---|---|---|---|---|---|---|
| `gemma3:4b` | Google DeepMind | 4.3B | 128K | `a2af6cc3eb7f` | 3.3 GB | Gemma Terms of Use | Answers "Case 1" every time in free text |
| `llama3.2:3b` | Meta | 3.2B | 128K | `a80c4f17acd5` | 2.0 GB | Llama 3.2 Community License | Refuses most free-text dilemmas |
| `qwen3:4b` | Alibaba (Qwen) | 4.0B | 256K | `359d7dd4bcda` | 2.5 GB | Apache 2.0 | Reasoning model; run with thinking off |
| `phi4-mini:3.8b` | Microsoft | 3.8B | 128K | `78fad5d182a7` | 2.5 GB | MIT | |
| `mistral:7b` | Mistral AI | 7.2B | 32K | `6577803aa9a0` | 4.4 GB | Apache 2.0 | Tends to hedge in free text |

Decoding: temperature 0 for dilemmas (deterministic), temperature 0.7 with seeds 0 to 9 for stated preferences.

## Method

**Revealed arm.** Each dilemma is asked in both orders (side a as Case 1, then side b as Case 1). Two calls per order:

1. *Free answer* (40 tokens): records whether the model refuses or picks a case.
2. *Forced choice*: the reply is prefilled with "Case" and we read P("1") and P("2") from the next-token
   log-probabilities.

Model: logit P(Case 1) = β − δ in order 0 and β + δ in order 1, where β is position bias and δ is the preference
for sparing side a. So δ = (logit₁ − logit₀)/2 and the debiased P(spare a) = σ(δ). Effect per attribute =
2·mean(P(spare preferred side)) − 1, on the paper's Delta P scale; 95% CI by bootstrap over scenarios.

**Stated arm.** For each attribute, two framings, 10 samples each: "how much *should* the car favor sparing X over Y"
and "how much *would you* favor…", from −100 to 100, divided by 100 to put it on the Delta P scale.

**Measures (fixed before results came in):**
- Effect per attribute vs. global humans; Spearman rank correlation; Euclidean distance; the 3 nearest countries.
- Self-knowledge: correlation and mean absolute error between stated and revealed effects, per framing.
- Free-text refusal rate; free-answer order consistency; mean |β| (position bias).

## Run

```bash
uv sync
PYTHONPATH=. uv run python -m mm.run --models gemma3:4b llama3.2:3b qwen3:4b phi4-mini:3.8b mistral:7b \
    --arm both --limit 325 --samples 10
PYTHONPATH=. uv run python -m mm.analyze      # writes results/effects.csv, summary.csv, scenario_level.csv
```

Optional: `--persona Japan` adds "Answer as a typical person from Japan would" to the dilemmas.

## Limitations

- Small 4-bit models; nothing here generalizes to frontier models.
- The forced choice measures a preference the model may not express in free text. That is the point of arm 1 vs. 2,
  but say so when reporting.
- The paper's AMCE comes from conjoint regression on outcome rows; ours is a paired within-type difference. Same
  target under random assignment, not the same estimator.
- English only. The human data mixes 10 languages.
