# Experiment 5 prediction (written 2026-11-14, before any self-report or intervention query was run)

Question: when a model rates how much each factor influenced its choice, do the ratings match what actually changes
its choice when that factor is changed and nothing else is?

Design (fixed before data): 50 scenarios × 2 case orders × 5 models. After the model's own forced choice (from
Experiment 1), in the same conversation, it rates 11 factors from 0 to 9 and, for each factor we can test, answers
"if this changed, would you switch?" (yes/no). Both are read from next-token log-probabilities. Then we change one
factor at a time and re-measure the choice: case order, the people in the two groups (attributed to the scenario's
character factor: number, age, species, gender, fitness or status), stay vs. swerve, passengers vs. pedestrians,
legal vs. illegal signal, and plain vs. moral wording.

What I had seen: Experiment 1 shows that case order decides the answer in 42% to 97% of dilemmas, depending on the
model. Nothing from this experiment had been queried.

## Expectations

1. **Order goes unreported.** For at least 4 of 5 models, the mean rating for "the order in which the two cases were
   listed" is among the 3 lowest of the 11 factors, while its measured effect is the largest or second largest of
   the tested factors for at least 3 of 5.
2. **Ratings don't track effects.** Item-level Spearman correlation between a factor's rating and its measured effect
   is below 0.3 for all 5 models, and its 95% CI includes 0 for at least 3.
3. **Counterfactual answers are near chance.** Balanced accuracy of "would you switch?" against the actual switch is
   between 0.40 and 0.60 for at least 4 of 5 models.
4. **They claim to count lives.** "The number of people" gets the highest mean rating for at least 3 of 5 models
   (consistent with the "minimize loss of life" explanations in Experiment 3).
