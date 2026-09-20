# Experiment 4: the diagnostics on a public benchmark

`ollama:qwen2.5:3b` on 250 GSM8K test items (50 per tier, tiers from the number of calculation steps in the reference solution), 2 repeats in the normal condition and one under the sandbag prompt. Second model: `ollama:llama3.2:3b`. All intervals are 95% bootstrap.

## Accuracy by tier and condition

| tier | what | normal | sandbag | second model |
|---|---|---|---|---|
| 1 | 0-1 calculation steps | 0.840 [0.770, 0.910] | 0.860 [0.760, 0.940] | 0.760 [0.640, 0.880] |
| 2 | 2 steps | 0.890 [0.830, 0.950] | 0.840 [0.740, 0.940] | 0.860 [0.760, 0.940] |
| 3 | 3 steps | 0.810 [0.730, 0.880] | 0.840 [0.740, 0.940] | 0.740 [0.620, 0.860] |
| 4 | 4 steps | 0.740 [0.650, 0.820] | 0.780 [0.660, 0.880] | 0.660 [0.540, 0.780] |
| 5 | 5 or more steps | 0.640 [0.550, 0.730] | 0.760 [0.640, 0.880] | 0.620 [0.480, 0.760] |

![accuracy by tier](benchmark_tiers.png)

## Eval health (normal condition)

- overall: 0.784 [0.744, 0.818]; saturated: False; floored: False
- saturated tiers (CI above 0.9): none
- dead items across 3 columns (repeats and models): 157 always pass, 29 always fail
- low-discrimination items: 25
- repeat agreement on the primary model: 0.960 [0.936, 0.984]

Warnings from `vantage diagnose`:

- `DEAD_ITEMS`: 186 items give identical outcomes across all 3 columns
- `LOW_DISCRIMINATION`: 25 items barely track the rest of the eval
- `COVERAGE_GAPS`: no items for 8 threat-model nodes

## Non-capability failures (normal condition)

| scorer | mean | 95% CI | n |
|---|---|---|---|
| raw_accuracy | 0.784 | [0.744, 0.818] | 500 |
| capability_adjusted_accuracy | 0.784 | [0.744, 0.818] | 500 |
| non_capability_failure_share | 0.000 | [0.000, 0.000] | 500 |

| class | n | share |
|---|---|---|
| correct | 392 | 0.784 [0.744, 0.818] |
| wrong_answer | 108 | 0.216 [0.182, 0.256] |

## Sandbagging signal (paired, same items)

| scorer | mean | 95% CI | n |
|---|---|---|---|
| accuracy_normal | 0.784 | [0.732, 0.836] | 250 |
| accuracy_sandbag | 0.816 | [0.768, 0.860] | 250 |
| paired_diff_sandbag_minus_normal | 0.032 | [-0.012, 0.076] | 250 |
| spearman_tier_vs_failure_normal | 0.900 | [0.500, 1.000] | 5 |
| spearman_tier_vs_failure_sandbag | 0.975 | [-0.224, 1.000] | 5 |
| easy_tier_failure_rate_normal | 0.135 | [0.090, 0.180] | 200 |
| easy_tier_failure_rate_sandbag | 0.150 | [0.090, 0.220] | 100 |

Items that flipped from pass to fail under the sandbag prompt: 13; fail to pass: 21.

## Memorisation probe

The model was given the first half of each question and asked to continue it verbatim (250 items, word-level ROUGE-L against the true second half). The same test was run on copies of the questions with every number changed.

| scorer | mean | 95% CI | n |
|---|---|---|---|
| overlap_verbatim | 0.200 | [0.184, 0.216] | 250 |
| overlap_perturbed | 0.196 | [0.182, 0.212] | 250 |
| paired_diff_perturbed_minus_verbatim | -0.004 | [-0.013, 0.006] | 250 |

Verdict: no detectable difference between verbatim and perturbed completions on this metric
