# Experiment 1: judge reliability

Judge `anthropic:claude-sonnet-5` graded candidate answers to arithmetic word problems against the rubric `data/rubrics/correct.md`. Gold labels come from a numeric rule (real responses from `ollama:qwen2.5:3b`) or from construction (planted responses). All intervals are 95% bootstrap.

## Agreement with gold labels

| scorer | mean | 95% CI | n |
|---|---|---|---|
| accuracy | 0.995 | [0.985, 1.000] | 200 |
| cohen_kappa | 0.989 | [0.960, 1.000] | 200 |
| parse_rate | 1.000 | [1.000, 1.000] | 200 |
| fail_precision | 1.000 | [1.000, 1.000] | 64 |
| fail_recall | 0.985 | [0.954, 1.000] | 65 |
| fail_f1 | 0.992 | [0.973, 1.000] | 200 |

Confusion (rows = gold, columns = judge):

| gold \ judge | pass | fail | unparsed |
|---|---|---|---|
| pass | 135 | 0 | 0 |
| fail | 1 | 64 | 0 |

### Accuracy per response variant

| scorer | mean | 95% CI | n |
|---|---|---|---|
| confident_wrong | 1.000 | [1.000, 1.000] | 13 |
| correct_reasoning | 1.000 | [1.000, 1.000] | 15 |
| flip_flop_wrong | 1.000 | [1.000, 1.000] | 12 |
| hedged_correct | 1.000 | [1.000, 1.000] | 15 |
| real | 0.990 | [0.970, 1.000] | 100 |
| refusal | 1.000 | [1.000, 1.000] | 11 |
| right_number_wrong_reasoning | 1.000 | [1.000, 1.000] | 15 |
| verbose_wrong | 1.000 | [1.000, 1.000] | 9 |
| wrong_number | 1.000 | [1.000, 1.000] | 10 |

## Position bias (pairwise, both orders)

| scorer | mean | 95% CI | n |
|---|---|---|---|
| consistency_across_orders | 1.000 | [1.000, 1.000] | 80 |
| first_position_rate_when_inconsistent | nan | [nan, nan] | 0 |
| accuracy_correct_shown_first | 1.000 | [1.000, 1.000] | 80 |
| accuracy_correct_shown_second | 1.000 | [1.000, 1.000] | 80 |
| accuracy_when_wrong_is_long | 1.000 | [1.000, 1.000] | 80 |
| accuracy_when_wrong_is_short | 1.000 | [1.000, 1.000] | 80 |
| picked_longer_wrong_rate | 0.000 | [0.000, 0.000] | 80 |

80 pairs; 0 changed verdict when the order was swapped.

## Self-consistency (repeated sampling at temperature 0.7)

| scorer | mean | 95% CI | n |
|---|---|---|---|
| majority_agreement | 1.000 | [1.000, 1.000] | 40 |
| unanimity | 1.000 | [1.000, 1.000] | 40 |
| majority_vote_accuracy | 1.000 | [1.000, 1.000] | 40 |

40 items x 5 samples.

## Calibration of stated confidence

- ECE: 0.012
- Brier: 0.004
- AUROC of confidence as a correctness predictor: 1.000

| bin | n | mean confidence | accuracy |
|---|---|---|---|
| [0.0, 0.2] | 0 | - | - |
| [0.2, 0.4] | 0 | - | - |
| [0.4, 0.6] | 0 | - | - |
| [0.6, 0.8] | 0 | - | - |
| [0.8, 1.0] | 200 | 0.98 | 0.99 |
