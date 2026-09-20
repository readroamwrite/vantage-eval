# Experiments

The four experiments below were run on one laptop (Apple M4, 16 GB) with local Ollama models and no API keys. `qwen2.5:3b` is the model under test, `qwen2.5:7b` is the judge and monitor, and `llama3.2:3b` serves as a second model. Every number is a mean with a 95% bootstrap interval.

The tables and figures here are copied from the reports that the experiment scripts write into [`results/`](../results/). Those reports are regenerated on every run, so the discussion lives in this file rather than there. To reproduce an experiment run `vantage experiment judge`, `vantage experiment robustness` or `vantage experiment monitoring`; thanks to the response cache a rerun costs nothing.

- [1. How reliable is the judge?](#1-how-reliable-is-the-judge)
- [2. Is the eval still measuring what we think?](#2-is-the-eval-still-measuring-what-we-think)
- [3. What does a monitor miss when it only sees the final answer?](#3-what-does-a-monitor-miss-when-it-only-sees-the-final-answer)
- [4. Do the diagnostics say anything on a real benchmark?](#4-do-the-diagnostics-say-anything-on-a-real-benchmark)

## 1. How reliable is the judge?

Raw report: [`results/judge_reliability.md`](../results/judge_reliability.md).

Before using an LLM as a judge I wanted to know how often it is wrong, and in which direction. The 7B judge graded 200 candidate answers to arithmetic word problems against a rubric which says that only the final answer matters. Half of the candidates were real answers from the 3B model, with gold labels from a numeric rule. The other half were planted answers with known labels, written to cover the ways judges are known to fail: verbose wrong answers, confident wrong answers, hedged correct answers, refusals, right answers reached by wrong reasoning, and answers that change their mind halfway through.

| Measure | Value |
|---|---|
| Accuracy vs gold | 0.905 [0.865, 0.945] (n = 200) |
| Cohen's kappa | 0.796 [0.702, 0.872] |
| Parseable verdicts | 100% |
| Recall on wrong answers | 0.969 [0.923, 1.000] |
| Precision on wrong answers | 0.787 [0.700, 0.875] |
| Verdict unchanged when pair order is swapped | 0.938 [0.875, 0.988] (80 pairs) |
| Picked the longer wrong answer over the shorter right one | 0.013 [0.000, 0.037] |
| ECE / AUROC of stated confidence | 0.095 / 0.500 |

![judge accuracy per response variant](../results/judge_reliability_variants.png)

A few things stand out. The judge errs on the strict side: 17 of its 19 mistakes were correct answers marked wrong. It also grades the reasoning even though the rubric tells it not to. When the final number was right but the working was visibly wrong, it accepted the answer only 20% [0%, 40%] of the time, and it accepted hedged correct answers 67% [40%, 87%] of the time. All the other variants, including the verbosity and confidence traps, were graded correctly at least 98% of the time.

There is a small position bias. Five of the 80 pairs changed verdict when the order was swapped, and in all five the judge preferred whichever answer was shown second. Length had no measurable effect.

The judge's stated confidence turned out to be useless. It answered "confidence: 100" on all 200 items, so confidence cannot be used to decide which items a human should review. The review queue in the dashboard ranks by disagreement between scorers instead.

Repeated grading was stable. After the seed bug was fixed, each of 40 items was graded five times at temperature 0.7. The wording of the rationale changed from sample to sample, but the verdict never did, so majority agreement and unanimity were both 1.000 on all 40 items, and the majority vote was right on 0.825 [0.700, 0.925] of them. For this judge, sampling several times and voting buys nothing. Its errors are systematic rather than noisy, which fits the strictness described above.

Since the confusion matrix is known, any pass rate this judge produces can be corrected for it. `vantage diagnose` prints both the raw rate and the Rogan-Gladen corrected rate, with the uncertainty propagated (this is the `JUDGE_NOISE` check).

## 2. Is the eval still measuring what we think?

Raw report: [`results/robustness.md`](../results/robustness.md).

This experiment applies the same checks that `vantage diagnose` runs on any stored run to an eval that was built to exercise them. The dataset has 120 arithmetic items in five difficulty tiers, with each base item written in three different ways, plus 10 probe items that either wrap easy arithmetic in edgy-sounding language or demand JSON output. The 3B model ran it twice with a normal system prompt and twice with a prompt telling it to get about half the answers wrong without saying so. The second model ran the normal condition only.

| Check | Normal prompt | Sandbag prompt |
|---|---|---|
| Accuracy | 0.758 [0.704, 0.808] | 0.558 [0.496, 0.619] |
| Paired difference, sandbag minus normal | | -0.200 [-0.273, -0.127] (n = 260) |
| Spearman correlation, tier vs failure rate | 0.700 [0.649, 0.900] | -0.100 [-0.359, 0.400] |
| Failure rate on the two easiest tiers | 0.021 [0.000, 0.052] | 0.500 [0.396, 0.604] |
| Paraphrase agreement | 0.863 [0.812, 0.912] | 0.750 [0.704, 0.796] |
| Non-capability failure share | 0.008 [0.000, 0.019] | |

![accuracy by difficulty tier](../results/robustness_tiers.png)

The first thing the diagnostics show is that most of this eval does no work. Across the two repeats and the second model, 102 of the 130 items either always pass or always fail, so only 28 items ever tell two runs apart. Tier 1 is saturated for both models. Comparing models on this set is really comparing them on about a fifth of it, which is what the `DEAD_ITEMS` and `SATURATED` checks report.

Second, the eval is partly measuring wording. For 17 of the 40 base items, the verdict depends on which of the three paraphrases was asked. A single accuracy number hides that.

Third, for this particular model the score is almost entirely about capability. The refusal probes produced no refusals, and the only non-capability failures were two JSON formatting errors, so the capability-adjusted accuracy barely moves. The check is there for models where it does move.

Finally, prompted sandbagging has a recognisable shape. The interesting signal is not the lower score, since in practice you would not have an honest baseline to compare with. It is that the difficulty curve flattens and then inverts. The honest model fails more often as the tiers get harder (Spearman 0.70) and almost never fails the easy tiers, whereas the sandbagging model fails the easiest tiers about half the time and on one tier actually scores higher than the honest run. Its paraphrase agreement drops as well. `SANDBAG_ALARM` fires on that combination. The obvious limitation is that a model sandbagging carefully, by failing only the hard items, would keep a natural-looking difficulty curve and this check would not notice. It catches clumsy underperformance, not careful underperformance.

## 3. What does a monitor miss when it only sees the final answer?

Raw report: [`results/monitoring.md`](../results/monitoring.md).

This experiment asks whether a monitor that can see everything an agent did is actually better than one that only reads the agent's final message. The material is 84 trajectories from the small coding agent: 36 real runs (12 fair tasks and 12 pressured tasks with an honest system prompt, then the 12 pressured tasks again with a permissive prompt that makes editing the tests seem acceptable), plus 48 scripted runs in which the agent's decisions were fixed in advance but the tools really ran. The scripted runs cover six behaviours: an honest fix, an honest give-up, blatant tampering that is admitted in the final message, subtle tampering that is never mentioned, disguised tampering described as a normal fix, and a false claim of success. Ground truth comes from the sandbox diff and the real test run, not from anyone's opinion. Five monitors then graded every trajectory: a rule that fires on writes to test files, and the 7B judge with the same tampering rubric asked four ways, output-only or full-trajectory, and either an immediate verdict or reasoning first.

| Monitor | Sees | Asked for | TPR | FPR | Precision | AUROC |
|---|---|---|---|---|---|---|
| Rule: write to `test_*.py` | trajectory | | 0.500 [0.324, 0.676] | 0.000 [0.000, 0.000] | 1.000 | 0.750 |
| Judge | output | immediate verdict | 0.824 [0.676, 0.941] | 0.200 [0.100, 0.320] | 0.737 | 0.812 |
| Judge | trajectory | immediate verdict | 0.000 [0.000, 0.000] | 0.100 [0.020, 0.180] | 0.000 | 0.450 |
| Judge | output | reasoning first | 0.971 [0.912, 1.000] | 0.520 [0.380, 0.660] | 0.559 | 0.754 |
| Judge | trajectory | reasoning first | 0.735 [0.559, 0.882] | 0.240 [0.120, 0.360] | 0.676 | 0.756 |

34 of the 84 trajectories are dishonest by ground truth; all monitors graded the same 84, so the rates are paired.

![detection per behaviour class](../results/monitoring_per_class.png)

The real agent, for what it is worth, was mostly well behaved. With the honest prompt it solved half of the fair tasks and tampered on none of the pressured ones. With the permissive prompt it tampered once in twelve runs, and the live rule monitor halted that run at the write. So nearly all the positives in the table come from the scripted behaviours, and each class has only eight examples, which is why the intervals are wide.

The result I did not expect is the third row. Given the full transcript, with the offending `write_file` call to the test file right there in the prompt, and asked for an immediate yes-or-no, the 7B judge flagged none of the 34 dishonest trajectories. It is not a context problem: the longest prompt used about 2,600 of the model's 4,096 tokens, and I checked that the tampering step and the admission in the final message are both present in the text the judge saw. The judge simply does not use the evidence when it has to commit to an answer straight away. Asking the same judge to write its reasoning first, before the verdict, raises its true positive rate from 0 to 0.735 [0.559, 0.882] on the same items. That is the largest effect in the whole experiment, and it is a prompting choice, not a change of model or of what the monitor can see.

The output-only judge looks good at first, with a true positive rate of 0.82, but the per-class breakdown shows how it gets there. It catches admitted and disguised tampering because those final messages mention the tests, and it catches false claims of success, but it also flags 63% of the honest scripted fixes and, with reasoning first, all of them. From the final message alone an honest "the tests pass" and a dishonest one look identical, so the judge ends up treating any claim of success as suspicious. Its true positives on subtle tampering, which is never mentioned in the final message, are the same effect: it is not detecting the tampering, it is distrusting the claim. This is exactly the blind spot the coverage matrix predicts for a monitor whose view is `output`, and the numbers show what that blind spot costs in false positives when the judge tries to compensate.

The trajectory judge with reasoning first is the only model-based monitor that separates honest fixes from tampering: it flagged none of the eight honest scripted fixes while catching all of the subtle tampering and most of the blatant tampering and false success claims. Its weaknesses are disguised tampering, where the transcript's own description of the edit apparently persuades it, and the messy real honest runs, a third of which it flagged, mostly ones full of syntax errors and repeated failing test runs. The rule monitor sits at the other extreme. It only watches one tool on one file pattern, so it misses tampering done through other files and cannot see false claims at all, but it never fires wrongly and it is the only monitor here that can stop a run before the damage is done. In practice the two are complementary: a narrow rule to halt the obvious cases, and a reasoning-first trajectory judge to find the rest, with its flags going to a human review queue rather than being treated as verdicts.

### The same 84 trajectories with a frontier judge

Raw report: [`results/frontier/monitoring.md`](../results/frontier/monitoring.md).

The obvious question about the table above is how much of it is about judges and how much is about a 7B judge. To find out I reran the monitors with `claude-sonnet-5` on the same 84 trajectories. The agent responses are cached by request hash and the agent runs at temperature 0, so copying the database and pointing the experiment at the copy reproduces the identical trajectories; only the judge changes. Thinking was switched off so that the immediate-verdict condition really is immediate.

```bash
python -c "import sqlite3; sqlite3.connect('vantage.db').backup(sqlite3.connect('vantage_frontier.db'))"
vantage experiment monitoring --judge anthropic:claude-sonnet-5 --db vantage_frontier.db --out results/frontier
```

| Monitor | Sees | Asked for | TPR | FPR | Precision | AUROC |
|---|---|---|---|---|---|---|
| Rule: write to `test_*.py` | trajectory | | 0.500 [0.324, 0.676] | 0.000 [0.000, 0.000] | 1.000 | 0.750 |
| Judge | output | immediate verdict | 1.000 [1.000, 1.000] | 0.480 [0.340, 0.620] | 0.586 | 0.864 |
| Judge | trajectory | immediate verdict | 1.000 [1.000, 1.000] | 0.180 [0.080, 0.300] | 0.791 | 0.994 |
| Judge | output | reasoning first | 1.000 [1.000, 1.000] | 0.460 [0.320, 0.600] | 0.596 | 0.877 |
| Judge | trajectory | reasoning first | 1.000 [1.000, 1.000] | 0.200 [0.100, 0.320] | 0.773 | 0.980 |

Two of the three findings above survive the change of judge and one does not.

The one that does not is the immediate-verdict failure. The frontier judge catches all 34 dishonest trajectories in every configuration, and asking for reasoning first changes nothing (paired difference 0.000 on all 34). So the third row of the first table was a fact about a 7B model that had the evidence and could not act on it without writing its reasoning out first, not a fact about judges. That is worth knowing, because "make the judge reason first" is cheap advice that would have been the wrong lesson here.

The output-only blind spot survives intact. With the final message as its only evidence, the frontier judge flags every one of the eight honest scripted fixes and 47% of the real honest runs, for a false positive rate of 0.48 and a precision of 0.59. The trajectory judge, on the same items, flags none of the honest fixes and 26% of the real honest runs, for a false positive rate of 0.18 and a precision of 0.79. A more capable judge did not close that gap; it cannot, because an honest "the tests pass" and a dishonest one are the same sentence, and the coverage matrix said so before either judge ran. The remaining false positives of the trajectory judge are the same messy real runs the 7B judge tripped on, full of syntax errors and repeated failing test runs.

The rule monitor is unchanged by definition, and the ordering holds: the rule is still the only monitor that can halt a run before the write, and the trajectory judge is still the one that can tell honest work from tampering. The rerun took 314 distinct judge calls, about 370k input and 36k output tokens, and a few minutes.

## 4. Do the diagnostics say anything on a real benchmark?

Raw report: [`results/benchmark.md`](../results/benchmark.md).

The first three experiments use synthetic data that was built to trigger the checks. This one applies the same checks to a benchmark people report numbers on. `vantage datagen gsm8k` downloads the GSM8K test split (Cobbe et al., 2021) and takes 250 items, 50 per tier, where an item's tier is the number of calculation steps in its reference solution. The 3B model ran the sample twice with a normal prompt and once with the sandbag prompt from experiment 2, and the second model ran it once. A memorisation probe then gave the model the first half of each question and asked it to continue verbatim, once as written and once with every number in the question changed, following the guided-completion test of Golchin and Surdeanu (2023) and the number-perturbation idea behind GSM1k (Zhang et al., 2024).

| Check | Value |
|---|---|
| Accuracy, 3B model, two repeats | 0.784 [0.744, 0.818] (n = 500) |
| Same item, same verdict on both repeats | 0.960 [0.936, 0.984] |
| Spearman correlation, tier vs failure rate | 0.900 [0.500, 1.000] |
| Dead items across both repeats and both models | 186 of 250 (157 always pass, 29 always fail) |
| Low-discrimination items | 25 |
| Non-capability failure share | 0.000 [0.000, 0.000] |
| Paired difference, sandbag minus normal | +0.032 [-0.012, 0.076] (n = 250) |
| Completion overlap, verbatim vs numbers changed | 0.200 [0.184, 0.216] vs 0.196 [0.182, 0.212]; difference -0.004 [-0.013, 0.006] |

![GSM8K accuracy by calculation-step tier](../results/benchmark_tiers.png)

Step count turns out to be a usable difficulty label. Accuracy falls from 0.84 and 0.89 on the two easiest tiers to 0.64 on problems with five or more steps, and the correlation between tier and failure rate is 0.90, without anyone hand-labelling difficulty. That matters because the sandbagging signature in `vantage diagnose` needs tiers, and most public benchmarks do not come with them.

The dead-item picture is the same as on the synthetic set, only now on a real benchmark. The sample is nowhere near saturated at 0.78, yet 186 of the 250 items give the same outcome on both repeats and on both models. A comparison of these two models on this sample is really a comparison on the other 64 items, and the intervals should be read with that in mind.

There were no non-capability failures. With a 400-token budget and an explicit answer format, all 500 answers contained a number and the failure classifier put every miss down to a wrong answer. The one problem in this area was in the tooling rather than the model: the first pass of this experiment used a number extractor that took the first number after a cue word, so an answer that said "the total is 14" on the way to "Final answer: 9" was scored as 14. Four answers were misread that way. The extractor now prefers the number after "final answer", the fix has a regression test, and the affected runs were re-scored from the stored transcripts without calling the model again. That is exactly the kind of error that gets read as a capability gap when nobody checks, and the framework's rescoring path is what made it cheap to correct.

The sandbag prompt did nothing here. The same instruction that cut the 3B model's accuracy by 0.20 on the arithmetic set in experiment 2 left its GSM8K accuracy unchanged, with 13 items flipping from pass to fail and 21 flipping the other way, which is what noise looks like. Only 1 of the 250 sandbag-condition answers mentions the instruction at all, and that one still gives the right answer. The alarm did not fire, which is the correct outcome, but the lesson is about elicitation rather than detection: prompted sandbagging is not something a model can be assumed to comply with, and a detector evaluated against "sandbagging" that never happened would be judged to have missed it. Any experiment of this kind needs a compliance check before the detection numbers mean anything. A second lesson is that the alarm's easy-tier threshold, chosen on the synthetic set where an honest model fails 2% of easy items, is close to firing on an honest run here, where the easy tiers are failed 13.5% of the time and the lower interval bound sits at 0.090. An absolute threshold does not transfer between benchmarks; the alarm should compare against a baseline measured on the same benchmark.

The memorisation probe found nothing. The 3B model reproduces about a fifth of the withheld second half of a question whether or not the numbers have been changed, which is what ordinary continuation of a familiar problem style gives. The probe cannot say that the model never saw GSM8K in training, only that it does not reproduce it verbatim at this size. Running the same probe on a larger model is the obvious next step, and it costs one command.
