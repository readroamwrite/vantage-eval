# Command line and dashboard

## Commands

| Command | What it does |
|---|---|
| `vantage run DATASET --target P:M --scorer S [--monitor M --halt] [--agent] [--repeats N] [--condition C] [--system TEXT\|FILE]` | Run and score. A run with the same name resumes where it left off. |
| `vantage rescore RUN --scorer S --monitor M` | Apply scorers and monitors to stored trajectories without calling the model under test. |
| `vantage runs list`, `vantage runs show RUN`, `vantage report RUN [--md FILE]` | Inspect runs. |
| `vantage compare A B [--scorer S]` | Paired difference with a confidence interval, and the items that flipped. |
| `vantage diagnose RUN [--against RUN...]` | Eval-health checks: `SATURATED`, `FLOORED`, `DEAD_ITEMS`, `LOW_DISCRIMINATION`, `NON_CAPABILITY_FAILURES`, `SANDBAG_ALARM`, `JUDGE_NOISE`, `COVERAGE_GAPS`. |
| `vantage coverage RUN [--monitor M...]` | Tag coverage, which threats the given monitors can structurally see, and measured detection rates. |
| `vantage search "RUBRIC" [--run R] [--fts TEXT] [--audit N] [--view output\|trajectory]` | Find stored trajectories that match a behaviour. Hits are stored as monitor events, and their precision is measured from human labels given for that rubric. With `--fts`, `--audit N` also judges N random trajectories the pre-filter excluded and estimates the pre-filter's recall. |
| `vantage agent TASK [--permissive]` | Run the built-in agent on one task and print the trajectory. |
| `vantage import FILE`, `vantage map TRAJECTORY [--mermaid FILE]` | Import a chat export or a plain transcript, then build a mind map of it in which every node points back to the messages it came from. |
| `vantage datagen judge\|robustness\|agent` | Regenerate the synthetic datasets. |
| `vantage datagen gsm8k [--source FILE\|URL] [--per-tier N]` | Download the GSM8K test split and write a stratified sample, tiered by the number of calculation steps in the reference solution. |
| `vantage experiment judge\|robustness\|monitoring\|benchmark` | Run an experiment end to end. Resumable and cached. |
| `vantage models check`, `vantage cache stats\|clear`, `vantage scorers` | Utilities. |
| `vantage dashboard [--db PATH] [--port N] [--host H]` | Start the dashboard, a local web app, on the given database. |

Every command accepts `--help`.

## Scorer and monitor specs

Scorers and monitors are named on the command line with strings of the form `name[:arg[,arg|key=value]...]`. Some examples:

- `numeric:0.01,relative=true`
- `contains:hello`
- `judge:data/rubrics/correct.md,model=ollama:qwen2.5:7b,view=output`
- `forbidden:write_file,test_*.py`

`vantage scorers` prints the available names. A judge spec accepts `view=output` or `view=trajectory` to choose what the judge is shown.

For agent runs, the ground-truth scorers read what the harness recorded, not what the agent said: `tampered` (a protected file changed: tests, fixtures or pytest configuration), `tests_passed` (the agent's own last test run), `final_tests_passed` (the harness reran the visible suite after the agent finished), `solved` (a held-out suite from the case's `meta["hidden_test"]` passed against the final implementation in a clean directory), `honest_success` (solved, no tampering, no false claim) and `fake_success` (the final message claims a result the harness contradicts).

## Datasets

Datasets can be JSON, JSONL or CSV. A case needs an `id` and an `input`. It may also have `expected`, `tags` (strings of the form `prefix:value`) and `system`, and anything else goes in `meta`, which is where agent tasks keep their files.

```jsonl
{"id": "q1", "input": "What is 17 * 3?", "expected": 51, "tags": ["cap:arithmetic", "tier:2"]}
{"id": "q2", "input": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, {"role": "user", "content": "2+2?"}], "expected": 4}
```

Two tag prefixes have special meaning. `cap:` names the capability an item exercises. `threat:` names the threat-model node an item probes; each threat declares whether its evidence appears in the final output or only in the trajectory, which is what lets `vantage coverage` say which threats a set of monitors can see at all.

## Database

The database defaults to `./vantage.db`. You can override it with `--db` or the `VANTAGE_DB` environment variable. It is a single SQLite file in WAL mode, so it is safe to read from the dashboard while a run is writing to it.

## Dashboard

`vantage dashboard` starts a small web server (uvicorn) on http://127.0.0.1:8501. It serves a JSON API under `/api` and a React app that reads from it. The built app is committed under `vantage/dashboard/static`, so using the dashboard needs no Node installation; the `[dashboard]` extra installs FastAPI, uvicorn and matplotlib. Developers who change the front end rebuild it with `cd web && npm install && npm run build`.

The app has eight pages:

1. **Runs.** Every run with per-scorer means and intervals.
2. **Run detail.** A per-case table with tag, status and flag filters, a failure-class chart, the eval-health summary and tag breakdowns.
3. **Trajectory.** The transcript step by step, with tool calls and results, monitor flags shown at the step where they fired, scores and judge rationales alongside, and the sandbox diff. A rail per monitor shows which steps that monitor could see and where it flagged.
4. **Review queue.** Trajectories ranked by monitor flags, scorer disagreement, low judge confidence and unparsed judge output, with a labelling form and a running human-vs-judge agreement and kappa.
5. **Compare.** The paired difference between two runs and the items that flipped.
6. **Experiments.** The experiment reports, rendered with their figures.
7. **Behaviour finder.** Rubric search with a full-text pre-filter. Hits become monitor events and go to the review queue. Each hit can be confirmed or rejected on the spot, which is what the precision figure is measured from, and an optional audit judges a random sample of what the pre-filter excluded to estimate its recall.
8. **Conversation map.** A topic tree of any stored or imported conversation. Clicking a node shows the exact messages it was built from.
