import json

import vantage.config as config
from vantage.experiments import benchmark
from vantage.experiments.benchmark import BenchmarkConfig, run_experiment
from vantage.models.mock import MockClient
from vantage.store import Store


def _rows(n: int = 40) -> list[dict]:
    rows = []
    for i in range(n):
        steps = i % 6
        calc = " ".join(f"<<{k}+1={k + 1}>>" for k in range(steps))
        rows.append(
            {
                "question": f"Sam has {i + 2} apples and buys {i + 3} more at the shop. How many apples does Sam have now?",
                "answer": f"{calc}\n#### {2 * i + 5}",
            }
        )
    return rows


def test_gsm8k_dataset_is_stratified_and_deterministic():
    ds = benchmark.gsm8k_dataset(_rows(), per_tier=3, seed=1)
    tiers = [c.tag_value("tier") for c in ds]
    assert sorted(set(tiers)) == ["1", "2", "3", "4", "5"] and tiers.count("3") == 3
    assert all(isinstance(c.expected, int) for c in ds)
    assert ds.ids == benchmark.gsm8k_dataset(_rows(), per_tier=3, seed=1).ids
    assert benchmark.tier_for_steps(0) == 1 and benchmark.tier_for_steps(9) == 5


def test_completion_probe_pairs_ids_and_perturbs_numbers():
    ds = benchmark.gsm8k_dataset(_rows(), per_tier=2, seed=0)
    verbatim, perturbed = benchmark.completion_datasets(ds, seed=0)
    assert verbatim.ids == ds.ids == perturbed.ids
    v, p = verbatim.cases[0], perturbed.cases[0]
    assert v.input + " " + v.expected == ds.cases[0].input
    assert p.input != v.input and p.system == benchmark.COMPLETION_SYSTEM
    assert "probe:completion" in p.tags and p.meta["condition"] == "perturbed"


def test_load_rows_from_file(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _rows(3)) + "\n")
    assert len(benchmark.load_gsm8k_rows(path)) == 3


async def test_experiment_end_to_end_with_mock(tmp_path, monkeypatch):
    rows = _rows()
    answers = {r["question"]: r["answer"].rsplit("####", 1)[1].strip() for r in rows}

    def responder(request):
        system = request.messages[0]["content"] if request.messages[0]["role"] == "system" else ""
        question = request.messages[-1]["content"]
        if "continuation" in system:
            return question.split(" and ")[-1] if "and" in question else "nothing"
        if "wrong" in system:
            return "Final answer: 1"
        want = answers.get(question)
        if want is None or int(question.split()[2]) % 7 == 0:
            return "I think it is 3. Final answer: 3"
        return f"Working: {want}. Final answer: {want}"

    monkeypatch.setattr(
        config, "_PROVIDERS", {**config._PROVIDERS, "fake": lambda m: MockClient(responder)}
    )
    ds = benchmark.gsm8k_dataset(rows, per_tier=3, seed=0)
    ds.save(tmp_path / "gsm8k.jsonl")
    cfg = BenchmarkConfig(
        dataset=tmp_path / "gsm8k.jsonl",
        target="fake:a",
        second="fake:b",
        repeats=2,
        out_dir=tmp_path / "r",
    )
    with Store(tmp_path / "t.db") as store:
        results = await run_experiment(store, cfg, progress=None)
        assert len(store.get_experiment("benchmark")["run_ids"]) == 5
    assert results["n_items"] == 15 and results["per_tier"] == 3
    assert results["compare"]["diff"].point < 0
    assert results["repeat_agreement"].point == 1.0
    assert results["memorisation"]["n_pairs"] == 15
    report = (tmp_path / "r" / "benchmark.md").read_text()
    assert "Memorisation probe" in report and "Sandbagging signal" in report
    assert "Warnings from `vantage diagnose`" in report
    assert (tmp_path / "r" / "benchmark_tiers.png").exists()
