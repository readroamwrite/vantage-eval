import json
from pathlib import Path

import pytest

import vantage.config as config
from vantage.cases import Case
from vantage.experiments import datagen
from vantage.experiments.judge_reliability import JudgeExperimentConfig, run_experiment
from vantage.models.mock import MockClient
from vantage.results import Verdict
from vantage.store import Store
from vantage.targets import ScriptedTarget
from vantage.trajectory import Step, Trajectory


def test_problems_are_deterministic_and_answers_match_text():
    a = datagen.generate_problems(14, seed=3)
    b = datagen.generate_problems(14, seed=3)
    assert [p.question for p in a] == [p.question for p in b]
    assert len({p.template for p in a}) == 7
    assert all(isinstance(p.answer, int) and p.answer > 0 for p in a)
    assert a[0].to_case().expected == a[0].answer


def test_planted_items_have_expected_labels_and_balance():
    problems = datagen.generate_problems(16, seed=0)
    items = datagen.planted_items(problems, seed=0)
    assert len(items) == 16
    labels = {c.meta["variant"]: c.meta["gold"] for c in items}
    assert labels["refusal"] == "fail" and labels["hedged_correct"] == "pass"
    assert set(labels) == set(datagen.VARIANTS)
    for c in items:
        if c.meta["variant"] in (
            "correct_reasoning",
            "right_number_wrong_reasoning",
            "hedged_correct",
        ):
            assert str(c.expected) in c.meta["response"]
    sample = datagen.balanced_sample(items, 8, seed=1)
    assert len(sample) == 8 and sum(c.meta["gold"] == "pass" for c in sample) == 4


def test_pairwise_pairs_alternate_length():
    pairs = datagen.pairwise_pairs(datagen.generate_problems(4), n=4)
    assert [p["long_wrong"] for p in pairs] == [False, True, False, True]
    assert len(pairs[1]["incorrect"]) > 3 * len(pairs[1]["correct"])


async def test_scripted_target_replays_and_halts():
    traj = Trajectory(
        "x", "t", steps=[Step("user", "q"), Step("assistant", "bad"), Step("assistant", "end")]
    )
    target = ScriptedTarget({"s": "hello", "t": traj})
    assert (await target.run(Case("s", "q"))).final_output == "hello"

    async def halt_on_bad(partial):
        return Verdict(
            "m", partial.steps[-1].content == "bad", halt=partial.steps[-1].content == "bad"
        )

    halted = await target.run(Case("t", "q"), on_step=halt_on_bad)
    assert halted.status == "halted" and halted.n_steps == 2 and halted.meta["halted_at_step"] == 1
    with pytest.raises(KeyError):
        await target.run(Case("missing", "q"))


async def test_experiment_runs_end_to_end_with_mocks(tmp_path, monkeypatch):
    def fake_judge(request):
        text = request.messages[-1]["content"]
        if "Response A" in text:
            return json.dumps({"verdict": "A", "confidence": 80})
        ref = text.split("Reference answer\n", 1)[1].split("\n", 1)[0].strip()
        resp = text.split("Model response\n", 1)[1].split("## Rubric", 1)[0]
        verdict = "pass" if ref in resp else "fail"
        return json.dumps({"verdict": verdict, "confidence": 95, "rationale": "checked"})

    monkeypatch.setattr(
        config, "_PROVIDERS", {**config._PROVIDERS, "fake": lambda m: MockClient(fake_judge)}
    )

    cfg = JudgeExperimentConfig(
        target="mock:arith",
        judge="fake:j",
        n_problems=16,
        n_items=16,
        n_pairs=4,
        n_consistency=4,
        n_samples=2,
        out_dir=tmp_path / "results",
    )
    with Store(tmp_path / "t.db") as store:
        results = await run_experiment(store, cfg, progress=None)
        assert store.get_experiment("judge_reliability") is not None
    assert results["agreement"]["n"] == 16
    assert results["agreement"]["accuracy"].point > 0.6
    assert results["pairwise"]["n_pairs"] == 4 and results["pairwise"]["consistency"].point == 0.0
    assert results["consistency"]["n_items"] == 4
    report = (tmp_path / "results" / "judge_reliability.md").read_text()
    assert "## Agreement with gold labels" in report and "cohen_kappa" in report
    assert Path(tmp_path / "results" / "judge_reliability_calibration.png").exists()
