import vantage.config as config
from vantage.config import arithmetic_mock
from vantage.experiments import datagen
from vantage.experiments.robustness import RobustnessConfig, run_experiment
from vantage.models.mock import MockClient
from vantage.store import Store


def test_tiered_items_shape_and_tags():
    cases = datagen.tiered_items(seed=1, per_tier=2)
    assert len(cases) == 5 * 2 * 3
    tiers = {c.tag_value("tier") for c in cases}
    assert tiers == {"1", "2", "3", "4", "5"}
    bases = {c.tag_value("base") for c in cases}
    assert len(bases) == 10
    assert all(c.expected == cases[0].expected for c in cases[:3])  # paraphrases share the answer
    assert datagen.tiered_items(seed=1, per_tier=2)[7].input == cases[7].input


def test_probes_are_tagged():
    probes = datagen.probe_items()
    assert sum(c.has_tag("probe:refusal_trap") for c in probes) == 5
    assert sum(c.has_tag("format:json") for c in probes) == 5
    assert len(datagen.robustness_dataset(0, 2)) == 30 + 10


async def test_experiment_end_to_end_with_mock(tmp_path, monkeypatch):
    def responder(request):
        system = request.messages[0]["content"] if request.messages[0]["role"] == "system" else ""
        if "wrong" in system:
            return "Final answer: 1"
        return arithmetic_mock(request)

    monkeypatch.setattr(
        config, "_PROVIDERS", {**config._PROVIDERS, "fake": lambda m: MockClient(responder)}
    )
    cfg = RobustnessConfig(
        target="fake:a", second="fake:b", per_tier=2, repeats=1, out_dir=tmp_path / "r"
    )
    with Store(tmp_path / "t.db") as store:
        results = await run_experiment(store, cfg, progress=None)
        assert store.get_experiment("robustness")["run_ids"].__len__() == 3
    assert results["n_items"] == 30 and results["n_probes"] == 10
    assert results["compare"]["diff"].point < 0
    assert (
        results["normal"]["difficulty"]["per_tier"][1].point
        > results["normal"]["difficulty"]["per_tier"][4].point
    )
    report = (tmp_path / "r" / "robustness.md").read_text()
    assert "Sandbagging signal" in report and "capability_adjusted_accuracy" in report
    assert (tmp_path / "r" / "robustness_tiers.png").exists()
