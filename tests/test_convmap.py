import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import vantage.config as config
from vantage.cli import app
from vantage.convmap import ConvMap, Node, build, validate
from vantage.importers import import_chat, load_chat, parse_text_transcript
from vantage.models.mock import MockClient
from vantage.store import Store

SAMPLE = Path(__file__).parent.parent / "samples" / "chat.json"
TREE = {
    "title": "Reading tracker app",
    "kind": "topic",
    "summary": "Planning a small app.",
    "steps": [0],
    "children": [
        {
            "title": "Database choice",
            "kind": "topic",
            "steps": [0, 1],
            "children": [{"title": "Use SQLite", "kind": "decision", "steps": [2]}],
        },
        {
            "title": "Frontend",
            "kind": "direction",
            "steps": [2, 3],
            "children": [{"title": "Templates", "kind": "decision", "steps": [4], "children": []}],
        },
        {"title": "Backups", "kind": "question", "steps": [4, 5, 99], "children": []},
        {"title": "Schema (draft)", "kind": "outcome", "steps": [7]},
    ],
}


def test_importers_read_json_and_text(tmp_path):
    traj = load_chat(SAMPLE)
    assert (
        traj.n_steps == 8 and traj.steps[0].role == "user" and traj.final_output.startswith("Sure")
    )
    text = tmp_path / "t.txt"
    text.write_text("User: hi\nthere\nAssistant: hello\nHuman: bye\nAI: ok")
    steps = parse_text_transcript(text.read_text())
    assert [s.role for s in steps] == ["user", "assistant", "user", "assistant"] and steps[
        0
    ].content == "hi\nthere"
    with pytest.raises(ValueError):
        parse_text_transcript("no prefixes here")
    with Store(":memory:") as store:
        run_id, traj_id = import_chat(store, SAMPLE)
        assert store.get_run(run_id)["name"] == "imported/chat"
        assert store.get_trajectory(traj_id).trajectory.n_steps == 8
        assert import_chat(store, SAMPLE)[0] == run_id  # idempotent


def test_validate_and_renderers():
    node = validate(Node.from_dict(TREE), 8)
    backups = node.children[2]
    assert backups.step_refs == [4, 5]  # 99 dropped
    conv = ConvMap(node, "m", 8)
    mermaid = conv.to_mermaid()
    assert (
        mermaid.startswith("mindmap\n  root((Reading tracker app))")
        and "Schema Schema" not in mermaid
    )
    assert "Use SQLite [2]" in mermaid and "(" not in mermaid.splitlines()[-1]
    dot = conv.to_dot()
    assert dot.count("->") == 6 and "steps 4, 5" in dot
    assert ConvMap.from_dict(conv.to_dict()).to_mermaid() == mermaid


async def test_build_single_and_chunked():
    traj = load_chat(SAMPLE)
    client = MockClient(default=json.dumps(TREE))
    conv = await build(traj, client, "m")
    assert conv.n_steps == 8 and len(conv.root.walk()) == 7 and len(client.calls) == 1
    chunked = await build(traj, MockClient(default=json.dumps(TREE)), "m", chunk_steps=3)
    assert chunked.root.title == "Reading tracker app"
    assert {r for n in chunked.root.walk() for r in n.step_refs} <= set(range(8))
    with pytest.raises(ValueError):
        await build(traj, MockClient(default="not json"), "m")


def test_import_and_map_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config,
        "_PROVIDERS",
        {**config._PROVIDERS, "fake": lambda m: MockClient(default=json.dumps(TREE))},
    )
    db = tmp_path / "t.db"
    runner = CliRunner()
    res = runner.invoke(app, ["import", str(SAMPLE), "--db", str(db)])
    assert res.exit_code == 0 and "trajectory 1" in res.output
    out = tmp_path / "map.md"
    res = runner.invoke(
        app, ["map", "1", "--model", "fake:m", "--mermaid", str(out), "--db", str(db)]
    )
    assert res.exit_code == 0, res.output
    assert out.read_text().startswith("```mermaid\nmindmap")
    res = runner.invoke(app, ["map", "1", "--model", "fake:m", "--db", str(db)])
    assert res.exit_code == 0 and "7 nodes" in res.output
