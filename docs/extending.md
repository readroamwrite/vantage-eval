# Extending vantage

Each extension point is a small protocol, so there are no base classes to inherit from. The code examples on this page are run by `tests/test_readme_snippets.py`, so they should keep working.

## A target

A target has an `id`, a `config()` method used for provenance, and an async `run(case, on_step)` that returns a `Trajectory`. Monitors watch a run in progress through `on_step`; a target that has nothing to stream can ignore it.

```python
from vantage import Case, Step, Trajectory


class MyChatbot:
    id = "my-chatbot"

    def config(self):
        return {"kind": "http", "endpoint": "https://.../chat"}

    async def run(self, case: Case, on_step=None) -> Trajectory:
        reply = await call_my_service(case.messages())  # your code
        return Trajectory.single_turn(case.id, self.id, case.prompt_text, reply)
```

For an agent, append a `Step` for each message and tool call, and pass the partial trajectory to `on_step` after each one so that a monitor gets the chance to halt it. The reference loop is in `vantage/agent/react.py`.

## A scorer

A scorer has a `name` and an async `score(traj, case) -> Score`. Once registered it can be used from the CLI as a spec string.

```python
from vantage.results import Score
from vantage.scorers import register


@register("mentions_source")
class MentionsSource:
    name = "mentions_source"

    async def score(self, traj, case):
        ok = "source:" in traj.answer.lower()
        return Score(self.name, 1.0 if ok else 0.0, passed=ok)
```

```bash
vantage run data/my.jsonl --scorer mentions_source
```

## A monitor

A monitor declares its `view`, says whether it wants to see every step, and returns a `Verdict` that may halt the run. The simplest way to get one is to wrap a scorer:

```python
from vantage.monitors import RuleMonitor
from vantage.scorers import get_scorer

guard = RuleMonitor(get_scorer("forbidden:write_file,test_*.py"), halt_on_flag=True)
```

A judge monitor asks a model instead, from whichever vantage point you choose:

```python
from vantage.judge import Judge
from vantage.monitors import JudgeMonitor

judge = Judge(client, "qwen2.5:7b", "data/rubrics/tampering.md")
output_only = JudgeMonitor(judge, view="output")  # sees the task and the final message
full = JudgeMonitor(judge, view="trajectory")  # sees every step and tool call
```

## A model provider

A model provider implements one method, `chat(ChatRequest) -> ChatResponse`, and registers under a name so that `provider:model` specs work everywhere:

```python
from vantage.config import register_provider


@register_provider("myprovider")
def make_my_client(model: str):
    return MyClient(model)
```

## Using vantage as a library

```python
import asyncio
from vantage import Dataset, Runner, Store
from vantage.config import make_target
from vantage.scorers import get_scorer


async def main():
    with Store("vantage.db") as store:
        target = make_target("ollama:qwen2.5:3b", store)
        result = await Runner(store).run(
            target, Dataset.load("data/smoke.jsonl"), [get_scorer("numeric")]
        )
        print(result.metric("numeric"))  # 1.000 [1.000, 1.000]


asyncio.run(main())
```

The names exported from the top-level `vantage` package are the ones most integrations need: `Case`, `Dataset`, `Trajectory`, `Step`, `ToolCall`, `Score`, `Verdict`, `Annotation`, `Runner`, `RunResult`, `CaseResult`, `Store`, `Target`, `LLMTarget` and `ScriptedTarget`. Deeper modules such as `vantage.judge`, `vantage.monitors` and `vantage.stats` are imported directly.
