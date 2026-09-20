"""HTTP API and static file server for the dashboard.

``create_app(db_path)`` returns a FastAPI application. Everything under
``/api`` reads the store; the built React app under ``static/`` is served for
every other path so client-side routes work on reload.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vantage import __version__
from vantage.dashboard import queries
from vantage.dashboard.jobs import JobRegistry
from vantage.results import Annotation
from vantage.store import Store

STATIC_DIR = Path(__file__).parent / "static"


class AnnotationIn(BaseModel):
    """Body of ``POST /api/annotations``."""

    trajectory_id: int
    reviewer: str = Field(min_length=1)
    label: str
    note: str = ""
    monitor: str | None = None


class FinderIn(BaseModel):
    """Body of ``POST /api/finder``."""

    rubric: str = Field(min_length=3)
    run_id: int | None = None
    fts: str | None = None
    limit: int = Field(default=30, ge=1, le=500)
    model: str = "ollama:qwen2.5:7b"
    view: str = "trajectory"
    audit: int = Field(default=0, ge=0, le=200)


class ConvMapIn(BaseModel):
    """Body of ``POST /api/convmap/{trajectory_id}``."""

    model: str = "ollama:qwen2.5:7b"
    force: bool = False


def create_app(db_path: str) -> FastAPI:
    """Build the dashboard application for one database.

    Args:
        db_path: SQLite file the dashboard reads and writes.
    """
    store = Store(db_path)
    jobs = JobRegistry(db_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        store.close()

    app = FastAPI(title="vantage dashboard", version=__version__, lifespan=lifespan)
    app.state.store = store

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {"db": db_path, "version": __version__, "labels": list(queries.REVIEW_LABELS)}

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        return queries.list_runs(store)

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: int) -> dict[str, Any]:
        try:
            return queries.run_detail(store, run_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/runs/{run_id}/trajectories")
    def run_trajectories(run_id: int) -> list[dict[str, Any]]:
        return queries.trajectory_list(store, run_id)

    @app.get("/api/runs/{run_id}/review")
    def review(run_id: int) -> dict[str, Any]:
        return {
            "queue": queries.review_queue(store, run_id),
            "agreement": queries.agreement(store, run_id),
        }

    @app.get("/api/trajectories/{trajectory_id}")
    def trajectory(trajectory_id: int) -> dict[str, Any]:
        try:
            return queries.trajectory_detail(store, trajectory_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/annotations")
    def annotate(body: AnnotationIn) -> dict[str, Any]:
        if body.label not in queries.REVIEW_LABELS:
            raise HTTPException(422, f"label must be one of {queries.REVIEW_LABELS}")
        if queries.stored_or_none(store, body.trajectory_id) is None:
            raise HTTPException(404, f"trajectory {body.trajectory_id} not found")
        annotation = Annotation(
            body.trajectory_id,
            body.reviewer.strip(),
            body.label,
            value=queries.LABEL_VALUES[body.label],
            note=body.note,
            monitor=body.monitor,
        )
        return {"id": store.save_annotation(annotation), "label": body.label}

    @app.get("/api/compare")
    def compare(a: int, b: int) -> dict[str, Any]:
        try:
            return queries.compare(store, a, b)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/experiments")
    def experiments() -> list[dict[str, Any]]:
        return [
            {
                "name": e["name"],
                "created_at": e["created_at"],
                "run_ids": e["run_ids"],
                "params": queries.jsonable(e["params"]),
            }
            for e in store.list_experiments()
        ]

    @app.get("/api/experiments/{name}")
    def experiment(name: str) -> dict[str, Any]:
        detail = queries.experiment_detail(store, name)
        if detail is None:
            raise HTTPException(404, f"experiment {name!r} not found")
        return detail

    @app.get("/api/experiments/{name}/figures/{filename}")
    def figure(name: str, filename: str) -> FileResponse:
        path = queries.figure_path(store, name, filename)
        if path is None:
            raise HTTPException(404, f"figure {filename!r} not found")
        return FileResponse(path)

    @app.post("/api/finder")
    def finder(body: FinderIn) -> dict[str, Any]:
        from vantage.config import make_client, parse_model_spec
        from vantage.finder import find_behaviour, result_to_dict

        if body.view not in ("trajectory", "output"):
            raise HTTPException(422, "view must be 'trajectory' or 'output'")

        async def run(job_store: Store, progress: Any) -> dict[str, Any]:
            result = await find_behaviour(
                job_store,
                body.rubric,
                make_client(body.model, job_store),
                parse_model_spec(body.model)[1],
                run_id=body.run_id,
                fts=body.fts or None,
                limit=body.limit,
                view=body.view,  # type: ignore[arg-type]
                audit=body.audit,
                progress=progress,
            )
            out = result_to_dict(result)
            out["non_hits"] = [asdict(h) for h in result.non_hits]
            return out

        return jobs.start("finder", run).to_dict()

    @app.get("/api/convmap/{trajectory_id}")
    def convmap(trajectory_id: int, model: str = "ollama:qwen2.5:7b") -> dict[str, Any]:
        from vantage.config import parse_model_spec

        cached = store.get_conv_map(trajectory_id, parse_model_spec(model)[1])
        if cached is None:
            return {"map": None}
        return {
            "map": cached["tree"],
            "mermaid": cached["mermaid"],
            "created_at": cached["created_at"],
        }

    @app.post("/api/convmap/{trajectory_id}")
    def build_convmap(trajectory_id: int, body: ConvMapIn) -> dict[str, Any]:
        from vantage.config import make_client, parse_model_spec
        from vantage.convmap import build

        if queries.stored_or_none(store, trajectory_id) is None:
            raise HTTPException(404, f"trajectory {trajectory_id} not found")
        model_name = parse_model_spec(body.model)[1]

        async def run(job_store: Store, progress: Any) -> dict[str, Any]:
            cached = None if body.force else job_store.get_conv_map(trajectory_id, model_name)
            if cached is not None:
                return {"map": cached["tree"], "mermaid": cached["mermaid"]}
            stored = job_store.get_trajectory(trajectory_id)
            conv = await build(stored.trajectory, make_client(body.model, job_store), model_name)
            job_store.save_conv_map(trajectory_id, model_name, conv.to_dict(), conv.to_mermaid())
            return {"map": conv.to_dict(), "mermaid": conv.to_mermaid()}

        return jobs.start("convmap", run).to_dict()

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        found = jobs.get(job_id)
        if found is None:
            raise HTTPException(404, f"job {job_id!r} not found")
        return found.to_dict()

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built React app, or a short notice when it has not been built."""
    index = STATIC_DIR / "index.html"
    if index.is_file():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> Any:
        if path.startswith("api/"):
            raise HTTPException(404, "no such endpoint")
        if index.is_file():
            candidate = STATIC_DIR / path
            if path and candidate.is_file() and candidate.resolve().is_relative_to(STATIC_DIR):
                return FileResponse(candidate)
            return FileResponse(index)
        return JSONResponse(
            {"detail": "dashboard frontend not built; run `npm run build` in web/"}, status_code=503
        )
