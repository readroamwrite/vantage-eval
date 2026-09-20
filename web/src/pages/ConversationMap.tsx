/** Conversation map: a topic tree of one trajectory; clicking a node shows the messages it came from. */

import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, waitForJob, type ConvMap, type MapNode, type TrajectoryDetail } from "../api";
import { Empty, ErrorNote, Loading, useLoad, useLocal } from "../components";
import { RunPicker, useDefaultRun } from "../RunPicker";

export function ConversationMap() {
  const [params, setParams] = useSearchParams();
  const fallback = useDefaultRun();
  const runId = params.get("run") ? Number(params.get("run")) : fallback;
  const chosen = params.get("t") ? Number(params.get("t")) : null;
  const [stubs] = useLoad(() => (runId === null ? Promise.resolve([]) : api.runTrajectories(runId)), [runId]);
  const trajectoryId = chosen ?? (stubs.state === "ok" && stubs.data.length > 0 ? stubs.data[0].id : null);

  return (
    <>
      <div className="page-head">
        <h1>Conversation map</h1>
        <span className="context">
          Topics, directions and decisions, each pointing back at the messages it was built from. Import outside chats with{" "}
          <span className="mono">vantage import FILE</span>.
        </span>
      </div>
      <div className="controls">
        <RunPicker value={runId} onChange={(id) => setParams(id === null ? {} : { run: String(id) })} />
        {stubs.state === "ok" && stubs.data.length > 0 && (
          <label className="field">
            Trajectory
            <select value={trajectoryId ?? ""} onChange={(e) => setParams({ run: String(runId), t: e.target.value })}>
              {stubs.data.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.id}: {s.case_id}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {runId === null && <Empty>No runs stored yet.</Empty>}
      {stubs.state === "ok" && runId !== null && stubs.data.length === 0 && <Empty>This run has no trajectories.</Empty>}
      {trajectoryId !== null && <MapPanel trajectoryId={trajectoryId} />}
    </>
  );
}

function MapPanel({ trajectoryId }: { trajectoryId: number }) {
  const [model, setModel] = useLocal("vantage.judge_model", "ollama:qwen2.5:7b");
  const [cached, reload] = useLoad(() => api.convmap(trajectoryId, model), [trajectoryId, model]);
  const [detail] = useLoad(() => api.trajectory(trajectoryId), [trajectoryId]);
  const [building, setBuilding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<MapNode | null>(null);
  useEffect(() => setSelected(null), [trajectoryId]);

  const build = async (force: boolean) => {
    setBuilding(true);
    setError(null);
    try {
      const job = await api.buildConvmap(trajectoryId, { model, force });
      await waitForJob(job.id, () => undefined);
      reload();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBuilding(false);
    }
  };

  const map = cached.state === "ok" ? cached.data.map : null;
  return (
    <>
      <div className="controls">
        <label className="field">
          Model
          <input type="text" value={model} onChange={(e) => setModel(e.target.value)} />
        </label>
        <button type="button" className="primary" disabled={building} onClick={() => build(map !== null)}>
          {building ? "Mapping the conversation" : map ? "Rebuild map" : "Build map"}
        </button>
      </div>
      {error && <p className="error">Could not build the map: {error}</p>}
      {cached.state === "loading" && <Loading what="the stored map" />}
      {cached.state === "error" && <ErrorNote message={cached.message} />}
      {cached.state === "ok" && !map && !building && (
        <Empty>No map stored for this trajectory and model yet. Build one; it takes a few model calls and is then cached.</Empty>
      )}
      {map && (
        <>
          <Tree map={map} selected={selected} onSelect={setSelected} />
          {selected && detail.state === "ok" && <NodeDetail node={selected} detail={detail.data} />}
          {!selected && <p className="small muted">Click a node to see the messages it cites.</p>}
          <details>
            <summary>Mermaid, for a GitHub readme</summary>
            <pre>{"```mermaid\n" + (cached.state === "ok" ? (cached.data.mermaid ?? "") : "") + "\n```"}</pre>
          </details>
        </>
      )}
    </>
  );
}

interface Placed {
  node: MapNode;
  x: number;
  y: number;
  parent: Placed | null;
}

const NODE_W = 190;
const NODE_H = 40;
const COL = 240;
const ROW = 52;

/** Lay the tree out left to right: leaves get a row each, parents sit at their children's midpoint. */
function layout(root: MapNode): Placed[] {
  const placed: Placed[] = [];
  let row = 0;
  const visit = (node: MapNode, depth: number, parent: Placed | null): Placed => {
    const p: Placed = { node, x: depth * COL, y: 0, parent };
    if (node.children.length === 0) {
      p.y = row * ROW;
      row += 1;
    } else {
      const kids = node.children.map((c) => visit(c, depth + 1, p));
      p.y = (kids[0].y + kids[kids.length - 1].y) / 2;
    }
    placed.push(p);
    return p;
  };
  visit(root, 0, null);
  return placed;
}

function Tree({ map, selected, onSelect }: { map: ConvMap; selected: MapNode | null; onSelect: (n: MapNode) => void }) {
  const placed = useMemo(() => layout(map.root), [map]);
  const width = Math.max(...placed.map((p) => p.x)) + NODE_W + 8;
  const height = Math.max(...placed.map((p) => p.y)) + NODE_H + 8;
  return (
    <div className="tree table-scroll">
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="tree" aria-label="Conversation map">
        {placed
          .filter((p) => p.parent)
          .map((p, i) => {
            const x1 = p.parent!.x + NODE_W;
            const y1 = p.parent!.y + NODE_H / 2 + 4;
            const x2 = p.x;
            const y2 = p.y + NODE_H / 2 + 4;
            const mx = (x1 + x2) / 2;
            return <path key={i} className="edge" d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`} />;
          })}
        {placed.map((p, i) => (
          <g
            key={i}
            className={`node${p.node === selected ? " selected" : ""}`}
            transform={`translate(${p.x + 4},${p.y + 4})`}
            role="treeitem"
            tabIndex={0}
            aria-selected={p.node === selected}
            onClick={() => onSelect(p.node)}
            onKeyDown={(e) => e.key === "Enter" && onSelect(p.node)}
          >
            <title>
              {p.node.kind}: {p.node.title}
              {p.node.step_refs.length ? ` (steps ${p.node.step_refs.join(", ")})` : ""}
            </title>
            <rect width={NODE_W} height={NODE_H} />
            <text className="kind" x={8} y={14}>
              {p.node.kind}
              {p.node.step_refs.length ? ` · steps ${p.node.step_refs.slice(0, 4).join(", ")}${p.node.step_refs.length > 4 ? "…" : ""}` : ""}
            </text>
            <text x={8} y={30}>
              {p.node.title.length > 26 ? p.node.title.slice(0, 25) + "…" : p.node.title}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function NodeDetail({ node, detail }: { node: MapNode; detail: TrajectoryDetail }) {
  const steps = node.step_refs.filter((i) => i >= 0 && i < detail.trajectory.steps.length);
  return (
    <>
      <h2>
        <span className="muted">{node.kind}</span> {node.title}
      </h2>
      {node.summary && <p>{node.summary}</p>}
      {steps.length === 0 ? (
        <p className="muted small">This node cites no specific messages.</p>
      ) : (
        steps.map((i) => {
          const s = detail.trajectory.steps[i];
          return (
            <div key={i} style={{ marginBottom: 12 }}>
              <div className="small muted">
                <span className="mono">{String(i).padStart(2, "0")}</span> {s.role}
              </div>
              <pre>{s.content.slice(0, 2000)}</pre>
            </div>
          );
        })
      )}
      <p className="small">
        <Link to={`/trajectories/${detail.id}`}>Open the full transcript</Link>
      </p>
    </>
  );
}
