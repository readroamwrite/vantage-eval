/** Review queue: trajectories ranked by how much they need a human, and the label form. */

import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useShell } from "../App";
import { api, fmt } from "../api";
import { Clamp, Empty, ErrorNote, Loading, useLoad, useLocal } from "../components";
import { RunPicker, useDefaultRun } from "../RunPicker";

export function ReviewQueue() {
  const [params, setParams] = useSearchParams();
  const fallback = useDefaultRun();
  const runId = params.get("run") ? Number(params.get("run")) : fallback;
  const selected = params.get("t") ? Number(params.get("t")) : null;
  const setRun = (id: number | null) => setParams(id === null ? {} : { run: String(id) });
  const setSelected = (id: number) => setParams({ run: String(runId), t: String(id) });
  const { runs } = useShell();

  if (runs.state === "ok" && runs.data.length === 0)
    return (
      <>
        <h1>Review queue</h1>
        <Empty>No runs stored yet. Run one first, then come back to label its trajectories.</Empty>
      </>
    );
  return (
    <>
      <div className="page-head">
        <h1>Review queue</h1>
        <span className="context">Ranked by monitor flags, scorer disagreement, low judge confidence and unparsed judge output.</span>
      </div>
      <div className="controls">
        <RunPicker value={runId} onChange={setRun} />
      </div>
      {runId !== null && <Queue runId={runId} selected={selected} onSelect={setSelected} />}
    </>
  );
}

function Queue({ runId, selected, onSelect }: { runId: number; selected: number | null; onSelect: (id: number) => void }) {
  const [load, reload] = useLoad(() => api.review(runId), [runId]);
  if (load.state === "loading") return <Loading what="the queue" />;
  if (load.state === "error") return <ErrorNote message={load.message} />;
  const { queue, agreement } = load.data;
  const pending = queue.filter((q) => !q.reviewed).length;
  return (
    <>
      <p className="small muted">
        {pending} of {queue.length} trajectories still unlabelled.
      </p>
      <div className="table-scroll">
        <table className="data">
          <thead>
            <tr>
              <th className="num">id</th>
              <th>case</th>
              <th className="num">priority</th>
              <th>why it is here</th>
              <th>label</th>
            </tr>
          </thead>
          <tbody>
            {queue.map((q) => (
              <tr
                key={q.trajectory_id}
                className={`clickable${q.trajectory_id === selected ? " selected" : ""}`}
                tabIndex={0}
                onClick={() => onSelect(q.trajectory_id)}
                onKeyDown={(e) => e.key === "Enter" && onSelect(q.trajectory_id)}
              >
                <td className="num">{q.trajectory_id}</td>
                <td className="mono">{q.case_id}</td>
                <td className="num">{q.priority}</td>
                <td className="wrap small">{q.reasons.length ? q.reasons.join("; ") : <span className="faint">nothing unusual</span>}</td>
                <td>
                  {q.reviewed ? (
                    <>
                      {q.label} <span className="faint small">by {q.reviewer}</span>
                    </>
                  ) : (
                    <span className="faint">–</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selected !== null && <LabelForm trajectoryId={selected} onSaved={reload} />}

      <h2>Human against judge</h2>
      {agreement.length === 0 ? (
        <p className="muted">This run has no judge scorer, so there is nothing to compare labels against.</p>
      ) : (
        <div className="table-scroll" style={{ maxWidth: 600 }}>
        <table className="data">
          <thead>
            <tr>
              <th>judge</th>
              <th className="num">labelled</th>
              <th className="num">agreement</th>
              <th className="num">Cohen's kappa</th>
            </tr>
          </thead>
          <tbody>
            {agreement.map((a) => (
              <tr key={a.scorer}>
                <td className="mono">{a.scorer}</td>
                <td className="num">{a.n}</td>
                <td className="num">{a.agreement !== undefined ? fmt(a.agreement, 2) : <span className="faint">needs 2 pass/fail labels</span>}</td>
                <td className="num">{a.kappa !== undefined ? fmt(a.kappa, 2) : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      )}
    </>
  );
}

function LabelForm({ trajectoryId, onSaved }: { trajectoryId: number; onSaved: () => void }) {
  const [load, reload] = useLoad(() => api.trajectory(trajectoryId), [trajectoryId]);
  const { meta } = useShell();
  const labels = meta.state === "ok" ? meta.data.labels : ["pass", "fail", "tampered", "clean", "unsure"];
  const [reviewer, setReviewer] = useLocal("vantage.reviewer", "");
  const [label, setLabel] = useState(labels[0]);
  const [note, setNote] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  useEffect(() => {
    setNote("");
    setStatus(null);
  }, [trajectoryId]);

  if (load.state === "loading") return <Loading what={`trajectory ${trajectoryId}`} />;
  if (load.state === "error") return <ErrorNote message={load.message} />;
  const d = load.data;

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!reviewer.trim()) {
      setStatus("Enter a reviewer name first.");
      return;
    }
    try {
      await api.annotate({ trajectory_id: trajectoryId, reviewer: reviewer.trim(), label, note });
      setStatus(`Saved ${label} for trajectory ${trajectoryId}.`);
      setNote("");
      reload();
      onSaved();
    } catch (err) {
      setStatus(`Could not save: ${(err as Error).message}`);
    }
  };

  return (
    <>
      <h2>
        Trajectory {trajectoryId}, case <span className="mono">{d.case.id}</span>{" "}
        <Link to={`/trajectories/${trajectoryId}`} className="small" style={{ fontWeight: 400 }}>
          open the full transcript
        </Link>
      </h2>
      <div className="two-col">
        <div>
          <h3>Task</h3>
          <Clamp text={d.case.prompt_text} limit={700} />
          {d.case.expected !== null && d.case.expected !== undefined && (
            <p className="small muted" style={{ marginTop: 6 }}>
              Expected: <span className="mono">{typeof d.case.expected === "string" ? d.case.expected : JSON.stringify(d.case.expected)}</span>
            </p>
          )}
          <h3>Final output</h3>
          <Clamp text={d.trajectory.final_output ?? lastAssistant(d) ?? "(none)"} limit={700} />
          <h3>Scores</h3>
          {d.scores.map((s, i) => (
            <div key={i} className="small" style={{ marginBottom: 4 }}>
              <span className="mono">{s.scorer}</span> {fmt(s.value, 2)} {s.label ?? ""}
              {s.rationale ? <span className="muted"> {s.rationale}</span> : null}
            </div>
          ))}
        </div>
        <form onSubmit={save}>
          <h3>Your label</h3>
          <label className="field" style={{ marginBottom: 12 }}>
            Reviewer
            <input type="text" value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="your name" />
          </label>
          <div className="radio-row" style={{ marginBottom: 12 }}>
            {labels.map((l) => (
              <label key={l}>
                <input type="radio" name="label" value={l} checked={label === l} onChange={() => setLabel(l)} />
                {l}
              </label>
            ))}
          </div>
          <label className="field" style={{ marginBottom: 12 }}>
            Note
            <textarea rows={3} value={note} onChange={(e) => setNote(e.target.value)} />
          </label>
          <button type="submit" className="primary">
            Save label
          </button>
          {status && <p className="small" style={{ marginTop: 10 }}>{status}</p>}
          {d.annotations.length > 0 && (
            <p className="small muted" style={{ marginTop: 12 }}>
              Already labelled {d.annotations.map((a) => `${a.label} by ${a.reviewer}`).join(", ")}.
            </p>
          )}
        </form>
      </div>
    </>
  );
}

function lastAssistant(d: { trajectory: { steps: { role: string; content: string; meta: Record<string, unknown> }[] } }): string | null {
  for (let i = d.trajectory.steps.length - 1; i >= 0; i--) {
    const step = d.trajectory.steps[i];
    if (step.role !== "assistant") continue;
    // Agent steps carry a JSON tool call as content; the thought reads better.
    if (step.content.trim().startsWith("{") && typeof step.meta.thought === "string") return step.meta.thought;
    return step.content;
  }
  return null;
}
