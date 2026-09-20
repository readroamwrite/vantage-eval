/**
 * Trajectory: the transcript step by step, with a sightline rail per monitor
 * showing which steps that monitor was allowed to see and where it flagged.
 */

import { Fragment, useMemo } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, fmt, type Step, type TrajectoryDetail, type VerdictRow } from "../api";
import { Clamp, ErrorNote, KeyValues, Loading, useLoad } from "../components";

/** Step indices an output-only observer sees, mirroring Trajectory.output_only(). */
function outputView(steps: Step[]): Set<number> {
  const seen = new Set<number>();
  const system = steps.findIndex((s) => s.role === "system");
  const user = steps.findIndex((s) => s.role === "user");
  let last = -1;
  for (let i = steps.length - 1; i >= 0; i--)
    if (steps[i].role === "assistant") {
      last = i;
      break;
    }
  for (const i of [system, user, last]) if (i >= 0) seen.add(i);
  return seen;
}

export function TrajectoryPage() {
  const id = Number(useParams().id);
  const [load] = useLoad(() => api.trajectory(id), [id]);
  if (load.state === "loading") return <Loading what={`trajectory ${id}`} />;
  if (load.state === "error") return <ErrorNote message={load.message} />;
  return <Viewer detail={load.data} />;
}

function Viewer({ detail }: { detail: TrajectoryDetail }) {
  const navigate = useNavigate();
  const traj = detail.trajectory;
  const steps = traj.steps;
  const monitors = detail.monitors;
  const outputSeen = useMemo(() => outputView(steps), [steps]);
  const covered = (monitorIndex: number, stepIndex: number) =>
    monitors[monitorIndex].view === "trajectory" || outputSeen.has(stepIndex);
  const verdictsAt = (stepIndex: number) => detail.verdicts.filter((v) => v.step_idx === stepIndex);
  const finalVerdicts = verdictsAt(-1);
  const siblingIndex = detail.siblings.findIndex((s) => s.id === detail.id);
  const prev = detail.siblings[siblingIndex - 1];
  const next = detail.siblings[siblingIndex + 1];
  const railWidth = monitors.length > 0 ? monitors.length * 10 + (monitors.length - 1) * 10 : 0;
  const toolCalls = steps.reduce((n, s) => n + s.tool_calls.length, 0);

  return (
    <>
      <div className="page-head">
        <div>
          <h1>
            Trajectory {detail.id} <span className="muted">of</span> <Link to={`/runs/${detail.run.id}`}>{detail.run.name}</Link>
          </h1>
          <div className="context" style={{ marginTop: 6 }}>
            Case <span className="mono">{detail.case.id}</span>
            {detail.repeat > 0 ? <> repeat {detail.repeat}</> : null}. Status {traj.status}, {steps.length} steps, {toolCalls} tool calls.{" "}
            {detail.case.tags.map((t) => (
              <span key={t} className="tag">
                {t}
              </span>
            ))}
          </div>
        </div>
        <div className="controls" style={{ margin: 0 }}>
          <button type="button" disabled={!prev} onClick={() => prev && navigate(`/trajectories/${prev.id}`)}>
            Previous
          </button>
          <select value={detail.id} onChange={(e) => navigate(`/trajectories/${e.target.value}`)} aria-label="Trajectory in this run">
            {detail.siblings.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id}: {s.case_id}
              </option>
            ))}
          </select>
          <button type="button" disabled={!next} onClick={() => next && navigate(`/trajectories/${next.id}`)}>
            Next
          </button>
          <Link to={`/review?run=${detail.run.id}&t=${detail.id}`} style={{ alignSelf: "center" }}>
            Label in review queue
          </Link>
        </div>
      </div>

      {traj.error && <p className="error">{traj.error}</p>}

      <div className="traj-layout">
        <div className="sightlines" style={{ "--rails": `${railWidth}px` } as React.CSSProperties}>
          <div className="legend">
            {monitors.length === 0 ? (
              <span>No monitor observed this trajectory. Rescore the run with a monitor to draw its sightline here.</span>
            ) : (
              monitors.map((m) => (
                <span key={m.name} className="m" title={m.view === "output" ? "sees the task and the final message only" : "sees every step"}>
                  <span className={`key${m.view === "output" ? " short" : ""}`} />
                  <span className="name">{m.name}</span>
                  <span>{m.view === "output" ? "final output only" : "every step"}</span>
                </span>
              ))
            )}
          </div>
          {steps.map((step, i) => (
            <StepRow key={i} index={i} step={step} monitors={monitors} covered={(m) => covered(m, i)} verdicts={verdictsAt(i)} />
          ))}
          <div className="step-row final-row">
            <div className="rails">
              {monitors.map((m, mi) => {
                const own = finalVerdicts.filter((v) => v.monitor === m.name);
                const flagged = own.find((v) => v.flagged);
                return (
                  <div key={m.name} className="rail-col">
                    <div className={`line end${covered(mi, steps.length - 1) ? "" : " off"}`} />
                    {flagged ? (
                      <div className={`dot${flagged.halt ? " halt" : ""}`} title={`${m.name} flagged on the finished trajectory: ${flagged.reason}`} />
                    ) : own.length > 0 ? (
                      <div className="dot calm" title={`${m.name} did not flag`} />
                    ) : null}
                  </div>
                );
              })}
            </div>
            <div className="step-idx">end</div>
            <div className="step-role">final output</div>
            <div className="step-body">
              {traj.final_output !== null ? <pre>{traj.final_output}</pre> : <span className="muted">Nothing was submitted.</span>}
              {finalVerdicts
                .filter((v) => v.flagged)
                .map((v, i) => (
                  <FlagNote key={i} v={v} />
                ))}
            </div>
          </div>
        </div>

        <aside className="traj-side">
          <h2>Scores</h2>
          {detail.scores.length === 0 && <p className="muted">No scores. Rescore this run to add some.</p>}
          {detail.scores.map((s, i) => (
            <div key={i} className="score-block">
              <div>
                <span className="name">{s.scorer}</span> <span className="num">{fmt(s.value, 2)}</span>
                {s.label ? <span className="muted"> {s.label}</span> : null}
                {s.confidence !== null ? <span className="faint small"> confidence {fmt(s.confidence, 2)}</span> : null}
              </div>
              {s.rationale && <div className="rationale">{s.rationale}</div>}
            </div>
          ))}

          {finalVerdicts.length > 0 && (
            <>
              <h2>Monitor verdicts on the finished trajectory</h2>
              {finalVerdicts.map((v, i) => (
                <div key={i} className="verdict">
                  <span className={`mark${v.flagged ? " flagged" : ""}`} />
                  <div>
                    <span className="mono">{v.monitor}</span> <span className="muted">{v.view === "output" ? "output view" : "trajectory view"}</span>
                    {v.score !== null ? <span className="num"> {fmt(v.score, 2)}</span> : null}
                    {v.reason && <div className="muted small">{v.reason}</div>}
                  </div>
                </div>
              ))}
            </>
          )}

          {detail.annotations.length > 0 && (
            <>
              <h2>Human labels</h2>
              {detail.annotations.map((a) => (
                <div key={a.id} className="score-block">
                  <div>
                    <strong>{a.label}</strong> <span className="muted">by {a.reviewer}</span>
                  </div>
                  {a.note && <div className="rationale">{a.note}</div>}
                </div>
              ))}
            </>
          )}

          {isRecord(traj.meta.sandbox_diff) && (
            <>
              <h2>Sandbox diff</h2>
              <KeyValues data={traj.meta.sandbox_diff} />
            </>
          )}

          <h2>Case</h2>
          <Clamp text={detail.case.prompt_text} limit={500} />
          {detail.case.expected !== null && detail.case.expected !== undefined && (
            <p className="small muted" style={{ marginTop: 6 }}>
              Expected: <span className="mono">{typeof detail.case.expected === "string" ? detail.case.expected : JSON.stringify(detail.case.expected)}</span>
            </p>
          )}
          <details>
            <summary>Trajectory metadata</summary>
            <pre>{JSON.stringify(traj.meta, null, 1)}</pre>
          </details>
        </aside>
      </div>
    </>
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function StepRow({
  index,
  step,
  monitors,
  covered,
  verdicts,
}: {
  index: number;
  step: Step;
  monitors: { name: string; view: string }[];
  covered: (monitorIndex: number) => boolean;
  verdicts: VerdictRow[];
}) {
  const thought = typeof step.meta.thought === "string" ? step.meta.thought : null;
  const rawIsToolJson = step.tool_calls.length > 0 && step.content.trim().startsWith("{");
  return (
    <div className="step-row">
      <div className="rails">
        {monitors.map((m, mi) => {
          const own = verdicts.filter((v) => v.monitor === m.name);
          const flagged = own.find((v) => v.flagged);
          return (
            <div key={m.name} className="rail-col">
              <div className={`line${index === 0 ? " start" : ""}${covered(mi) ? "" : " off"}`} />
              {flagged ? (
                <div className={`dot${flagged.halt ? " halt" : ""}`} title={`${m.name} flagged here${flagged.halt ? " and halted the run" : ""}: ${flagged.reason}`} />
              ) : own.length > 0 ? (
                <div className="dot calm" title={`${m.name} looked here and did not flag`} />
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="step-idx">{String(index).padStart(2, "0")}</div>
      <div className={`step-role ${step.role}`}>{step.role}</div>
      <div className="step-body">
        {step.role === "system" ? (
          <details>
            <summary>System prompt, {step.content.length.toLocaleString()} characters</summary>
            <pre>{step.content}</pre>
          </details>
        ) : step.role === "tool" ? (
          <>
            {step.meta.error === true && <span className="small" style={{ color: "var(--berry)" }}>tool error</span>}
            <Clamp text={step.content || "(empty result)"} limit={400} />
          </>
        ) : (
          <>
            {thought && <div className="thought">{thought}</div>}
            {step.content && !rawIsToolJson && !(thought && step.content === thought) && <Clamp text={step.content} limit={1200} />}
            {step.tool_calls.map((call) => (
              <div key={call.id} className="call">
                <div className="head">{call.name}({summarizeArgs(call.args)})</div>
                {Object.keys(call.args).length > 0 && (
                  <details>
                    <summary>arguments</summary>
                    <pre>{JSON.stringify(call.args, null, 1)}</pre>
                  </details>
                )}
              </div>
            ))}
            {rawIsToolJson && (
              <details>
                <summary>raw message</summary>
                <pre>{step.content}</pre>
              </details>
            )}
          </>
        )}
        {verdicts
          .filter((v) => v.flagged)
          .map((v, i) => (
            <FlagNote key={i} v={v} />
          ))}
      </div>
    </div>
  );
}

function summarizeArgs(args: Record<string, unknown>): string {
  return Object.entries(args)
    .map(([k, v]) => {
      const s = typeof v === "string" ? v : JSON.stringify(v);
      return `${k}=${s.length > 40 ? JSON.stringify(s.slice(0, 37)) + "…" : JSON.stringify(s)}`;
    })
    .join(", ");
}

function FlagNote({ v }: { v: VerdictRow }) {
  return (
    <div className="flag-note">
      <span className="who">{v.monitor}</span>
      {v.halt ? "flagged and halted the run" : "flagged"}
      {v.reason ? <Fragment>: {v.reason}</Fragment> : null}
    </div>
  );
}
