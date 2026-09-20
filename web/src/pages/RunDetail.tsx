/** Run detail: the header, per-scorer metrics, eval health and the per-case table. */

import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, fmt, fmtEstimate, type CaseRow, type Diagnosis, type Estimate } from "../api";
import { CountBars, Empty, ErrorNote, Est, IntervalPlot, Loading, useLoad } from "../components";

export function RunDetail() {
  const id = Number(useParams().id);
  const [load] = useLoad(() => api.run(id), [id]);
  if (load.state === "loading") return <Loading what={`run ${id}`} />;
  if (load.state === "error") return <ErrorNote message={load.message} />;
  const { run, cases, diagnosis, scorers } = load.data;
  return (
    <>
      <div className="page-head">
        <h1>{run.name}</h1>
        <span className="context">
          <span className="mono">{run.target}</span> on <span className="mono">{run.dataset}</span>
          {run.condition ? <> under condition {run.condition}</> : null}. Status {run.status}, {run.n} trajectories.
        </span>
      </div>

      <div className="table-scroll" style={{ maxWidth: 680 }}>
      <table className="data">
        <thead>
          <tr>
            <th>scorer</th>
            <th>mean and interval</th>
            <th className="num">n</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(run.metrics).map(([name, est]) => (
            <tr key={name}>
              <td className="mono">{name}</td>
              <td>
                <Est est={est} />
              </td>
              <td className="num">{est.n}</td>
            </tr>
          ))}
          {Object.keys(run.metrics).length === 0 && (
            <tr>
              <td colSpan={3} className="muted">
                No scores stored. Rescore this run with <code>vantage rescore</code>.
              </td>
            </tr>
          )}
        </tbody>
      </table>
      </div>
      <p className="small muted" style={{ marginTop: 8 }}>
        Outcomes:{" "}
        {Object.entries(run.statuses)
          .map(([k, v]) => `${v} ${k}`)
          .join(", ") || "none"}
        .
      </p>

      {diagnosis && diagnosis.scorer && <Health diag={diagnosis} />}

      <h2>Cases</h2>
      {cases.length === 0 ? <Empty>No trajectories stored for this run.</Empty> : <CaseTable cases={cases} scorers={scorers} />}
    </>
  );
}

function Health({ diag }: { diag: Diagnosis }) {
  const sat = diag.saturation;
  const tagGroups = Object.entries(diag.by_tag ?? {}).filter(([prefix]) => prefix !== "base" && prefix !== "para");
  const tiers = diag.difficulty?.per_tier;
  return (
    <>
      <h2>Eval health, judged on {diag.scorer}</h2>
      {diag.warnings && diag.warnings.length > 0 ? (
        <ul className="warnings">
          {diag.warnings.map((w) => {
            const [code, ...rest] = w.split(": ");
            return (
              <li key={w}>
                <span className="code">{code}</span>
                {rest.join(": ")}
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="muted">No warnings. The eval still separates this target from the ceiling and the floor.</p>
      )}
      {sat && (
        <p className="small">
          Mean <span className="num">{fmtEstimate(sat.mean)}</span>, headroom {fmt(sat.headroom, 2)}.{sat.saturated ? " Saturated." : ""}
          {sat.floored ? " Floored." : ""}
        </p>
      )}

      {diag.failures && (
        <>
          <h3>Why items failed</h3>
          <p className="small">
            Raw accuracy <span className="num">{fmtEstimate(diag.failures.raw_accuracy)}</span> becomes{" "}
            <span className="num">{fmtEstimate(diag.failures.adjusted_accuracy)}</span> once {diag.failures.n_excluded} failure
            {diag.failures.n_excluded === 1 ? "" : "s"} unrelated to capability {diag.failures.n_excluded === 1 ? "is" : "are"} set aside.
          </p>
          <CountBars rows={Object.entries(diag.failures.classes).map(([label, c]) => ({ label, n: c.n }))} />
        </>
      )}

      {tiers && diag.difficulty && (
        <>
          <h3>By difficulty tier</h3>
          <p className="small">
            Spearman correlation between tier and score <span className="num">{fmtEstimate(diag.difficulty.spearman_ci)}</span>
            {diag.difficulty.monotone ? ", falling monotonically with difficulty" : ", not monotone"}. Failure rate on the easiest tier{" "}
            <span className="num">{fmtEstimate(diag.difficulty.easy_failure_rate)}</span>.
          </p>
          <IntervalPlot rows={Object.entries(tiers).map(([tier, est]) => ({ label: `tier ${tier}`, est }))} />
        </>
      )}

      {diag.judge_corrected && diag.judge_corrected.available === true && (
        <p className="small">
          Judge-corrected pass rate: raw <span className="num">{fmtEstimate(diag.judge_corrected.raw as Estimate)}</span> becomes{" "}
          <span className="num">{fmtEstimate(diag.judge_corrected.corrected as Estimate)}</span> after adjusting for the judge's measured
          error rates (kappa against gold <span className="num">{fmt(diag.judge_corrected.kappa, 2)}</span>).
        </p>
      )}

      {diag.paraphrase && (
        <p className="small">
          Paraphrase agreement <span className="num">{fmtEstimate(diag.paraphrase.agreement)}</span> over {diag.paraphrase.n_bases} base items;{" "}
          {diag.paraphrase.inconsistent.length} inconsistent.
        </p>
      )}
      {diag.dead_items && (
        <p className="small">
          Dead items across {diag.n_columns} columns: {diag.dead_items.all_pass.length} always pass, {diag.dead_items.all_fail.length} always fail.
        </p>
      )}

      {tagGroups.map(([prefix, tags]) => (
        <details key={prefix}>
          <summary>
            By {prefix} tag ({Object.keys(tags).length})
          </summary>
          <IntervalPlot rows={Object.entries(tags).map(([tag, est]) => ({ label: tag.replace(`${prefix}:`, ""), est: est as Estimate }))} />
        </details>
      ))}
      {diag.coverage_gaps && diag.coverage_gaps.length > 0 && (
        <details>
          <summary>Threat-model nodes with no items ({diag.coverage_gaps.length})</summary>
          <p className="small">
            {diag.coverage_gaps.map((g) => (
              <span key={g} className="tag">
                {g}
              </span>
            ))}
          </p>
        </details>
      )}
    </>
  );
}

function CaseTable({ cases, scorers }: { cases: CaseRow[]; scorers: string[] }) {
  const navigate = useNavigate();
  const statuses = useMemo(() => Array.from(new Set(cases.map((c) => c.status))).sort(), [cases]);
  const [tag, setTag] = useState("");
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [onlyFlagged, setOnlyFlagged] = useState(false);
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 }>({ key: "trajectory_id", dir: 1 });

  const shown = useMemo(() => {
    const rows = cases.filter(
      (c) => !hidden.has(c.status) && (!onlyFlagged || c.flagged) && (!tag || c.tags.some((t) => t.includes(tag))),
    );
    const value = (c: CaseRow): number | string => {
      if (sort.key in c) return (c as unknown as Record<string, number | string>)[sort.key];
      return c.scores[sort.key]?.value ?? -1;
    };
    return rows.sort((a, b) => {
      const va = value(a);
      const vb = value(b);
      return (va < vb ? -1 : va > vb ? 1 : 0) * sort.dir;
    });
  }, [cases, hidden, onlyFlagged, tag, sort]);

  const toggleSort = (key: string) => setSort((s) => (s.key === key ? { key, dir: s.dir === 1 ? -1 : 1 } : { key, dir: 1 }));
  const th = (key: string, label: string, num = false) => (
    <th className={`sortable${num ? " num" : ""}`} onClick={() => toggleSort(key)} aria-sort={sort.key === key ? (sort.dir === 1 ? "ascending" : "descending") : "none"}>
      {label}
      {sort.key === key ? (sort.dir === 1 ? " ↑" : " ↓") : ""}
    </th>
  );
  return (
    <>
      <div className="controls">
        <label className="field">
          Tag contains
          <input type="text" value={tag} onChange={(e) => setTag(e.target.value)} placeholder="threat:" />
        </label>
        {statuses.map((s) => (
          <label key={s} className="field check">
            <input
              type="checkbox"
              checked={!hidden.has(s)}
              onChange={(e) => {
                const next = new Set(hidden);
                if (e.target.checked) next.delete(s);
                else next.add(s);
                setHidden(next);
              }}
            />
            {s}
          </label>
        ))}
        <label className="field check">
          <input type="checkbox" checked={onlyFlagged} onChange={(e) => setOnlyFlagged(e.target.checked)} />
          only monitor-flagged
        </label>
        <span className="muted small" style={{ paddingBottom: 6 }}>
          {shown.length} of {cases.length}
        </span>
      </div>
      <div className="table-scroll">
        <table className="data">
          <thead>
            <tr>
              {th("trajectory_id", "id", true)}
              {th("case_id", "case")}
              {th("repeat", "rep", true)}
              {th("status", "status")}
              {th("steps", "steps", true)}
              {th("tool_calls", "tools", true)}
              {scorers.map((s) => th(s, s, true))}
              <th>tags</th>
              <th>answer</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((c) => (
              <tr
                key={c.trajectory_id}
                className="clickable"
                tabIndex={0}
                onClick={() => navigate(`/trajectories/${c.trajectory_id}`)}
                onKeyDown={(e) => e.key === "Enter" && navigate(`/trajectories/${c.trajectory_id}`)}
              >
                <td className="num">
                  {c.flagged && <span className="flag" title="a monitor flagged this trajectory" />}
                  <Link to={`/trajectories/${c.trajectory_id}`}>{c.trajectory_id}</Link>
                </td>
                <td className="mono">{c.case_id}</td>
                <td className="num">{c.repeat}</td>
                <td>{c.status}</td>
                <td className="num">{c.steps}</td>
                <td className="num">{c.tool_calls}</td>
                {scorers.map((s) => {
                  const cell = c.scores[s];
                  return (
                    <td key={s} className="num">
                      {cell ? (
                        <>
                          {fmt(cell.value, 2)}
                          {cell.label ? <span className="faint"> {cell.label}</span> : null}
                        </>
                      ) : (
                        <span className="faint">–</span>
                      )}
                    </td>
                  );
                })}
                <td className="tags">
                  {c.tags.map((t) => (
                    <span key={t} className="tag">
                      {t}
                    </span>
                  ))}
                </td>
                <td className="clip small muted" title={c.answer}>
                  {c.answer}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
