/** Compare: the paired difference between two runs and the items that flipped. */

import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useShell } from "../App";
import { api, type TrajectoryStub } from "../api";
import { Empty, ErrorNote, Est, IntervalPlot, Loading, useLoad } from "../components";
import { RunPicker } from "../RunPicker";

export function CompareRuns() {
  const { runs } = useShell();
  const [params, setParams] = useSearchParams();
  const ids = runs.state === "ok" ? runs.data.map((r) => r.id) : [];
  const a = params.get("a") ? Number(params.get("a")) : (ids[1] ?? null);
  const b = params.get("b") ? Number(params.get("b")) : (ids[0] ?? null);
  const set = (key: "a" | "b", id: number | null) => {
    const next = new URLSearchParams(params);
    if (id === null) next.delete(key);
    else next.set(key, String(id));
    setParams(next);
  };
  if (runs.state === "ok" && runs.data.length < 2)
    return (
      <>
        <h1>Compare runs</h1>
        <Empty>Store at least two runs to compare them. The comparison pairs items by case id, so the runs should share a dataset.</Empty>
      </>
    );
  return (
    <>
      <div className="page-head">
        <h1>Compare runs</h1>
        <span className="context">Paired on shared case ids. The difference is B minus A with a 95% bootstrap interval.</span>
      </div>
      <div className="controls">
        <RunPicker label="Run A (baseline)" value={a} onChange={(id) => set("a", id)} />
        <RunPicker label="Run B" value={b} onChange={(id) => set("b", id)} />
      </div>
      {a !== null && b !== null && <Result a={a} b={b} />}
    </>
  );
}

function Result({ a, b }: { a: number; b: number }) {
  const [load] = useLoad(() => Promise.all([api.compare(a, b), api.runTrajectories(b)]), [a, b]);
  const [scorer, setScorer] = useState<string | null>(null);
  if (load.state === "loading") return <Loading what="the comparison" />;
  if (load.state === "error") return <ErrorNote message={load.message} />;
  const [result, stubs] = load.data;
  const names = Object.keys(result.scorers);
  if (names.length === 0) return <Empty>These runs share no scorer or no case ids, so there is nothing to pair.</Empty>;
  const chosen = scorer && names.includes(scorer) ? scorer : names[0];
  const block = result.scorers[chosen];
  const limit = Math.max(0.1, ...names.flatMap((n) => [Math.abs(result.scorers[n].diff.lo ?? 0), Math.abs(result.scorers[n].diff.hi ?? 0)]));
  return (
    <>
      <div className="table-scroll">
        <table className="data">
          <thead>
            <tr>
              <th>scorer</th>
              <th>A: {result.run_a.name}</th>
              <th>B: {result.run_b.name}</th>
              <th>difference, B minus A</th>
              <th className="num">pairs</th>
              <th className="num">flipped up</th>
              <th className="num">flipped down</th>
            </tr>
          </thead>
          <tbody>
            {names.map((n) => {
              const s = result.scorers[n];
              return (
                <tr key={n} className={n === chosen ? "selected clickable" : "clickable"} onClick={() => setScorer(n)}>
                  <td className="mono">{n}</td>
                  <td>
                    <Est est={s.mean_a} />
                  </td>
                  <td>
                    <Est est={s.mean_b} />
                  </td>
                  <td>
                    <Est est={s.diff} min={-1} max={1} />
                  </td>
                  <td className="num">{s.n_pairs}</td>
                  <td className="num">{s.flipped_up.length}</td>
                  <td className="num">{s.flipped_down.length}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <h3>Difference per scorer</h3>
      <IntervalPlot rows={names.map((n) => ({ label: n, est: result.scorers[n].diff }))} min={-limit} max={limit} zero />

      <h2>Items that flipped on {chosen}</h2>
      <div className="two-col">
        <FlipList title="Failed in A, passed in B" ids={block.flipped_up} stubs={stubs} />
        <FlipList title="Passed in A, failed in B" ids={block.flipped_down} stubs={stubs} />
      </div>
    </>
  );
}

function FlipList({ title, ids, stubs }: { title: string; ids: string[]; stubs: TrajectoryStub[] }) {
  const byCase = useMemo(() => {
    const m = new Map<string, number>();
    for (const s of stubs) if (!m.has(s.case_id)) m.set(s.case_id, s.id);
    return m;
  }, [stubs]);
  const unique = Array.from(new Set(ids));
  return (
    <div>
      <h3>
        {title} <span className="muted">({unique.length})</span>
      </h3>
      {unique.length === 0 ? (
        <p className="muted small">None.</p>
      ) : (
        <p className="small">
          {unique.map((id) => {
            const tid = byCase.get(id);
            return (
              <span key={id} className="tag">
                {tid ? <Link to={`/trajectories/${tid}`}>{id}</Link> : id}
              </span>
            );
          })}
        </p>
      )}
    </div>
  );
}
