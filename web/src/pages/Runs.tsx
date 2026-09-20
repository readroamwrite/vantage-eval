/** Runs: every stored run with its per-scorer means and intervals. */

import { useNavigate } from "react-router-dom";
import { useShell } from "../App";
import type { Estimate, Run } from "../api";
import { Empty, ErrorNote, Est, Loading } from "../components";

export function Runs() {
  const { runs } = useShell();
  const navigate = useNavigate();
  if (runs.state === "loading") return <Loading what="runs" />;
  if (runs.state === "error") return <ErrorNote message={runs.message} />;
  // One headline metric per run keeps the table readable; the run page has every scorer.
  const headline = (run: Run): [string, Estimate] | null => {
    const names = Object.keys(run.metrics);
    const first = names.find((n) => n !== "gold" && n !== "failure") ?? names[0];
    return first ? [first, run.metrics[first]] : null;
  };
  return (
    <>
      <div className="page-head">
        <h1>Runs</h1>
        <span className="context">Means with 95% bootstrap intervals. Open a run for its cases and eval health.</span>
      </div>
      {runs.data.length === 0 ? (
        <Empty>
          No runs stored yet. Run one first, for example <code>vantage run data/smoke.jsonl --target mock:arith --scorer numeric</code>, then reload this page.
        </Empty>
      ) : (
        <div className="table-scroll fit">
          <table className="data">
            <thead>
              <tr>
                <th className="num">id</th>
                <th>run</th>
                <th>target</th>
                <th>dataset</th>
                <th>condition</th>
                <th>status</th>
                <th className="num">n</th>
                <th>headline metric</th>
              </tr>
            </thead>
            <tbody>
              {runs.data.map((run) => (
                <tr
                  key={run.id}
                  className="clickable"
                  tabIndex={0}
                  onClick={() => navigate(`/runs/${run.id}`)}
                  onKeyDown={(e) => e.key === "Enter" && navigate(`/runs/${run.id}`)}
                >
                  <td className="num">{run.id}</td>
                  <td className="name">{run.name}</td>
                  <td className="mono">{run.target}</td>
                  <td className="mono">{run.dataset}</td>
                  <td>{run.condition || <span className="faint">none</span>}</td>
                  <td>{run.status}</td>
                  <td className="num">{run.n}</td>
                  <td>
                    {(() => {
                      const h = headline(run);
                      return h ? (
                        <>
                          <span className="mono muted">{h[0]}</span> <Est est={h[1]} />
                        </>
                      ) : (
                        <span className="faint">no scores yet</span>
                      );
                    })()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
