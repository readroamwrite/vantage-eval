/** Experiments: the stored reports, rendered with their figures. */

import { useCallback } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api";
import { Empty, ErrorNote, Loading, Markdown, useLoad } from "../components";

export function Experiments() {
  const [list] = useLoad(() => api.experiments(), []);
  const [params, setParams] = useSearchParams();
  if (list.state === "loading") return <Loading what="experiments" />;
  if (list.state === "error") return <ErrorNote message={list.message} />;
  if (list.data.length === 0)
    return (
      <>
        <h1>Experiments</h1>
        <Empty>
          No experiments stored. Run one with <code>vantage experiment judge</code>, <code>vantage experiment robustness</code> or{" "}
          <code>vantage experiment monitoring</code>.
        </Empty>
      </>
    );
  const chosen = params.get("name") && list.data.some((e) => e.name === params.get("name")) ? params.get("name")! : list.data[0].name;
  return (
    <>
      <div className="page-head">
        <h1>Experiments</h1>
        <div className="controls" style={{ margin: 0 }}>
          <label className="field">
            Experiment
            <select value={chosen} onChange={(e) => setParams({ name: e.target.value })}>
              {list.data.map((e) => (
                <option key={e.name} value={e.name}>
                  {e.name}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>
      <Report name={chosen} />
    </>
  );
}

function Report({ name }: { name: string }) {
  const [load] = useLoad(() => api.experiment(name), [name]);
  const resolveImage = useCallback((src: string) => `/api/experiments/${encodeURIComponent(name)}/figures/${encodeURIComponent(src.split("/").pop() ?? src)}`, [name]);
  if (load.state === "loading") return <Loading what={name} />;
  if (load.state === "error") return <ErrorNote message={load.message} />;
  const e = load.data;
  return (
    <>
      <p className="small muted">
        Created {e.created_at}. Runs {e.run_ids.join(", ")}.
      </p>
      {e.report ? (
        <Markdown text={e.report} resolveImage={resolveImage} />
      ) : (
        <p className="notice">
          The report file {e.report_path} is not on disk, so only the raw results below are shown. Rerun the experiment to write it.
        </p>
      )}
      <details>
        <summary>Parameters</summary>
        <pre>{JSON.stringify(e.params, null, 1)}</pre>
      </details>
      <details>
        <summary>Raw results</summary>
        <pre>{JSON.stringify(e.results, null, 1)}</pre>
      </details>
    </>
  );
}
