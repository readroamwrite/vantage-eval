/** Frame: the left rail with navigation and the run list, and the page routes. */

import { createContext, useContext, useEffect } from "react";
import { BrowserRouter, Navigate, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { api, type Meta, type Run } from "./api";
import { ErrorBoundary, useLoad, type Load } from "./components";
import { CompareRuns } from "./pages/CompareRuns";
import { ConversationMap } from "./pages/ConversationMap";
import { Experiments } from "./pages/Experiments";
import { Finder } from "./pages/Finder";
import { ReviewQueue } from "./pages/ReviewQueue";
import { RunDetail } from "./pages/RunDetail";
import { Runs } from "./pages/Runs";
import { TrajectoryPage } from "./pages/Trajectory";

interface Shell {
  runs: Load<Run[]>;
  meta: Load<Meta>;
  reloadRuns: () => void;
}

const ShellContext = createContext<Shell>({
  runs: { state: "loading" },
  meta: { state: "loading" },
  reloadRuns: () => undefined,
});

export const useShell = () => useContext(ShellContext);

const PAGES = [
  ["/runs", "Runs"],
  ["/review", "Review queue"],
  ["/compare", "Compare"],
  ["/experiments", "Experiments"],
  ["/finder", "Behaviour finder"],
  ["/map", "Conversation map"],
] as const;

/** The viewfinder mark: a sightline with a flag on it. */
function Mark() {
  return (
    <svg className="mark" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
      <rect x="7" y="0" width="2" height="16" fill="var(--steel)" />
      <circle cx="8" cy="11" r="3" fill="var(--berry)" stroke="var(--paper)" strokeWidth="1.5" />
    </svg>
  );
}

function Rail() {
  const { runs, meta, reloadRuns } = useShell();
  const location = useLocation();
  const activeRun = location.pathname.match(/^\/runs\/(\d+)/)?.[1];
  const dbName = meta.state === "ok" ? meta.data.db.split("/").pop() : "";
  return (
    <aside className="rail">
      <div className="wordmark">
        <Mark />
        <span>vantage</span>
      </div>
      <nav className="nav" aria-label="Pages">
        {PAGES.map(([to, label]) => (
          <NavLink key={to} to={to} className={({ isActive }) => (isActive ? "active" : "")}>
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="rail-runs" aria-label="Stored runs">
        <div className="head">
          <span>{runs.state === "ok" ? `${runs.data.length} stored run${runs.data.length === 1 ? "" : "s"}` : "Stored runs"}</span>
          <button type="button" className="quiet" onClick={reloadRuns} title="Re-read the database">
            Refresh
          </button>
        </div>
        {runs.state === "ok" &&
          runs.data.map((run) => (
            <NavLink key={run.id} to={`/runs/${run.id}`} className={activeRun === String(run.id) ? "active" : ""} title={run.name}>
              <span className="id">{run.id}</span>
              <span>
                <span className="name" style={{ display: "block" }}>
                  {run.name}
                </span>
                <span className="sub" style={{ display: "block" }}>
                  {run.n} on {run.dataset}
                </span>
              </span>
            </NavLink>
          ))}
      </div>
      <div className="rail-foot" title={meta.state === "ok" ? meta.data.db : ""}>
        {dbName}
        {meta.state === "ok" ? ` · vantage ${meta.data.version}` : ""}
      </div>
    </aside>
  );
}

function Shell() {
  const [runs, reloadRuns] = useLoad(() => api.runs(), []);
  const [meta] = useLoad(() => api.meta(), []);
  const location = useLocation();

  // New runs written by the CLI appear without a manual reload: re-read the
  // run list on every navigation and whenever the window regains focus.
  useEffect(() => {
    reloadRuns();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.pathname]);
  useEffect(() => {
    const onFocus = () => reloadRuns();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <ShellContext.Provider value={{ runs, meta, reloadRuns }}>
      <div className="frame">
        <Rail />
        <main>
          <ErrorBoundary resetKey={location.pathname + location.search}>
            <Routes>
              <Route path="/" element={<Navigate to="/runs" replace />} />
              <Route path="/runs" element={<Runs />} />
              <Route path="/runs/:id" element={<RunDetail />} />
              <Route path="/trajectories/:id" element={<TrajectoryPage />} />
              <Route path="/review" element={<ReviewQueue />} />
              <Route path="/compare" element={<CompareRuns />} />
              <Route path="/experiments" element={<Experiments />} />
              <Route path="/finder" element={<Finder />} />
              <Route path="/map" element={<ConversationMap />} />
              <Route path="*" element={<Navigate to="/runs" replace />} />
            </Routes>
          </ErrorBoundary>
        </main>
      </div>
    </ShellContext.Provider>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Shell />
    </BrowserRouter>
  );
}
