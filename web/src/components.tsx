/** Shared pieces: data loading, interval marks, small charts, markdown. */

import { Component, useEffect, useMemo, useRef, useState, type ErrorInfo, type ReactNode } from "react";
import { marked } from "marked";
import type { Estimate } from "./api";
import { fmt, fmtEstimate } from "./api";

export type Load<T> = { state: "loading" } | { state: "error"; message: string } | { state: "ok"; data: T };

/** Fetch once per dependency change and expose the three states. */
export function useLoad<T>(fetcher: () => Promise<T>, deps: unknown[]): [Load<T>, () => void] {
  const [load, setLoad] = useState<Load<T>>({ state: "loading" });
  const [tick, setTick] = useState(0);
  const depsKey = JSON.stringify(deps);
  const lastKey = useRef(depsKey);
  useEffect(() => {
    let alive = true;
    // A dependency change means different data: show "loading". A manual
    // reload of the same data keeps what is on screen until the new copy lands.
    if (lastKey.current !== depsKey) setLoad({ state: "loading" });
    lastKey.current = depsKey;
    fetcher().then(
      (data) => alive && setLoad({ state: "ok", data }),
      (err: Error) => alive && setLoad({ state: "error", message: err.message }),
    );
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  return [load, () => setTick((t) => t + 1)];
}

export function Loading({ what }: { what: string }) {
  return <p className="muted">Loading {what}.</p>;
}

export function ErrorNote({ message }: { message: string }) {
  return <p className="error">Could not load: {message}</p>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

/** A point estimate with its interval drawn as a short range on a 0 to 1 track. */
export function Est({ est, max = 1, min = 0 }: { est: Estimate | null | undefined; max?: number; min?: number }) {
  if (!est || est.n === 0 || est.point === null) return <span className="num faint">n/a</span>;
  const scale = (v: number) => ((Math.min(max, Math.max(min, v)) - min) / (max - min)) * 64;
  const lo = scale(est.lo ?? est.point);
  const hi = scale(est.hi ?? est.point);
  const pt = scale(est.point);
  return (
    <span className="est" title={`${fmtEstimate(est)} from n = ${est.n}`}>
      <span className="num">{fmtEstimate(est)}</span>
      <svg viewBox="0 0 64 10" aria-hidden="true">
        <line x1="0" x2="64" y1="5" y2="5" stroke="var(--hairline)" strokeWidth="1" />
        <line x1={lo} x2={hi} y1="5" y2="5" stroke="var(--steel)" strokeWidth="2" strokeLinecap="round" />
        <circle cx={pt} cy="5" r="3" fill="var(--steel)" stroke="var(--paper)" strokeWidth="1.5" />
      </svg>
    </span>
  );
}

export interface IntervalRow {
  label: string;
  est: Estimate;
  context?: boolean;
}

/**
 * Horizontal interval plot: one row per label, point and range on a shared axis.
 * A single hue for the series; grey for rows that are only context.
 */
export function IntervalPlot({
  rows,
  min = 0,
  max = 1,
  zero = false,
  width = 560,
}: {
  rows: IntervalRow[];
  min?: number;
  max?: number;
  zero?: boolean;
  width?: number;
}) {
  const labelW = Math.min(220, 8 + 7.2 * Math.max(6, ...rows.map((r) => r.label.length)));
  const valueW = 130;
  const plotW = Math.max(160, width - labelW - valueW);
  const rowH = 24;
  const height = rows.length * rowH + 20;
  const x = (v: number) => labelW + ((Math.min(max, Math.max(min, v)) - min) / (max - min)) * plotW;
  const ticks = zero ? [min, 0, max] : [min, (min + max) / 2, max];
  return (
    <div className="chart" role="img" aria-label={rows.map((r) => `${r.label} ${fmtEstimate(r.est)}`).join("; ")}>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {ticks.map((t) => (
          <g key={t}>
            <line className={`grid${t === 0 && zero ? " zero" : ""}`} x1={x(t)} x2={x(t)} y1={0} y2={rows.length * rowH} />
            <text x={x(t)} y={rows.length * rowH + 14} textAnchor="middle">
              {fmt(t, 1)}
            </text>
          </g>
        ))}
        {rows.map((r, i) => {
          const cy = i * rowH + rowH / 2;
          const ok = r.est.n > 0 && r.est.point !== null;
          const ctx = r.context ? " ctx" : "";
          return (
            <g key={r.label}>
              <title>{`${r.label}: ${fmtEstimate(r.est)}, n = ${r.est.n}`}</title>
              <text x={labelW - 10} y={cy + 4} textAnchor="end">
                {r.label.length > 28 ? r.label.slice(0, 27) + "…" : r.label}
              </text>
              {ok && (
                <>
                  <line className={`range${ctx}`} x1={x(r.est.lo ?? r.est.point!)} x2={x(r.est.hi ?? r.est.point!)} y1={cy} y2={cy} />
                  <circle className={`pt${ctx}`} cx={x(r.est.point!)} cy={cy} r={4} />
                </>
              )}
              <text className="value" x={labelW + plotW + 10} y={cy + 4}>
                {ok ? fmtEstimate(r.est) : "n/a"}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** Horizontal bars for counts. One hue; value at the tip. */
export function CountBars({ rows, width = 480 }: { rows: { label: string; n: number }[]; width?: number }) {
  const max = Math.max(1, ...rows.map((r) => r.n));
  const labelW = Math.min(220, 8 + 7.2 * Math.max(6, ...rows.map((r) => r.label.length)));
  const plotW = Math.max(120, width - labelW - 50);
  const rowH = 22;
  const barH = 14;
  return (
    <div className="chart" role="img" aria-label={rows.map((r) => `${r.label} ${r.n}`).join("; ")}>
      <svg width={width} height={rows.length * rowH} viewBox={`0 0 ${width} ${rows.length * rowH}`}>
        {rows.map((r, i) => {
          const w = (r.n / max) * plotW;
          const y = i * rowH + (rowH - barH) / 2;
          return (
            <g key={r.label}>
              <title>{`${r.label}: ${r.n}`}</title>
              <text x={labelW - 10} y={y + barH - 3} textAnchor="end">
                {r.label}
              </text>
              <path className="bar" d={`M${labelW},${y} h${Math.max(0, w - 4)} a4,4 0 0 1 4,4 v${barH - 8} a4,4 0 0 1 -4,4 h${-Math.max(0, w - 4)} z`} />
              <text className="value" x={labelW + w + 8} y={y + barH - 3}>
                {r.n}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** Rendered markdown. Image sources are rewritten through `resolveImage`. */
export function Markdown({ text, resolveImage }: { text: string; resolveImage?: (src: string) => string }) {
  const html = useMemo(() => {
    const renderer = new marked.Renderer();
    if (resolveImage) {
      renderer.image = ({ href, title, text: alt }) =>
        `<img src="${resolveImage(href)}" alt="${alt}"${title ? ` title="${title}"` : ""} />`;
    }
    return marked.parse(text, { renderer, gfm: true, async: false }) as string;
  }, [text, resolveImage]);
  return (
    <div className="md-wrap">
      <div className="md" dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
}

/** A key/value list for small dictionaries. */
export function KeyValues({ data }: { data: Record<string, unknown> }) {
  return (
    <dl className="kv">
      {Object.entries(data).map(([k, v]) => (
        <span key={k} style={{ display: "contents" }}>
          <dt>{k}</dt>
          <dd className={typeof v === "number" ? "num" : ""}>{typeof v === "string" ? v : JSON.stringify(v)}</dd>
        </span>
      ))}
    </dl>
  );
}

/** Text that collapses past a length with a control to expand. */
export function Clamp({ text, limit = 600 }: { text: string; limit?: number }) {
  const [open, setOpen] = useState(false);
  if (text.length <= limit) return <pre>{text}</pre>;
  return (
    <>
      <pre>{open ? text : text.slice(0, limit) + "…"}</pre>
      <div className="more">
        <button type="button" onClick={() => setOpen(!open)}>
          {open ? "Show less" : `Show all ${text.length.toLocaleString()} characters`}
        </button>
      </div>
    </>
  );
}

/** Persist a small string in the browser only, for conveniences like the reviewer name. */
export function useLocal(key: string, initial: string): [string, (v: string) => void] {
  const [value, setValue] = useState<string>(() => {
    try {
      return localStorage.getItem(key) ?? initial;
    } catch {
      return initial;
    }
  });
  const set = (v: string) => {
    setValue(v);
    try {
      localStorage.setItem(key, v);
    } catch {
      /* storage unavailable */
    }
  };
  return [value, set];
}

/** Shows the error a page threw instead of leaving it blank. Reset by changing `resetKey`. */
export class ErrorBoundary extends Component<{ children: ReactNode; resetKey: string }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidUpdate(prev: { resetKey: string }) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null });
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("page failed to render", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="empty">
        <p className="error">This page hit an error while rendering, so it stopped.</p>
        <pre>{String(this.state.error.stack ?? this.state.error.message)}</pre>
        <p className="small muted">Reload the page to try again. If it keeps happening, the message above is what to report.</p>
      </div>
    );
  }
}
