/** Typed fetch helpers for the dashboard API. Every shape mirrors vantage/dashboard/queries.py. */

export interface Estimate {
  point: number | null;
  lo: number | null;
  hi: number | null;
  n: number;
  level: number;
}

export interface Run {
  id: number;
  name: string;
  target: string;
  dataset: string;
  condition: string;
  status: string;
  n: number;
  created_at: string | null;
  metrics: Record<string, Estimate>;
  statuses: Record<string, number>;
}

export interface ScoreCell {
  value: number;
  label: string | null;
  passed: boolean | null;
}

export interface CaseRow {
  trajectory_id: number;
  case_id: string;
  repeat: number;
  status: string;
  steps: number;
  tool_calls: number;
  flagged: boolean;
  tags: string[];
  answer: string;
  scores: Record<string, ScoreCell>;
}

export interface Diagnosis {
  scorer: string | null;
  n?: number;
  saturation?: { mean: Estimate; saturated: boolean; floored: boolean; headroom: number };
  by_tag?: Record<string, Record<string, Estimate>>;
  n_columns?: number;
  dead_items?: { all_pass: string[]; all_fail: string[] } | null;
  low_discrimination?: string[];
  paraphrase?: { n_bases: number; agreement: Estimate; inconsistent: string[] } | null;
  difficulty?: {
    per_tier: Record<string, Estimate>;
    spearman: number | null;
    spearman_ci: Estimate;
    monotone: boolean;
    easy_failure_rate: Estimate;
  } | null;
  failures?: {
    n: number;
    raw_accuracy: Estimate;
    adjusted_accuracy: Estimate;
    non_capability_share: Estimate;
    n_excluded: number;
    classes: Record<string, { n: number; share: Estimate }>;
  } | null;
  coverage_gaps?: string[];
  judge_corrected?: Record<string, unknown> | null;
  warnings?: string[];
}

export interface RunDetail {
  run: Run;
  scorers: string[];
  primary_scorer: string | null;
  cases: CaseRow[];
  diagnosis: Diagnosis | null;
}

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export interface Step {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_calls: ToolCall[];
  tool_call_id: string | null;
  meta: Record<string, unknown>;
}

export interface Trajectory {
  case_id: string;
  target_id: string;
  steps: Step[];
  final_output: string | null;
  status: string;
  error: string | null;
  meta: Record<string, unknown>;
}

export interface ScoreRow {
  scorer: string;
  value: number;
  passed: boolean | null;
  label: string | null;
  confidence: number | null;
  rationale: string | null;
  trajectory_id: number;
}

export interface VerdictRow {
  monitor: string;
  view: "output" | "trajectory";
  step_idx: number;
  flagged: boolean;
  halt: boolean;
  score: number | null;
  reason: string;
  trajectory_id: number;
}

export interface AnnotationRow {
  id: number;
  trajectory_id: number;
  reviewer: string;
  label: string;
  value: number | null;
  note: string;
  created_at: string;
}

export interface TrajectoryDetail {
  id: number;
  run: { id: number; name: string };
  repeat: number;
  case: { id: string; prompt_text: string; expected: unknown; tags: string[] };
  trajectory: Trajectory;
  scores: ScoreRow[];
  verdicts: VerdictRow[];
  monitors: { name: string; view: "output" | "trajectory" }[];
  annotations: AnnotationRow[];
  siblings: { id: number; case_id: string }[];
}

export interface TrajectoryStub {
  id: number;
  case_id: string;
  repeat: number;
  status: string;
}

export interface QueueRow {
  trajectory_id: number;
  case_id: string;
  priority: number;
  reasons: string[];
  reviewed: boolean;
  label: string | null;
  reviewer: string | null;
  status: string;
}

export interface AgreementRow {
  scorer: string;
  n: number;
  agreement?: number;
  kappa?: number | null;
}

export interface CompareBlock {
  mean_a: Estimate;
  mean_b: Estimate;
  diff: Estimate;
  n_pairs: number;
  flipped_up: string[];
  flipped_down: string[];
}

export interface CompareResult {
  run_a: Run;
  run_b: Run;
  scorers: Record<string, CompareBlock>;
}

export interface ExperimentStub {
  name: string;
  created_at: string;
  run_ids: number[];
  params: Record<string, unknown>;
}

export interface ExperimentDetail extends ExperimentStub {
  results: Record<string, unknown>;
  report: string | null;
  report_path: string;
  out_dir: string;
}

export interface Hit {
  trajectory_id: number;
  run_id: number;
  case_id: string;
  flagged: boolean;
  score: number;
  rationale: string;
}

export interface FinderAudit {
  n_excluded: number;
  n_audited: number;
  n_flagged: number;
  flag_rate: Estimate;
  estimated_missed: number;
  prefilter_recall: Estimate;
  hits: Hit[];
}

export interface FinderResult {
  monitor: string;
  examined: number;
  hits: Hit[];
  non_hits: Hit[];
  precision: Estimate;
  n_labelled: number;
  audit?: FinderAudit | null;
}

export interface MapNode {
  title: string;
  kind: string;
  summary: string;
  step_refs: number[];
  children: MapNode[];
}

export interface ConvMap {
  root: MapNode;
  model: string;
  n_steps: number;
}

export interface Job<T> {
  id: string;
  kind: string;
  status: "running" | "done" | "error";
  done: number;
  total: number;
  result: T | null;
  error: string | null;
}

export interface Meta {
  db: string;
  version: string;
  labels: string[];
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, { headers: { "content-type": "application/json" }, ...init });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body && typeof body.detail === "string") detail = body.detail;
    } catch {
      /* body was not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export const api = {
  meta: () => request<Meta>("/api/meta"),
  runs: () => request<Run[]>("/api/runs"),
  run: (id: number) => request<RunDetail>(`/api/runs/${id}`),
  runTrajectories: (id: number) => request<TrajectoryStub[]>(`/api/runs/${id}/trajectories`),
  review: (id: number) =>
    request<{ queue: QueueRow[]; agreement: AgreementRow[] }>(`/api/runs/${id}/review`),
  trajectory: (id: number) => request<TrajectoryDetail>(`/api/trajectories/${id}`),
  annotate: (body: { trajectory_id: number; reviewer: string; label: string; note: string; monitor?: string | null }) =>
    request<{ id: number; label: string }>("/api/annotations", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  compare: (a: number, b: number) => request<CompareResult>(`/api/compare?a=${a}&b=${b}`),
  experiments: () => request<ExperimentStub[]>("/api/experiments"),
  experiment: (name: string) =>
    request<ExperimentDetail>(`/api/experiments/${encodeURIComponent(name)}`),
  finder: (body: {
    rubric: string;
    run_id: number | null;
    fts: string | null;
    limit: number;
    model: string;
    view: string;
    audit: number;
  }) => request<Job<FinderResult>>("/api/finder", { method: "POST", body: JSON.stringify(body) }),
  convmap: (trajectoryId: number, model: string) =>
    request<{ map: ConvMap | null; mermaid?: string }>(
      `/api/convmap/${trajectoryId}?model=${encodeURIComponent(model)}`,
    ),
  buildConvmap: (trajectoryId: number, body: { model: string; force: boolean }) =>
    request<Job<{ map: ConvMap; mermaid: string }>>(`/api/convmap/${trajectoryId}`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  job: <T>(id: string) => request<Job<T>>(`/api/jobs/${id}`),
};

/** Poll a job until it finishes, reporting progress along the way. */
export async function waitForJob<T>(
  id: string,
  onProgress: (done: number, total: number) => void,
  intervalMs = 800,
): Promise<T> {
  for (;;) {
    const job = await api.job<T>(id);
    onProgress(job.done, job.total);
    if (job.status === "done") return job.result as T;
    if (job.status === "error") throw new Error(job.error ?? "job failed");
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

/** Format a number for display: three decimals for fractions, integers as is. */
export function fmt(value: unknown, digits = 3): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(digits);
}

/** Format an estimate as "point [lo, hi]". */
export function fmtEstimate(est: Estimate | null | undefined): string {
  if (!est || typeof est.point !== "number" || est.n === 0) return "n/a";
  return `${est.point.toFixed(3)} [${(est.lo ?? est.point).toFixed(2)}, ${(est.hi ?? est.point).toFixed(2)}]`;
}
