/** A select over stored runs, shared by the pages that start from a run. */

import { useShell } from "./App";
import type { Run } from "./api";

export function RunPicker({
  value,
  onChange,
  label = "Run",
  allowAll = false,
}: {
  value: number | null;
  onChange: (id: number | null) => void;
  label?: string;
  allowAll?: boolean;
}) {
  const { runs } = useShell();
  const options: Run[] = runs.state === "ok" ? runs.data : [];
  return (
    <label className="field">
      {label}
      <select value={value ?? ""} onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}>
        {allowAll && <option value="">all runs</option>}
        {!allowAll && value === null && <option value="">choose a run</option>}
        {options.map((run) => (
          <option key={run.id} value={run.id}>
            {run.id}: {run.name}
          </option>
        ))}
      </select>
    </label>
  );
}

/** The newest run id, used as a default when a page is opened without one. */
export function useDefaultRun(): number | null {
  const { runs } = useShell();
  return runs.state === "ok" && runs.data.length > 0 ? runs.data[0].id : null;
}
