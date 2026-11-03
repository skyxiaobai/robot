"""Append-only experiment review overlays. Raw metric rows are never rewritten."""
import json
from pathlib import Path


def reviewed_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    notes = [r for r in records if r.get("record_type") == "review_note"]
    rows = []
    for record in records:
        if record.get("record_type") == "review_note":
            continue
        row = dict(record)
        for note in notes:
            if row.get("id") in note.get("target_ids", []):
                row.update(note.get("annotations", {}))
                row.setdefault("review_notes", []).append(note["id"])
        rows.append(row)
    return rows


def decision_status(row):
    """Phase-0 never grants eligibility; old/window-level intervals cannot win.

    Future protocol code must explicitly attest independent units, a locked
    minimum useful improvement and enough units. Seeds do not increase n_units.
    """
    if row.get("status") != "ok":
        return row.get("status") or ""
    if row.get("evidence_status") == "evidence_insufficient" or not row.get("decision_eligible"):
        return "evidence_insufficient"
    if row.get("ci_method") != "episode_session_cluster_bootstrap":
        return "evidence_insufficient"
    minimum = row.get("min_useful_improvement_cm")
    required = row.get("min_eval_units")
    if minimum is None or minimum <= 0 or required is None or required < 4:
        return "evidence_insufficient"
    if (row.get("n_moving_eval_units") or 0) < required:
        return "evidence_insufficient"
    ci = row.get("delta_moving_ci95")
    if not ci or any(v is None for v in ci):
        return "evidence_insufficient"
    if ci[1] < -minimum:
        return "win_moving"
    if ci[0] > minimum:
        return "loss_moving"
    return "inconclusive"
