from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from etappe02_modelle.core.pareto import (
    dedupe_by_objectives_keep_min_solution_id,
    dedupe_by_solution_id,
    non_dominated,
)
from etappe02_modelle.procedures import normalize_verfahren
from tools.validation import ValidationReport, read_json


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict):
                yield rec


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_manifest(path: Path) -> dict[str, Any]:
    report = ValidationReport()
    data = read_json(path, report=report, level="full", must_be_object=True, what=str(path))
    report.raise_if_errors()
    assert isinstance(data, dict)
    return data


def _export_A_v(*, out_root: Path, dataset_id: str, case_id: str, verfahren: str) -> None:
    proc_dir = out_root / case_id / verfahren
    sol_path = proc_dir / "solutions.jsonl"
    export_path = proc_dir / "exports" / "A_v.jsonl"

    records = list(_iter_jsonl(sol_path))
    records = dedupe_by_solution_id(records, get_solution_id=lambda r: str(r.get("solution_id", "")))
    records = dedupe_by_objectives_keep_min_solution_id(
        records,
        get_F=lambda r: r.get("F", {}),
        get_solution_id=lambda r: str(r.get("solution_id", "")),
    )
    records = non_dominated(records, get_F=lambda r: r.get("F", {}), get_solution_id=lambda r: str(r.get("solution_id", "")))

    # Minimaler Export: genug für ND/Contribution/Coverage in Etappe 3.
    out_recs: list[dict[str, Any]] = []
    for r in records:
        out_recs.append(
            {
                "dataset_id": dataset_id,
                "case_id": case_id,
                "verfahren": verfahren,
                "solution_id": r.get("solution_id"),
                "F": r.get("F"),
                "schedule_delta": r.get("schedule_delta"),
                "source": r.get("source"),
            }
        )
    _write_jsonl(export_path, out_recs)
    print(f"[export] A_v {dataset_id}/{case_id}/{verfahren}: {export_path} ({len(out_recs)} Punkte)")


def _export_P_star(*, out_root: Path, dataset_id: str, case_id: str) -> None:
    export_path = out_root / case_id / "exports" / "P_star.jsonl"

    all_records: list[dict[str, Any]] = []
    for verfahren in ["V1", "V2", "V3", "V4"]:
        sol_path = out_root / case_id / verfahren / "solutions.jsonl"
        all_records.extend(list(_iter_jsonl(sol_path)))

    all_records = dedupe_by_solution_id(all_records, get_solution_id=lambda r: str(r.get("solution_id", "")))
    all_records = dedupe_by_objectives_keep_min_solution_id(
        all_records,
        get_F=lambda r: r.get("F", {}),
        get_solution_id=lambda r: str(r.get("solution_id", "")),
    )
    all_records = non_dominated(
        all_records,
        get_F=lambda r: r.get("F", {}),
        get_solution_id=lambda r: str(r.get("solution_id", "")),
    )

    out_recs: list[dict[str, Any]] = []
    for r in all_records:
        out_recs.append(
            {
                "dataset_id": dataset_id,
                "case_id": case_id,
                "verfahren": r.get("verfahren"),
                "solution_id": r.get("solution_id"),
                "F": r.get("F"),
                "schedule_delta": r.get("schedule_delta"),
                "source": r.get("source"),
            }
        )

    _write_jsonl(export_path, out_recs)
    print(f"[export] P_star {dataset_id}/{case_id}: {export_path} ({len(out_recs)} Punkte)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Etappe 02: Exportiert Approximationsmengen A_v bzw. die empirische Referenzmenge P* (Dedupe + ND-Filter)."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Pfad zu manifest.json (z.B. logs/daten/<dataset_id>/manifest.json).")
    parser.add_argument("--out", type=Path, required=True, help="Output-Root der Läufe, z.B. logs/laeufe/<dataset_id> oder /tmp/laeufe/<dataset_id>.")
    parser.add_argument("--export", type=str, required=True, choices=["A_v", "P_star"], help="Export-Typ: A_v (pro Verfahren) oder P_star (Union, ND-gefiltert).")
    parser.add_argument("--verfahren", type=str, help="Nur für A_v: V1|V2|V3|V4.")
    args = parser.parse_args()

    manifest = _load_manifest(args.dataset)
    manifest_report = ValidationReport()
    dataset_id = str(manifest.get("dataset_id", ""))
    cases_raw = manifest.get("cases")
    if not dataset_id:
        manifest_report.add_error("manifest.json ungültig: dataset_id fehlt/leer.")
    if not isinstance(cases_raw, list):
        manifest_report.add_error("manifest.json ungültig: cases ist kein Array.")
    manifest_report.raise_if_errors()
    assert isinstance(cases_raw, list)
    cases: list[dict[str, Any]] = [entry for entry in cases_raw if isinstance(entry, dict)]

    cases_sorted = sorted(cases, key=lambda e: str(e.get("case_id", "")))

    if args.export == "A_v":
        export_report = ValidationReport()
        if not args.verfahren:
            export_report.add_error("--verfahren ist für --export A_v erforderlich.")
        export_report.raise_if_errors()
        verfahren = normalize_verfahren(args.verfahren)
        for entry in cases_sorted:
            case_id = str(entry.get("case_id", ""))
            if not case_id:
                continue
            _export_A_v(out_root=args.out, dataset_id=dataset_id, case_id=case_id, verfahren=verfahren)
        return 0

    if args.export == "P_star":
        for entry in cases_sorted:
            case_id = str(entry.get("case_id", ""))
            if not case_id:
                continue
            _export_P_star(out_root=args.out, dataset_id=dataset_id, case_id=case_id)
        return 0

    raise SystemExit("Unbekannter Export-Typ.")


if __name__ == "__main__":
    raise SystemExit(main())
