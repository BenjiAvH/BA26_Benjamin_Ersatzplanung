from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from etappe02_modelle.scripts.run_case import run_case
from tools.validation import ValidationReport, read_json


def _pick_existing_path(path: Path, *, bases: list[Path]) -> Path:
    """
    Löst relative Pfade robust auf, ohne sie blind umzubasen.

    Auflösungsreihenfolge für relative Pfade:
      1) erster existierender Treffer unter den angegebenen Basispfaden (in Reihenfolge)
      2) unverändert (relativ zum aktuellen Arbeitsverzeichnis) – aber nur, wenn der Pfad existiert
      3) Fallback: Original beibehalten (für klare Fehlermeldungen)
    """

    if path.is_absolute():
        return path

    for base in bases:
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()

    if path.exists():
        return path.resolve()

    return path


def _load_manifest(path: Path) -> dict[str, Any]:
    report = ValidationReport()
    data = read_json(path, report=report, level="full", must_be_object=True, what=str(path))
    report.raise_if_errors()
    assert isinstance(data, dict)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Etappe 02: Führt einen Run-Plan über ein Dataset aus (deterministische Case-Reihenfolge; ruft run_case pro (Case, Verfahren) auf)."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Pfad zu manifest.json (z.B. logs/daten/<dataset_id>/manifest.json).")
    parser.add_argument("--plan", type=Path, required=True, help="Run-Plan (JSON), z.B. etappe02_modelle/configs/run_dataset.final.json.")
    parser.add_argument("--out", type=Path, required=True, help="Output-Root für Läufe, z.B. logs/laeufe/<dataset_id> oder /tmp/laeufe/<dataset_id>.")
    parser.add_argument("--seed", type=int, required=True, help="Globaler Seed (Reproduzierbarkeit).")
    parser.add_argument("--budget", type=float, required=True, help="Wall-Clock-Gesamtbudget pro (Case, Verfahren) in Sekunden.")
    parser.add_argument("--resume", action="store_true", help="Fortsetzen (Resume): überspringt Gruppen mit group_end(status=ok).")
    parser.add_argument(
        "--allow-nonstrict",
        action="store_true",
        help="Deaktiviert das Strict-Gate (akzeptiert validation_ok=True, auch wenn validation_ok_strict=False). Standard: strikt.",
    )
    args = parser.parse_args()

    # Hinweis: run_dataset kann (mit absoluten --dataset/--plan) aus beliebigem CWD gestartet werden
    # (Config-/Case-Pfade werden robust aufgelöst).
    # Manueller Test (CWD-Unabhängigkeit für Konfigurations-/Case-Pfade):
    # - Start im Repository-Root mit dem Standardplan (Root-relative Pfade in Plan+Manifest)
    # - Start aus einem anderen CWD mit absoluten --dataset/--plan Pfaden (gleicher Plan-/Manifest-Inhalt)
    # Erwartung: Konfigurations-/Case-Pfade werden ohne doppelte Pfadsegmente aufgelöst.

    repo_root = Path(__file__).resolve().parents[2]
    plan_dir = args.plan.resolve().parent
    manifest_dir = args.dataset.resolve().parent

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

    plan_report = ValidationReport()
    plan = read_json(args.plan, report=plan_report, level="full", must_be_object=True, what=str(args.plan))
    plan_report.raise_if_errors()
    assert isinstance(plan, dict)
    procedures = plan.get("procedures")
    if not isinstance(procedures, list) or not procedures:
        plan_report.add_error("Run-Plan ungültig: procedures fehlt/leer.")
        plan_report.raise_if_errors()

    proc_items: list[tuple[str, Path]] = []
    for idx, item in enumerate(procedures):
        if not isinstance(item, dict):
            plan_report.add_error(f"Run-Plan ungültig: procedures[{idx}] ist kein Objekt.")
            continue
        verfahren = str(item.get("verfahren", "")).upper()
        cfg = item.get("config")
        if not verfahren or cfg is None:
            plan_report.add_error(f"Run-Plan ungültig: procedures[{idx}] braucht verfahren+config.")
            continue
        cfg_path = _pick_existing_path(Path(str(cfg)), bases=[plan_dir, repo_root])
        proc_items.append((verfahren, cfg_path))
    plan_report.raise_if_errors()

    cases_sorted = sorted(cases, key=lambda e: str(e.get("case_id", "")))
    failures = 0
    for entry in cases_sorted:
        case_id = str(entry.get("case_id", ""))
        case_file = entry.get("file")
        if not case_id or not case_file:
            continue
        case_path = _pick_existing_path(Path(str(case_file)), bases=[manifest_dir, repo_root])

        strict_ok = bool(entry.get("validation_ok_strict", False))
        ok = bool(entry.get("validation_ok", False))
        if not args.allow_nonstrict and not strict_ok:
            print(f"[skip/validation_strict] {dataset_id}/{case_id}")
            continue
        if args.allow_nonstrict and not ok:
            print(f"[skip/validation_ok] {dataset_id}/{case_id}")
            continue

        for verfahren, cfg_path in proc_items:
            rc = run_case(
                case_path=case_path,
                verfahren=verfahren,
                config_path=cfg_path,
                out_root=args.out,
                seed_global=int(args.seed),
                budget_total_seconds=float(args.budget),
                resume=bool(args.resume),
                allow_nonstrict=bool(args.allow_nonstrict),
            )
            if rc != 0:
                failures += 1

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
