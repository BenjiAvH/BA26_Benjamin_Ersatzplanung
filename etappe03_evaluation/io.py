from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

# === IS-Definition  ==================================
#
# Runs/Lösungen sind über (dataset_id, case_id) eindeutig
# identifizierbar (Beispielpfad: logs/laeufe/<dataset_id>/<case_id>/<verfahren>/...).
# Damit ist ein Instanz–Szenario (IS) in der Evaluation als Tupel:
#
#   IS_ID_FIELDS = ("dataset_id", "case_id")
#
# definiert. Diese Felder sind in runs.jsonl/solutions.jsonl vorhanden.
IS_ID_FIELDS: tuple[str, str] = ("dataset_id", "case_id")


def make_is_id(dataset_id: str, case_id: str) -> str:
    return f"{dataset_id}__{case_id}"


def normalize_verfahren(value: Any) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    s_up = s.upper()
    if s_up in {"V1", "V2", "V3", "V4"}:
        return s_up
    if s_up.startswith("V") and len(s_up) == 2 and s_up[1].isdigit():
        return s_up
    return s_up


@dataclass(frozen=True)
class CaseMeta:
    dataset_id: str
    case_id: str
    size_class: str | None = None
    severity: str | None = None
    availability_level: str | None = None
    demand_pattern: str | None = None
    seeds_instance: int | None = None
    seeds_scenario: int | None = None
    I: int | None = None
    T: int | None = None
    S: int | None = None


@dataclass(frozen=True)
class SolutionRecord:
    dataset_id: str
    case_id: str
    verfahren: str
    group_id: str | None
    subrun_id: str | None
    solution_id: str | None
    F: tuple[float, float, float, float]
    feasible: bool
    elapsed_seconds: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunGroupRecord:
    dataset_id: str
    case_id: str
    verfahren: str
    group_id: str
    seed_global: int | None = None
    seed_case: int | None = None
    seed_group: int | None = None
    budget_total_seconds: float | None = None
    status: str | None = None
    wall_seconds_used: float | None = None
    solutions_found: int | None = None
    time_to_first_feasible_seconds: float | None = None
    raw_start: dict[str, Any] = field(default_factory=dict)
    raw_end: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubrunEndRecord:
    dataset_id: str
    case_id: str
    verfahren: str
    group_id: str
    subrun_id: str
    status: str | None = None
    wall_seconds_used: float | None = None
    solutions_found: int | None = None
    time_to_first_feasible_seconds: float | None = None
    solver_status: str | None = None
    termination_reason: str | None = None
    mip_gap: float | None = None
    best_bound: float | None = None
    best_obj_scalar: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedData:
    case_meta: dict[tuple[str, str], CaseMeta]
    solutions: list[SolutionRecord]
    run_groups: list[RunGroupRecord]
    subrun_ends: list[SubrunEndRecord]
    omegas_v1: list[tuple[float, float, float, float]]
    warnings: list[str]


class _Warnings:
    def __init__(self) -> None:
        self._seen: set[str] = set()
        self.items: list[str] = []

    def add_once(self, key: str, msg: str) -> None:
        if key in self._seen:
            return
        self._seen.add(key)
        self.items.append(msg)


def _iter_jsonl(path: Path, *, warnings: _Warnings) -> Iterator[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    warnings.add_once(f"jsonl:{path}", f"[WARN] JSONL enthält ungültige Zeilen: {path} (z.B. Zeile {lineno})")
                    continue
                if isinstance(rec, dict):
                    yield rec
    except Exception as e:
        warnings.add_once(f"read:{path}", f"[WARN] Konnte Datei nicht lesen: {path} ({e})")
        return


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _parse_F(rec: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    F = rec.get("F")
    if not isinstance(F, Mapping):
        return None
    try:
        return (float(F["f_stab"]), float(F["f_ot"]), float(F["f_pref"]), float(F["f_fair"]))
    except Exception:
        return None


def load_case_meta_from_manifests(manifest_paths: Iterable[Path], *, warnings: _Warnings) -> dict[tuple[str, str], CaseMeta]:
    out: dict[tuple[str, str], CaseMeta] = {}
    for path in sorted({Path(p) for p in manifest_paths}):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            warnings.add_once(f"manifest:{path}", f"[WARN] manifest.json konnte nicht gelesen werden: {path} ({e})")
            continue
        if not isinstance(data, dict):
            warnings.add_once(f"manifest:{path}", f"[WARN] manifest.json ungültig (Top-Level muss Objekt sein): {path}")
            continue

        dataset_id = str(data.get("dataset_id", "")).strip()
        cases = data.get("cases")
        if not dataset_id or not isinstance(cases, list):
            warnings.add_once(f"manifest_schema:{path}", f"[WARN] manifest.json Schema unerwartet: {path}")
            continue

        for entry in cases:
            if not isinstance(entry, dict):
                continue
            case_id = str(entry.get("case_id", "")).strip()
            if not case_id:
                continue

            tags = entry.get("tags") if isinstance(entry.get("tags"), dict) else {}
            scenario = entry.get("scenario") if isinstance(entry.get("scenario"), dict) else {}
            seeds = entry.get("seeds") if isinstance(entry.get("seeds"), dict) else {}

            size_class = str(tags.get("size_class", "")).strip() or None
            availability_level = str(tags.get("availability_level", "")).strip() or None
            demand_pattern = str(tags.get("demand_pattern", "")).strip() or None
            severity = str(scenario.get("severity", "")).strip() or None

            seeds_instance = _parse_int(seeds.get("instance"))
            seeds_scenario = _parse_int(seeds.get("scenario"))

            dims = entry.get("dimensions") if isinstance(entry.get("dimensions"), dict) else {}
            I = _parse_int(dims.get("I") if isinstance(dims, dict) else None)
            T = _parse_int(dims.get("T") if isinstance(dims, dict) else None)
            S = _parse_int(dims.get("S") if isinstance(dims, dict) else None)

            out[(dataset_id, case_id)] = CaseMeta(
                dataset_id=dataset_id,
                case_id=case_id,
                size_class=size_class,
                severity=severity,
                availability_level=availability_level,
                demand_pattern=demand_pattern,
                seeds_instance=seeds_instance,
                seeds_scenario=seeds_scenario,
                I=I,
                T=T,
                S=S,
            )
    return out


def discover_input_files(input_root: Path) -> dict[str, list[Path]]:
    root = Path(input_root)
    manifests = sorted(root.rglob("manifest.json"))
    runs = sorted(root.rglob("runs.jsonl"))
    solutions = sorted(root.rglob("solutions.jsonl"))
    return {"manifests": manifests, "runs": runs, "solutions": solutions}


def load_input(
    input_root: Path,
    *,
    methods: set[str] | None = None,
    seed_global: int | None = None,
) -> LoadedData:
    """
    Liest Ergebnisse aus Etappe 1/2 ein und vereinheitlicht sie.

    Unterstützte, im Projekt vorhandene Artefakte (Ist-Stand):
      - Case-Metadaten: logs/daten/<dataset_id>/manifest.json (optional, aber empfohlen)
      - Runs:           logs/laeufe/<dataset_id>/<case_id>/<verfahren>/runs.jsonl
      - Lösungen:       logs/laeufe/<dataset_id>/<case_id>/<verfahren>/solutions.jsonl

    Filter:
      - methods: Menge aus {"V1","V2","V3","V4"} (falls None: alles laden)
      - seed_global: nur Run-Groups mit diesem seed_global (falls vorhanden)
    """

    warnings = _Warnings()
    files = discover_input_files(Path(input_root))

    case_meta = load_case_meta_from_manifests(files["manifests"], warnings=warnings)

    # --- Runs laden -----------------------------------------------------------
    run_groups: list[RunGroupRecord] = []
    subrun_ends: list[SubrunEndRecord] = []
    omegas_v1_set: set[tuple[float, float, float, float]] = set()

    for path in files["runs"]:
        group_start_by_id: dict[str, dict[str, Any]] = {}
        group_end_by_id: dict[str, dict[str, Any]] = {}

        for rec in _iter_jsonl(path, warnings=warnings):
            event = str(rec.get("event", "")).strip()
            group_id = str(rec.get("group_id", "")).strip()
            if not group_id:
                continue

            verfahren = normalize_verfahren(rec.get("verfahren"))
            if not verfahren:
                warnings.add_once(f"no_verfahren:{path}", f"[WARN] runs.jsonl ohne Feld 'verfahren': {path}")
                continue
            if methods is not None and verfahren not in methods:
                continue

            if event == "group_start":
                group_start_by_id[group_id] = rec

                # ω-Extraktion für V1 (Win-Rate-Heatmap; Kap. 5.3: Metriken & Vergleichsauswertung).
                if verfahren == "V1":
                    subruns = rec.get("subruns")
                    if isinstance(subruns, list):
                        for sr in subruns:
                            if not isinstance(sr, dict):
                                continue
                            params = sr.get("params")
                            if not isinstance(params, dict):
                                continue
                            omega = params.get("omega")
                            if isinstance(omega, (list, tuple)) and len(omega) == 4:
                                try:
                                    omegas_v1_set.add((float(omega[0]), float(omega[1]), float(omega[2]), float(omega[3])))
                                except Exception:
                                    pass

                    cfg = rec.get("config_snapshot")
                    if isinstance(cfg, dict):
                        omegas = cfg.get("omegas")
                        if isinstance(omegas, list):
                            for omega in omegas:
                                if isinstance(omega, (list, tuple)) and len(omega) == 4:
                                    try:
                                        omegas_v1_set.add((float(omega[0]), float(omega[1]), float(omega[2]), float(omega[3])))
                                    except Exception:
                                        pass

            elif event == "group_end":
                group_end_by_id[group_id] = rec

            elif event == "subrun_end":
                dataset_id = str(rec.get("dataset_id") or "").strip()
                case_id = str(rec.get("case_id") or "").strip()
                if not dataset_id or not case_id:
                    warnings.add_once(f"subrun_ids_missing:{path}", f"[WARN] subrun_end ohne dataset_id/case_id: {path}")
                    continue

                # seed-filter hier bewusst NICHT anwenden: Subrun-End-Infos sind Debug.
                subrun_id = str(rec.get("subrun_id", "")).strip()
                if not subrun_id:
                    continue

                subrun_ends.append(
                    SubrunEndRecord(
                        dataset_id=dataset_id,
                        case_id=case_id,
                        verfahren=verfahren,
                        group_id=group_id,
                        subrun_id=subrun_id,
                        status=str(rec.get("status", "")).strip() or None,
                        wall_seconds_used=_parse_float(rec.get("wall_seconds_used")),
                        solutions_found=_parse_int(rec.get("solutions_found")),
                        time_to_first_feasible_seconds=_parse_float(rec.get("time_to_first_feasible_seconds")),
                        solver_status=str(rec.get("solver_status", "")).strip() or None,
                        termination_reason=str(rec.get("termination_reason", "")).strip() or None,
                        mip_gap=_parse_float(rec.get("mip_gap")),
                        best_bound=_parse_float(rec.get("best_bound")),
                        best_obj_scalar=_parse_float(rec.get("best_obj_scalar")),
                        raw=dict(rec),
                    )
                )

        # Gruppen materialisieren (union aus start/end).
        group_ids = sorted(set(group_start_by_id) | set(group_end_by_id))
        for group_id in group_ids:
            start = group_start_by_id.get(group_id, {})
            end = group_end_by_id.get(group_id, {})

            dataset_id = str((end.get("dataset_id") or start.get("dataset_id") or "") or "").strip()
            case_id = str((end.get("case_id") or start.get("case_id") or "") or "").strip()
            verfahren = normalize_verfahren((end.get("verfahren") or start.get("verfahren") or "") or "")
            if not dataset_id or not case_id or not verfahren:
                warnings.add_once(
                    f"group_ids_missing:{path}",
                    f"[WARN] group_start/group_end ohne dataset_id/case_id/verfahren: {path}",
                )
                continue
            if methods is not None and verfahren not in methods:
                continue

            seed_val = _parse_int(start.get("seed_global"))
            if seed_global is not None and seed_val is not None and seed_val != seed_global:
                continue

            run_groups.append(
                RunGroupRecord(
                    dataset_id=dataset_id,
                    case_id=case_id,
                    verfahren=verfahren,
                    group_id=group_id,
                    seed_global=seed_val,
                    seed_case=_parse_int(start.get("seed_case")),
                    seed_group=_parse_int(start.get("seed_group")),
                    budget_total_seconds=_parse_float(start.get("budget_total_seconds")),
                    status=str(end.get("status", "")).strip() or None,
                    wall_seconds_used=_parse_float(end.get("wall_seconds_used")),
                    solutions_found=_parse_int(end.get("solutions_found")),
                    time_to_first_feasible_seconds=_parse_float(end.get("time_to_first_feasible_seconds")),
                    raw_start=dict(start) if isinstance(start, dict) else {},
                    raw_end=dict(end) if isinstance(end, dict) else {},
                )
            )

    # --- Lösungen laden -------------------------------------------------------
    solutions: list[SolutionRecord] = []

    for path in files["solutions"]:
        for rec in _iter_jsonl(path, warnings=warnings):
            dataset_id = str(rec.get("dataset_id") or "").strip()
            case_id = str(rec.get("case_id") or "").strip()
            verfahren = normalize_verfahren(rec.get("verfahren"))
            if not dataset_id or not case_id or not verfahren:
                warnings.add_once(f"solution_ids_missing:{path}", f"[WARN] solutions.jsonl ohne dataset_id/case_id/verfahren: {path}")
                continue
            if methods is not None and verfahren not in methods:
                continue

            F = _parse_F(rec)
            if F is None:
                warnings.add_once(f"no_F:{path}", f"[WARN] solutions.jsonl ohne gültigen Zielvektor F entdeckt: {path}")
                continue

            feasible_raw = rec.get("feasible", None)
            if not isinstance(feasible_raw, bool):
                warnings.add_once(
                    f"feasible_missing_or_invalid:{path}",
                    f"[WARN] solutions.jsonl ohne boolsches Feld 'feasible': {path}",
                )
                continue
            feasible = bool(feasible_raw)

            group_id = str(rec.get("group_id", "")).strip() or None
            subrun_id = str(rec.get("subrun_id", "")).strip() or None
            solution_id = str(rec.get("solution_id", "")).strip() or None

            solutions.append(
                SolutionRecord(
                    dataset_id=dataset_id,
                    case_id=case_id,
                    verfahren=verfahren,
                    group_id=group_id,
                    subrun_id=subrun_id,
                    solution_id=solution_id,
                    F=F,
                    feasible=feasible,
                    elapsed_seconds=_parse_float(rec.get("elapsed_seconds")),
                    raw=dict(rec),
                )
            )

    omegas_v1 = sorted(omegas_v1_set)
    return LoadedData(
        case_meta=case_meta,
        solutions=sorted(
            solutions,
            key=lambda r: (r.dataset_id, r.case_id, r.verfahren, str(r.group_id or ""), r.F, str(r.solution_id or "")),
        ),
        run_groups=sorted(run_groups, key=lambda r: (r.dataset_id, r.case_id, r.verfahren, r.group_id)),
        subrun_ends=sorted(subrun_ends, key=lambda r: (r.dataset_id, r.case_id, r.verfahren, r.group_id, r.subrun_id)),
        omegas_v1=omegas_v1,
        warnings=warnings.items,
    )
