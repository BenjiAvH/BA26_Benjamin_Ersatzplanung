from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from etappe01_simulation.simulator.schema import Case, SchemaError
from etappe01_simulation.simulator.validate import validate_case


class CaseLoadError(ValueError):
    """Fehler beim Laden/Validieren eines Case-JSONs (Etappe 2)."""


def _derive_default_weeks(T: int) -> list[list[int]]:
    # Identisch zur Fallback-Logik in etappe01_simulation.simulator.validate.validate_case.
    return [list(range(w * 7 + 1, min(T, (w + 1) * 7) + 1)) for w in range((T + 6) // 7)]


def _normalize_weeks(T: int, weeks_raw: Any) -> tuple[list[list[int]], list[str]]:
    warnings: list[str] = []

    if not isinstance(weeks_raw, list) or not weeks_raw:
        weeks = _derive_default_weeks(T)
        warnings.append("weeks.T_w fehlt/leer/ungültig; wird deterministisch neu abgeleitet (7-Tage-Blöcke).")
        return weeks, warnings

    weeks: list[list[int]] = []
    for idx, days in enumerate(weeks_raw):
        if not isinstance(days, list):
            warnings.append(f"weeks.T_w[{idx}] ist kein Array; wird ignoriert und Wochen werden neu abgeleitet.")
            weeks = _derive_default_weeks(T)
            return weeks, warnings
        weeks.append([int(t) for t in days])

    all_days = set(range(1, T + 1))
    seen: set[int] = set()
    overlap = False
    out_of_range: list[int] = []
    for days in weeks:
        for t in days:
            if t in seen:
                overlap = True
            seen.add(t)
            if t not in all_days:
                out_of_range.append(int(t))

    cover_ok = seen == all_days
    if overlap or (not cover_ok) or out_of_range:
        weeks_new = _derive_default_weeks(T)
        missing = sorted(all_days - seen)
        msg = "weeks.T_w ist nicht Cover+NoOverlap"
        details: list[str] = []
        if overlap:
            details.append("overlap=True")
        if not cover_ok:
            details.append(f"missing_days={missing}")
        if out_of_range:
            details.append(f"out_of_range={sorted(set(out_of_range))}")
        warnings.append(f"{msg} ({', '.join(details)}); wird deterministisch neu abgeleitet (7-Tage-Blöcke).")
        return weeks_new, warnings

    return weeks, warnings


def _ensure_P_subset_S_plus(case: Case) -> None:
    S_plus_raw = case.sets.get("S_plus")
    if isinstance(S_plus_raw, list) and S_plus_raw:
        S_plus = {int(s) for s in S_plus_raw}
    else:
        S_plus = set(range(1, int(case.dimensions.S)))

    P_raw = case.params.get("P", [])
    if not isinstance(P_raw, list):
        raise CaseLoadError("Case ungültig: params.P ist kein Array.")

    bad: list[tuple[int, int]] = []
    for pair in P_raw:
        try:
            s, sp = int(pair[0]), int(pair[1])
        except Exception as e:
            raise CaseLoadError(f"Case ungültig: params.P enthält kein Paar: {pair}") from e
        if s not in S_plus or sp not in S_plus:
            bad.append((s, sp))

    if bad:
        raise CaseLoadError(f"Case abgelehnt: params.P enthält Paare außerhalb S_plus×S_plus: {bad}")


@dataclass(frozen=True)
class LoadedCase:
    case: Case
    case_dict: dict[str, Any]
    validation: dict[str, Any]
    weeks_T_w: list[list[int]]
    warnings: list[str] = field(default_factory=list)


def load_case(
    path: Path,
    *,
    strict_gate: bool = True,
) -> LoadedCase:
    """
    Lädt ein Case-JSON, parst es via Case.from_dict und führt harte Prüfungen aus.

    Standardmäßig ist das Gate strikt: validation.ok_strict == True (Kap. 5.1: Forschungsdesign & Evaluationsstrategie; fairer Vergleich).
    """

    try:
        case_dict = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise CaseLoadError(f"Case konnte nicht gelesen werden: {path}") from e

    try:
        case = Case.from_dict(case_dict)
    except SchemaError as e:
        raise CaseLoadError(f"Case-JSON entspricht nicht dem Schema: {path}") from e

    validation = validate_case(case_dict)
    key = "ok_strict" if strict_gate else "ok"
    ok = bool(validation.get(key, False))
    if not ok:
        notes = validation.get("notes", [])
        raise CaseLoadError(
            "Case abgelehnt durch Validierungs-Gate "
            f"({key}=False). Notes: {notes if isinstance(notes, list) else [str(notes)]}"
        )

    T = int(case.dimensions.T)
    weeks_raw = case.weeks.get("T_w")
    weeks_T_w, warnings = _normalize_weeks(T, weeks_raw)

    _ensure_P_subset_S_plus(case)

    return LoadedCase(case=case, case_dict=case_dict, validation=validation, weeks_T_w=weeks_T_w, warnings=warnings)


def load_config_json(path: Path) -> dict[str, Any]:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Config konnte nicht gelesen werden: {path}") from e

    if not isinstance(cfg, dict):
        raise ValueError(f"Config muss ein JSON-Objekt sein: {path}")
    return cfg


def ensure_manifest_schema(manifest: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    dataset_id = str(manifest.get("dataset_id", ""))
    cases = manifest.get("cases")
    if not dataset_id:
        raise ValueError("manifest.json ungültig: dataset_id fehlt/leer.")
    if not isinstance(cases, list):
        raise ValueError("manifest.json ungültig: cases ist kein Array.")
    out: list[dict[str, Any]] = []
    for entry in cases:
        if not isinstance(entry, dict):
            continue
        out.append(entry)
    return dataset_id, out
