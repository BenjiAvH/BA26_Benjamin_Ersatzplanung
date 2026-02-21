"""
Zentrale Validierungen (Vorprüfung/Nachprüfung) für dieses Projekt.

Wichtig: Diese Prüfungen dürfen **keine** Ergebnisse verändern. Sie sind rein defensiv:
- früh abbrechen (Fehler) oder
- warnen (Hinweis),
ohne Daten zu transformieren, RNG/Seeds zu beeinflussen oder Timing-Logik zu ändern.

validate_level:
  - "none"  : keine Validierung
  - "light" : günstige Prüfungen (Existenz, JSON parsebar)
  - "full"  : striktere Prüfungen (Schema-/Vollständigkeit, wo möglich)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

ValidateLevel = Literal["none", "light", "full"]

_LEVEL_RANK: dict[str, int] = {"none": 0, "light": 1, "full": 2}


def normalize_validate_level(value: str | None, *, default: ValidateLevel = "full") -> ValidateLevel:
    v = str(value or "").strip().lower()
    if not v:
        return default
    if v in _LEVEL_RANK:
        return v
    raise ValueError(f"Ungültiges validate_level: {value!r} (erwartet: none|light|full).")


def _enabled(level: ValidateLevel, required: ValidateLevel) -> bool:
    return _LEVEL_RANK[str(level)] >= _LEVEL_RANK[str(required)]


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return not self.errors

    def add_error(self, msg: str) -> None:
        self.errors.append(str(msg))

    def add_warning(self, msg: str) -> None:
        self.warnings.append(str(msg))

    def emit_warnings(self, *, sink: Callable[[str], None] | None = None, prefix: str = "[WARN] ") -> None:
        if sink is None:
            return
        for w in self.warnings:
            sink(f"{prefix}{w}")

    def raise_if_errors(self) -> None:
        if not self.errors:
            return
        if len(self.errors) == 1:
            raise SystemExit(self.errors[0])
        joined = "\n".join(f"- {e}" for e in self.errors)
        raise SystemExit(f"Validierung fehlgeschlagen:\n{joined}")


def require_path_exists(
    path: Path,
    *,
    report: ValidationReport,
    level: ValidateLevel,
    required: ValidateLevel = "light",
    kind: Literal["file", "dir"] | None = None,
    what: str | None = None,
) -> None:
    if not _enabled(level, required):
        return
    p = Path(path)
    label = what or str(p)
    if kind == "file":
        if not p.is_file():
            report.add_error(f"Datei nicht gefunden: {label}")
        return
    if kind == "dir":
        if not p.is_dir():
            report.add_error(f"Ordner nicht gefunden: {label}")
        return
    if not p.exists():
        report.add_error(f"Pfad nicht gefunden: {label}")


def read_json(
    path: Path,
    *,
    report: ValidationReport,
    level: ValidateLevel,
    required: ValidateLevel = "light",
    must_be_object: bool = False,
    what: str | None = None,
) -> Any | None:
    if not _enabled(level, required):
        return None
    p = Path(path)
    label = what or str(p)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        report.add_error(f"JSON konnte nicht gelesen werden: {label} ({type(e).__name__}: {e})")
        return None
    if must_be_object and not isinstance(data, Mapping):
        report.add_error(f"JSON ungültig (Top-Level muss Objekt sein): {label}")
        return None
    return data
