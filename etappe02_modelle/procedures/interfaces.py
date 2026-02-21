from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from etappe02_modelle.core.case_loader import LoadedCase
from etappe02_modelle.core.schema import ScheduleDense, SubrunSpec


class Procedure(Protocol):
    """
    Schnittstelle für Verfahren V1–V4.

    Das Etappe-2-Grundgerüst implementiert nur Orchestrierung/Framework; konkrete
    Solver-/Heuristiklogik liegt in den Verfahren selbst.
    """

    verfahren: str

    def plan_subruns(self, config_snapshot: dict[str, Any]) -> list[SubrunSpec]:
        """Erzeugt einen deterministischen Subrun-Plan (für Budget-Splitting und Logging)."""

    def run_subrun(
        self,
        *,
        loaded_case: LoadedCase,
        subrun: SubrunSpec,
        seed_subrun: int,
        deadline_monotonic: float,
    ) -> Iterable[ScheduleDense]:
        """Liefert (ggf. inkrementell) Kandidaten-Schedules; muss Budget/Deadline respektieren."""


def _subruns_from_explicit_list(cfg: dict[str, Any]) -> list[SubrunSpec] | None:
    raw = cfg.get("subruns")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("Config ungültig: subruns muss ein Array sein.")

    subruns: list[SubrunSpec] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Config ungültig: subruns[{idx}] ist kein Objekt.")
        subrun_id = item.get("subrun_id", str(idx))
        params = item.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"Config ungültig: subruns[{idx}].params ist kein Objekt.")
        subruns.append(SubrunSpec(subrun_id=str(subrun_id), params=dict(params)))
    return subruns


__all__ = ["Procedure", "_subruns_from_explicit_list"]

