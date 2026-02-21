from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping


def canonical_json_dumps(data: Any) -> str:
    """
    Kanonisches JSON-Encoding (deterministisch).

    Wichtig für Config-Hashes und ID-Bildung (Kap. 5.1: Forschungsdesign & Evaluationsstrategie; Reproduzierbarkeit).
    """

    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def compute_config_hash(config_snapshot: Mapping[str, Any]) -> str:
    """
    Hash einer vollständig aufgelösten Config (kanonisch; JSON).
    """

    return sha256_hex(canonical_json_dumps(dict(config_snapshot)).encode("utf-8"))


def compute_group_id(
    *,
    dataset_id: str,
    case_id: str,
    verfahren: str,
    config_hash: str,
    seed_global: int,
    budget_total_seconds: float,
) -> str:
    """
    Deterministische Group-ID:
      sha256(dataset_id|case_id|verfahren|config_hash|seed_global|budget_total)

    Für das Budget wird eine millisekundengenaue, kanonische Darstellung verwendet,
    um Float-Repräsentationsartefakte zu vermeiden.
    """

    total_ms = int(round(float(budget_total_seconds) * 1000.0))
    payload = f"{dataset_id}|{case_id}|{verfahren}|{config_hash}|{int(seed_global)}|{total_ms}"
    return sha256_hex(payload.encode("utf-8"))


ScheduleDense = list[list[int]]


@dataclass(frozen=True)
class ObjectiveVector:
    f_stab: int
    f_ot: int
    f_pref: int
    f_fair: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (int(self.f_stab), int(self.f_ot), int(self.f_pref), int(self.f_fair))

    def to_dict(self) -> dict[str, int]:
        return {
            "f_stab": int(self.f_stab),
            "f_ot": int(self.f_ot),
            "f_pref": int(self.f_pref),
            "f_fair": int(self.f_fair),
        }


@dataclass(frozen=True)
class SubrunSpec:
    subrun_id: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"subrun_id": str(self.subrun_id), "params": dict(self.params)}


@dataclass(frozen=True)
class PlannedSubrun:
    subrun_id: str
    seed_subrun: int
    budget_seconds: float
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subrun_id": str(self.subrun_id),
            "seed_subrun": int(self.seed_subrun),
            "budget_seconds": float(self.budget_seconds),
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class RunPlan:
    dataset_id: str
    case_id: str
    verfahren: str
    group_id: str
    seed_global: int
    seed_case: int
    seed_group: int
    budget_total_seconds: float
    subruns: list[PlannedSubrun]
    config_snapshot: dict[str, Any]
    config_hash: str
    env: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "dataset_id": str(self.dataset_id),
            "case_id": str(self.case_id),
            "verfahren": str(self.verfahren),
            "group_id": str(self.group_id),
            "seed_global": int(self.seed_global),
            "seed_case": int(self.seed_case),
            "seed_group": int(self.seed_group),
            "budget_total_seconds": float(self.budget_total_seconds),
            "subruns": [sr.to_dict() for sr in self.subruns],
            "config_snapshot": dict(self.config_snapshot),
            "config_hash": str(self.config_hash),
            "env": dict(self.env),
        }
        if self.warnings:
            out["warnings"] = list(self.warnings)
        return out


@dataclass(frozen=True)
class RunEvent:
    ts_utc: str
    event: str
    dataset_id: str
    case_id: str
    verfahren: str
    group_id: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ts_utc": str(self.ts_utc),
            "event": str(self.event),
            "dataset_id": str(self.dataset_id),
            "case_id": str(self.case_id),
            "verfahren": str(self.verfahren),
            "group_id": str(self.group_id),
        }
        out.update(dict(self.payload))
        return out


@dataclass(frozen=True)
class SolutionRecord:
    ts_utc: str
    elapsed_seconds: float
    dataset_id: str
    case_id: str
    verfahren: str
    group_id: str
    subrun_id: str
    solution_id: str
    F: ObjectiveVector
    schedule_delta: dict[str, Any]
    source: dict[str, Any]
    schedule_dense: ScheduleDense | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ts_utc": str(self.ts_utc),
            "elapsed_seconds": float(self.elapsed_seconds),
            "dataset_id": str(self.dataset_id),
            "case_id": str(self.case_id),
            "verfahren": str(self.verfahren),
            "group_id": str(self.group_id),
            "subrun_id": str(self.subrun_id),
            "solution_id": str(self.solution_id),
            "F": self.F.to_dict(),
            "schedule_delta": dict(self.schedule_delta),
            "source": dict(self.source),
        }
        if self.schedule_dense is not None:
            out["schedule_dense"] = self.schedule_dense
        return out
