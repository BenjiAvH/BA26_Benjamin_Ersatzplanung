from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from typing import Any, Mapping


class SchemaError(ValueError):
    """Fehler beim (De-)Serialisieren eines Case-JSONs."""


def dumps_json(data: Any) -> str:
    if hasattr(data, "to_dict"):
        data = data.to_dict()
    elif is_dataclass(data):
        data = asdict(data)
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"


@dataclass(frozen=True)
class Seeds:
    global_seed: int
    instance: int
    scenario: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Seeds":
        try:
            return cls(
                global_seed=int(data["global"]),
                instance=int(data["instance"]),
                scenario=int(data["scenario"]),
            )
        except Exception as e:
            raise SchemaError(f"Ungültiges seeds-Objekt: {data}") from e

    def to_dict(self) -> dict[str, int]:
        return {"global": int(self.global_seed), "instance": int(self.instance), "scenario": int(self.scenario)}


@dataclass(frozen=True)
class Dimensions:
    I: int
    T: int
    S: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Dimensions":
        try:
            return cls(I=int(data["I"]), T=int(data["T"]), S=int(data["S"]))
        except Exception as e:
            raise SchemaError(f"Ungültiges dimensions-Objekt: {data}") from e

    def to_dict(self) -> dict[str, int]:
        return {"I": int(self.I), "T": int(self.T), "S": int(self.S)}


@dataclass(frozen=True)
class ScenarioEvent:
    i: int
    t_start: int
    duration: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenarioEvent":
        try:
            return cls(i=int(data["i"]), t_start=int(data["t_start"]), duration=int(data["duration"]))
        except Exception as e:
            raise SchemaError(f"Ungültiges scenario.events-Element: {data}") from e

    def to_dict(self) -> dict[str, int]:
        return {"i": int(self.i), "t_start": int(self.t_start), "duration": int(self.duration)}


@dataclass(frozen=True)
class Scenario:
    severity: str
    model: str
    events: list[ScenarioEvent]
    status: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Scenario":
        try:
            events_raw = list(data.get("events", []))
            return cls(
                severity=str(data["severity"]),
                model=str(data["model"]),
                events=[ScenarioEvent.from_dict(ev) for ev in events_raw],
                status=str(data.get("status", "unknown")),
            )
        except Exception as e:
            raise SchemaError(f"Ungültiges scenario-Objekt: {data}") from e

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": str(self.severity),
            "model": str(self.model),
            "events": [ev.to_dict() for ev in self.events],
            "status": str(self.status),
        }


@dataclass(frozen=True)
class Tags:
    size_class: str
    availability_level: str
    demand_pattern: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Tags":
        try:
            return cls(
                size_class=str(data["size_class"]),
                availability_level=str(data["availability_level"]),
                demand_pattern=str(data["demand_pattern"]),
            )
        except Exception as e:
            raise SchemaError(f"Ungültiges tags-Objekt: {data}") from e

    def to_dict(self) -> dict[str, str]:
        return {
            "size_class": str(self.size_class),
            "availability_level": str(self.availability_level),
            "demand_pattern": str(self.demand_pattern),
        }


@dataclass(frozen=True)
class Case:
    schema_version: str
    dataset_id: str
    case_id: str
    seeds: Seeds
    dimensions: Dimensions
    sets: dict[str, Any]
    weeks: dict[str, Any]
    params: dict[str, Any]
    scenario: Scenario
    tags: Tags
    validation: dict[str, Any] | None = None
    generated_at: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Case":
        try:
            return cls(
                schema_version=str(data["schema_version"]),
                dataset_id=str(data["dataset_id"]),
                case_id=str(data["case_id"]),
                seeds=Seeds.from_dict(data["seeds"]),
                dimensions=Dimensions.from_dict(data["dimensions"]),
                sets=dict(data["sets"]),
                weeks=dict(data["weeks"]),
                params=dict(data["params"]),
                scenario=Scenario.from_dict(data["scenario"]),
                tags=Tags.from_dict(data["tags"]),
                validation=dict(data["validation"]) if "validation" in data else None,
                generated_at=str(data["generated_at"]) if "generated_at" in data else None,
            )
        except Exception as e:
            raise SchemaError("Ungültiges Case-JSON (Top-Level).") from e

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": str(self.schema_version),
            "dataset_id": str(self.dataset_id),
            "case_id": str(self.case_id),
            "seeds": self.seeds.to_dict(),
            "dimensions": self.dimensions.to_dict(),
            "sets": self.sets,
            "weeks": self.weeks,
            "params": self.params,
            "scenario": self.scenario.to_dict(),
            "tags": self.tags.to_dict(),
        }
        if self.generated_at is not None:
            out["generated_at"] = str(self.generated_at)
        if self.validation is not None:
            out["validation"] = self.validation
        return out
