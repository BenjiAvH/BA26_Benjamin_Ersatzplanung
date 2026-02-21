from __future__ import annotations

import argparse
import json
from pathlib import Path

from etappe01_simulation.simulator.generator import generate_case
from etappe01_simulation.simulator.schema import dumps_json


def _load_config(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise SystemExit(f"Unbekanntes Config-Format: {path} (erwartet: .json)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Etappe 01: Generiert synthetische Instanzen und Störungsszenarien (vgl. schriftliche Arbeit, Kap. 5.2)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config-Datei (.json), z.B. etappe01_simulation/configs/praxisnah_v1.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output-Ordner (Case-JSONs + manifest.json + generation_log.jsonl), z.B. logs/daten/praxisnah_v1.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Globaler Seed (Reproduzierbarkeit; deterministische Erzeugung).")
    parser.add_argument("--overwrite", action="store_true", help="Überschreibt vorhandene Case-Dateien.")
    parser.add_argument(
        "--include-timestamp",
        action="store_true",
        help="Schreibt generated_at in die Case-JSONs (nicht bit-identische Ausgabe).",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)

    dataset_id = str(cfg["dataset_id"])
    schema_version = str(cfg.get("schema_version", "1.0"))
    global_seed = int(args.seed)

    fix = cfg["fix_horizon"]
    shifts = cfg["shifts"]
    limits = cfg["limits"]
    qualification = cfg["qualification"]
    availability = cfg["availability"]
    demand = cfg["demand"]
    sizes = cfg["size_classes"]
    disturbance = cfg["disturbance"]
    prefs = cfg.get("preferences", {})

    h = [int(v) for v in shifts["h"]]
    S_cfg = int(shifts.get("S", len(h)))
    if S_cfg != len(h):
        raise SystemExit(f"Config inkonsistent: shifts.S={S_cfg}, aber len(shifts.h)={len(h)}.")
    if S_cfg < 2:
        raise SystemExit(f"Config ungültig: shifts.S={S_cfg} (erwartet S>=2 inkl. 0=frei).")
    if int(h[0]) != 0:
        raise SystemExit("Config ungültig: shifts.h[0] muss 0 sein (Schicht 0 = frei).")

    P_forbidden = [[int(a), int(b)] for a, b in shifts["P_forbidden"]]
    for s, sp in P_forbidden:
        if not (0 <= int(s) < S_cfg and 0 <= int(sp) < S_cfg):
            raise SystemExit(f"Config ungültig: P_forbidden enthält {(s, sp)} außerhalb 0..{S_cfg-1}.")

    split_work = demand.get("split_work")
    if split_work is not None:
        split_work = [float(x) for x in split_work]

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    log_path = args.out / "generation_log.jsonl"

    availability_levels: dict[str, float] = dict(availability["levels"])
    demand_patterns: list[str] = list(demand["patterns"])
    severities: dict = disturbance["severity"]

    manifest_entries: list[dict] = []

    with log_path.open("a", encoding="utf-8") as log_f:
        for size_class, size_cfg in sizes.items():
            I = int(size_cfg["I"])
            T = int(size_cfg["T"])

            for availability_level, prob in availability_levels.items():
                for pattern in demand_patterns:
                    if pattern == "weekday_high":
                        weekday_frac = float(demand["weekday_high_weekday_frac"])
                        weekend_frac = float(demand["weekday_high_weekend_frac"])
                    elif pattern == "weekend_high":
                        weekday_frac = float(demand["weekend_high_weekday_frac"])
                        weekend_frac = float(demand["weekend_high_weekend_frac"])
                    else:
                        raise SystemExit(f"Unbekanntes demand pattern: {pattern}")

                    for severity_name, sev_cfg in severities.items():
                        case_id = f"sc-{size_class}__av-{availability_level}__dp-{pattern}__sev-{severity_name}"
                        out_file = args.out / f"{case_id}.json"

                        status = "written"
                        error: dict | None = None
                        if out_file.exists() and not args.overwrite:
                            status = "exists"
                            try:
                                case = json.loads(out_file.read_text(encoding="utf-8"))
                            except Exception as e:
                                case = {}
                                error = {"type": type(e).__name__, "message": str(e)}
                        else:
                            try:
                                case = generate_case(
                                    dataset_id=dataset_id,
                                    schema_version=schema_version,
                                    global_seed=global_seed,
                                    size_class=str(size_class),
                                    I=I,
                                    T=T,
                                    availability_level=str(availability_level),
                                    availability_prob=float(prob),
                                    demand_pattern=str(pattern),
                                    weekday_frac=weekday_frac,
                                    weekend_frac=weekend_frac,
                                    split_work=split_work,
                                    split_frueh=float(demand["split_frueh"]),
                                    split_spaet=float(demand["split_spaet"]),
                                    split_nacht=float(
                                        demand.get(
                                            "split_nacht",
                                            1.0 - float(demand["split_frueh"]) - float(demand["split_spaet"]),
                                        )
                                    ),
                                    q_night=float(qualification["q_night"]),
                                    slack_min=int(availability["slack_min"]),
                                    slack_fraction=float(availability["slack_fraction"]),
                                    fix_fraction=float(fix["fraction"]),
                                    fix_min_days=int(fix["min_days"]),
                                    fix_max_days=int(fix["max_days"]),
                                    h=h,
                                    H_max_value=int(limits["H_max"]),
                                    P_forbidden=P_forbidden,
                                    disturbance_model=str(disturbance["default_model"]),
                                    disturbance_attempts=int(disturbance["attempts"]),
                                    severity=str(severity_name),
                                    n_absent_fraction=float(sev_cfg["n_absent_fraction"]),
                                    n_absent_min=int(sev_cfg["n_absent_min"]),
                                    duration_days=int(sev_cfg["duration_days"]),
                                    preferences_noise=bool(prefs.get("noise", False)),
                                    include_timestamp=bool(args.include_timestamp),
                                )
                                out_file.write_text(dumps_json(case), encoding="utf-8")
                            except Exception as e:
                                case = {}
                                status = "generation_failed"
                                error = {"type": type(e).__name__, "message": str(e)}

                        tags_fallback = {
                            "size_class": str(size_class),
                            "availability_level": str(availability_level),
                            "demand_pattern": str(pattern),
                        }
                        scenario_fallback = {
                            "severity": str(severity_name),
                            "model": str(disturbance["default_model"]),
                            "events": [],
                            "status": "generation_failed" if status == "generation_failed" else "unknown",
                        }

                        validation = case.get("validation", {}) if isinstance(case, dict) else {}
                        validation_ok = bool(validation.get("ok", False))
                        post_feasible = bool(validation.get("post_disturbance_feasibility_ok", False))
                        validation_ok_strict = bool(validation.get("ok_strict", validation_ok and post_feasible))

                        entry = {
                            "case_id": case_id,
                            "file": str(out_file),
                            "tags": case.get("tags", tags_fallback),
                            "scenario": case.get("scenario", scenario_fallback),
                            "seeds": case.get("seeds", {"global": global_seed}),
                            "validation_ok": validation_ok,
                            "validation_ok_strict": validation_ok_strict,
                            "post_disturbance_feasibility_ok": post_feasible,
                            "status": status,
                        }
                        if error is not None:
                            entry["error"] = error

                        manifest_entries.append(entry)
                        log_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        print(f"[{status}] {case_id}")

    manifest_path.write_text(
        json.dumps({"dataset_id": dataset_id, "cases": manifest_entries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")
    print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
