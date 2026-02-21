from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback
from typing import Any, Iterable

from etappe02_modelle.core.budget import split_budget_seconds, split_budget_seconds_weighted, SubrunTimer
from etappe02_modelle.core.case_loader import CaseLoadError, load_case, load_config_json
from etappe02_modelle.core.feasibility import check_feasible
from etappe02_modelle.core.logging_jsonl import JsonlWriter, env_snapshot, utc_now_iso
from etappe02_modelle.core.objective import compute_solution_id, evaluate_F, schedule_delta_vs_sigma
from etappe02_modelle.core.schema import PlannedSubrun, RunEvent, RunPlan, SubrunSpec, compute_config_hash, compute_group_id
from etappe02_modelle.core.seeding import seed_case, seed_group, seed_subrun
from etappe02_modelle.procedures import get_procedure, normalize_verfahren
from tools.validation import ValidationReport


def _has_ok_group_end(runs_path: Path, group_id: str) -> bool:
    if not runs_path.exists():
        return False
    try:
        with runs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("event") == "group_end" and rec.get("group_id") == group_id and rec.get("status") == "ok":
                    return True
    except Exception:
        return False
    return False


def _iter_schedules(
    schedules: Iterable[list[list[int]]],
    *,
    deadline_monotonic: float,
) -> Iterable[list[list[int]]]:
    for sched in schedules:
        if time.monotonic() >= deadline_monotonic:
            break
        yield sched


def run_case(
    *,
    case_path: Path,
    verfahren: str,
    config_path: Path,
    out_root: Path,
    seed_global: int,
    budget_total_seconds: float,
    resume: bool,
    allow_nonstrict: bool,
) -> int:
    verfahren_norm = normalize_verfahren(verfahren)
    config_snapshot = load_config_json(config_path)
    config_hash = compute_config_hash(config_snapshot)

    strict_gate = not bool(allow_nonstrict)
    loaded = load_case(case_path, strict_gate=strict_gate)
    case = loaded.case

    dataset_id = str(case.dataset_id)
    case_id = str(case.case_id)

    group_id = compute_group_id(
        dataset_id=dataset_id,
        case_id=case_id,
        verfahren=verfahren_norm,
        config_hash=config_hash,
        seed_global=int(seed_global),
        budget_total_seconds=float(budget_total_seconds),
    )

    run_dir = out_root / case_id / verfahren_norm
    runs_path = run_dir / "runs.jsonl"
    solutions_path = run_dir / "solutions.jsonl"

    if resume and _has_ok_group_end(runs_path, group_id):
        print(f"[skip/resume] {dataset_id}/{case_id}/{verfahren_norm} group_id={group_id}")
        return 0

    procedure = get_procedure(verfahren_norm)
    subrun_specs = procedure.plan_subruns(config_snapshot)
    if not subrun_specs:
        subrun_specs = [SubrunSpec(subrun_id="subrun_0", params={})]

    any_weight = any("budget_weight" in (sr.params or {}) for sr in subrun_specs)
    if any_weight:
        weights: list[float] = []
        preflight = ValidationReport()
        for sr in subrun_specs:
            if "budget_weight" not in (sr.params or {}):
                preflight.add_error(
                    "Subrun-Plan ungültig: budget_weight muss für alle Subruns gesetzt sein (oder für keinen)."
                )
                break
            weights.append(float(sr.params["budget_weight"]))
        if preflight.errors:
            raise ValueError(preflight.errors[0])
        budgets = split_budget_seconds_weighted(float(budget_total_seconds), weights)
    else:
        budgets = split_budget_seconds(float(budget_total_seconds), len(subrun_specs))

    s_case = seed_case(int(seed_global), dataset_id, case_id)
    s_group = seed_group(s_case, verfahren_norm, config_hash)

    planned_subruns: list[PlannedSubrun] = []
    for spec, b in zip(subrun_specs, budgets, strict=True):
        planned_subruns.append(
            PlannedSubrun(
                subrun_id=str(spec.subrun_id),
                seed_subrun=seed_subrun(s_group, str(spec.subrun_id)),
                budget_seconds=float(b),
                params=dict(spec.params),
            )
        )

    solver_name: str | None = None
    solver_cfg = config_snapshot.get("solver")
    if isinstance(solver_cfg, dict):
        name = solver_cfg.get("name")
        if name is not None and str(name).strip():
            solver_name = str(name).strip()
    env = env_snapshot(cwd=Path.cwd(), solver=solver_name)

    plan = RunPlan(
        dataset_id=dataset_id,
        case_id=case_id,
        verfahren=verfahren_norm,
        group_id=group_id,
        seed_global=int(seed_global),
        seed_case=int(s_case),
        seed_group=int(s_group),
        budget_total_seconds=float(budget_total_seconds),
        subruns=planned_subruns,
        config_snapshot=config_snapshot,
        config_hash=config_hash,
        env=env,
        warnings=list(loaded.warnings),
    )

    runs_writer = JsonlWriter(runs_path)
    solutions_writer = JsonlWriter(solutions_path)

    group_start_mon = time.monotonic()
    runs_writer.append(
        RunEvent(
            ts_utc=utc_now_iso(),
            event="group_start",
            dataset_id=dataset_id,
            case_id=case_id,
            verfahren=verfahren_norm,
            group_id=group_id,
            payload=plan.to_dict(),
        ).to_dict()
    )

    group_status = "ok"
    group_solutions = 0
    group_first_feasible_elapsed: float | None = None
    group_any_timeout = False

    for spec, planned in zip(subrun_specs, planned_subruns, strict=True):
        subrun_start_mon = time.monotonic()
        timer = SubrunTimer(start_monotonic=subrun_start_mon, budget_seconds=float(planned.budget_seconds))

        runs_writer.append(
            RunEvent(
                ts_utc=utc_now_iso(),
                event="subrun_start",
                dataset_id=dataset_id,
                case_id=case_id,
                verfahren=verfahren_norm,
                group_id=group_id,
                payload={
                    "subrun_id": str(planned.subrun_id),
                    "seed_subrun": int(planned.seed_subrun),
                    "budget_seconds": float(planned.budget_seconds),
                    "params": dict(spec.params),
                },
            ).to_dict()
        )

        subrun_status = "ok"
        subrun_solutions = 0
        subrun_first_feasible_elapsed: float | None = None

        try:
            schedules = procedure.run_subrun(
                loaded_case=loaded,
                subrun=spec,
                seed_subrun=int(planned.seed_subrun),
                deadline_monotonic=float(timer.deadline),
            )

            for schedule in _iter_schedules(schedules, deadline_monotonic=float(timer.deadline)):
                feas = check_feasible(case, schedule, weeks_T_w=loaded.weeks_T_w)
                if not feas.ok:
                    continue

                F = evaluate_F(case, schedule, weeks_T_w=loaded.weeks_T_w)
                sol_id = compute_solution_id(schedule)
                delta = schedule_delta_vs_sigma(case, schedule)

                now_mon = time.monotonic()
                elapsed = now_mon - group_start_mon
                if group_first_feasible_elapsed is None:
                    group_first_feasible_elapsed = elapsed
                if subrun_first_feasible_elapsed is None:
                    subrun_first_feasible_elapsed = now_mon - subrun_start_mon

                solutions_writer.append(
                    {
                        "ts_utc": utc_now_iso(),
                        "elapsed_seconds": float(elapsed),
                        "dataset_id": dataset_id,
                        "case_id": case_id,
                        "verfahren": verfahren_norm,
                        "group_id": group_id,
                        "subrun_id": str(spec.subrun_id),
                        "solution_id": sol_id,
                        "F": F.to_dict(),
                        # Hinweis: In Etappe 2 werden nur zulässige Lösungen geloggt.
                        # Das Flag ist für Etappe 3 hilfreich (per_group-Grouping, Debug).
                        "feasible": True,
                        "schedule_delta": delta,
                        "source": {"params": dict(spec.params), "seed_subrun": int(planned.seed_subrun)},
                    }
                )
                subrun_solutions += 1
                group_solutions += 1

                if timer.is_expired():
                    break

            if timer.is_expired():
                subrun_status = "timeout"
                group_any_timeout = True

        except NotImplementedError as e:
            subrun_status = "error"
            group_status = "error"
            runs_writer.append(
                RunEvent(
                    ts_utc=utc_now_iso(),
                    event="error",
                    dataset_id=dataset_id,
                    case_id=case_id,
                    verfahren=verfahren_norm,
                    group_id=group_id,
                    payload={
                        "subrun_id": str(spec.subrun_id),
                        "error_type": type(e).__name__,
                        "message": str(e),
                    },
                ).to_dict()
            )
        except Exception as e:
            subrun_status = "error"
            group_status = "error"
            runs_writer.append(
                RunEvent(
                    ts_utc=utc_now_iso(),
                    event="error",
                    dataset_id=dataset_id,
                    case_id=case_id,
                    verfahren=verfahren_norm,
                    group_id=group_id,
                    payload={
                        "subrun_id": str(spec.subrun_id),
                        "error_type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                ).to_dict()
            )

        wall_used = time.monotonic() - subrun_start_mon
        subrun_meta: dict[str, Any] | None = None
        get_meta = getattr(procedure, "get_subrun_meta", None)
        if callable(get_meta):
            try:
                raw = get_meta(str(spec.subrun_id))
            except Exception:
                raw = None
            if isinstance(raw, dict):
                subrun_meta = raw

        payload: dict[str, Any] = {
            "subrun_id": str(spec.subrun_id),
            "status": str(subrun_status),
            "wall_seconds_used": float(wall_used),
            "solutions_found": int(subrun_solutions),
            "time_to_first_feasible_seconds": subrun_first_feasible_elapsed,
        }
        if subrun_meta:
            payload.update(subrun_meta)

        runs_writer.append(
            RunEvent(
                ts_utc=utc_now_iso(),
                event="subrun_end",
                dataset_id=dataset_id,
                case_id=case_id,
                verfahren=verfahren_norm,
                group_id=group_id,
                payload=payload,
            ).to_dict()
        )

        if subrun_status == "error":
            break

    group_wall_used = time.monotonic() - group_start_mon
    if group_solutions == 0 and group_status == "ok":
        # Kein Verfahrenserfolg: unterscheide "timeout" (Budget erschöpft) vs. "infeasible" (ohne Timeout).
        group_status = "timeout" if group_any_timeout else "infeasible"

    group_meta: dict[str, Any] | None = None
    get_group_meta = getattr(procedure, "get_group_meta", None)
    if callable(get_group_meta):
        try:
            raw = get_group_meta()
        except Exception:
            raw = None
        if isinstance(raw, dict):
            group_meta = raw

    group_payload: dict[str, Any] = {
        "status": str(group_status),
        "wall_seconds_used": float(group_wall_used),
        "solutions_found": int(group_solutions),
        "time_to_first_feasible_seconds": group_first_feasible_elapsed,
    }
    if group_meta:
        group_payload.update(group_meta)

    runs_writer.append(
        RunEvent(
            ts_utc=utc_now_iso(),
            event="group_end",
            dataset_id=dataset_id,
            case_id=case_id,
            verfahren=verfahren_norm,
            group_id=group_id,
            payload=group_payload,
        ).to_dict()
    )

    return 0 if group_status == "ok" else 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Etappe 02: Führt ein Verfahren (V1–V4) auf einem einzelnen Case aus und loggt die Ergebnisse append-only.\n"
            "Subruns (z.B. Bounds-Phase, ω/ε-Sweeps) werden innerhalb des Gesamtbudgets geplant.\n"
            "Hinweis: Logging/Seed-/Budget-Handling ist verfahrensneutral; die Such-/Solverlogik liegt im jeweiligen Verfahren (vgl. schriftliche Arbeit, Kap. 5.1)."
        )
    )
    parser.add_argument("--case", type=Path, required=True, help="Pfad zu einem Case-JSON (Output aus Etappe 01).")
    parser.add_argument("--verfahren", type=str, required=True, help="Verfahren: V1|V2|V3|V4.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Verfahrens-Config (JSON), z.B. etappe02_modelle/configs/v1_ws_milp.final.json.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output-Root für Läufe, z.B. logs/laeufe/<dataset_id> oder /tmp/laeufe/<dataset_id>.")
    parser.add_argument("--seed", type=int, required=True, help="Globaler Seed (Reproduzierbarkeit).")
    parser.add_argument("--budget", type=float, required=True, help="Wall-Clock-Gesamtbudget pro (Case, Verfahren) in Sekunden.")
    parser.add_argument("--resume", action="store_true", help="Fortsetzen (Resume): überspringt Gruppen mit group_end(status=ok).")
    parser.add_argument(
        "--allow-nonstrict",
        action="store_true",
        help="Deaktiviert das Strict-Gate (akzeptiert validation['ok']=True, auch wenn validation['ok_strict']=False). Standard: strikt.",
    )
    args = parser.parse_args()

    try:
        return run_case(
            case_path=args.case,
            verfahren=args.verfahren,
            config_path=args.config,
            out_root=args.out,
            seed_global=int(args.seed),
            budget_total_seconds=float(args.budget),
            resume=bool(args.resume),
            allow_nonstrict=bool(args.allow_nonstrict),
        )
    except CaseLoadError as e:
        raise SystemExit(f"Case-Fehler: {e}") from e
    except Exception as e:
        raise SystemExit(f"Fehler: {type(e).__name__}: {e}") from e


if __name__ == "__main__":
    raise SystemExit(main())
