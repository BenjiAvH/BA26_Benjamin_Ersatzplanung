from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
import math
import time
from typing import Any

from etappe02_modelle.core.case_loader import LoadedCase
from etappe02_modelle.core.feasibility import check_feasible
from etappe02_modelle.core.objective import evaluate_F
from etappe02_modelle.core.schema import ScheduleDense, SubrunSpec
from etappe02_modelle.procedures.interfaces import _subruns_from_explicit_list
from etappe02_modelle.procedures.milp_scip_core import (
    build_base_model_scip,
    extract_best_schedule_dense_scip,
    extract_solver_meta_scip,
)


_OBJ_KEYS = ["f_stab", "f_ot", "f_pref", "f_fair"]


def _normalize_obj_key(raw: Any) -> str:
    key = str(raw).strip()
    if key not in set(_OBJ_KEYS):
        raise ValueError(f"V2: Objective-Key ungültig: {raw!r} (erwartet: {', '.join(_OBJ_KEYS)})")
    return key


def _parse_solver_threads(cfg: dict[str, Any]) -> int:
    solver_cfg = cfg.get("solver", {})
    if not isinstance(solver_cfg, dict):
        return 1
    threads = solver_cfg.get("threads", 1)
    try:
        threads_int = int(threads)
    except Exception:
        return 1
    return max(1, threads_int)


def _parse_bounds_cfg(cfg: dict[str, Any]) -> tuple[bool, float, str]:
    bounds_cfg = cfg.get("bounds", None)
    if bounds_cfg is None:
        return True, 0.10, "extreme_runs_minmax"
    if not isinstance(bounds_cfg, dict):
        raise ValueError("Config ungültig: bounds muss ein Objekt sein.")
    enabled = bool(bounds_cfg.get("enabled", True))
    alpha = bounds_cfg.get("budget_fraction", 0.10)
    try:
        alpha_f = float(alpha)
    except Exception as e:
        raise ValueError("Config ungültig: bounds.budget_fraction ist keine Zahl.") from e
    if alpha_f < 0.0 or alpha_f > 1.0:
        raise ValueError("Config ungültig: bounds.budget_fraction muss in [0,1] liegen.")

    method = str(bounds_cfg.get("method", "extreme_runs_minmax")).strip() or "extreme_runs_minmax"
    if method != "extreme_runs_minmax":
        raise ValueError("V2: bounds.method aktuell nur 'extreme_runs_minmax' unterstützt.")

    return enabled, alpha_f, method


def _parse_primary_objectives(cfg: dict[str, Any]) -> list[str]:
    raw_list = cfg.get("primary_objectives")
    raw_single = cfg.get("primary_objective")

    if raw_list is None and raw_single is None:
        return list(_OBJ_KEYS)

    if raw_list is None:
        return [_normalize_obj_key(raw_single)]

    if not isinstance(raw_list, list) or not raw_list:
        raise ValueError("Config ungültig: primary_objectives muss ein nicht-leeres Array sein.")

    out: list[str] = []
    seen: set[str] = set()
    for item in raw_list:
        key = _normalize_obj_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _parse_eps_levels(cfg: dict[str, Any]) -> dict[str, list[float]]:
    default = [0.25, 0.5, 0.75]
    raw = cfg.get("eps_levels", None)

    def _validate_levels(levels: Any, *, name: str) -> list[float]:
        if not isinstance(levels, list) or not levels:
            raise ValueError(f"Config ungültig: {name} muss ein nicht-leeres Array sein.")
        out: list[float] = []
        for idx, v in enumerate(levels):
            try:
                q = float(v)
            except Exception as e:
                raise ValueError(f"Config ungültig: {name}[{idx}] ist keine Zahl: {v!r}") from e
            if not math.isfinite(q) or q < 0.0 or q > 1.0:
                raise ValueError(f"Config ungültig: {name}[{idx}] muss in [0,1] liegen (Wert={q}).")
            out.append(float(q))
        return out

    if raw is None:
        levels_all = _validate_levels(default, name="eps_levels")
        return {k: list(levels_all) for k in _OBJ_KEYS}

    if isinstance(raw, list):
        levels_all = _validate_levels(raw, name="eps_levels")
        return {k: list(levels_all) for k in _OBJ_KEYS}

    if isinstance(raw, dict):
        out: dict[str, list[float]] = {}
        for obj in _OBJ_KEYS:
            if obj in raw:
                out[obj] = _validate_levels(raw[obj], name=f"eps_levels.{obj}")
            else:
                out[obj] = _validate_levels(default, name=f"eps_levels.{obj}")
        return out

    raise ValueError("Config ungültig: eps_levels muss Array, Objekt oder null sein.")


def _parse_eps_plan(cfg: dict[str, Any]) -> list[dict[str, float]] | None:
    raw = cfg.get("eps_plan", None)
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError("Config ungültig: eps_plan muss ein nicht-leeres Array sein (oder null).")

    out: list[dict[str, float]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Config ungültig: eps_plan[{idx}] ist kein Objekt.")
        levels: dict[str, float] = {}
        for k, v in item.items():
            obj = _normalize_obj_key(k)
            try:
                q = float(v)
            except Exception as e:
                raise ValueError(f"Config ungültig: eps_plan[{idx}].{k} ist keine Zahl: {v!r}") from e
            if not math.isfinite(q) or q < 0.0 or q > 1.0:
                raise ValueError(f"Config ungültig: eps_plan[{idx}].{k} muss in [0,1] liegen (Wert={q}).")
            levels[obj] = float(q)
        out.append(levels)
    return out


def _parse_eps_max_from_config(entry: dict[str, Any]) -> dict[str, int]:
    eps_max: dict[str, int] = {}
    for raw_k, raw_v in entry.items():
        k = str(raw_k).strip()
        if k.endswith("_max"):
            obj = _normalize_obj_key(k[: -len("_max")])
        else:
            obj = _normalize_obj_key(k)

        try:
            fv = float(raw_v)
        except Exception as e:
            raise ValueError(f"Config ungültig: eps_max.{k} ist keine Zahl: {raw_v!r}") from e
        if not math.isfinite(fv) or fv < 0.0:
            raise ValueError(f"Config ungültig: eps_max.{k} muss finite und >=0 sein (Wert={fv}).")
        if abs(fv - round(fv)) > 1e-9:
            raise ValueError(f"Config ungültig: eps_max.{k} muss ganzzahlig sein (Wert={fv}).")
        eps_max[obj] = int(round(fv))
    return eps_max


@dataclass(frozen=True)
class _EpsBounds:
    LB: dict[str, int]
    UB: dict[str, int]
    source: str
    budget_fraction: float


def _materialize_eps_from_levels(bounds: _EpsBounds, eps_level: dict[str, float], *, primary_objective: str) -> dict[str, int]:
    eps: dict[str, int] = {}
    for obj in _OBJ_KEYS:
        if obj == primary_objective:
            continue
        if obj not in eps_level:
            raise ValueError(f"V2: eps_level fehlt für Ziel {obj!r} (primary_objective={primary_objective!r}).")
        q = float(eps_level[obj])
        LB = int(bounds.LB[obj])
        UB = int(bounds.UB[obj])
        val = int(math.floor(float(LB) + float(q) * float(UB - LB)))
        val = max(LB, min(UB, val))
        eps[obj] = int(val)
    return eps


def _fallback_case_bounds(*, loaded_case: LoadedCase) -> _EpsBounds:
    """
    Deterministische, solverfreie Fallback-Bounds aus dem Case.

    Hinweis: Bounds sind bewusst grob; primär dienen sie zur Skalierung/Materialisierung
    von ε-Levels, wenn keine Extrempunkte vorliegen.
    """

    case = loaded_case.case

    I = int(case.dimensions.I)
    T_rep = [int(t) for t in case.params["T_rep"]]
    weeks_T_w = loaded_case.weeks_T_w
    sigma = case.params["sigma"]
    h = case.params["h"]
    H_max = case.params["H_max"]
    a = case.params["a"]
    c = case.params["c"]

    LB = {"f_stab": 0, "f_ot": 0, "f_pref": 0, "f_fair": 0}
    UB_stab = I * len(T_rep)
    UB_fair = len(T_rep)

    # UB_ot: Σ_i Σ_w max(0, H_i^max - H_ref_{i,w})
    UB_ot = 0
    for i in range(I):
        limit = int(H_max[i])
        for w_days in weeks_T_w:
            ref_hours = 0
            for t in w_days:
                s_ref = int(sigma[i][int(t)])
                if s_ref != 0:
                    ref_hours += int(h[s_ref])
            UB_ot += max(0, limit - ref_hours)

    # UB_pref: Σ_i Σ_{t∈T_rep} max_{s: a_{i,t,s}=1} c_{i,t,s}
    UB_pref = 0
    for i in range(I):
        for t in T_rep:
            best = 0
            for s in range(int(case.dimensions.S)):
                if int(a[i][t][s]) == 1:
                    best = max(best, int(c[i][t][s]))
            UB_pref += best

    UB = {"f_stab": int(UB_stab), "f_ot": int(UB_ot), "f_pref": int(UB_pref), "f_fair": int(UB_fair)}
    return _EpsBounds(LB=LB, UB=UB, source="fallback_case_bounds", budget_fraction=0.0)


class V2EPSMILPProcedure:
    """
    V2 — EPS-MILP (ε-Restriktionen).

    Kernidee:
    - (optional) Bounds-Phase (Extrempunkte) zur instanzspezifischen ε-Skalierung
    - EPS-Phase: minimiere primäres Ziel und setze für alle anderen Ziele ε-Constraints
    - deterministisches Budget-Splitting über `params.budget_weight` (Orchestrator)
    - MILP via SCIP/PySCIPOpt (Kernel identisch zu V1; gemeinsames Core-Modul)

    Bezug: Kap. 5.1 (Forschungsdesign & Evaluationsstrategie).
    """

    verfahren = "V2"

    def plan_subruns(self, config_snapshot: dict[str, Any]) -> list[SubrunSpec]:
        self._solver_threads = _parse_solver_threads(config_snapshot)
        self._bounds_enabled, self._bounds_budget_fraction, self._bounds_method = _parse_bounds_cfg(config_snapshot)

        # Reset pro Group/Run: Procedure-Objekt ist pro run_case neu, aber explizit halten.
        self._subrun_meta: dict[str, dict[str, Any]] = {}
        self._extreme_F: dict[str, dict[str, int]] = {}
        self._eps_bounds: _EpsBounds | None = None
        self._eps_counts = {"eps_subruns_total": 0, "eps_subruns_infeasible": 0, "eps_subruns_no_incumbent": 0}

        explicit = _subruns_from_explicit_list(config_snapshot)
        if explicit is not None:
            return explicit

        primary_objectives = _parse_primary_objectives(config_snapshot)

        # EPS-Plan: entweder explizite numerische ε-Maxima, oder (Default) Levels + Kartesisches Raster.
        eps_entries: list[tuple[str, int, dict[str, Any]]] = []

        eps_configs = config_snapshot.get("eps_configs")
        if isinstance(eps_configs, list) and eps_configs:
            for k in primary_objectives:
                required = set(_OBJ_KEYS) - {k}
                for idx, raw in enumerate(eps_configs):
                    if not isinstance(raw, dict):
                        raise ValueError(f"Config ungültig: eps_configs[{idx}] ist kein Objekt.")
                    eps_max = _parse_eps_max_from_config(raw)
                    if not required.issubset(set(eps_max.keys())):
                        missing = sorted(required - set(eps_max.keys()))
                        raise ValueError(
                            f"Config ungültig: eps_configs[{idx}] fehlt ε-Maximum für {missing} "
                            f"(primary_objective={k})."
                        )
                    eps_entries.append((k, idx, {"eps_max": eps_max}))
        else:
            eps_plan = _parse_eps_plan(config_snapshot)
            if eps_plan is not None:
                for k in primary_objectives:
                    for idx, levels in enumerate(eps_plan):
                        eps_entries.append((k, idx, {"eps_level": dict(levels)}))
            else:
                levels_map = _parse_eps_levels(config_snapshot)
                mode = str(config_snapshot.get("eps_plan_mode", "cartesian_3d")).strip() or "cartesian_3d"
                if mode != "cartesian_3d":
                    raise ValueError("V2: eps_plan_mode aktuell nur 'cartesian_3d' unterstützt.")

                for k in primary_objectives:
                    keys = [obj for obj in _OBJ_KEYS if obj != k]
                    grids = [levels_map[obj] for obj in keys]
                    for idx, combo in enumerate(product(*grids)):
                        eps_level = {obj: float(q) for obj, q in zip(keys, combo, strict=True)}
                        eps_entries.append((k, idx, {"eps_level": eps_level}))

        if not eps_entries:
            raise ValueError("V2: EPS-Plan ist leer (eps_configs/eps_plan/eps_levels).")

        alpha = float(self._bounds_budget_fraction) if self._bounds_enabled else 0.0
        bounds_weight = (alpha / 4.0) if self._bounds_enabled else 0.0
        eps_weight = ((1.0 - alpha) / float(len(eps_entries))) if len(eps_entries) > 0 else 0.0

        subruns: list[SubrunSpec] = []
        if self._bounds_enabled:
            for obj in _OBJ_KEYS:
                subruns.append(
                    SubrunSpec(
                        subrun_id=f"bound_{obj}",
                        params={
                            "phase": "bounds",
                            "primary_objective": obj,
                            "budget_weight": bounds_weight,
                        },
                    )
                )

        for k, idx, spec in eps_entries:
            params: dict[str, Any] = {"phase": "eps", "primary_objective": k, "budget_weight": eps_weight}
            params.update(spec)
            subruns.append(SubrunSpec(subrun_id=f"eps_{k}_{idx}", params=params))

        return subruns

    def run_subrun(
        self,
        *,
        loaded_case: LoadedCase,
        subrun: SubrunSpec,
        seed_subrun: int,
        deadline_monotonic: float,
    ) -> Iterable[ScheduleDense]:
        # Für Group-Meta (Bounds/Logging) auch dann setzen, wenn der Solver-Import fehlschlägt.
        self._last_loaded_case = loaded_case

        now = time.monotonic()
        time_limit = float(deadline_monotonic) - float(now)
        if time_limit <= 0.0:
            self._subrun_meta[str(subrun.subrun_id)] = {
                "solver_name": "scip",
                "solver_status": "skipped",
                "termination_reason": "time_limit",
                "incumbent_found": False,
            }
            return

        phase = str(subrun.params.get("phase", "eps")).strip() or "eps"
        primary = _normalize_obj_key(subrun.params.get("primary_objective", ""))

        case = loaded_case.case
        I = int(case.dimensions.I)
        T = int(case.dimensions.T)
        S = int(case.dimensions.S)

        m, vars_scip, objs = build_base_model_scip(
            loaded_case=loaded_case,
            seed_subrun=int(seed_subrun),
            solver_threads=int(getattr(self, "_solver_threads", 1)),
            model_name=f"V2_{subrun.subrun_id}",
        )

        subrun_meta: dict[str, Any] = {"phase": phase, "primary_objective": primary}

        if phase == "eps":
            bounds = self._ensure_eps_bounds(loaded_case)

            eps_level_raw = subrun.params.get("eps_level")
            eps_max_raw = subrun.params.get("eps_max")

            eps_constraints: dict[str, int]
            if eps_level_raw is not None:
                if not isinstance(eps_level_raw, dict):
                    raise ValueError("V2: Subrun-Params ungültig: eps_level muss ein Objekt sein.")
                eps_level: dict[str, float] = {}
                for k, v in eps_level_raw.items():
                    obj = _normalize_obj_key(k)
                    try:
                        q = float(v)
                    except Exception as e:
                        raise ValueError(f"V2: Subrun-Params ungültig: eps_level.{k} ist keine Zahl: {v!r}") from e
                    if not math.isfinite(q) or q < 0.0 or q > 1.0:
                        raise ValueError(f"V2: Subrun-Params ungültig: eps_level.{k} muss in [0,1] liegen (Wert={q}).")
                    eps_level[obj] = float(q)
                eps_constraints = _materialize_eps_from_levels(bounds, eps_level, primary_objective=primary)
                subrun_meta["eps_level"] = dict(eps_level)
            elif eps_max_raw is not None:
                if not isinstance(eps_max_raw, dict):
                    raise ValueError("V2: Subrun-Params ungültig: eps_max muss ein Objekt sein.")
                eps_max = _parse_eps_max_from_config({str(k): v for k, v in eps_max_raw.items()})
                required = set(_OBJ_KEYS) - {primary}
                if not required.issubset(set(eps_max.keys())):
                    missing = sorted(required - set(eps_max.keys()))
                    raise ValueError(f"V2: Subrun-Params ungültig: eps_max fehlt für {missing} (primary_objective={primary}).")
                eps_constraints = {obj: int(eps_max[obj]) for obj in sorted(required)}
                subrun_meta["eps_max"] = dict(eps_max)
            else:
                raise ValueError("V2: Subrun-Params ungültig: entweder eps_level oder eps_max ist erforderlich.")

            # ε-Constraints: für alle Ziele j != k
            for obj in _OBJ_KEYS:
                if obj == primary:
                    continue
                eps_val = int(eps_constraints[obj])
                m.addCons(objs[obj] <= eps_val)

            subrun_meta["epsilon_vector"] = dict(eps_constraints)
            subrun_meta["epsilon_bounds"] = {"LB": dict(bounds.LB), "UB": dict(bounds.UB)}
            subrun_meta["epsilon_bounds_source"] = str(bounds.source)
            subrun_meta["bounds_budget_fraction"] = float(bounds.budget_fraction)

        elif phase == "bounds":
            pass
        else:
            raise ValueError(f"V2: Unbekannte phase in Subrun-Params: {phase!r}")

        # Primäres Ziel setzen (EPS oder Bounds).
        m.setObjective(objs[primary], "minimize")

        remaining = float(deadline_monotonic) - float(time.monotonic())
        if remaining <= 0.0:
            subrun_meta.update(
                {
                    "solver_name": "scip",
                    "solver_status": "skipped",
                    "termination_reason": "time_limit",
                    "incumbent_found": False,
                }
            )
            self._subrun_meta[str(subrun.subrun_id)] = dict(subrun_meta)
            return

        # Budget als TimeLimit (Wall-Clock, Sek.). Deadline enthält auch Modellbau-Zeit.
        m.setParam("limits/time", max(0.0, remaining))
        m.optimize()

        solver_meta = extract_solver_meta_scip(m)
        subrun_meta.update(solver_meta)
        self._subrun_meta[str(subrun.subrun_id)] = dict(subrun_meta)

        if phase == "eps":
            self._eps_counts["eps_subruns_total"] += 1
            term = str(subrun_meta.get("termination_reason", ""))
            if term == "infeasible":
                self._eps_counts["eps_subruns_infeasible"] += 1
            if term == "no_incumbent":
                self._eps_counts["eps_subruns_no_incumbent"] += 1

        schedule = extract_best_schedule_dense_scip(m, vars_scip.x, I=I, T=T, S=S)
        if schedule is None:
            return

        weeks_T_w = loaded_case.weeks_T_w
        feas = check_feasible(case, schedule, weeks_T_w=weeks_T_w)
        if feas.ok and phase == "bounds":
            F = evaluate_F(case, schedule, weeks_T_w=weeks_T_w).to_dict()
            self._extreme_F[primary] = dict(F)
            self._subrun_meta[str(subrun.subrun_id)] = {
                **dict(subrun_meta),
                "F_extreme": dict(F),
            }

        yield schedule

    def _ensure_eps_bounds(self, loaded_case: LoadedCase) -> _EpsBounds:
        if self._eps_bounds is not None:
            return self._eps_bounds

        if len(self._extreme_F) >= 2 and self._bounds_method == "extreme_runs_minmax":
            Fs = list(self._extreme_F.values())
            LB = {k: min(int(F[k]) for F in Fs) for k in _OBJ_KEYS}
            UB = {k: max(int(F[k]) for F in Fs) for k in _OBJ_KEYS}
            self._eps_bounds = _EpsBounds(
                LB=LB,
                UB=UB,
                source="extreme_runs_minmax",
                budget_fraction=float(self._bounds_budget_fraction),
            )
            return self._eps_bounds

        fb = _fallback_case_bounds(loaded_case=loaded_case)
        self._eps_bounds = _EpsBounds(
            LB=dict(fb.LB),
            UB=dict(fb.UB),
            source=str(fb.source),
            budget_fraction=float(self._bounds_budget_fraction),
        )
        return self._eps_bounds

    def get_subrun_meta(self, subrun_id: str) -> dict[str, Any] | None:
        return self._subrun_meta.get(str(subrun_id))

    def get_group_meta(self) -> dict[str, Any] | None:
        loaded = getattr(self, "_last_loaded_case", None)
        if isinstance(loaded, LoadedCase):
            bounds = self._ensure_eps_bounds(loaded)
        else:
            bounds = None

        out: dict[str, Any] = {}
        if bounds is not None:
            out.update(
                {
                    "eps_bounds": {"LB": dict(bounds.LB), "UB": dict(bounds.UB)},
                    "eps_bounds_source": str(bounds.source),
                    "bounds_budget_fraction": float(bounds.budget_fraction),
                }
            )
        out.update(dict(getattr(self, "_eps_counts", {})))
        return out or None


__all__ = ["V2EPSMILPProcedure"]
