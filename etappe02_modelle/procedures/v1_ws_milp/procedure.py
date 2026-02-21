from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
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


def _omega21() -> list[list[float]]:
    """
    Deterministische ω-Menge (21 Punkte), Simplex-nah.

    Reihenfolge: (f_stab, f_ot, f_pref, f_fair).
    Bezug: Kap. 5.1 (Forschungsdesign & Evaluationsstrategie) — strukturierte Gewichtungswahl (WS-MILP).
    """

    w: list[list[float]] = []
    # Extreme (4)
    w.extend([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # Paarweise 50/50 (6)
    w.extend(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0, 0.5],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [0.0, 0.0, 0.5, 0.5],
        ]
    )
    # Dreier gleich (4)
    third = 1.0 / 3.0
    w.extend(
        [
            [third, third, third, 0.0],
            [third, third, 0.0, third],
            [third, 0.0, third, third],
            [0.0, third, third, third],
        ]
    )
    # Uniform (1)
    w.append([0.25, 0.25, 0.25, 0.25])
    # Stabilitäts-betont (6)
    w.extend(
        [
            [0.7, 0.2, 0.1, 0.0],
            [0.7, 0.2, 0.0, 0.1],
            [0.7, 0.1, 0.2, 0.0],
            [0.7, 0.1, 0.0, 0.2],
            [0.7, 0.0, 0.2, 0.1],
            [0.7, 0.0, 0.1, 0.2],
        ]
    )
    return w


def _validate_and_normalize_omega(raw: Any) -> list[float]:
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("Config ungültig: omega muss ein Array mit 4 Einträgen sein (stab, ot, pref, fair).")

    vals: list[float] = []
    for idx, v in enumerate(raw):
        try:
            fv = float(v)
        except Exception as e:
            raise ValueError(f"Config ungültig: omega[{idx}] ist keine Zahl: {v!r}") from e
        if fv < 0.0:
            raise ValueError(f"Config ungültig: omega[{idx}] < 0 ist nicht erlaubt: {fv}")
        vals.append(fv)

    s = float(sum(vals))
    if not math.isfinite(s) or s <= 0.0:
        raise ValueError("Config ungültig: Summe(omega) muss > 0 sein.")
    return [v / s for v in vals]


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


def _parse_bounds_cfg(cfg: dict[str, Any]) -> tuple[bool, float]:
    bounds_cfg = cfg.get("bounds", None)
    if bounds_cfg is None:
        # Default: Normierung standardmäßig aktiv (Kap. 5.1: Skalen/Normierung bei Gewichtung).
        return True, 0.10
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
    return enabled, alpha_f


@dataclass(frozen=True)
class _WsBounds:
    LB: dict[str, int]
    UB: dict[str, int]
    source: str
    budget_fraction: float


class V1WSMILPProcedure:
    """
    V1 — WS-MILP (Gewichtete Summenbildung).

    Kernidee:
    - zwei Phasen (Bounds + Ω)
    - deterministisches Budget-Splitting über `params.budget_weight` (Orchestrator)
    - MILP via SCIP/PySCIPOpt

    Bezug: Kap. 5.1 (Forschungsdesign & Evaluationsstrategie).
    """

    verfahren = "V1"

    def plan_subruns(self, config_snapshot: dict[str, Any]) -> list[SubrunSpec]:
        self._solver_threads = _parse_solver_threads(config_snapshot)
        self._bounds_enabled, self._bounds_budget_fraction = _parse_bounds_cfg(config_snapshot)

        # Reset pro Group/Run: Procedure-Objekt ist pro run_case neu, aber explizit halten.
        self._subrun_meta: dict[str, dict[str, Any]] = {}
        self._extreme_F: dict[str, dict[str, int]] = {}
        self._ws_bounds: _WsBounds | None = None

        explicit = _subruns_from_explicit_list(config_snapshot)
        if explicit is not None:
            return explicit

        omegas_raw = config_snapshot.get("omegas")
        if omegas_raw is None:
            omegas = _omega21()
        elif isinstance(omegas_raw, list) and omegas_raw:
            omegas = omegas_raw
        else:
            raise ValueError("Config ungültig: omegas fehlt/leer (oder null für Default).")

        omegas_norm = [_validate_and_normalize_omega(w) for w in omegas]
        if not omegas_norm:
            raise ValueError("Config ungültig: omegas ist leer.")

        alpha = float(self._bounds_budget_fraction) if self._bounds_enabled else 0.0
        bounds_weight = (alpha / 4.0) if self._bounds_enabled else 0.0
        omega_weight = ((1.0 - alpha) / float(len(omegas_norm))) if len(omegas_norm) > 0 else 0.0

        subruns: list[SubrunSpec] = []
        if self._bounds_enabled:
            for obj_key in ["f_stab", "f_ot", "f_pref", "f_fair"]:
                subruns.append(
                    SubrunSpec(
                        subrun_id=f"bound_{obj_key}",
                        params={
                            "phase": "bounds",
                            "primary_objective": obj_key,
                            "budget_weight": bounds_weight,
                        },
                    )
                )

        for idx, omega in enumerate(omegas_norm):
            subruns.append(
                SubrunSpec(
                    subrun_id=f"omega_{idx}",
                    params={
                        "phase": "omega",
                        "omega": omega,
                        "budget_weight": omega_weight,
                    },
                )
            )

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
        self._last_weeks_T_w = loaded_case.weeks_T_w

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

        case = loaded_case.case
        I = int(case.dimensions.I)
        T = int(case.dimensions.T)
        S = int(case.dimensions.S)

        weeks_T_w = loaded_case.weeks_T_w

        phase = str(subrun.params.get("phase", "omega"))

        # Bounds vorbereiten, falls eine ω-Phase Normierung braucht.
        if phase == "omega":
            omega = subrun.params.get("omega", None)
            omega_norm = _validate_and_normalize_omega(omega) if omega is not None else [0.25, 0.25, 0.25, 0.25]
            weights = self._omega_scaled_weights(loaded_case, omega_norm)
        elif phase == "bounds":
            primary = str(subrun.params.get("primary_objective", "")).strip()
            weights = self._primary_objective_weights(primary)
        else:
            raise ValueError(f"V1: Unbekannte phase in Subrun-Params: {phase!r}")

        m, vars_scip, objs = build_base_model_scip(
            loaded_case=loaded_case,
            seed_subrun=int(seed_subrun),
            solver_threads=int(getattr(self, "_solver_threads", 1)),
            model_name=f"V1_{subrun.subrun_id}",
        )

        # Zielfunktion: gewichtete Summe der Komponenten (ggf. skaliert/normiert).
        w_stab, w_ot, w_pref, w_fair = (float(v) for v in weights)
        m.setObjective(
            w_stab * objs["f_stab"] + w_ot * objs["f_ot"] + w_pref * objs["f_pref"] + w_fair * objs["f_fair"],
            "minimize",
        )

        # Solve
        remaining = float(deadline_monotonic) - float(time.monotonic())
        if remaining <= 0.0:
            self._subrun_meta[str(subrun.subrun_id)] = {
                "solver_name": "scip",
                "solver_status": "skipped",
                "termination_reason": "time_limit",
                "incumbent_found": False,
            }
            return

        # Budget als TimeLimit (Wall-Clock, Sek.). Deadline enthält auch Modellbau-Zeit.
        m.setParam("limits/time", max(0.0, remaining))
        m.optimize()

        meta = extract_solver_meta_scip(m)
        self._subrun_meta[str(subrun.subrun_id)] = meta

        schedule = extract_best_schedule_dense_scip(m, vars_scip.x, I=I, T=T, S=S)
        if schedule is None:
            return

        # Optional: interne Bounds aus Bounds-Phase aktualisieren (nur wenn Schedule auch wirklich zulässig ist).
        feas = check_feasible(case, schedule, weeks_T_w=weeks_T_w)
        if feas.ok and phase == "bounds":
            primary = str(subrun.params.get("primary_objective", "")).strip()
            F = evaluate_F(case, schedule, weeks_T_w=weeks_T_w).to_dict()
            self._extreme_F[primary] = dict(F)
            meta["primary_objective"] = primary
            meta["F_extreme"] = dict(F)
            self._subrun_meta[str(subrun.subrun_id)] = meta

        yield schedule

    def get_subrun_meta(self, subrun_id: str) -> dict[str, Any] | None:
        return self._subrun_meta.get(str(subrun_id))

    def get_group_meta(self) -> dict[str, Any] | None:
        if self._ws_bounds is None and self._bounds_enabled:
            # Falls die ω-Phase nie lief (z.B. Abbruch), trotzdem deterministisch finalisieren.
            loaded = getattr(self, "_last_loaded_case", None)
            if isinstance(loaded, LoadedCase):
                self._ws_bounds = self._compute_ws_bounds(loaded)
            else:
                self._ws_bounds = self._compute_ws_bounds_fallback_only()
        if self._ws_bounds is None:
            return None
        return {
            "ws_bounds": {"LB": dict(self._ws_bounds.LB), "UB": dict(self._ws_bounds.UB)},
            "ws_bounds_source": str(self._ws_bounds.source),
            "bounds_budget_fraction": float(self._ws_bounds.budget_fraction),
        }

    def _primary_objective_weights(self, primary: str) -> list[float]:
        key = str(primary).strip()
        if key == "f_stab":
            return [1.0, 0.0, 0.0, 0.0]
        if key == "f_ot":
            return [0.0, 1.0, 0.0, 0.0]
        if key == "f_pref":
            return [0.0, 0.0, 1.0, 0.0]
        if key == "f_fair":
            return [0.0, 0.0, 0.0, 1.0]
        raise ValueError(f"V1: primary_objective ungültig: {primary!r} (erwartet: f_stab|f_ot|f_pref|f_fair)")

    def _omega_scaled_weights(self, loaded_case: LoadedCase, omega_simplex: list[float]) -> list[float]:
        omega = list(omega_simplex)
        if not self._bounds_enabled:
            return omega

        if self._ws_bounds is None:
            self._ws_bounds = self._compute_ws_bounds(loaded_case)

        # Min–Max-Normierung: omega'_j = omega_j / max(eps, UB_j-LB_j), Konstanten entfallen im Optimum.
        eps = 1.0
        den_stab = max(eps, float(self._ws_bounds.UB["f_stab"] - self._ws_bounds.LB["f_stab"]))
        den_ot = max(eps, float(self._ws_bounds.UB["f_ot"] - self._ws_bounds.LB["f_ot"]))
        den_pref = max(eps, float(self._ws_bounds.UB["f_pref"] - self._ws_bounds.LB["f_pref"]))
        den_fair = max(eps, float(self._ws_bounds.UB["f_fair"] - self._ws_bounds.LB["f_fair"]))

        return [
            float(omega[0]) / den_stab,
            float(omega[1]) / den_ot,
            float(omega[2]) / den_pref,
            float(omega[3]) / den_fair,
        ]

    def _compute_ws_bounds(self, loaded_case: LoadedCase) -> _WsBounds:
        # Extrempunkte vorhanden?
        if len(self._extreme_F) >= 2:
            Fs = list(self._extreme_F.values())
            LB = {
                "f_stab": min(int(F["f_stab"]) for F in Fs),
                "f_ot": min(int(F["f_ot"]) for F in Fs),
                "f_pref": min(int(F["f_pref"]) for F in Fs),
                "f_fair": min(int(F["f_fair"]) for F in Fs),
            }
            UB = {
                "f_stab": max(int(F["f_stab"]) for F in Fs),
                "f_ot": max(int(F["f_ot"]) for F in Fs),
                "f_pref": max(int(F["f_pref"]) for F in Fs),
                "f_fair": max(int(F["f_fair"]) for F in Fs),
            }
            return _WsBounds(LB=LB, UB=UB, source="extreme_runs_minmax", budget_fraction=float(self._bounds_budget_fraction))
        return self._compute_ws_bounds_fallback_only()

    def _compute_ws_bounds_fallback_only(self) -> _WsBounds:
        # Deterministische Fallback-Bounds aus dem Case (solverfrei; für Normierung/Skalierung).
        case = self._last_loaded_case.case if hasattr(self, "_last_loaded_case") else None
        if case is None:
            # Fallback ohne Case: minimal gültig, aber wenig aussagekräftig.
            LB = {"f_stab": 0, "f_ot": 0, "f_pref": 0, "f_fair": 0}
            UB = {"f_stab": 1, "f_ot": 1, "f_pref": 1, "f_fair": 1}
            return _WsBounds(LB=LB, UB=UB, source="fallback_case_bounds", budget_fraction=float(self._bounds_budget_fraction))

        I = int(case.dimensions.I)
        T_rep = [int(t) for t in case.params["T_rep"]]
        weeks_T_w = getattr(self, "_last_weeks_T_w", [])
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
        return _WsBounds(LB=LB, UB=UB, source="fallback_case_bounds", budget_fraction=float(self._bounds_budget_fraction))

__all__ = ["V1WSMILPProcedure"]
