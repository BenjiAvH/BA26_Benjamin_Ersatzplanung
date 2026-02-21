from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from etappe02_modelle.core.case_loader import LoadedCase
from etappe02_modelle.core.schema import ScheduleDense


def _require_pyscipopt() -> tuple[Any, Any]:
    try:
        from pyscipopt import Model, quicksum  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PySCIPOpt/SCIP ist nicht verfügbar. Bitte Conda-Umgebung gemäß etappe02_modelle/README.md einrichten "
            "(environment.yml) und `python -m etappe02_modelle.scripts.check_scip` ausführen."
        ) from e
    return Model, quicksum


@dataclass(frozen=True)
class ScipVars:
    x: dict[tuple[int, int, int], Any]
    u: dict[tuple[int, int], Any]
    o: dict[tuple[int, int], Any]
    U_max: Any


def set_solver_params_scip(model: Any, *, threads: int, seed_subrun: int) -> None:
    model.setParam("parallel/maxnthreads", int(max(1, int(threads))))

    # Seed: nicht jedes SCIP-Build hat alle Parameter; daher tolerant setzen.
    for key in ["randomization/randomseedshift", "randomization/permutationseed"]:
        try:
            model.setParam(key, int(seed_subrun))
        except Exception:
            pass


def build_base_model_scip(
    *,
    loaded_case: LoadedCase,
    seed_subrun: int,
    solver_threads: int,
    model_name: str,
) -> tuple[Any, ScipVars, dict[str, Any]]:
    """
    Baut den gemeinsamen MILP-Kern für V1/V2 (SCIP/PySCIPOpt).

    Enthält:
    - Variablen x, u, o, U_max
    - Harte Nebenbedingungen (Bachelorarbeit Kap. 4.4: Nebenbedingungen)
    - Kopplungen/Definitionen für u/o/U^max (Bachelorarbeit Kap. 4.4: Nebenbedingungen; Weiche Restriktionen)
    - Liefert die vier Objective-Komponenten als lineare Ausdrücke zurück.
    """

    Model, quicksum = _require_pyscipopt()

    case = loaded_case.case
    I = int(case.dimensions.I)
    T = int(case.dimensions.T)
    S = int(case.dimensions.S)

    S_plus_raw = case.sets.get("S_plus")
    if isinstance(S_plus_raw, list) and S_plus_raw:
        S_plus = [int(s) for s in S_plus_raw]
    else:
        S_plus = list(range(1, S))

    r = case.params["r"]
    a = case.params["a"]
    sigma = case.params["sigma"]
    h = case.params["h"]
    H_max = case.params["H_max"]
    P = [(int(pair[0]), int(pair[1])) for pair in case.params["P"]]
    T_fix = {int(t) for t in case.params["T_fix"]}
    T_rep = sorted(int(t) for t in case.params["T_rep"])
    weeks_T_w = loaded_case.weeks_T_w
    c = case.params["c"]

    m = Model(str(model_name))
    m.hideOutput()
    set_solver_params_scip(m, threads=int(solver_threads), seed_subrun=int(seed_subrun))

    # Variablen
    x: dict[tuple[int, int, int], Any] = {}
    for i in range(I):
        for t in range(1, T + 1):
            for s in range(S):
                x[(i, t, s)] = m.addVar(vtype="BINARY", name=f"x_{i}_{t}_{s}")

    u: dict[tuple[int, int], Any] = {}
    for i in range(I):
        for t in T_rep:
            u[(i, t)] = m.addVar(vtype="BINARY", name=f"u_{i}_{t}")

    o: dict[tuple[int, int], Any] = {}
    ref_hours_by_iw: dict[tuple[int, int], int] = {}
    for i in range(I):
        for w_idx in range(len(weeks_T_w)):
            limit = int(H_max[i])
            ref_hours = 0
            for t in weeks_T_w[w_idx]:
                s_ref = int(sigma[i][int(t)])
                if s_ref in S_plus:
                    ref_hours += int(h[s_ref])
            ub = max(0, limit - ref_hours)
            ref_hours_by_iw[(i, w_idx)] = int(ref_hours)
            o[(i, w_idx)] = m.addVar(vtype="CONTINUOUS", lb=0.0, ub=float(ub), name=f"o_{i}_{w_idx}")

    U_max = m.addVar(vtype="INTEGER", lb=0, ub=float(len(T_rep)), name="U_max")

    # Harte Restriktionen (Bachelorarbeit Kap. 4.4: Nebenbedingungen)
    # 1) Genau eine Tageszuordnung
    for i in range(I):
        for t in range(1, T + 1):
            m.addCons(quicksum(x[(i, t, s)] for s in range(S)) == 1)

    # 2) Bedarfsdeckung
    for t in range(1, T + 1):
        for s in S_plus:
            required = int(r[t][s])
            m.addCons(quicksum(x[(i, t, s)] for i in range(I)) >= required)

    # 3) Verfügbarkeit nach Störung
    for i in range(I):
        for t in range(1, T + 1):
            for s in range(S):
                m.addCons(x[(i, t, s)] <= int(a[i][t][s]))

    # 4) Fixierung in T_fix
    for i in range(I):
        for t in T_fix:
            s_ref = int(sigma[i][t])
            m.addCons(x[(i, t, s_ref)] == 1)

    # 5) Wochenstundenlimit
    for i in range(I):
        limit = int(H_max[i])
        for w_days in weeks_T_w:
            m.addCons(quicksum(int(h[s]) * x[(i, int(t), s)] for t in w_days for s in S_plus) <= limit)

    # 6) Verbotene Folgen
    for i in range(I):
        for t in range(1, T):
            for s, sp in P:
                m.addCons(x[(i, t, s)] + x[(i, t + 1, sp)] <= 1)

    # Weiche Kopplungen (Bachelorarbeit Kap. 4.4: Nebenbedingungen; Weiche Restriktionen)
    # u_{i,t} = 1 - x_{i,t,sigma_{i,t}} für t in T_rep
    for i in range(I):
        for t in T_rep:
            s_ref = int(sigma[i][t])
            m.addCons(u[(i, t)] + x[(i, t, s_ref)] == 1)

    # Σ_{t∈T_rep} u_{i,t} <= U_max
    for i in range(I):
        m.addCons(quicksum(u[(i, t)] for t in T_rep) <= U_max)

    # o_{i,w} >= Σ_{t∈T_w} Σ_{s∈S+} h_s x_{i,t,s} - Σ_{t∈T_w} h_{sigma_{i,t}}
    for i in range(I):
        for w_idx, w_days in enumerate(weeks_T_w):
            m.addCons(
                o[(i, w_idx)]
                >= quicksum(int(h[s]) * x[(i, int(t), s)] for t in w_days for s in S_plus) - int(ref_hours_by_iw[(i, w_idx)])
            )

    # Objective-Komponenten (Bachelorarbeit Kap. 4.3: Zielfunktion)
    obj_stab = quicksum(u[(i, t)] for i in range(I) for t in T_rep)
    obj_ot = quicksum(o[(i, w_idx)] for i in range(I) for w_idx in range(len(weeks_T_w)))
    obj_pref = quicksum(int(c[i][t][s]) * x[(i, t, s)] for i in range(I) for t in T_rep for s in range(S))
    obj_fair = U_max

    vars_out = ScipVars(x=x, u=u, o=o, U_max=U_max)
    objs_out = {"f_stab": obj_stab, "f_ot": obj_ot, "f_pref": obj_pref, "f_fair": obj_fair}
    return m, vars_out, objs_out


def extract_best_schedule_dense_scip(model: Any, x_vars: dict[tuple[int, int, int], Any], *, I: int, T: int, S: int) -> ScheduleDense | None:
    sol = None
    try:
        sol = model.getBestSol()
    except Exception:
        sol = None
    if sol is None:
        return None

    schedule: ScheduleDense = [[0] * (T + 1) for _ in range(I)]
    for i in range(I):
        for t in range(1, T + 1):
            best_s = 0
            best_v = -1.0
            for s in range(S):
                v = float(model.getSolVal(sol, x_vars[(i, t, s)]))
                if v > best_v:
                    best_v = v
                    best_s = s
            schedule[i][t] = int(best_s)
    return schedule


def extract_solver_meta_scip(model: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {"solver_name": "scip"}

    try:
        if hasattr(model, "getVersion"):
            meta["solver_version"] = str(model.getVersion())
    except Exception:
        pass

    status = None
    try:
        status = str(model.getStatus())
    except Exception:
        status = None
    meta["solver_status"] = status

    incumbent_found = False
    try:
        incumbent_found = int(model.getNSols()) > 0
    except Exception:
        incumbent_found = False
    meta["incumbent_found"] = bool(incumbent_found)

    status_l = str(status or "").lower()
    if "infeasible" in status_l:
        term = "infeasible"
    elif "optimal" in status_l:
        term = "optimal"
    elif not incumbent_found:
        term = "no_incumbent"
    elif "time" in status_l and "limit" in status_l:
        term = "time_limit"
    elif "timelimit" in status_l:
        term = "time_limit"
    else:
        term = "error"
    meta["termination_reason"] = term

    # Optional: MILP-Stand (Gap/Bounds/Obj).
    try:
        meta["mip_gap"] = float(model.getGap())
    except Exception:
        pass
    try:
        meta["best_bound"] = float(model.getDualbound())
    except Exception:
        pass
    try:
        meta["best_obj_scalar"] = float(model.getObjVal())
    except Exception:
        pass

    return meta


__all__ = [
    "ScipVars",
    "build_base_model_scip",
    "extract_best_schedule_dense_scip",
    "extract_solver_meta_scip",
    "set_solver_params_scip",
]
