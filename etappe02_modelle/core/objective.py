from __future__ import annotations

import hashlib
from typing import Any

from etappe01_simulation.simulator.schema import Case

from etappe02_modelle.core.schema import ObjectiveVector, ScheduleDense


def encode_schedule_dense(schedule: ScheduleDense) -> bytes:
    """
    Kanonische Byte-Repräsentation eines dichten Dienstplans (ScheduleDense; I×(T+1)).

    Verwendet für `solution_id = sha256(schedule_dense)`.
    """

    # Keine Whitespace-Variation; deterministisch.
    rows = [",".join(str(int(v)) for v in row) for row in schedule]
    return ("\n".join(rows)).encode("utf-8")


def compute_solution_id(schedule: ScheduleDense) -> str:
    return hashlib.sha256(encode_schedule_dense(schedule)).hexdigest()


def schedule_delta_vs_sigma(case: Case, schedule: ScheduleDense) -> dict[str, Any]:
    """
    Delta-Encoding für Logs: nur Änderungen ggü. Referenzdienstplan σ, nur t in T_rep.
    Sortierung: zuerst nach t, dann nach i.
    """

    sigma = case.params["sigma"]
    T_rep = {int(t) for t in case.params["T_rep"]}

    changes: list[dict[str, int]] = []
    I = int(case.dimensions.I)
    for i in range(I):
        for t in sorted(T_rep):
            s_ref = int(sigma[i][t])
            s_new = int(schedule[i][t])
            if s_new != s_ref:
                changes.append({"i": int(i), "t": int(t), "s": int(s_new)})

    changes.sort(key=lambda c: (int(c["t"]), int(c["i"])))
    return {"encoding": "delta_vs_sigma", "changes": changes}


def evaluate_F(case: Case, schedule: ScheduleDense, *, weeks_T_w: list[list[int]] | None = None) -> ObjectiveVector:
    """
    Zielvektor F(x) gemäß Bachelorarbeit (Kap. 4.3: Zielfunktion; Kap. 5.1: Forschungsdesign & Evaluationsstrategie):
      f_stab = Σ_i Σ_{t∈T_rep} u_{i,t}
      f_ot   = Σ_i Σ_w o_{i,w}   (Mehrarbeit, nur positive Abweichung)
      f_pref = Σ_i Σ_{t∈T_rep} Σ_s c_{i,t,s} x_{i,t,s}  (äquivalent: c_{i,t,shift_{i,t}})
      f_fair = U^max = max_i Σ_{t∈T_rep} u_{i,t}
    """

    I = int(case.dimensions.I)
    T = int(case.dimensions.T)
    S = int(case.dimensions.S)

    if len(schedule) != I:
        raise ValueError(f"Schedule-Shape ungültig: len(schedule)={len(schedule)} != I={I}")
    for i in range(I):
        if len(schedule[i]) != T + 1:
            raise ValueError(f"Schedule-Shape ungültig: len(schedule[{i}])={len(schedule[i])} != T+1={T+1}")
        for t in range(T + 1):
            s = int(schedule[i][t])
            if s < 0 or s >= S:
                raise ValueError(f"Schedule enthält ungültige Schicht-ID: schedule[{i}][{t}]={s} (S={S})")

    sigma = case.params["sigma"]
    c = case.params["c"]
    h = case.params["h"]

    S_plus_raw = case.sets.get("S_plus")
    if isinstance(S_plus_raw, list) and S_plus_raw:
        S_plus = {int(s) for s in S_plus_raw}
    else:
        S_plus = set(range(1, S))

    if weeks_T_w is None:
        weeks_raw = case.weeks.get("T_w")
        if isinstance(weeks_raw, list) and weeks_raw:
            weeks_T_w = [[int(t) for t in days] for days in weeks_raw]
        else:
            weeks_T_w = []

    T_rep = {int(t) for t in case.params["T_rep"]}

    u_by_i: list[int] = [0] * I
    f_pref = 0

    for i in range(I):
        for t in T_rep:
            if int(schedule[i][t]) != int(sigma[i][t]):
                u_by_i[i] += 1
            f_pref += int(c[i][t][int(schedule[i][t])])

    f_stab = sum(u_by_i)
    f_fair = max(u_by_i) if u_by_i else 0

    # Mehrarbeit: pro Woche nur positive Abweichung ggü. Referenzdienstplan.
    f_ot = 0
    for i in range(I):
        for w_days in weeks_T_w:
            plan_hours = 0
            ref_hours = 0
            for t in w_days:
                s_plan = int(schedule[i][t])
                s_ref = int(sigma[i][t])
                if s_plan in S_plus:
                    plan_hours += int(h[s_plan])
                if s_ref in S_plus:
                    ref_hours += int(h[s_ref])
            if plan_hours > ref_hours:
                f_ot += int(plan_hours - ref_hours)

    return ObjectiveVector(f_stab=int(f_stab), f_ot=int(f_ot), f_pref=int(f_pref), f_fair=int(f_fair))
