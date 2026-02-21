from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from etappe01_simulation.simulator.schema import Case

from etappe02_modelle.core.schema import ScheduleDense


@dataclass(frozen=True)
class FeasibilityResult:
    ok: bool
    violations: list[dict[str, Any]] = field(default_factory=list)


def check_feasible(case: Case, schedule: ScheduleDense, *, weeks_T_w: list[list[int]] | None = None) -> FeasibilityResult:
    """
    Harte Restriktionen gemäß Bachelorarbeit Kap. 4.4: Nebenbedingungen (harte Restriktionen).

    Hinweis: Diese Funktion ist verfahrensneutral; nur zulässige Lösungen werden geloggt.
    """

    violations: list[dict[str, Any]] = []

    I = int(case.dimensions.I)
    T = int(case.dimensions.T)
    S = int(case.dimensions.S)

    if len(schedule) != I:
        violations.append({"code": "shape", "message": "len(schedule) != I", "I": I, "len": len(schedule)})
        return FeasibilityResult(ok=False, violations=violations)

    for i in range(I):
        if len(schedule[i]) != T + 1:
            violations.append(
                {"code": "shape", "message": "len(schedule[i]) != T+1", "i": i, "T": T, "len": len(schedule[i])}
            )
            return FeasibilityResult(ok=False, violations=violations)
        for t in range(T + 1):
            s = int(schedule[i][t])
            if s < 0 or s >= S:
                violations.append(
                    {"code": "shift_range", "message": "Schicht-ID außerhalb 0..S-1", "i": i, "t": t, "s": s, "S": S}
                )
                return FeasibilityResult(ok=False, violations=violations)

    S_plus_raw = case.sets.get("S_plus")
    if isinstance(S_plus_raw, list) and S_plus_raw:
        S_plus = [int(s) for s in S_plus_raw]
    else:
        S_plus = list(range(1, S))

    if weeks_T_w is None:
        weeks_raw = case.weeks.get("T_w")
        if isinstance(weeks_raw, list) and weeks_raw:
            weeks_T_w = [[int(t) for t in days] for days in weeks_raw]
        else:
            weeks_T_w = []

    r = case.params["r"]
    a = case.params["a"]
    sigma = case.params["sigma"]
    h = case.params["h"]
    H_max = case.params["H_max"]
    P_raw = case.params["P"]
    T_fix = {int(t) for t in case.params["T_fix"]}

    # Bedarfsdeckung
    for t in range(1, T + 1):
        counts = [0] * S
        for i in range(I):
            counts[int(schedule[i][t])] += 1
        for s in S_plus:
            required = int(r[t][s])
            assigned = int(counts[s])
            if assigned < required:
                violations.append(
                    {"code": "demand", "t": int(t), "s": int(s), "required": int(required), "assigned": int(assigned)}
                )

    # Verfügbarkeit nach Störung
    for i in range(I):
        for t in range(1, T + 1):
            s = int(schedule[i][t])
            if int(a[i][t][s]) != 1:
                violations.append({"code": "availability", "i": int(i), "t": int(t), "s": int(s)})

    # Fixierung
    for i in range(I):
        for t in T_fix:
            s_ref = int(sigma[i][t])
            s_plan = int(schedule[i][t])
            if s_plan != s_ref:
                violations.append(
                    {"code": "fix", "i": int(i), "t": int(t), "s_plan": int(s_plan), "s_ref": int(s_ref)}
                )

    # Wochenstundenlimit
    for i in range(I):
        limit = int(H_max[i])
        for w_idx, w_days in enumerate(weeks_T_w):
            hours = 0
            for t in w_days:
                s = int(schedule[i][t])
                if s in S_plus:
                    hours += int(h[s])
            if hours > limit:
                violations.append(
                    {"code": "week_hours", "i": int(i), "w": int(w_idx), "hours": int(hours), "limit": int(limit)}
                )

    # Verbotene Folgen
    P_set = {(int(s), int(sp)) for s, sp in P_raw}
    for i in range(I):
        for t in range(1, T):
            s = int(schedule[i][t])
            sp = int(schedule[i][t + 1])
            if (s, sp) in P_set:
                violations.append({"code": "forbidden_sequence", "i": int(i), "t": int(t), "s": int(s), "sp": int(sp)})

    return FeasibilityResult(ok=len(violations) == 0, violations=violations)
