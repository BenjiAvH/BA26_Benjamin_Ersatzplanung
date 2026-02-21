from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _set_from_list(values: Iterable[int]) -> set[int]:
    return set(int(v) for v in values)


def validate_case(case: dict[str, Any]) -> dict[str, Any]:
    """
    Harte Prüfungen für ein Instanz–Szenario-JSON (Case).

    Fokus: Vollständigkeit und Konsistenz (Kap. 4.4: Nebenbedingungen; Kap. 5.2: Datengrundlage & Szenarien).
    Konvention: Störung reduziert die Verfügbarkeit a gegenüber a_base (keine Erhöhungen).
    """

    validation: dict[str, Any] = {"ok": False, "notes": []}

    required_top = [
        "schema_version",
        "dataset_id",
        "case_id",
        "seeds",
        "dimensions",
        "sets",
        "weeks",
        "params",
        "scenario",
        "tags",
    ]
    missing = [k for k in required_top if k not in case]
    validation["schema_ok"] = len(missing) == 0
    if missing:
        validation["notes"].append(f"Fehlende Top-Level Keys: {missing}")
        return validation

    dims = case["dimensions"]
    I = int(dims["I"])
    T = int(dims["T"])
    S = int(dims["S"])
    S_plus = list(range(1, S))

    params = case["params"]
    required_params = ["r", "a_base", "a", "sigma", "c", "h", "H_max", "P", "T_fix", "T_rep"]
    missing_params = [k for k in required_params if k not in params]
    validation["params_ok"] = len(missing_params) == 0
    if missing_params:
        validation["notes"].append(f"Fehlende params Keys: {missing_params}")
        return validation

    r = params["r"]
    a_base = params["a_base"]
    a = params["a"]
    sigma = params["sigma"]
    c = params["c"]
    h = params["h"]
    H_max = params["H_max"]
    P = params["P"]
    T_fix = _set_from_list(params["T_fix"])
    T_rep = _set_from_list(params["T_rep"])

    shapes_ok = True
    if len(r) != T + 1:
        shapes_ok = False
    else:
        shapes_ok = all(len(row) == S for row in r)

    if len(sigma) != I:
        shapes_ok = False
    else:
        shapes_ok = shapes_ok and all(len(row) == T + 1 for row in sigma)

    def _check_3d_shape(arr: Any) -> bool:
        if len(arr) != I:
            return False
        for i in range(I):
            if len(arr[i]) != T + 1:
                return False
            for t in range(T + 1):
                if len(arr[i][t]) != S:
                    return False
        return True

    shapes_ok = shapes_ok and _check_3d_shape(a_base) and _check_3d_shape(a) and _check_3d_shape(c)
    shapes_ok = shapes_ok and len(h) == S and len(H_max) == I
    validation["shapes_ok"] = shapes_ok
    if not shapes_ok:
        validation["notes"].append("Shape-Check fehlgeschlagen (r/sigma/a/a_base/c/h/H_max).")
        return validation

    # Partition T_fix/T_rep
    all_days = set(range(1, T + 1))
    T_partition_ok = (T_fix & T_rep) == set() and (T_fix | T_rep) == all_days
    validation["T_partition_ok"] = T_partition_ok
    if not T_partition_ok:
        validation["notes"].append("T_fix/T_rep sind keine disjunkte Partition von T.")

    # Wochenstruktur
    weeks = case["weeks"].get("T_w")
    if not isinstance(weeks, list) or not weeks:
        weeks = [list(range(w * 7 + 1, min(T, (w + 1) * 7) + 1)) for w in range((T + 6) // 7)]

    # Referenzdienstplan: Bedarf, Verfügbarkeit, verbotene Sequenzen, Wochenstunden
    demand_viol = 0
    availability_viol = 0
    week_hours_viol = 0
    forbidden_viol = 0

    P_set = {(int(s), int(sp)) for s, sp in P}

    for t in range(1, T + 1):
        counts = [0] * S
        for i in range(I):
            s = int(sigma[i][t])
            if s < 0 or s >= S:
                availability_viol += 1
                continue
            counts[s] += 1
            if int(a_base[i][t][s]) != 1:
                availability_viol += 1
        for s in S_plus:
            if counts[s] < int(r[t][s]):
                demand_viol += 1

    for i in range(I):
        for t in range(1, T):
            s = int(sigma[i][t])
            sp = int(sigma[i][t + 1])
            if (s, sp) in P_set:
                forbidden_viol += 1

    for i in range(I):
        for w_days in weeks:
            hours = 0
            for t in w_days:
                s = int(sigma[i][t])
                if s in S_plus:
                    hours += int(h[s])
            if hours > int(H_max[i]):
                week_hours_viol += 1

    validation["reference_demand_ok"] = demand_viol == 0
    validation["reference_demand_violations"] = demand_viol
    validation["reference_availability_ok"] = availability_viol == 0
    validation["reference_availability_violations"] = availability_viol
    validation["reference_forbidden_sequences_ok"] = forbidden_viol == 0
    validation["reference_forbidden_sequence_violations"] = forbidden_viol
    validation["reference_week_hours_ok"] = week_hours_viol == 0
    validation["reference_week_hours_violations"] = week_hours_viol

    # Bedarf: Freischicht hat keinen Bedarf (r[t][0] = 0).
    r0_viol = 0
    for t in range(1, T + 1):
        if int(r[t][0]) != 0:
            r0_viol += 1
    validation["demand_free_shift_ok"] = r0_viol == 0
    validation["demand_free_shift_violations"] = r0_viol

    a0_viol = 0
    a_binary_viol = 0
    a_increase_viol = 0
    for i in range(I):
        for t in range(T + 1):
            if int(a_base[i][t][0]) != 1 or int(a[i][t][0]) != 1:
                a0_viol += 1
            for s in range(S):
                if int(a_base[i][t][s]) not in (0, 1) or int(a[i][t][s]) not in (0, 1):
                    a_binary_viol += 1
                # Störungskonvention: a darf a_base nur unterschreiten (keine Erhöhungen der Verfügbarkeit).
                if int(a[i][t][s]) > int(a_base[i][t][s]):
                    a_increase_viol += 1
    validation["availability_free_shift_ok"] = a0_viol == 0
    validation["availability_free_shift_violations"] = a0_viol
    validation["availability_binary_ok"] = a_binary_viol == 0
    validation["availability_binary_violations"] = a_binary_viol
    validation["disturbance_only_reduces_availability_ok"] = a_increase_viol == 0
    validation["disturbance_only_reduces_availability_violations"] = a_increase_viol

    c0_viol = 0
    c_negative_viol = 0
    for i in range(I):
        for t in range(T + 1):
            if int(c[i][t][0]) != 0:
                c0_viol += 1
            for s in range(S):
                if int(c[i][t][s]) < 0:
                    c_negative_viol += 1
    validation["preference_free_shift_ok"] = c0_viol == 0
    validation["preference_free_shift_violations"] = c0_viol
    validation["preference_nonnegative_ok"] = c_negative_viol == 0
    validation["preference_negative_violations"] = c_negative_viol

    # Fix-Kollisionen nach Störung: Referenzdienstplan σ muss auf T_fix weiterhin zulässig sein (a=1).
    fix_viol = 0
    for i in range(I):
        for t in T_fix:
            s = int(sigma[i][t])
            if int(a[i][t][s]) != 1:
                fix_viol += 1
    validation["fix_ok_after_disturbance"] = fix_viol == 0
    validation["fix_violations"] = fix_viol

    events = list(case.get("scenario", {}).get("events", []))
    events_viol = 0
    for ev in events:
        try:
            i = int(ev["i"])
            t_start = int(ev["t_start"])
            duration = int(ev["duration"])
        except Exception:
            events_viol += 1
            continue
        if i < 0 or i >= I or duration < 1:
            events_viol += 1
            continue
        for dt in range(duration):
            t = t_start + dt
            if t not in T_rep:
                events_viol += 1
                break
    validation["events_in_T_rep_ok"] = events_viol == 0
    validation["events_in_T_rep_violations"] = events_viol

    # Schnelltest nach Störung: pro (t,s) ausreichend Verfügbarkeiten für den Bedarf.
    feas_viol = 0
    for t in range(1, T + 1):
        for s in S_plus:
            avail = 0
            for i in range(I):
                avail += int(a[i][t][s])
            if avail < int(r[t][s]):
                feas_viol += 1
    validation["post_disturbance_feasibility_ok"] = feas_viol == 0
    validation["post_disturbance_feasibility_violations"] = feas_viol

    validation["ok"] = (
        validation["schema_ok"]
        and validation["params_ok"]
        and validation["shapes_ok"]
        and validation["T_partition_ok"]
        and validation["reference_demand_ok"]
        and validation["reference_availability_ok"]
        and validation["reference_forbidden_sequences_ok"]
        and validation["reference_week_hours_ok"]
        and validation["demand_free_shift_ok"]
        and validation["availability_free_shift_ok"]
        and validation["availability_binary_ok"]
        and validation["disturbance_only_reduces_availability_ok"]
        and validation["preference_free_shift_ok"]
        and validation["preference_nonnegative_ok"]
        and validation["events_in_T_rep_ok"]
        and validation["fix_ok_after_disturbance"]
    )
    validation["ok_strict"] = bool(validation["ok"] and validation["post_disturbance_feasibility_ok"])
    return validation
