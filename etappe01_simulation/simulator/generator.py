from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any

from etappe01_simulation.simulator.seeding import derive_seed
from etappe01_simulation.simulator.validate import validate_case


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def _ceil_int(x: float) -> int:
    return int(math.ceil(x))


def build_weeks(T: int) -> list[list[int]]:
    weeks: list[list[int]] = []
    w = 0
    while w * 7 < T:
        start = w * 7 + 1
        end = min(T, (w + 1) * 7)
        weeks.append(list(range(start, end + 1)))
        w += 1
    return weeks


def compute_fix_rep(T: int, fraction: float, min_days: int, max_days: int) -> tuple[list[int], list[int]]:
    fix_days = min(max_days, max(min_days, _ceil_int(fraction * T)))
    T_fix = list(range(1, fix_days + 1))
    T_rep = list(range(fix_days + 1, T + 1))
    return T_fix, T_rep


def generate_demand(
    *,
    T: int,
    I: int,
    S: int,
    weekday_frac: float,
    weekend_frac: float,
    split_work: list[float] | None = None,
    split_frueh: float | None = None,
    split_spaet: float | None = None,
    split_nacht: float | None = None,
) -> list[list[int]]:
    # r[t][s] mit t=0 unbenutzt
    if S < 2:
        raise ValueError("Ungültige Schichtanzahl S: erwartet S>=2 (inkl. 0=frei).")
    r: list[list[int]] = [[0] * S for _ in range(T + 1)]

    if split_work is None:
        # Default: 3 Arbeitsschichten {1=Früh,2=Spät,3=Nacht}
        if S != 4:
            raise ValueError(
                "Demand-Config unvollständig: Für S != 4 muss 'split_work' (Länge S-1) angegeben werden."
            )
        if split_frueh is None or split_spaet is None or split_nacht is None:
            raise ValueError("Demand-Config unvollständig: split_frueh/split_spaet/split_nacht fehlen.")
        split_work = [float(split_frueh), float(split_spaet), float(split_nacht)]
    else:
        if len(split_work) != S - 1:
            raise ValueError(f"Demand-Config ungültig: split_work hat Länge {len(split_work)}, erwartet {S-1}.")
        split_work = [float(x) for x in split_work]

    if any(x < 0 for x in split_work):
        raise ValueError("Demand-Config ungültig: split_work enthält negative Anteile.")
    total_split = float(sum(split_work))
    if total_split <= 0:
        raise ValueError("Demand-Config ungültig: Summe(split_work) muss > 0 sein.")
    weights = [x / total_split for x in split_work]

    for t in range(1, T + 1):
        day_of_week = (t - 1) % 7  # 0=Mo, ..., 5=Sa, 6=So
        is_weekend = day_of_week in (5, 6)
        frac = weekend_frac if is_weekend else weekday_frac
        total = _round_half_up(frac * I)

        # Erst runden, dann deterministisch auf Summe=total korrigieren (Rest auf die erste Arbeitsschicht).
        demands = [_round_half_up(total * w) for w in weights]
        diff = int(total) - int(sum(demands))
        if diff > 0:
            demands[0] += diff
        elif diff < 0:
            need = -diff
            # Reduziere zuerst die größten Blöcke, um Negativwerte zu vermeiden.
            for idx in sorted(range(len(demands)), key=lambda i: demands[i], reverse=True):
                take = min(int(demands[idx]), int(need))
                demands[idx] -= take
                need -= take
                if need == 0:
                    break
            if need != 0:
                raise RuntimeError("Interner Fehler: Demand-Rundung konnte nicht korrigiert werden.")

        for s in range(1, S):
            r[t][s] = int(demands[s - 1])

    return r


def generate_qualification(I: int, q_night: float, rng: random.Random) -> set[int]:
    n_night = _round_half_up(q_night * I)
    n_night = max(0, min(I, n_night))
    return set(rng.sample(list(range(I)), k=n_night))


def generate_availability_base(
    *,
    I: int,
    T: int,
    S: int,
    night_qualified: set[int],
    prob: float,
    r: list[list[int]],
    slack_min: int,
    slack_fraction: float,
    rng: random.Random,
    sigma: list[list[int]],
) -> list[list[list[int]]]:
    a: list[list[list[int]]] = [[[0] * S for _ in range(T + 1)] for _ in range(I)]

    for i in range(I):
        a[i][0][0] = 1
        for t in range(1, T + 1):
            a[i][t][0] = 1
            for s in range(1, S):
                if s == 3 and i not in night_qualified:
                    a[i][t][s] = 0
                else:
                    a[i][t][s] = 1 if (rng.random() < prob) else 0
            # Referenzdienstplan muss zulässig bleiben: a_base(i,t, sigma(i,t)) = 1
            s_ref = int(sigma[i][t])
            if 0 <= s_ref < S:
                a[i][t][s_ref] = 1

    slack = max(slack_min, _ceil_int(slack_fraction * I))
    for t in range(1, T + 1):
        for s in range(1, S):
            required = int(r[t][s]) + slack
            have = sum(int(a[i][t][s]) for i in range(I))
            if have >= required:
                continue

            candidates: list[int] = []
            for i in range(I):
                if a[i][t][s] == 1:
                    continue
                if s == 3 and i not in night_qualified:
                    continue
                candidates.append(i)

            rng.shuffle(candidates)
            need = required - have
            if len(candidates) < need:
                raise RuntimeError(f"Verfügbarkeits-Guard nicht erfüllbar: t={t}, s={s}, need={need}, cand={len(candidates)}")
            for i in candidates[:need]:
                a[i][t][s] = 1

    return a


def generate_preference_costs(
    *,
    I: int,
    T: int,
    S: int,
    night_qualified: set[int],
    noise: bool,
    rng: random.Random,
) -> list[list[list[int]]]:
    c: list[list[list[int]]] = [[[0] * S for _ in range(T + 1)] for _ in range(I)]

    preferred: list[int] = [1] * I
    disliked: list[int] = [2] * I

    for i in range(I):
        allowed_pref = [s for s in range(1, S) if not (s == 3 and i not in night_qualified)]
        if not allowed_pref:
            raise RuntimeError("Keine Arbeitsschichten für Präferenzkosten verfügbar (prüfe S/h/Qualifikation).")
        preferred[i] = rng.choice(allowed_pref)
        allowed_disliked = [s for s in allowed_pref if s != preferred[i]]
        disliked[i] = rng.choice(allowed_disliked) if allowed_disliked else preferred[i]

    for i in range(I):
        c[i][0][0] = 0
        for t in range(1, T + 1):
            c[i][t][0] = 0
            for s in range(1, S):
                if s == preferred[i]:
                    base = rng.randint(0, 1)
                elif s == disliked[i]:
                    base = rng.randint(5, 8)
                else:
                    base = rng.randint(2, 4)
                if noise:
                    base = max(0, base + rng.choice([-1, 0, 1]))
                c[i][t][s] = base
    return c


def construct_reference_plan(
    *,
    I: int,
    T: int,
    S_plus: list[int],
    r: list[list[int]],
    night_qualified: set[int],
    P_forbidden: list[list[int]],
    weeks: list[list[int]],
    h: list[int],
    H_max: list[int],
    base_seed: int,
    max_restarts: int,
) -> list[list[int]]:
    """
    Solverfreie Konstruktion eines zulässigen Referenzdienstplans σ_{i,t}.

    Heuristik: tagweise Greedy-Zuweisung (Früh->Nacht->Spät), mit Restart bei Fehlschlag.
    """

    P_set = {(int(s), int(sp)) for s, sp in P_forbidden}

    for attempt in range(max_restarts):
        rng = random.Random(derive_seed(base_seed, "refplan", f"attempt_{attempt}"))
        sigma: list[list[int]] = [[0] * (T + 1) for _ in range(I)]

        week_hours = [[0 for _ in range(len(weeks))] for _ in range(I)]

        ok = True
        for t in range(1, T + 1):
            w_idx = (t - 1) // 7
            unassigned = set(range(I))

            # Reihenfolge ist wichtig: Früh zuerst (wegen (Spät/Nacht)->Früh-Verboten),
            # danach Nacht (Qualifikation), zuletzt Spät.
            preferred_order: list[int] = []
            for s in [1, 3, 2]:
                if s in S_plus:
                    preferred_order.append(s)
            for s in S_plus:
                if s not in preferred_order:
                    preferred_order.append(s)

            for s in preferred_order:
                required = int(r[t][s])
                if required <= 0:
                    continue

                candidates: list[int] = []
                for i in range(I):
                    if i not in unassigned:
                        continue
                    if s == 3 and i not in night_qualified:
                        continue
                    prev = int(sigma[i][t - 1]) if t > 1 else 0
                    if (prev, s) in P_set:
                        continue
                    if week_hours[i][w_idx] + int(h[s]) > int(H_max[i]):
                        continue
                    candidates.append(i)

                if len(candidates) < required:
                    ok = False
                    break

                candidates.sort(key=lambda i: (week_hours[i][w_idx], rng.random()))
                chosen = candidates[:required]
                for i in chosen:
                    sigma[i][t] = s
                    unassigned.remove(i)
                    week_hours[i][w_idx] += int(h[s])

            if not ok:
                break

        if ok:
            return sigma

    raise RuntimeError(f"Referenzplan konnte nach {max_restarts} Restarts nicht konstruiert werden.")


def _post_disturbance_feasible(
    *,
    I: int,
    T: int,
    S_plus: list[int],
    r: list[list[int]],
    a: list[list[list[int]]],
) -> bool:
    for t in range(1, T + 1):
        for s in S_plus:
            avail = 0
            for i in range(I):
                avail += int(a[i][t][s])
            if avail < int(r[t][s]):
                return False
    return True


def generate_events(
    *,
    I: int,
    T_rep: list[int],
    n_absent: int,
    duration_days: int,
    rng: random.Random,
) -> list[dict[str, int]]:
    if not T_rep:
        return []
    rep_start = min(T_rep)
    rep_end = max(T_rep)
    latest_start = rep_end - duration_days + 1
    if latest_start < rep_start:
        raise RuntimeError("Repair-Horizont zu kurz für die gewählte Störungsdauer.")

    employees = rng.sample(list(range(I)), k=min(n_absent, I))
    events: list[dict[str, int]] = []
    for i in employees:
        t_start = rng.randint(rep_start, latest_start)
        events.append({"i": int(i), "t_start": int(t_start), "duration": int(duration_days)})
    return events


def apply_disturbance(
    *,
    a_base: list[list[list[int]]],
    sigma: list[list[int]],
    events: list[dict[str, int]],
    model: str,
) -> list[list[list[int]]]:
    a = copy.deepcopy(a_base)
    S = len(a[0][0])
    S_plus = list(range(1, S))

    for ev in events:
        i = int(ev["i"])
        t_start = int(ev["t_start"])
        duration = int(ev["duration"])
        for dt in range(duration):
            t = t_start + dt
            if model == "shift_specific":
                s = int(sigma[i][t])
                if s in S_plus:
                    a[i][t][s] = 0
            else:
                for s in S_plus:
                    a[i][t][s] = 0
            a[i][t][0] = 1

    return a


@dataclass(frozen=True)
class CaseTags:
    size_class: str
    availability_level: str
    demand_pattern: str
    severity: str


def generate_case(
    *,
    dataset_id: str,
    schema_version: str,
    global_seed: int,
    size_class: str,
    I: int,
    T: int,
    availability_level: str,
    availability_prob: float,
    demand_pattern: str,
    weekday_frac: float,
    weekend_frac: float,
    split_frueh: float | None = None,
    split_spaet: float | None = None,
    split_nacht: float | None = None,
    q_night: float,
    slack_min: int,
    slack_fraction: float,
    fix_fraction: float,
    fix_min_days: int,
    fix_max_days: int,
    h: list[int],
    H_max_value: int,
    P_forbidden: list[list[int]],
    disturbance_model: str,
    disturbance_attempts: int,
    severity: str,
    n_absent_fraction: float,
    n_absent_min: int,
    duration_days: int,
    preferences_noise: bool,
    split_work: list[float] | None = None,
    include_timestamp: bool = False,
) -> dict[str, Any]:
    if not h:
        raise ValueError("Ungültige Schichtdefinition: h ist leer.")
    if int(h[0]) != 0:
        raise ValueError("Ungültige Schichtdefinition: h[0] muss 0 sein (Schicht 0 = frei).")
    S = int(len(h))
    for pair in P_forbidden:
        if len(pair) != 2:
            raise ValueError(f"Ungültiger Eintrag in P_forbidden: {pair} (erwartet 2 Integers).")
        s, sp = int(pair[0]), int(pair[1])
        if not (0 <= s < S and 0 <= sp < S):
            raise ValueError(f"Ungültiger Eintrag in P_forbidden: {(s, sp)} (Indices außerhalb 0..{S-1}).")

    tags = CaseTags(
        size_class=size_class,
        availability_level=availability_level,
        demand_pattern=demand_pattern,
        severity=severity,
    )
    case_id = f"sc-{tags.size_class}__av-{tags.availability_level}__dp-{tags.demand_pattern}__sev-{tags.severity}"

    instance_seed = derive_seed(global_seed, "instance", tags.size_class, tags.availability_level, tags.demand_pattern)
    scenario_seed = derive_seed(global_seed, "scenario", tags.size_class, tags.availability_level, tags.demand_pattern, tags.severity)

    rng_instance = random.Random(instance_seed)
    weeks = build_weeks(T)
    T_fix, T_rep = compute_fix_rep(T, fix_fraction, fix_min_days, fix_max_days)

    r = generate_demand(
        T=T,
        I=I,
        S=S,
        weekday_frac=weekday_frac,
        weekend_frac=weekend_frac,
        split_work=split_work,
        split_frueh=split_frueh,
        split_spaet=split_spaet,
        split_nacht=split_nacht,
    )

    night_qualified = generate_qualification(I, q_night, rng_instance)
    H_max = [int(H_max_value) for _ in range(I)]
    sigma = construct_reference_plan(
        I=I,
        T=T,
        S_plus=list(range(1, S)),
        r=r,
        night_qualified=night_qualified,
        P_forbidden=P_forbidden,
        weeks=weeks,
        h=h,
        H_max=H_max,
        base_seed=instance_seed,
        max_restarts=30,
    )

    a_base = generate_availability_base(
        I=I,
        T=T,
        S=S,
        night_qualified=night_qualified,
        prob=availability_prob,
        r=r,
        slack_min=slack_min,
        slack_fraction=slack_fraction,
        rng=rng_instance,
        sigma=sigma,
    )

    c = generate_preference_costs(
        I=I,
        T=T,
        S=S,
        night_qualified=night_qualified,
        noise=preferences_noise,
        rng=rng_instance,
    )

    n_absent = max(int(n_absent_min), _ceil_int(n_absent_fraction * I))
    chosen_events: list[dict[str, int]] = []
    chosen_a: list[list[list[int]]] | None = None
    scenario_status = "ok"

    for attempt in range(max(1, int(disturbance_attempts))):
        rng_scenario = random.Random(derive_seed(scenario_seed, "disturbance", f"attempt_{attempt}"))
        events = generate_events(
            I=I,
            T_rep=T_rep,
            n_absent=n_absent,
            duration_days=int(duration_days),
            rng=rng_scenario,
        )
        disturbed = apply_disturbance(a_base=a_base, sigma=sigma, events=events, model=disturbance_model)
        chosen_events = events
        chosen_a = disturbed
        if _post_disturbance_feasible(I=I, T=T, S_plus=list(range(1, S)), r=r, a=disturbed):
            scenario_status = "ok"
            break
        scenario_status = "likely_infeasible"

    assert chosen_a is not None

    case: dict[str, Any] = {
        "schema_version": str(schema_version),
        "dataset_id": str(dataset_id),
        "case_id": case_id,
        "seeds": {
            "global": int(global_seed),
            "instance": int(instance_seed),
            "scenario": int(scenario_seed),
        },
        "dimensions": {"I": int(I), "T": int(T), "S": int(S)},
        "sets": {
            "I": list(range(I)),
            "T": list(range(1, T + 1)),
            "S": list(range(S)),
            "S_plus": list(range(1, S)),
        },
        "weeks": {"W": list(range(1, len(weeks) + 1)), "T_w": weeks},
        "params": {
            "h": [int(v) for v in h],
            "H_max": H_max,
            "P": [[int(s), int(sp)] for s, sp in P_forbidden],
            "T_fix": [int(t) for t in T_fix],
            "T_rep": [int(t) for t in T_rep],
            "r": r,
            "a_base": a_base,
            "a": chosen_a,
            "sigma": sigma,
            "c": c,
        },
        "scenario": {
            "severity": str(severity),
            "model": str(disturbance_model),
            "events": [
                {"i": int(ev["i"]), "t_start": int(ev["t_start"]), "duration": int(ev["duration"])} for ev in chosen_events
            ],
            "status": scenario_status,
        },
        "tags": {
            "size_class": tags.size_class,
            "availability_level": tags.availability_level,
            "demand_pattern": tags.demand_pattern,
        },
    }

    if include_timestamp:
        import datetime as _dt

        case["generated_at"] = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()

    case["validation"] = validate_case(case)
    return case
