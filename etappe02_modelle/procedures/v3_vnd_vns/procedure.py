from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
import random
import time
from typing import Any

from etappe01_simulation.simulator.seeding import derive_seed

from etappe02_modelle.core.case_loader import LoadedCase
from etappe02_modelle.core.feasibility import check_feasible
from etappe02_modelle.core.objective import compute_solution_id, evaluate_F
from etappe02_modelle.core.pareto import dominates
from etappe02_modelle.core.schema import ScheduleDense, SubrunSpec
from etappe02_modelle.procedures.interfaces import _subruns_from_explicit_list


def _omega21() -> list[list[float]]:
    """
    Deterministische ω-Menge (21 Punkte), simplex-nah.

    Reihenfolge: (f_stab, f_ot, f_pref, f_fair).

    Hinweis: Diese Liste ist bewusst identisch zur Default-Ω in V1,
    damit die ω-Impulse über Verfahren hinweg konsistent interpretierbar bleiben.
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
    # Stabilitäts-betont (6; Adaption, aber strukturkonform zu "simplex-nahem" Ω)
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
        raise ValueError("V3-Config ungültig: omega muss ein Array mit 4 Einträgen sein (stab, ot, pref, fair).")

    vals: list[float] = []
    for idx, v in enumerate(raw):
        try:
            fv = float(v)
        except Exception as e:
            raise ValueError(f"V3-Config ungültig: omega[{idx}] ist keine Zahl: {v!r}") from e
        if fv < 0.0:
            raise ValueError(f"V3-Config ungültig: omega[{idx}] < 0 ist nicht erlaubt: {fv}")
        vals.append(fv)

    s = float(sum(vals))
    if not math.isfinite(s) or s <= 0.0:
        raise ValueError("V3-Config ungültig: Summe(omega) muss > 0 sein.")
    return [v / s for v in vals]


def _copy_schedule(schedule: ScheduleDense) -> ScheduleDense:
    return [list(row) for row in schedule]


def _compute_S_plus(S: int, S_plus_raw: Any) -> list[int]:
    if isinstance(S_plus_raw, list) and S_plus_raw:
        return [int(s) for s in S_plus_raw]
    return list(range(1, int(S)))


def _compute_day_to_week(*, weeks_T_w: list[list[int]], T: int) -> list[int]:
    out = [-1] * (int(T) + 1)
    for w_idx, days in enumerate(weeks_T_w):
        for t in days:
            tt = int(t)
            if 1 <= tt <= int(T):
                out[tt] = int(w_idx)
    return out


def _compute_ub_scales(*, loaded_case: LoadedCase) -> dict[str, int]:
    """
    Deterministische UB-Skalen für Z_ω_norm (Skalierung der ω-Leitsuche).
    """

    case = loaded_case.case
    I = int(case.dimensions.I)
    T = int(case.dimensions.T)
    S = int(case.dimensions.S)

    weeks_T_w = loaded_case.weeks_T_w

    S_plus = _compute_S_plus(S, case.sets.get("S_plus"))
    T_rep = [int(t) for t in case.params["T_rep"]]
    sigma = case.params["sigma"]
    c = case.params["c"]
    h = case.params["h"]

    UB_stab = int(I * len(T_rep))
    UB_fair = int(len(T_rep))

    UB_pref = 0
    for i in range(I):
        for t in T_rep:
            UB_pref += int(max(int(v) for v in c[i][t]))

    max_h = int(max((int(h[s]) for s in S_plus), default=0))
    UB_ot = 0
    for w_days in weeks_T_w:
        ub_plan_hours = int(len(w_days) * max_h)
        for i in range(I):
            ref_hours = 0
            for t in w_days:
                ref_hours += int(h[int(sigma[i][int(t)])])
            UB_ot += int(max(0, ub_plan_hours - ref_hours))

    def _max1(x: int) -> int:
        return int(x) if int(x) >= 1 else 1

    return {
        "f_stab": _max1(int(UB_stab)),
        "f_ot": _max1(int(UB_ot)),
        "f_pref": _max1(int(UB_pref)),
        "f_fair": _max1(int(UB_fair)),
    }


def _Z_omega_norm(*, F: Any, omega: list[float], UB: dict[str, int]) -> float:
    f_stab = float(F.f_stab)
    f_ot = float(F.f_ot)
    f_pref = float(F.f_pref)
    f_fair = float(F.f_fair)

    return float(
        float(omega[0]) * (f_stab / float(UB["f_stab"]))
        + float(omega[1]) * (f_ot / float(UB["f_ot"]))
        + float(omega[2]) * (f_pref / float(UB["f_pref"]))
        + float(omega[3]) * (f_fair / float(UB["f_fair"]))
    )


def _archive_add_if_nd(
    archive: list[tuple[tuple[int, int, int, int], str]],
    *,
    F_tup: tuple[int, int, int, int],
    solution_id: str,
) -> bool:
    # gleiche solution_id oder gleicher Zielvektor erweitert das Archiv nicht.
    for Ft, sid in archive:
        if sid == solution_id:
            return False
        if Ft == F_tup:
            return False
        if dominates(Ft, F_tup):
            return False

    # Entferne alle durch den Kandidaten dominierten Punkte.
    kept: list[tuple[tuple[int, int, int, int], str]] = []
    for Ft, sid in archive:
        if dominates(F_tup, Ft):
            continue
        kept.append((Ft, sid))

    kept.append((F_tup, str(solution_id)))
    archive[:] = kept
    return True


def _random_allowed_shift(*, a: list[list[list[int]]], i: int, t: int, S: int, rng: random.Random) -> int | None:
    allowed = [s for s in range(int(S)) if int(a[int(i)][int(t)][int(s)]) == 1]
    if not allowed:
        return None
    return int(rng.choice(allowed))


def _repair_to_feasible(
    *,
    loaded_case: LoadedCase,
    schedule: ScheduleDense,
    rng: random.Random,
    deadline_monotonic: float,
    max_restarts: int,
) -> ScheduleDense | None:
    """
    Strict-Repair bis Feasibility (nur T_rep veränderbar; T_fix wird eingefroren).

    Umsetzung ist solverfrei und bewusst heuristisch (V3).
    """

    case = loaded_case.case

    I = int(case.dimensions.I)
    T = int(case.dimensions.T)
    S = int(case.dimensions.S)

    weeks_T_w = loaded_case.weeks_T_w
    day_to_week = _compute_day_to_week(weeks_T_w=weeks_T_w, T=T)
    W = int(len(weeks_T_w))

    S_plus = _compute_S_plus(S, case.sets.get("S_plus"))
    S_plus_set = set(S_plus)

    r = case.params["r"]
    a = case.params["a"]
    sigma = case.params["sigma"]
    h = case.params["h"]
    H_max = case.params["H_max"]
    P_set = {(int(p[0]), int(p[1])) for p in case.params["P"]}

    T_fix = {int(t) for t in case.params["T_fix"]}
    T_rep = [int(t) for t in case.params["T_rep"]]
    T_rep_set = set(T_rep)

    def _init_counts_and_hours(sched: ScheduleDense) -> tuple[list[list[int]], list[list[int]]]:
        counts = [[0] * S for _ in range(T + 1)]
        for t in range(1, T + 1):
            for i in range(I):
                counts[t][int(sched[i][t])] += 1

        week_hours = [[0] * W for _ in range(I)]
        for i in range(I):
            for w_idx, w_days in enumerate(weeks_T_w):
                hours = 0
                for t in w_days:
                    s = int(sched[i][int(t)])
                    if s in S_plus_set:
                        hours += int(h[s])
                week_hours[i][int(w_idx)] = int(hours)
        return counts, week_hours

    def _set_shift(
        *,
        sched: ScheduleDense,
        counts: list[list[int]],
        week_hours: list[list[int]],
        i: int,
        t: int,
        s_new: int,
    ) -> None:
        s_old = int(sched[int(i)][int(t)])
        if s_old == int(s_new):
            return
        sched[int(i)][int(t)] = int(s_new)
        counts[int(t)][int(s_old)] -= 1
        counts[int(t)][int(s_new)] += 1

        w_idx = int(day_to_week[int(t)])
        if w_idx >= 0:
            if s_old in S_plus_set:
                week_hours[int(i)][w_idx] -= int(h[s_old])
            if int(s_new) in S_plus_set:
                week_hours[int(i)][w_idx] += int(h[int(s_new)])

    def _can_assign(*, sched: ScheduleDense, week_hours: list[list[int]], i: int, t: int, s_new: int) -> bool:
        if int(t) not in T_rep_set:
            return False
        if int(a[int(i)][int(t)][int(s_new)]) != 1:
            return False

        s_old = int(sched[int(i)][int(t)])
        w_idx = int(day_to_week[int(t)])
        if w_idx >= 0:
            limit = int(H_max[int(i)])
            delta = int(h[int(s_new)]) - int(h[int(s_old)])
            if int(week_hours[int(i)][w_idx]) + int(delta) > int(limit):
                return False

        prev = int(sched[int(i)][int(t - 1)]) if int(t) > 1 else 0
        nxt = int(sched[int(i)][int(t + 1)]) if int(t) < int(T) else 0
        if (int(prev), int(s_new)) in P_set:
            return False
        if (int(s_new), int(nxt)) in P_set:
            return False

        return True

    def _fix_forbidden_sequences(*, sched: ScheduleDense, counts: list[list[int]], week_hours: list[list[int]]) -> None:
        for i in range(I):
            for t in range(1, T):
                s = int(sched[i][t])
                sp = int(sched[i][t + 1])
                if (s, sp) not in P_set:
                    continue
                # Bevorzugt: t+1 auf frei setzen (falls im Repair-Horizont), sonst t.
                if int(t + 1) in T_rep_set:
                    _set_shift(sched=sched, counts=counts, week_hours=week_hours, i=i, t=t + 1, s_new=0)
                elif int(t) in T_rep_set:
                    _set_shift(sched=sched, counts=counts, week_hours=week_hours, i=i, t=t, s_new=0)

    def _fix_week_hours(*, sched: ScheduleDense, counts: list[list[int]], week_hours: list[list[int]]) -> None:
        for i in range(I):
            limit = int(H_max[i])
            for w_idx, w_days in enumerate(weeks_T_w):
                while int(week_hours[i][int(w_idx)]) > int(limit) and time.monotonic() < float(deadline_monotonic):
                    # Setze einen Arbeitstag in T_rep auf frei (randomisiert, deterministisch über rng).
                    candidates = [int(t) for t in w_days if int(t) in T_rep_set and int(sched[i][int(t)]) in S_plus_set]
                    if not candidates:
                        break
                    t = int(rng.choice(candidates))
                    _set_shift(sched=sched, counts=counts, week_hours=week_hours, i=i, t=t, s_new=0)

    c = case.params["c"]

    def _fill_demand(*, sched: ScheduleDense, counts: list[list[int]], week_hours: list[list[int]]) -> bool:
        def _all_ok() -> bool:
            for tt in range(1, T + 1):
                for ss in S_plus:
                    if int(counts[int(tt)][int(ss)]) < int(r[int(tt)][int(ss)]):
                        return False
            return True

        # Mehrere Pässe, da "unsichere" Moves (safe_move=1) neue Defizite erzeugen können.
        for _pass in range(3):
            changed_any = False

            for t in range(1, T + 1):
                for s in S_plus:
                    required = int(r[t][int(s)])
                    while int(counts[t][int(s)]) < int(required) and time.monotonic() < float(deadline_monotonic):
                        if int(t) in T_fix:
                            return False  # in Fix-Zeitraum kann nicht repariert werden

                        best0_i: int | None = None
                        best0_score: tuple[int, int, float] | None = None
                        best1_i: int | None = None
                        best1_score: tuple[int, int, float] | None = None

                        for i in range(I):
                            if int(sched[i][int(t)]) == int(s):
                                continue
                            if not _can_assign(sched=sched, week_hours=week_hours, i=i, t=int(t), s_new=int(s)):
                                continue

                            s_old = int(sched[i][int(t)])
                            from_free = 0 if s_old == 0 else 1
                            pref = int(c[i][int(t)][int(s)])
                            score = (from_free, pref, float(rng.random()))

                            safe_move = 0
                            if s_old in S_plus_set and int(counts[t][int(s_old)]) - 1 < int(r[t][int(s_old)]):
                                safe_move = 1

                            if safe_move == 0:
                                if best0_score is None or score < best0_score:
                                    best0_score = score
                                    best0_i = int(i)
                            else:
                                if best1_score is None or score < best1_score:
                                    best1_score = score
                                    best1_i = int(i)

                        best_i = best0_i if best0_i is not None else best1_i
                        if best_i is None:
                            return False

                        _set_shift(sched=sched, counts=counts, week_hours=week_hours, i=int(best_i), t=int(t), s_new=int(s))
                        changed_any = True

            if _all_ok():
                return True
            if not changed_any:
                break

        return _all_ok()

    base = _copy_schedule(schedule)

    for attempt in range(int(max_restarts)):
        if time.monotonic() >= float(deadline_monotonic):
            return None

        sched = _copy_schedule(base) if attempt == 0 else _copy_schedule(base)
        counts, week_hours = _init_counts_and_hours(sched)

        # (A) Fixiere T_fix strikt auf σ.
        for t in T_fix:
            for i in range(I):
                s_ref = int(sigma[i][int(t)])
                if int(sched[i][int(t)]) != int(s_ref):
                    _set_shift(sched=sched, counts=counts, week_hours=week_hours, i=i, t=int(t), s_new=int(s_ref))

        # (B) Availability: setze ungültige Einsätze in T_rep deterministisch auf frei.
        for i in range(I):
            for t in T_rep:
                s_cur = int(sched[i][int(t)])
                if int(a[i][int(t)][int(s_cur)]) != 1:
                    _set_shift(sched=sched, counts=counts, week_hours=week_hours, i=i, t=int(t), s_new=0)

        # (C) Bei Restart>0: leichte Randomisierung (nur T_rep), um Sackgassen zu verlassen.
        if attempt > 0:
            shake_steps = min(10, max(2, int(round(0.001 * I * max(1, len(T_rep))))))
            for _ in range(int(shake_steps)):
                if time.monotonic() >= float(deadline_monotonic):
                    break
                i = int(rng.randrange(I))
                t = int(rng.choice(T_rep))
                s_new = _random_allowed_shift(a=a, i=i, t=t, S=S, rng=rng)
                if s_new is None:
                    continue
                _set_shift(sched=sched, counts=counts, week_hours=week_hours, i=i, t=t, s_new=int(s_new))

        # (D) Folge-/Wochenstunden-Konflikte lösen (setzt auf frei; Demand wird später geschlossen).
        _fix_forbidden_sequences(sched=sched, counts=counts, week_hours=week_hours)
        _fix_week_hours(sched=sched, counts=counts, week_hours=week_hours)

        # (E) Demand schließen.
        if not _fill_demand(sched=sched, counts=counts, week_hours=week_hours):
            continue

        # Final: harter Check als Gate.
        feas = check_feasible(case, sched, weeks_T_w=weeks_T_w)
        if feas.ok:
            return sched

    return None


def _default_perturb_steps(*, I: int, T_rep_len: int) -> int:
    # Default: min(50, max(5, round(0.01*|I|*|T_rep|))).
    return int(min(50, max(5, int(round(0.01 * float(I) * float(max(1, T_rep_len)))))))


@dataclass(frozen=True)
class _V3MoveCtx:
    I: int
    T: int
    S: int
    S_plus: list[int]
    S_plus_set: set[int]
    T_rep: list[int]
    r: list[list[int]]
    a: list[list[list[int]]]
    h: list[int]
    H_max: list[int]
    P_set: set[tuple[int, int]]
    weeks_T_w: list[list[int]]
    day_to_week: list[int]


def _work_hours(s: int, *, S_plus_set: set[int], h: list[int]) -> int:
    return int(h[int(s)]) if int(s) in S_plus_set else 0


def _compute_counts_and_week_hours(
    schedule: ScheduleDense,
    *,
    ctx: _V3MoveCtx,
) -> tuple[list[list[int]], list[list[int]]]:
    counts = [[0] * int(ctx.S) for _ in range(int(ctx.T) + 1)]
    for t in range(1, int(ctx.T) + 1):
        row = counts[int(t)]
        for i in range(int(ctx.I)):
            row[int(schedule[i][int(t)])] += 1

    W = int(len(ctx.weeks_T_w))
    week_hours = [[0] * W for _ in range(int(ctx.I))]
    for i in range(int(ctx.I)):
        for w_idx, w_days in enumerate(ctx.weeks_T_w):
            hours = 0
            for t in w_days:
                hours += _work_hours(int(schedule[int(i)][int(t)]), S_plus_set=ctx.S_plus_set, h=ctx.h)
            week_hours[int(i)][int(w_idx)] = int(hours)

    return counts, week_hours


def _apply_deltas_to_counts_and_week_hours(
    deltas: list[tuple[int, int, int, int]],
    *,
    counts: list[list[int]],
    week_hours: list[list[int]],
    ctx: _V3MoveCtx,
) -> None:
    for i, t, s_old, s_new in deltas:
        counts[int(t)][int(s_old)] -= 1
        counts[int(t)][int(s_new)] += 1
        w_idx = int(ctx.day_to_week[int(t)])
        if w_idx >= 0:
            week_hours[int(i)][int(w_idx)] += int(
                _work_hours(int(s_new), S_plus_set=ctx.S_plus_set, h=ctx.h)
                - _work_hours(int(s_old), S_plus_set=ctx.S_plus_set, h=ctx.h)
            )


class V3VNDVNSProcedure:
    """
    V3 — VND/VNS-Heuristik (Multi-Start).

    Kernidee:
    - Multi-Start (start_0..start_{n-1})
    - pro Start ein ω-Impuls (Z_ω_norm als Suchleitgröße)
    - strict-feasible Repair + VND/VNS (einfach gehalten, solverfrei)
    """

    verfahren = "V3"

    def plan_subruns(self, config_snapshot: dict[str, Any]) -> list[SubrunSpec]:
        # Meta-Container pro Group-Run (Procedure-Objekt wird pro run_case instanziiert).
        self._subrun_meta: dict[str, dict[str, Any]] = {}
        self._group_meta: dict[str, Any] = {}

        explicit = _subruns_from_explicit_list(config_snapshot)
        if explicit is not None:
            return explicit

        # V3-Parameter (werden aus config_snapshot gelesen und in der Procedure gespeichert,
        # ohne sie pro Lösung in solutions.jsonl zu duplizieren).
        self._cfg_start_design = str(config_snapshot.get("start_design", "sigma_then_perturb")).strip() or "sigma_then_perturb"
        self._cfg_neighborhoods = config_snapshot.get(
            "neighborhoods",
            [
                "changeShift",
                "swapShift",
                "assignDeleteShift",
                "changeAssignMissingShift",
            ],
        )
        if not isinstance(self._cfg_neighborhoods, list) or not self._cfg_neighborhoods:
            raise ValueError("V3-Config ungültig: neighborhoods muss ein nicht-leeres Array sein.")
        self._cfg_neighborhoods = [str(x) for x in self._cfg_neighborhoods]

        self._cfg_K_MAX = max(1, int(config_snapshot.get("K_MAX", 4)))
        self._cfg_STAGNATION_LIMIT = max(1, int(config_snapshot.get("STAGNATION_LIMIT", 30)))
        self._cfg_MAX_LOCAL_STEPS = max(1, int(config_snapshot.get("MAX_LOCAL_STEPS", 100)))
        self._cfg_max_tries_per_neighborhood = max(1, int(config_snapshot.get("max_tries_per_neighborhood", 200)))

        perturb_steps = config_snapshot.get("perturb_steps", None)
        if perturb_steps is None:
            self._cfg_perturb_steps = None
        else:
            try:
                self._cfg_perturb_steps = int(perturb_steps)
            except Exception as e:
                raise ValueError("V3-Config ungültig: perturb_steps muss int oder null sein.") from e
            if self._cfg_perturb_steps < 0:
                raise ValueError("V3-Config ungültig: perturb_steps muss >=0 sein (oder null für Default).")

        n_starts = int(config_snapshot.get("n_starts", 3))
        if n_starts < 1:
            n_starts = 1

        omega_set_id = str(config_snapshot.get("omega_set_id", "Omega21")).strip() or "Omega21"
        omega_assignment = str(config_snapshot.get("omega_assignment", "cycle")).strip() or "cycle"
        acceptance_mode = str(config_snapshot.get("acceptance_mode", "Z_omega_norm_strict")).strip() or "Z_omega_norm_strict"
        yield_mode = str(config_snapshot.get("yield_mode", "nd_archive_only")).strip() or "nd_archive_only"
        yield_period_raw = config_snapshot.get("yield_period", 1)
        try:
            yield_period = int(yield_period_raw)
        except Exception as e:
            raise ValueError("V3-Config ungültig: yield_period muss eine ganze Zahl >= 1 sein.") from e
        if yield_period < 1:
            raise ValueError("V3-Config ungültig: yield_period muss >= 1 sein.")

        omegas_raw = config_snapshot.get("omegas", None)
        if omegas_raw is None:
            if omega_set_id != "Omega21":
                raise ValueError("V3-Config ungültig: omega_set_id unbekannt (erwartet: Omega21 oder omegas-Liste).")
            omegas = _omega21()
        else:
            if not isinstance(omegas_raw, list) or not omegas_raw:
                raise ValueError("V3-Config ungültig: omegas muss ein nicht-leeres Array sein (oder null für Default).")
            omegas = omegas_raw

        omegas_norm = [_validate_and_normalize_omega(w) for w in omegas]
        if not omegas_norm:
            raise ValueError("V3-Config ungültig: omegas ist leer.")

        if omega_assignment != "cycle":
            raise ValueError("V3-Config ungültig: omega_assignment aktuell nur cycle unterstützt.")

        subruns: list[SubrunSpec] = []
        for k in range(n_starts):
            omega_idx = int(k) % int(len(omegas_norm))
            subruns.append(
                SubrunSpec(
                    subrun_id=f"start_{k}",
                    params={
                        "start_index": int(k),
                        "omega_set_id": str(omega_set_id),
                        "omega_assignment": str(omega_assignment),
                        "omega_set_index": int(omega_idx),
                        "omega": list(omegas_norm[omega_idx]),
                        "acceptance_mode": str(acceptance_mode),
                        "yield_mode": str(yield_mode),
                        "yield_period": int(yield_period),
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
        case = loaded_case.case
        weeks_T_w = loaded_case.weeks_T_w

        now = time.monotonic()
        if now >= float(deadline_monotonic):
            self._subrun_meta[str(subrun.subrun_id)] = {
                "termination_reason": "time_limit",
                "incumbent_found": False,
            }
            return

        I = int(case.dimensions.I)
        T = int(case.dimensions.T)
        S = int(case.dimensions.S)

        sigma = case.params["sigma"]
        a = case.params["a"]

        start_index = int(subrun.params.get("start_index", 0))
        omega = _validate_and_normalize_omega(subrun.params.get("omega", [0.25, 0.25, 0.25, 0.25]))
        acceptance_mode = str(subrun.params.get("acceptance_mode", "Z_omega_norm_strict")).strip() or "Z_omega_norm_strict"
        yield_mode = str(subrun.params.get("yield_mode", "nd_archive_only")).strip() or "nd_archive_only"
        yield_period = int(subrun.params.get("yield_period", 1))
        if yield_period < 1:
            yield_period = 1

        if acceptance_mode != "Z_omega_norm_strict":
            raise ValueError("V3: acceptance_mode aktuell nur 'Z_omega_norm_strict' unterstützt.")
        if yield_mode not in {"nd_archive_only", "incumbent_improvements"}:
            raise ValueError("V3: yield_mode ungültig (erwartet: nd_archive_only|incumbent_improvements).")

        rng_init = random.Random(int(derive_seed(int(seed_subrun), "init")))
        rng_pert = random.Random(int(derive_seed(int(seed_subrun), "perturb")))
        rng_move = random.Random(int(derive_seed(int(seed_subrun), "move")))
        rng_shake = random.Random(int(derive_seed(int(seed_subrun), "shake")))

        UB = _compute_ub_scales(loaded_case=loaded_case)
        # Group-Meta (einmal pro Case/Verfahren sinnvoll).
        self._group_meta.setdefault("Z_omega_norm_UB", dict(UB))

        # 1) Startlösung: σ, ggf. Perturbation in T_rep, dann strict Repair.
        schedule = _copy_schedule(sigma)
        T_rep = [int(t) for t in case.params["T_rep"]]
        start_design = str(getattr(self, "_cfg_start_design", "sigma_then_perturb"))
        if start_design == "sigma_then_perturb" and int(start_index) >= 1:
            steps = (
                int(self._cfg_perturb_steps)
                if getattr(self, "_cfg_perturb_steps", None) is not None
                else _default_perturb_steps(I=I, T_rep_len=len(T_rep))
            )
            for _ in range(int(steps)):
                if time.monotonic() >= float(deadline_monotonic):
                    break
                i = int(rng_pert.randrange(I))
                t = int(rng_pert.choice(T_rep))
                s_new = _random_allowed_shift(a=a, i=i, t=t, S=S, rng=rng_pert)
                if s_new is None:
                    continue
                schedule[i][int(t)] = int(s_new)

        # Repair bis Feasibility.
        repaired = _repair_to_feasible(
            loaded_case=loaded_case,
            schedule=schedule,
            rng=rng_init,
            deadline_monotonic=float(deadline_monotonic),
            max_restarts=5,
        )
        if repaired is None:
            self._subrun_meta[str(subrun.subrun_id)] = {
                "termination_reason": "no_feasible_found",
                "incumbent_found": False,
                "omega": list(omega),
                "start_index": int(start_index),
            }
            return

        incumbent = repaired
        incumbent_F = evaluate_F(case, incumbent, weeks_T_w=weeks_T_w)
        incumbent_Z = _Z_omega_norm(F=incumbent_F, omega=omega, UB=UB)
        incumbent_id = compute_solution_id(incumbent)

        S_plus = _compute_S_plus(S, case.sets.get("S_plus"))
        ctx = _V3MoveCtx(
            I=int(I),
            T=int(T),
            S=int(S),
            S_plus=list(S_plus),
            S_plus_set=set(int(s) for s in S_plus),
            T_rep=list(T_rep),
            r=case.params["r"],
            a=case.params["a"],
            h=case.params["h"],
            H_max=case.params["H_max"],
            P_set={(int(p[0]), int(p[1])) for p in case.params["P"]},
            weeks_T_w=list(weeks_T_w),
            day_to_week=_compute_day_to_week(weeks_T_w=weeks_T_w, T=T),
        )
        counts, week_hours = _compute_counts_and_week_hours(incumbent, ctx=ctx)

        nd_archive: list[tuple[tuple[int, int, int, int], str]] = []
        F_tup = incumbent_F.as_tuple()
        _archive_add_if_nd(nd_archive, F_tup=F_tup, solution_id=incumbent_id)

        # Yield 1. zulässige Lösung (TTFF).
        yield _copy_schedule(incumbent)
        yielded_solution_ids: set[str] = {str(incumbent_id)}

        # 2) VND/VNS Suche
        best_Z = float(incumbent_Z)
        best_F = incumbent_F
        best_id = str(incumbent_id)
        best_schedule = incumbent

        local_steps = 0
        accepted_improvements = 0
        stagnant = 0
        shake_k = 1

        neighborhoods = list(getattr(self, "_cfg_neighborhoods", []))
        K_MAX = int(getattr(self, "_cfg_K_MAX", 4))
        STAGNATION_LIMIT = int(getattr(self, "_cfg_STAGNATION_LIMIT", 30))
        MAX_LOCAL_STEPS = int(getattr(self, "_cfg_MAX_LOCAL_STEPS", 100))
        max_tries = int(getattr(self, "_cfg_max_tries_per_neighborhood", 200))

        while time.monotonic() < float(deadline_monotonic) and int(local_steps) < int(MAX_LOCAL_STEPS):
            improved = False

            for neigh in neighborhoods:
                if time.monotonic() >= float(deadline_monotonic):
                    break

                neigh_name = str(neigh)
                for _ in range(int(max_tries)):
                    if time.monotonic() >= float(deadline_monotonic):
                        break
                    sampled = self._sample_neighbor(
                        neigh=neigh_name,
                        current=incumbent,
                        counts=counts,
                        week_hours=week_hours,
                        ctx=ctx,
                        loaded_case=loaded_case,
                        rng=rng_move,
                        deadline_monotonic=float(deadline_monotonic),
                    )
                    if sampled is None:
                        continue
                    cand, deltas = sampled
                    cand_F = evaluate_F(case, cand, weeks_T_w=weeks_T_w)
                    cand_Z = _Z_omega_norm(F=cand_F, omega=omega, UB=UB)

                    if float(cand_Z) < float(incumbent_Z) - 1e-12:
                        # Sicherheitsnetz: finale harte Prüfung (inkrementelle Prüfungen sind bewusst minimal).
                        feas = check_feasible(case, cand, weeks_T_w=weeks_T_w)
                        if not feas.ok:
                            continue
                        incumbent = cand
                        _apply_deltas_to_counts_and_week_hours(deltas, counts=counts, week_hours=week_hours, ctx=ctx)
                        incumbent_F = cand_F
                        incumbent_Z = float(cand_Z)
                        accepted_improvements += 1

                        cand_id = compute_solution_id(cand)
                        added_nd = _archive_add_if_nd(nd_archive, F_tup=cand_F.as_tuple(), solution_id=cand_id)
                        if yield_mode == "nd_archive_only":
                            if added_nd:
                                yield _copy_schedule(cand)
                        else:
                            if accepted_improvements == 1 or (int(accepted_improvements) % int(yield_period) == 0):
                                if str(cand_id) not in yielded_solution_ids:
                                    yielded_solution_ids.add(str(cand_id))
                                    yield _copy_schedule(cand)

                        if float(incumbent_Z) < float(best_Z) - 1e-12:
                            best_Z = float(incumbent_Z)
                            best_F = incumbent_F
                            best_id = str(cand_id)
                            best_schedule = incumbent

                        improved = True
                        stagnant = 0
                        shake_k = 1
                        break

                if improved:
                    break  # VND: restart Neighborhood-Reihenfolge

            local_steps += 1
            if improved:
                continue

            stagnant += 1
            if int(stagnant) >= int(STAGNATION_LIMIT):
                shaken = self._shake(
                    current=incumbent,
                    loaded_case=loaded_case,
                    rng=rng_shake,
                    k=int(shake_k),
                    deadline_monotonic=float(deadline_monotonic),
                )
                stagnant = 0
                shake_k = int(min(int(K_MAX), int(shake_k) + 1))
                if shaken is None:
                    break
                incumbent = shaken
                counts, week_hours = _compute_counts_and_week_hours(incumbent, ctx=ctx)
                incumbent_F = evaluate_F(case, incumbent, weeks_T_w=weeks_T_w)
                incumbent_Z = _Z_omega_norm(F=incumbent_F, omega=omega, UB=UB)

                cand_id = compute_solution_id(incumbent)
                added_nd = _archive_add_if_nd(nd_archive, F_tup=incumbent_F.as_tuple(), solution_id=cand_id)
                if yield_mode == "nd_archive_only":
                    if added_nd:
                        yield _copy_schedule(incumbent)

                if float(incumbent_Z) < float(best_Z) - 1e-12:
                    best_Z = float(incumbent_Z)
                    best_F = incumbent_F
                    best_id = str(cand_id)
                    best_schedule = incumbent

        # Für yield_mode=incumbent_improvements: bestes gefundenes Schedule sicherstellen.
        if yield_mode == "incumbent_improvements":
            if time.monotonic() < float(deadline_monotonic) and str(best_id) not in yielded_solution_ids:
                yielded_solution_ids.add(str(best_id))
                yield _copy_schedule(best_schedule)

        self._subrun_meta[str(subrun.subrun_id)] = {
            "omega": list(omega),
            "start_index": int(start_index),
            "best_solution_id": str(best_id),
            "best_Z_omega_norm": float(best_Z),
            "best_F": best_F.to_dict(),
            "nd_archive_size": int(len(nd_archive)),
            "termination_reason": "budget_or_steps",
            "incumbent_found": True,
        }

    def _sample_neighbor(
        self,
        *,
        neigh: str,
        current: ScheduleDense,
        counts: list[list[int]],
        week_hours: list[list[int]],
        ctx: _V3MoveCtx,
        loaded_case: LoadedCase,
        rng: random.Random,
        deadline_monotonic: float,
    ) -> tuple[ScheduleDense, list[tuple[int, int, int, int]]] | None:
        if time.monotonic() >= float(deadline_monotonic):
            return None

        neigh = str(neigh)

        def _check_forbidden(*, i: int, t: int, s_new: int) -> bool:
            prev = int(current[int(i)][int(t - 1)]) if int(t) > 1 else 0
            nxt = int(current[int(i)][int(t + 1)]) if int(t) < int(ctx.T) else 0
            if (int(prev), int(s_new)) in ctx.P_set:
                return False
            if (int(s_new), int(nxt)) in ctx.P_set:
                return False
            return True

        def _check_week_hours(*, i: int, t: int, s_old: int, s_new: int) -> bool:
            w_idx = int(ctx.day_to_week[int(t)])
            if w_idx < 0:
                return True
            new_hours = int(week_hours[int(i)][int(w_idx)]) + int(
                _work_hours(int(s_new), S_plus_set=ctx.S_plus_set, h=ctx.h)
                - _work_hours(int(s_old), S_plus_set=ctx.S_plus_set, h=ctx.h)
            )
            return int(new_hours) <= int(ctx.H_max[int(i)])

        def _materialize(deltas: list[tuple[int, int, int, int]]) -> ScheduleDense:
            cand = list(current)
            touched: set[int] = set()
            for i, t, _s_old, s_new in deltas:
                if int(i) not in touched:
                    cand[int(i)] = list(cand[int(i)])
                    touched.add(int(i))
                cand[int(i)][int(t)] = int(s_new)
            return cand  # type: ignore[return-value]

        if neigh == "changeShift":
            i = int(rng.randrange(int(ctx.I)))
            t = int(rng.choice(list(ctx.T_rep)))
            s_old = int(current[int(i)][int(t)])

            allowed = [s for s in range(int(ctx.S)) if int(ctx.a[int(i)][int(t)][int(s)]) == 1 and int(s) != int(s_old)]
            if not allowed:
                return None
            s_new = int(rng.choice(allowed))

            # Demand darf nicht unter Mindestbedarf fallen (nur relevant, wenn ein Arbeitsshift "verlassen" wird).
            if int(s_old) in ctx.S_plus_set:
                if int(counts[int(t)][int(s_old)]) - 1 < int(ctx.r[int(t)][int(s_old)]):
                    return None

            if not _check_forbidden(i=i, t=int(t), s_new=int(s_new)):
                return None
            if not _check_week_hours(i=i, t=int(t), s_old=int(s_old), s_new=int(s_new)):
                return None

            deltas = [(int(i), int(t), int(s_old), int(s_new))]
            return _materialize(deltas), deltas

        if neigh == "swapShift":
            t = int(rng.choice(list(ctx.T_rep)))
            i1 = int(rng.randrange(int(ctx.I)))
            i2 = int(rng.randrange(int(ctx.I)))
            if int(i1) == int(i2):
                return None

            s1 = int(current[int(i1)][int(t)])
            s2 = int(current[int(i2)][int(t)])
            if int(s1) == int(s2):
                return None

            if int(ctx.a[int(i1)][int(t)][int(s2)]) != 1:
                return None
            if int(ctx.a[int(i2)][int(t)][int(s1)]) != 1:
                return None

            if not _check_forbidden(i=int(i1), t=int(t), s_new=int(s2)):
                return None
            if not _check_forbidden(i=int(i2), t=int(t), s_new=int(s1)):
                return None

            if not _check_week_hours(i=int(i1), t=int(t), s_old=int(s1), s_new=int(s2)):
                return None
            if not _check_week_hours(i=int(i2), t=int(t), s_old=int(s2), s_new=int(s1)):
                return None

            deltas = [(int(i1), int(t), int(s1), int(s2)), (int(i2), int(t), int(s2), int(s1))]
            return _materialize(deltas), deltas

        if neigh == "assignDeleteShift":
            # Reassign/Move: verschiebe eine Schicht von Person i auf eine freie Person j (Tag fix, nur T_rep).
            t = int(rng.choice(list(ctx.T_rep)))
            i = int(rng.randrange(int(ctx.I)))
            s_old = int(current[int(i)][int(t)])
            if int(s_old) == 0:
                return None

            # Suche eine Person j, die an t frei ist.
            j: int | None = None
            for _ in range(30):
                cand_j = int(rng.randrange(int(ctx.I)))
                if int(cand_j) == int(i):
                    continue
                if int(current[int(cand_j)][int(t)]) != 0:
                    continue
                if int(ctx.a[int(cand_j)][int(t)][int(s_old)]) != 1:
                    continue
                j = int(cand_j)
                break
            if j is None:
                return None

            # Prüfungen für i -> frei, j -> s_old
            if not _check_forbidden(i=int(j), t=int(t), s_new=int(s_old)):
                return None
            if not _check_week_hours(i=int(j), t=int(t), s_old=0, s_new=int(s_old)):
                return None

            # i wird frei: forbidden/week-hours i werden i.d.R. nicht schlechter, aber prüfen wir deterministisch.
            if not _check_forbidden(i=int(i), t=int(t), s_new=0):
                return None
            if not _check_week_hours(i=int(i), t=int(t), s_old=int(s_old), s_new=0):
                return None

            deltas = [(int(i), int(t), int(s_old), 0), (int(j), int(t), 0, int(s_old))]
            return _materialize(deltas), deltas

        if neigh == "changeAssignMissingShift":
            # Kombi-Nachbarschaft: i wechselt Schicht, und eine freie Person j übernimmt i's alte Schicht (Coverage bleibt erhalten).
            t = int(rng.choice(list(ctx.T_rep)))
            i = int(rng.randrange(int(ctx.I)))
            s_old = int(current[int(i)][int(t)])
            if int(s_old) == 0:
                return None

            allowed = [s for s in range(int(ctx.S)) if int(ctx.a[int(i)][int(t)][int(s)]) == 1 and int(s) != int(s_old)]
            if not allowed:
                return None
            s_new = int(rng.choice(allowed))

            # Suche freie Person j, die s_old übernehmen kann.
            j: int | None = None
            for _ in range(40):
                cand_j = int(rng.randrange(int(ctx.I)))
                if int(cand_j) == int(i):
                    continue
                if int(current[int(cand_j)][int(t)]) != 0:
                    continue
                if int(ctx.a[int(cand_j)][int(t)][int(s_old)]) != 1:
                    continue
                j = int(cand_j)
                break
            if j is None:
                return None

            # Prüfungen: i -> s_new, j -> s_old
            if not _check_forbidden(i=int(i), t=int(t), s_new=int(s_new)):
                return None
            if not _check_forbidden(i=int(j), t=int(t), s_new=int(s_old)):
                return None

            if not _check_week_hours(i=int(i), t=int(t), s_old=int(s_old), s_new=int(s_new)):
                return None
            if not _check_week_hours(i=int(j), t=int(t), s_old=0, s_new=int(s_old)):
                return None

            deltas = [(int(i), int(t), int(s_old), int(s_new)), (int(j), int(t), 0, int(s_old))]
            return _materialize(deltas), deltas

        if neigh == "assignMissingShift":
            # Nur sinnvoll, wenn aktuelle Lösung Coverage-Verletzungen hat (sollte im strict-feasible Loop selten sein).
            deficits: list[tuple[int, int]] = []
            for t in ctx.T_rep:
                for s in ctx.S_plus:
                    if int(counts[int(t)][int(s)]) < int(ctx.r[int(t)][int(s)]):
                        deficits.append((int(t), int(s)))
            if not deficits:
                return None

            t, s = deficits[int(rng.randrange(len(deficits)))]
            # Suche freie Person i, die s übernehmen kann.
            i: int | None = None
            for _ in range(60):
                cand_i = int(rng.randrange(int(ctx.I)))
                if int(current[int(cand_i)][int(t)]) != 0:
                    continue
                if int(ctx.a[int(cand_i)][int(t)][int(s)]) != 1:
                    continue
                if not _check_forbidden(i=int(cand_i), t=int(t), s_new=int(s)):
                    continue
                if not _check_week_hours(i=int(cand_i), t=int(t), s_old=0, s_new=int(s)):
                    continue
                i = int(cand_i)
                break
            if i is None:
                return None

            deltas = [(int(i), int(t), 0, int(s))]
            return _materialize(deltas), deltas

        raise ValueError(f"V3: Unbekannte Neighborhood: {neigh!r}")

    def _shake(
        self,
        *,
        current: ScheduleDense,
        loaded_case: LoadedCase,
        rng: random.Random,
        k: int,
        deadline_monotonic: float,
    ) -> ScheduleDense | None:
        case = loaded_case.case
        I = int(case.dimensions.I)
        T = int(case.dimensions.T)
        S = int(case.dimensions.S)

        a = case.params["a"]
        T_rep = [int(t) for t in case.params["T_rep"]]

        cand = _copy_schedule(current)
        k_eff = max(1, int(k))
        for _ in range(k_eff):
            if time.monotonic() >= float(deadline_monotonic):
                break
            i = int(rng.randrange(I))
            t = int(rng.choice(T_rep))
            s_new = _random_allowed_shift(a=a, i=i, t=t, S=S, rng=rng)
            if s_new is None:
                continue
            cand[i][t] = int(s_new)

        return _repair_to_feasible(
            loaded_case=loaded_case,
            schedule=cand,
            rng=rng,
            deadline_monotonic=float(deadline_monotonic),
            max_restarts=5,
        )

    def get_subrun_meta(self, subrun_id: str) -> dict[str, Any] | None:
        return self._subrun_meta.get(str(subrun_id))

    def get_group_meta(self) -> dict[str, Any] | None:
        return dict(getattr(self, "_group_meta", {})) or None


__all__ = ["V3VNDVNSProcedure"]
