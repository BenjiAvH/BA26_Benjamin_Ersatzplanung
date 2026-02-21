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


def _default_perturb_steps(*, I: int, T_rep_len: int) -> int:
    # Kleine, deterministische Default-Heuristik (analog zu V3).
    return int(min(50, max(5, int(round(0.01 * float(I) * float(max(1, T_rep_len)))))))


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

    Hinweis (B/C Adaption): Pato/Moz (2008) arbeiten in UPGH mit Permutations-Encoding+Decoder.
    Hier wird direkt auf der dichten Planrepräsentation gearbeitet und am Ende mit
    `check_feasible` gegated.
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

        # Mehrere Pässe, da "unsichere" Moves neue Defizite erzeugen können.
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

        sched = _copy_schedule(base)
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
                if not T_rep:
                    break
                t = int(rng.choice(T_rep))
                s_new = _random_allowed_shift(a=a, i=i, t=t, S=S, rng=rng)
                if s_new is None:
                    continue
                _set_shift(sched=sched, counts=counts, week_hours=week_hours, i=i, t=int(t), s_new=int(s_new))

        # (D) Folge-/Wochenstunden-Konflikte lösen (setzt auf frei; Demand wird später geschlossen).
        _fix_forbidden_sequences(sched=sched, counts=counts, week_hours=week_hours)
        _fix_week_hours(sched=sched, counts=counts, week_hours=week_hours)

        # (E) Demand schließen.
        if not _fill_demand(sched=sched, counts=counts, week_hours=week_hours):
            continue

        feas = check_feasible(case, sched, weeks_T_w=weeks_T_w)
        if feas.ok:
            return sched

    return None


@dataclass(frozen=True)
class _Individual:
    schedule: ScheduleDense
    F: tuple[int, int, int, int]
    solution_id: str
    is_utopic_anchor: bool = False


def _archive_add_if_nd(archive: list[_Individual], cand: _Individual) -> bool:
    for other in archive:
        if other.solution_id == cand.solution_id:
            return False
        if other.F == cand.F:
            return False
        if dominates(other.F, cand.F):
            return False

    kept: list[_Individual] = []
    for other in archive:
        if dominates(cand.F, other.F):
            continue
        kept.append(other)
    kept.append(cand)
    archive[:] = kept
    return True


def _crowding_distance(items: list[_Individual]) -> dict[str, float]:
    """
    Crowding Distance (NSGA-II) als Diversitätsmaß innerhalb einer Front.

    Adaption (B): in UPGH (Pato/Moz 2008) nicht explizit beschrieben; hier nur genutzt,
    um ein zu großes ND-Archiv deterministisch zu truncaten.
    """

    if not items:
        return {}
    if len(items) <= 2:
        return {it.solution_id: math.inf for it in items}

    distances: dict[str, float] = {it.solution_id: 0.0 for it in items}
    objectives = [
        ("f_stab", 0),
        ("f_ot", 1),
        ("f_pref", 2),
        ("f_fair", 3),
    ]

    for _name, idx in objectives:
        items_sorted = sorted(items, key=lambda it: (int(it.F[idx]), str(it.solution_id)))
        distances[items_sorted[0].solution_id] = math.inf
        distances[items_sorted[-1].solution_id] = math.inf

        min_v = float(items_sorted[0].F[idx])
        max_v = float(items_sorted[-1].F[idx])
        denom = float(max_v - min_v)
        if denom <= 0.0:
            continue

        for k in range(1, len(items_sorted) - 1):
            if math.isinf(distances[items_sorted[k].solution_id]):
                continue
            prev_v = float(items_sorted[k - 1].F[idx])
            next_v = float(items_sorted[k + 1].F[idx])
            distances[items_sorted[k].solution_id] += float(next_v - prev_v) / float(denom)

    # Anchor (falls vorhanden) nie als "crowded" behandeln.
    for it in items:
        if it.is_utopic_anchor:
            distances[it.solution_id] = math.inf

    return distances


def _truncate_archive_by_crowding(archive: list[_Individual], *, max_size: int) -> None:
    if max_size < 1:
        archive.clear()
        return
    if len(archive) <= int(max_size):
        return

    dist = _crowding_distance(archive)

    while len(archive) > int(max_size):
        # Entferne kleinste Distance; bei Gleichstand: größtes solution_id (damit kleine IDs stabil bleiben).
        cand_idx: int | None = None
        cand_d: float | None = None
        cand_sid: str | None = None
        for idx, it in enumerate(archive):
            if it.is_utopic_anchor:
                continue
            d = float(dist.get(it.solution_id, 0.0))
            sid = str(it.solution_id)
            if cand_idx is None:
                cand_idx, cand_d, cand_sid = int(idx), float(d), str(sid)
                continue
            if d < float(cand_d):  # type: ignore[arg-type]
                cand_idx, cand_d, cand_sid = int(idx), float(d), str(sid)
            elif d == float(cand_d) and sid > str(cand_sid):
                cand_idx, cand_d, cand_sid = int(idx), float(d), str(sid)
        if cand_idx is None:
            # Nur Anchor im Archiv? Dann abbrechen.
            break
        archive.pop(int(cand_idx))


def _non_dominated_sort(pop: list[_Individual]) -> list[int]:
    """
    Pareto-Ranking (Front-Index / Rang) via Non-Dominated Sorting (NSGA-II-Standard).

    Rang ist 1-basiert (Front 1 = nicht dominiert).
    """

    n = len(pop)
    if n == 0:
        return []
    if n == 1:
        return [1]

    S: list[list[int]] = [[] for _ in range(n)]
    n_dom = [0] * n

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(pop[p].F, pop[q].F):
                S[p].append(q)
            elif dominates(pop[q].F, pop[p].F):
                n_dom[p] += 1

    rank = [0] * n
    front = [i for i in range(n) if n_dom[i] == 0]
    front.sort(key=lambda i: (pop[i].F, pop[i].solution_id))
    r = 1
    while front:
        next_front: list[int] = []
        for p in front:
            rank[p] = int(r)
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        r += 1
        front = sorted(set(next_front), key=lambda i: (pop[i].F, pop[i].solution_id))

    # Fallback (sollte nicht passieren): alles was keinen Rang bekam -> letzter Rang.
    last_rank = max(rank) if rank else 1
    for i in range(n):
        if rank[i] == 0:
            rank[i] = int(last_rank)
    return rank


def _roulette_select(pop: list[_Individual], fitness: list[float], *, rng: random.Random) -> _Individual:
    if not pop:
        raise ValueError("V4: interne Selektion auf leerer Population.")
    if len(pop) != len(fitness):
        raise ValueError("V4: interne Selektion: pop/fitness Längen passen nicht zusammen.")

    total = float(sum(fitness))
    if not math.isfinite(total) or total <= 0.0:
        return pop[int(rng.randrange(len(pop)))]

    r = float(rng.random()) * total
    acc = 0.0
    for it, w in zip(pop, fitness, strict=True):
        acc += float(w)
        if acc >= r:
            return it
    return pop[-1]


class V4UPGHProcedure:
    """
    V4 — Utopic Pareto Genetic Heuristic (populationsbasiert, Pareto-orientiert).

    Kernidee:
    - Pareto-Ranking + rangbasierte Fitness (≈ 1/rank) + Roulette-Selektion
    - zusätzlich ein „utopic/anchor“-Individuum im Pool
    - Variation (Adaption auf direkte Schedule-Repräsentation): Block-Rekombination + Swap-Mutation
    - feasible-only: jedes Kind wird via strict Repair in eine zulässige Lösung überführt
    - ND-Archiv (approximiert 𝓐_V4); Yields erfolgen nur bei neuen ND-Punkten
    """

    verfahren = "V4"

    def plan_subruns(self, config_snapshot: dict[str, Any]) -> list[SubrunSpec]:
        self._subrun_meta: dict[str, dict[str, Any]] = {}

        # Defaults (werden ggf. über Subrun-Params überschrieben).
        n_runs = int(config_snapshot.get("n_runs", 1))
        if n_runs < 1:
            n_runs = 1

        pop_size = int(config_snapshot.get("pop_size", 100))
        if pop_size < 2:
            pop_size = 2

        pc = float(config_snapshot.get("pc", 0.6))
        pm = float(config_snapshot.get("pm", 0.001))
        if pc < 0.0 or pc > 1.0:
            raise ValueError("V4-Config ungültig: pc muss in [0,1] liegen.")
        if pm < 0.0 or pm > 1.0:
            raise ValueError("V4-Config ungültig: pm muss in [0,1] liegen.")

        max_generations = int(config_snapshot.get("max_generations", 10_000))
        if max_generations < 1:
            max_generations = 1

        archive_max = int(config_snapshot.get("archive_max", 300))
        if archive_max < 1:
            archive_max = 1

        init_perturb_steps_raw = config_snapshot.get("init_perturb_steps", None)
        if init_perturb_steps_raw is None:
            init_perturb_steps = None
        else:
            try:
                init_perturb_steps = int(init_perturb_steps_raw)
            except Exception as e:
                raise ValueError("V4-Config ungültig: init_perturb_steps muss int oder null sein.") from e
            if init_perturb_steps < 0:
                raise ValueError("V4-Config ungültig: init_perturb_steps muss >=0 sein (oder null).")

        max_restarts = int(config_snapshot.get("max_restarts", 4))
        if max_restarts < 1:
            max_restarts = 1

        crossover_mode = str(config_snapshot.get("crossover_mode", "day_block")).strip() or "day_block"
        if crossover_mode not in {"day_block", "nurse_block"}:
            raise ValueError("V4-Config ungültig: crossover_mode muss 'day_block' oder 'nurse_block' sein.")

        # Im Objekt speichern (Procedure wird pro run_case instanziiert).
        self._cfg_pop_size = int(pop_size)
        self._cfg_pc = float(pc)
        self._cfg_pm = float(pm)
        self._cfg_max_generations = int(max_generations)
        self._cfg_archive_max = int(archive_max)
        self._cfg_init_perturb_steps = init_perturb_steps
        self._cfg_max_restarts = int(max_restarts)
        self._cfg_crossover_mode = str(crossover_mode)

        explicit = _subruns_from_explicit_list(config_snapshot)
        if explicit is not None:
            return explicit

        subruns: list[SubrunSpec] = []
        for k in range(n_runs):
            subruns.append(
                SubrunSpec(
                    subrun_id=f"run_{k}",
                    params={
                        "run_index": int(k),
                        "pop_size": int(self._cfg_pop_size),
                        "pc": float(self._cfg_pc),
                        "pm": float(self._cfg_pm),
                        "max_generations": int(self._cfg_max_generations),
                        "archive_max": int(self._cfg_archive_max),
                        "crossover_mode": str(self._cfg_crossover_mode),
                        "representation_type": "direct_schedule_dense",
                        "yield_mode": "nd_archive_only",
                        "feasible_only": True,
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

        I = int(case.dimensions.I)
        T = int(case.dimensions.T)
        S = int(case.dimensions.S)

        sigma: ScheduleDense = case.params["sigma"]
        a: list[list[list[int]]] = case.params["a"]
        T_fix = {int(t) for t in case.params["T_fix"]}
        T_rep = [int(t) for t in case.params["T_rep"]]

        # Seeds (deterministisch, aus seed_subrun abgeleitet).
        rng_init = random.Random(int(derive_seed(int(seed_subrun), "init")))
        rng_select = random.Random(int(derive_seed(int(seed_subrun), "select")))
        rng_var = random.Random(int(derive_seed(int(seed_subrun), "variation")))

        params = dict(getattr(subrun, "params", {}) or {})

        pop_size = int(params.get("pop_size", getattr(self, "_cfg_pop_size", 100)))
        pc = float(params.get("pc", getattr(self, "_cfg_pc", 0.6)))
        pm = float(params.get("pm", getattr(self, "_cfg_pm", 0.001)))
        max_generations = int(params.get("max_generations", getattr(self, "_cfg_max_generations", 10_000)))
        archive_max = int(params.get("archive_max", getattr(self, "_cfg_archive_max", 300)))
        max_restarts = int(params.get("max_restarts", getattr(self, "_cfg_max_restarts", 4)))
        crossover_mode = str(params.get("crossover_mode", getattr(self, "_cfg_crossover_mode", "day_block")))

        init_perturb_steps = params.get("init_perturb_steps", getattr(self, "_cfg_init_perturb_steps", None))
        if init_perturb_steps is None:
            init_perturb_steps = _default_perturb_steps(I=I, T_rep_len=len(T_rep))
        init_perturb_steps = int(init_perturb_steps)

        if pop_size < 2:
            pop_size = 2
        if max_generations < 1:
            max_generations = 1
        if archive_max < 1:
            archive_max = 1
        if max_restarts < 1:
            max_restarts = 1
        if pc < 0.0 or pc > 1.0:
            raise ValueError("V4-Subrun ungültig: pc muss in [0,1] liegen.")
        if pm < 0.0 or pm > 1.0:
            raise ValueError("V4-Subrun ungültig: pm muss in [0,1] liegen.")
        if crossover_mode not in {"day_block", "nurse_block"}:
            raise ValueError("V4-Subrun ungültig: crossover_mode muss 'day_block' oder 'nurse_block' sein.")

        evals_total = 0
        offspring_total = 0
        offspring_feasible = 0
        discarded_infeasible = 0

        def _enforce_fix(sched: ScheduleDense) -> None:
            for t in T_fix:
                for i in range(I):
                    sched[i][int(t)] = int(sigma[i][int(t)])

        # --- 1) Anchor/„utopic“-Individuum: starte bei σ und repaire strikt.
        anchor_start = _copy_schedule(sigma)
        anchor = _repair_to_feasible(
            loaded_case=loaded_case,
            schedule=anchor_start,
            rng=rng_init,
            deadline_monotonic=float(deadline_monotonic),
            max_restarts=max_restarts,
        )
        if anchor is None:
            self._subrun_meta[str(subrun.subrun_id)] = {
                "termination_reason": "no_feasible_anchor",
                "evals_total": 0,
                "archive_size_final": 0,
            }
            return

        anchor_id = str(compute_solution_id(anchor))
        anchor_F = evaluate_F(case, anchor, weeks_T_w=loaded_case.weeks_T_w).as_tuple()
        anchor_ind = _Individual(schedule=anchor, F=anchor_F, solution_id=anchor_id, is_utopic_anchor=True)

        archive: list[_Individual] = []
        population: list[_Individual] = []
        yielded: set[str] = set()
        seen: set[str] = set()

        population.append(anchor_ind)
        _archive_add_if_nd(archive, anchor_ind)
        _truncate_archive_by_crowding(archive, max_size=int(archive_max))
        yielded.add(anchor_ind.solution_id)
        seen.add(anchor_ind.solution_id)
        evals_total += 1

        # Yield 1. zulässige Lösung (TTFF).
        yield _copy_schedule(anchor_ind.schedule)

        # --- 2) Initialpopulation (diversifizieren durch Perturbation in T_rep + Repair).
        while len(population) < int(pop_size) and time.monotonic() < float(deadline_monotonic):
            cand = _copy_schedule(anchor_ind.schedule)
            for _ in range(int(init_perturb_steps)):
                if time.monotonic() >= float(deadline_monotonic):
                    break
                if not T_rep:
                    break
                i = int(rng_init.randrange(I))
                t = int(rng_init.choice(T_rep))
                s_new = _random_allowed_shift(a=a, i=i, t=t, S=S, rng=rng_init)
                if s_new is None:
                    continue
                cand[i][int(t)] = int(s_new)

            _enforce_fix(cand)
            repaired = _repair_to_feasible(
                loaded_case=loaded_case,
                schedule=cand,
                rng=rng_init,
                deadline_monotonic=float(deadline_monotonic),
                max_restarts=max(1, int(max_restarts // 2)),
            )
            if repaired is None:
                discarded_infeasible += 1
                continue

            sid = str(compute_solution_id(repaired))
            if sid in seen:
                continue
            seen.add(sid)

            F = evaluate_F(case, repaired, weeks_T_w=loaded_case.weeks_T_w).as_tuple()
            ind = _Individual(schedule=repaired, F=F, solution_id=sid)
            population.append(ind)
            evals_total += 1

            if _archive_add_if_nd(archive, ind):
                _truncate_archive_by_crowding(archive, max_size=int(archive_max))
                if ind.solution_id not in yielded:
                    yielded.add(ind.solution_id)
                    yield _copy_schedule(ind.schedule)

        # --- 3) Evolution (Zeit ist das primäre Abbruchkriterium, generations ist Safety-Cap).
        generations_done = 0

        def _crossover(parent_a: ScheduleDense, parent_b: ScheduleDense) -> ScheduleDense:
            child = _copy_schedule(parent_a)
            if not T_rep:
                return child

            if crossover_mode == "nurse_block":
                # Kopiere zufällige Teilmenge der Mitarbeitenden komplett aus parent_b.
                k = int(max(1, round(0.2 * float(I))))
                nurses = {int(rng_var.randrange(I)) for _ in range(int(k))}
                for i in nurses:
                    for t in T_rep:
                        child[int(i)][int(t)] = int(parent_b[int(i)][int(t)])
                return child

            # Default: day_block
            a_idx = int(rng_var.randrange(len(T_rep)))
            b_idx = int(rng_var.randrange(len(T_rep)))
            lo = min(a_idx, b_idx)
            hi = max(a_idx, b_idx)
            days = [int(T_rep[k]) for k in range(int(lo), int(hi) + 1)]
            for t in days:
                for i in range(I):
                    child[int(i)][int(t)] = int(parent_b[int(i)][int(t)])
            return child

        def _mutate_swap_day(sched: ScheduleDense, *, rng: random.Random) -> None:
            if not T_rep:
                return
            n_genes = int(I) * int(len(T_rep))
            expected_swaps = float(pm) * float(n_genes)
            n_swaps = int(math.floor(expected_swaps))
            frac = float(expected_swaps - float(n_swaps))
            if rng.random() < frac:
                n_swaps += 1
            n_swaps = int(max(0, n_swaps))
            if n_swaps == 0:
                return

            for _ in range(int(n_swaps)):
                if time.monotonic() >= float(deadline_monotonic):
                    return
                t = int(rng.choice(T_rep))
                i1 = int(rng.randrange(I))
                i2 = int(rng.randrange(I))
                if i1 == i2:
                    continue
                s1 = int(sched[int(i1)][int(t)])
                s2 = int(sched[int(i2)][int(t)])
                if s1 == s2:
                    continue
                # Availability-Schnellcheck (Rest macht Repair).
                if int(a[int(i1)][int(t)][int(s2)]) != 1:
                    continue
                if int(a[int(i2)][int(t)][int(s1)]) != 1:
                    continue
                sched[int(i1)][int(t)] = int(s2)
                sched[int(i2)][int(t)] = int(s1)

        # Um nicht zu "verhungern", limitieren wir Fehlversuche pro Generation.
        max_failed_children = int(pop_size) * 5

        while time.monotonic() < float(deadline_monotonic) and generations_done < int(max_generations):
            generations_done += 1
            if len(population) < 2:
                break

            ranks = _non_dominated_sort(population)
            fitness: list[float] = []
            for it, r in zip(population, ranks, strict=True):
                f = 1.0 / float(max(1, int(r)))
                if it.is_utopic_anchor:
                    f = 1.0
                fitness.append(float(f))

            offspring: list[_Individual] = []
            failed = 0
            while len(offspring) < int(pop_size) - 1 and time.monotonic() < float(deadline_monotonic):
                if failed >= int(max_failed_children):
                    break

                p1 = _roulette_select(population, fitness, rng=rng_select)
                p2 = _roulette_select(population, fitness, rng=rng_select)

                child = _copy_schedule(p1.schedule)
                if rng_var.random() < float(pc):
                    child = _crossover(p1.schedule, p2.schedule)

                _mutate_swap_day(child, rng=rng_var)
                _enforce_fix(child)

                repaired = _repair_to_feasible(
                    loaded_case=loaded_case,
                    schedule=child,
                    rng=rng_var,
                    deadline_monotonic=float(deadline_monotonic),
                    max_restarts=max(1, int(max_restarts // 2)),
                )
                offspring_total += 1
                if repaired is None:
                    discarded_infeasible += 1
                    failed += 1
                    continue

                if not check_feasible(case, repaired, weeks_T_w=loaded_case.weeks_T_w).ok:
                    discarded_infeasible += 1
                    failed += 1
                    continue

                offspring_feasible += 1
                sid = str(compute_solution_id(repaired))
                if sid in seen:
                    continue
                seen.add(sid)

                F = evaluate_F(case, repaired, weeks_T_w=loaded_case.weeks_T_w).as_tuple()
                ind = _Individual(schedule=repaired, F=F, solution_id=sid)
                offspring.append(ind)
                evals_total += 1

                if _archive_add_if_nd(archive, ind):
                    _truncate_archive_by_crowding(archive, max_size=int(archive_max))
                    if ind.solution_id not in yielded:
                        yielded.add(ind.solution_id)
                        yield _copy_schedule(ind.schedule)

            # Nächste Generation: elitistisch Anchor behalten, Rest sind Offspring.
            if offspring:
                population = [anchor_ind] + offspring
            else:
                break

        self._subrun_meta[str(subrun.subrun_id)] = {
            "termination_reason": "deadline" if time.monotonic() >= float(deadline_monotonic) else "max_generations",
            "generations_done": int(generations_done),
            "evals_total": int(evals_total),
            "offspring_total": int(offspring_total),
            "offspring_feasible": int(offspring_feasible),
            "discarded_infeasible": int(discarded_infeasible),
            "archive_size_final": int(len(archive)),
        }

    def get_subrun_meta(self, subrun_id: str) -> dict[str, Any] | None:
        return self._subrun_meta.get(str(subrun_id))


__all__ = ["V4UPGHProcedure"]

