from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any, Iterable

from etappe03_evaluation.io import (
    CaseMeta,
    LoadedData,
    SolutionRecord,
    RunGroupRecord,
    make_is_id,
)
from etappe03_evaluation.pareto import Point, dedupe_points, non_dominated, dominates


METHODS_DEFAULT_ORDER: tuple[str, str, str, str] = ("V1", "V2", "V3", "V4")


def _methods_sorted(methods: Iterable[str]) -> list[str]:
    ms = {m for m in (str(x).strip().upper() for x in methods) if m}
    ordered = [m for m in METHODS_DEFAULT_ORDER if m in ms]
    ordered.extend(sorted(ms - set(ordered)))
    return ordered


def _quantile(values: list[float], q: float) -> float | None:
    """
    Deterministische Quantile (linear, wie numpy/pandas Standardidee):
      pos = q*(n-1); interpolation zwischen floor/ceil.
    """

    xs = [float(v) for v in values if v is not None]  # type: ignore[comparison-overlap]
    if not xs:
        return None
    xs.sort()
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _median(values: list[float]) -> float | None:
    return _quantile(values, 0.5)


@dataclass(frozen=True)
class RunSummary:
    feasible_rate: float | None
    runtime_median_seconds: float | None
    ttff_median_seconds: float | None
    n_runs: int
    n_success: int


def summarize_runs(
    run_groups: list[RunGroupRecord],
) -> dict[tuple[str, str, str], RunSummary]:
    """
    Sekundärmetriken gemäß Bachelorarbeit (Kap. 5.3: Metriken & Vergleichsauswertung):
      - Laufzeit (wall-clock)
      - Time-to-first-feasible (TTFF)
      - Feasibility-Rate (Anteil erfolgreicher Runs)

    Erwartetes Log-Schema (Ist-Stand im Projekt):
      - runs.jsonl liefert wall_seconds_used, time_to_first_feasible_seconds und solutions_found pro Run-Group.
      - solutions.jsonl enthält die zulässigen Lösungen (für Pareto-Metriken).
    """

    # Index: (dataset, case, verfahren) -> list of per-run metrics
    per_method_runs: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for rg in run_groups:
        key_m = (rg.dataset_id, rg.case_id, rg.verfahren)
        success = bool((rg.solutions_found or 0) > 0)

        per_method_runs.setdefault(key_m, []).append(
            {
                "success": success,
                "runtime": rg.wall_seconds_used,
                "ttff": rg.time_to_first_feasible_seconds,
            }
        )

    out: dict[tuple[str, str, str], RunSummary] = {}
    for key_m, runs in per_method_runs.items():
        n_runs = len(runs)
        n_success = sum(1 for r in runs if r.get("success"))
        feasible_rate = (n_success / n_runs) if n_runs > 0 else None

        runtimes = [r["runtime"] for r in runs if r.get("runtime") is not None]
        ttffs = [r["ttff"] for r in runs if r.get("ttff") is not None and r.get("success")]

        out[key_m] = RunSummary(
            feasible_rate=feasible_rate,
            runtime_median_seconds=_median(runtimes) if runtimes else None,
            ttff_median_seconds=_median(ttffs) if ttffs else None,
            n_runs=n_runs,
            n_success=n_success,
        )
    return out


def coverage(A: list[Point], B: list[Point], *, eps: float = 1e-9) -> float | None:
    """
    Coverage (Dominance-Indikator) gemäß Bachelorarbeit (Kap. 5.3: Metriken & Vergleichsauswertung):

      C(A,B) = |{ b in B | ∃a in A: a ⪯ b }| / |B|

    mit ⪯ als Dominanzrelation (Minimierung, strikt in ≥1 Komponente besser).

    Randfall |B|=0:
      - mathematisch undefiniert -> None (wird in Aggregation als NA behandelt).
    """

    if not B:
        return None
    if not A:
        return 0.0

    covered = 0
    for b in B:
        if any(dominates(a.F, b.F, eps=eps) for a in A):
            covered += 1
    return covered / float(len(B))


def contribution_inclusive_unique(P_star: list[Point], *, methods: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    """
    Pareto-Contribution gemäß Bachelorarbeit (Kap. 5.3: Metriken & Vergleichsauswertung):
      (a) inclusive: |{p in P* : v ∈ producers(p)}| / |P*|
      (b) unique:    |{p in P* : producers(p) = {v}}| / |P*|
    """

    inc: dict[str, int] = {m: 0 for m in methods}
    uniq: dict[str, int] = {m: 0 for m in methods}

    if not P_star:
        return ({m: 0.0 for m in methods}, {m: 0.0 for m in methods})

    for p in P_star:
        for m in p.producers:
            if m in inc:
                inc[m] += 1
        if len(p.producers) == 1:
            only = next(iter(p.producers))
            if only in uniq:
                uniq[only] += 1

    denom = float(len(P_star))
    inc_f = {m: inc[m] / denom for m in methods}
    uniq_f = {m: uniq[m] / denom for m in methods}
    return inc_f, uniq_f


@dataclass(frozen=True)
class EvaluationResult:
    per_instance_rows: list[dict[str, Any]]
    coverage_rows: list[dict[str, Any]]
    pstar_rows: list[dict[str, Any]]
    winrate_rows: list[dict[str, Any]]
    winrate_is_rows: list[dict[str, Any]]


def _best_under_omega(
    points: list[Point],
    *,
    omega: tuple[float, float, float, float],
    mins: tuple[float, float, float, float],
    maxs: tuple[float, float, float, float],
) -> tuple[float, tuple[float, float, float, float]] | None:
    if not points:
        return None

    def _norm(f: float, mn: float, mx: float) -> float:
        if mx == mn:
            return 0.0
        return (f - mn) / (mx - mn)

    best_z: float | None = None
    best_F: tuple[float, float, float, float] | None = None
    for p in points:
        f = p.F
        f_norm = (
            _norm(f[0], mins[0], maxs[0]),
            _norm(f[1], mins[1], maxs[1]),
            _norm(f[2], mins[2], maxs[2]),
            _norm(f[3], mins[3], maxs[3]),
        )
        z = omega[0] * f_norm[0] + omega[1] * f_norm[1] + omega[2] * f_norm[2] + omega[3] * f_norm[3]
        if best_z is None or z < best_z or (z == best_z and f < (best_F or f)):
            best_z = z
            best_F = f
    if best_z is None or best_F is None:
        return None
    return best_z, best_F


def evaluate(
    data: LoadedData,
    *,
    eps: float = 1e-9,
    methods: Iterable[str] = METHODS_DEFAULT_ORDER,
    export_pstar_points: bool = False,
    set_mode: str = "union",
    replicate_key: str = "auto",
) -> EvaluationResult:
    set_mode_norm = str(set_mode).strip().lower()
    replicate_key_norm = str(replicate_key).strip().lower()
    if set_mode_norm not in {"union", "per_group"}:
        raise ValueError("set_mode muss 'union' oder 'per_group' sein.")
    if replicate_key_norm not in {"auto", "seed_group", "seed_global"}:
        raise ValueError("replicate_key muss 'auto', 'seed_group' oder 'seed_global' sein.")

    if set_mode_norm == "per_group":
        return _evaluate_per_group(
            data,
            eps=eps,
            methods=methods,
            export_pstar_points=export_pstar_points,
            replicate_key=replicate_key_norm,
        )

    methods_list = _methods_sorted(methods)
    run_summary = summarize_runs(data.run_groups)

    # Index solutions by IS+method
    sol_by_is_method: dict[tuple[str, str, str], list[SolutionRecord]] = {}
    for sol in data.solutions:
        sol_by_is_method.setdefault((sol.dataset_id, sol.case_id, sol.verfahren), []).append(sol)

    # IS list deterministisch
    is_keys = sorted({(s.dataset_id, s.case_id) for s in data.solutions} | {(g.dataset_id, g.case_id) for g in data.run_groups})

    per_instance_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    pstar_rows: list[dict[str, Any]] = []

    # Für Win-Rate Heatmap: pro IS Winner berechnen
    winrate_is_rows: list[dict[str, Any]] = []

    for dataset_id, case_id in is_keys:
        meta: CaseMeta | None = data.case_meta.get((dataset_id, case_id))
        size_class = meta.size_class if meta is not None else None
        severity = meta.severity if meta is not None else None
        is_id = make_is_id(dataset_id, case_id)

        # --- Pro Verfahren: feasible Punkte -> dedupe -> ND (A_v) -------------
        A_v: dict[str, list[Point]] = {}
        raw_feasible_counts: dict[str, int] = {}
        deduped_counts: dict[str, int] = {}

        # Für Normalisierung (Win-Rate): min/max über UNION aller FEASIBLE Lösungen aller Verfahren (pro IS)
        all_feasible_F: list[tuple[float, float, float, float]] = []

        for v in methods_list:
            sols = sol_by_is_method.get((dataset_id, case_id, v), [])
            feasible_pts = [Point(F=s.F, producers=frozenset({v})) for s in sols if s.feasible]
            raw_feasible_counts[v] = len(feasible_pts)
            for p in feasible_pts:
                all_feasible_F.append(p.F)

            feasible_pts = dedupe_points(feasible_pts, eps=eps)
            deduped_counts[v] = len(feasible_pts)
            A_v[v] = non_dominated(feasible_pts, eps=eps)

        # --- Empirische Referenzmenge P* -------------------------------------
        union_pts: list[Point] = []
        for v in methods_list:
            union_pts.extend(A_v.get(v, []))
        union_pts = dedupe_points(union_pts, eps=eps)
        P_star = non_dominated(union_pts, eps=eps)

        inc, uniq = contribution_inclusive_unique(P_star, methods=methods_list)

        # --- Coverage Matrix (pro IS) ----------------------------------------
        for a in methods_list:
            for b in methods_list:
                c = coverage(A_v.get(a, []), A_v.get(b, []), eps=eps)
                coverage_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "case_id": case_id,
                        "is_id": is_id,
                        "size_class": size_class,
                        "severity": severity,
                        "method_a": a,
                        "method_b": b,
                        "coverage": c,
                        "n_A": len(A_v.get(a, [])),
                        "n_B": len(A_v.get(b, [])),
                    }
                )

        # --- P*-Punkte (optional) --------------------------------------------
        if export_pstar_points:
            for p in sorted(P_star, key=lambda x: (x.F, ",".join(sorted(x.producers)))):
                pstar_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "case_id": case_id,
                        "is_id": is_id,
                        "f_stab": p.F[0],
                        "f_ot": p.F[1],
                        "f_pref": p.F[2],
                        "f_fair": p.F[3],
                        "producers": ",".join(sorted(p.producers)),
                    }
                )

        # --- Per-IS+Verfahren Metriken ---------------------------------------
        for v in methods_list:
            rs = run_summary.get((dataset_id, case_id, v))
            per_instance_rows.append(
                {
                    "dataset_id": dataset_id,
                    "case_id": case_id,
                    "is_id": is_id,
                    "size_class": size_class,
                    "severity": severity,
                    "method": v,
                    "n_feasible_raw": raw_feasible_counts.get(v, 0),
                    "n_feasible_deduped": deduped_counts.get(v, 0),
                    "nd_size": len(A_v.get(v, [])),
                    "p_star_size": len(P_star),
                    "contrib_inclusive": inc.get(v, 0.0),
                    "contrib_unique": uniq.get(v, 0.0),
                    "runtime_median_seconds": rs.runtime_median_seconds if rs else None,
                    "ttff_median_seconds": rs.ttff_median_seconds if rs else None,
                    "feasibility_rate": rs.feasible_rate if rs else None,
                    "n_runs": rs.n_runs if rs else 0,
                    "n_success": rs.n_success if rs else 0,
                    "I": meta.I if meta else None,
                    "T": meta.T if meta else None,
                    "S": meta.S if meta else None,
                    "seed_instance": meta.seeds_instance if meta else None,
                    "seed_scenario": meta.seeds_scenario if meta else None,
                }
            )

        # --- Win-Rate pro ω (wenn ω vorhanden & feasible-Lösungen existieren) -
        if data.omegas_v1 and all_feasible_F:
            mins = (
                min(f[0] for f in all_feasible_F),
                min(f[1] for f in all_feasible_F),
                min(f[2] for f in all_feasible_F),
                min(f[3] for f in all_feasible_F),
            )
            maxs = (
                max(f[0] for f in all_feasible_F),
                max(f[1] for f in all_feasible_F),
                max(f[2] for f in all_feasible_F),
                max(f[3] for f in all_feasible_F),
            )

            for omega in data.omegas_v1:
                best_by_method: dict[str, tuple[float, tuple[float, float, float, float]]] = {}
                for v in methods_list:
                    best = _best_under_omega(A_v.get(v, []), omega=omega, mins=mins, maxs=maxs)
                    if best is not None:
                        best_by_method[v] = best

                if not best_by_method:
                    winner = None
                    best_z = None
                else:
                    best_z = min(z for z, _ in best_by_method.values())
                    winners = sorted([m for m, (z, _) in best_by_method.items() if z == best_z])
                    winner = winners[0] if winners else None

                winrate_is_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "case_id": case_id,
                        "is_id": is_id,
                        "size_class": size_class,
                        "severity": severity,
                        "omega": omega,
                        "winner": winner,
                        "best_z": best_z,
                    }
                )

    # --- Win-Rate Aggregation ------------------------------------------------
    winrate_rows: list[dict[str, Any]] = []
    if data.omegas_v1 and winrate_is_rows:
        # (omega, method) -> counts
        counts: dict[tuple[tuple[float, float, float, float], str], int] = {}
        totals: dict[tuple[float, float, float, float], int] = {}
        for row in winrate_is_rows:
            omega = row["omega"]
            if not isinstance(omega, tuple):
                continue
            totals[omega] = totals.get(omega, 0) + 1
            winner = row.get("winner")
            if winner:
                counts[(omega, winner)] = counts.get((omega, winner), 0) + 1

        for omega in sorted(totals):
            total = totals[omega]
            for v in methods_list:
                wins = counts.get((omega, v), 0)
                winrate_rows.append(
                    {
                        "omega": omega,
                        "method": v,
                        "wins": wins,
                        "n_is": total,
                        "win_rate": (wins / total) if total > 0 else None,
                    }
                )

    return EvaluationResult(
        per_instance_rows=per_instance_rows,
        coverage_rows=coverage_rows,
        pstar_rows=pstar_rows,
        winrate_rows=winrate_rows,
        winrate_is_rows=winrate_is_rows,
    )


def _evaluate_per_group(
    data: LoadedData,
    *,
    eps: float,
    methods: Iterable[str],
    export_pstar_points: bool,
    replicate_key: str,
) -> EvaluationResult:
    """
    per_group-Modus:
      - Einheiten: (dataset_id, case_id, replicate_id)
      - pro Einheit: A_v je Verfahren nur aus den Run-Groups dieser Replicate
      - pro Einheit: P*, Coverage, Contribution wie im Union-Modus (Definitionen unverändert)

    replicate_id:
      - auto: seed_global, außer seed_group ist je seed_global konsistent
      - seed_group: nutzt seed_group (falls vorhanden)
      - seed_global: nutzt seed_global (falls vorhanden)

    Falls Replicate-IDs fehlen: WARN + Fallback auf union für dieses IS.
    """

    def _warn(msg: str) -> None:
        print(str(msg), file=sys.stderr)

    methods_list = _methods_sorted(methods)

    # Index: (dataset, case) -> run_groups
    run_groups_by_is: dict[tuple[str, str], list[RunGroupRecord]] = {}
    for rg in data.run_groups:
        run_groups_by_is.setdefault((rg.dataset_id, rg.case_id), []).append(rg)

    # Index solutions by (dataset, case, verfahren, group_id)
    sol_by_group: dict[tuple[str, str, str, str], list[SolutionRecord]] = {}
    for sol in data.solutions:
        if not sol.group_id:
            continue
        sol_by_group.setdefault((sol.dataset_id, sol.case_id, sol.verfahren, sol.group_id), []).append(sol)

    # IS list deterministisch
    is_keys = sorted({(s.dataset_id, s.case_id) for s in data.solutions} | {(g.dataset_id, g.case_id) for g in data.run_groups})

    per_instance_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    pstar_rows: list[dict[str, Any]] = []

    # per_group: Win-Rate ist bewusst deaktiviert (würde sonst Replicates vermischen).
    winrate_rows: list[dict[str, Any]] = []
    winrate_is_rows: list[dict[str, Any]] = []

    def _fallback_union_for_is(dataset_id: str, case_id: str, *, reason: str) -> None:
        meta: CaseMeta | None = data.case_meta.get((dataset_id, case_id))
        size_class = meta.size_class if meta is not None else None
        severity = meta.severity if meta is not None else None
        is_id = make_is_id(dataset_id, case_id)

        A_v_fb: dict[str, list[Point]] = {}
        raw_counts_fb: dict[str, int] = {}
        dedup_counts_fb: dict[str, int] = {}

        for v in methods_list:
            sols = [s for s in data.solutions if s.dataset_id == dataset_id and s.case_id == case_id and s.verfahren == v]
            feasible_pts = [Point(F=s.F, producers=frozenset({v})) for s in sols if s.feasible]
            raw_counts_fb[v] = len(feasible_pts)
            feasible_pts = dedupe_points(feasible_pts, eps=eps)
            dedup_counts_fb[v] = len(feasible_pts)
            A_v_fb[v] = non_dominated(feasible_pts, eps=eps)

        union_pts_fb: list[Point] = []
        for v in methods_list:
            union_pts_fb.extend(A_v_fb.get(v, []))
        union_pts_fb = dedupe_points(union_pts_fb, eps=eps)
        P_star_fb = non_dominated(union_pts_fb, eps=eps)
        inc_fb, uniq_fb = contribution_inclusive_unique(P_star_fb, methods=methods_list)

        for a in methods_list:
            for b in methods_list:
                c = coverage(A_v_fb.get(a, []), A_v_fb.get(b, []), eps=eps)
                coverage_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "case_id": case_id,
                        "is_id": is_id,
                        "size_class": size_class,
                        "severity": severity,
                        "replicate_id": None,
                        "replicate_key_used": "union_fallback",
                        "replicate_fallback_reason": reason,
                        "method_a": a,
                        "method_b": b,
                        "coverage": c,
                        "n_A": len(A_v_fb.get(a, [])),
                        "n_B": len(A_v_fb.get(b, [])),
                    }
                )

        if export_pstar_points:
            for p in sorted(P_star_fb, key=lambda x: (x.F, ",".join(sorted(x.producers)))):
                pstar_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "case_id": case_id,
                        "is_id": is_id,
                        "replicate_id": None,
                        "replicate_key_used": "union_fallback",
                        "replicate_fallback_reason": reason,
                        "f_stab": p.F[0],
                        "f_ot": p.F[1],
                        "f_pref": p.F[2],
                        "f_fair": p.F[3],
                        "producers": ",".join(sorted(p.producers)),
                    }
                )

        for v in methods_list:
            per_instance_rows.append(
                {
                    "dataset_id": dataset_id,
                    "case_id": case_id,
                    "is_id": is_id,
                    "size_class": size_class,
                    "severity": severity,
                    "replicate_id": None,
                    "replicate_key_used": "union_fallback",
                    "replicate_fallback_reason": reason,
                    "method": v,
                    "n_feasible_raw": raw_counts_fb.get(v, 0),
                    "n_feasible_deduped": dedup_counts_fb.get(v, 0),
                    "nd_size": len(A_v_fb.get(v, [])),
                    "p_star_size": len(P_star_fb),
                    "contrib_inclusive": inc_fb.get(v, 0.0),
                    "contrib_unique": uniq_fb.get(v, 0.0),
                    "runtime_seconds": None,
                    "ttff_seconds": None,
                    "runtime_median_seconds": None,
                    "ttff_median_seconds": None,
                    "feasibility_rate": None,
                    "n_runs": 0,
                    "n_success": 0,
                    "I": meta.I if meta else None,
                    "T": meta.T if meta else None,
                    "S": meta.S if meta else None,
                    "seed_instance": meta.seeds_instance if meta else None,
                    "seed_scenario": meta.seeds_scenario if meta else None,
                }
            )

    for dataset_id, case_id in is_keys:
        rgs_is = run_groups_by_is.get((dataset_id, case_id), [])
        if not rgs_is:
            _warn(f"[WARN] per_group: Keine runs.jsonl Gruppen für {dataset_id}/{case_id} gefunden -> union-Fallback.")
            _fallback_union_for_is(dataset_id, case_id, reason="no_run_groups")
            continue

        # replicate_key pro IS bestimmen
        if replicate_key == "auto":
            has_seed_global = any(rg.seed_global is not None for rg in rgs_is)
            has_seed_group = any(rg.seed_group is not None for rg in rgs_is)
            if has_seed_global:
                key_used = "seed_global"
                if has_seed_group:
                    # Konsistenzcheck: pro seed_global darf es höchstens einen seed_group geben.
                    consistent = True
                    for sg in sorted({rg.seed_global for rg in rgs_is if rg.seed_global is not None}):
                        seed_groups = {rg.seed_group for rg in rgs_is if rg.seed_global == sg and rg.seed_group is not None}
                        if len(seed_groups) > 1:
                            consistent = False
                            break
                    if consistent:
                        key_used = "seed_group"
            elif has_seed_group:
                key_used = "seed_group"
            else:
                key_used = None
        else:
            key_used = replicate_key

        if key_used not in {"seed_group", "seed_global"}:
            _warn(f"[WARN] per_group: Weder seed_group noch seed_global verfügbar für {dataset_id}/{case_id} -> union-Fallback.")
            _fallback_union_for_is(dataset_id, case_id, reason="no_seed_fields")
            continue

        # Replicate-IDs sammeln
        rep_vals: set[int] = set()
        for rg in rgs_is:
            val = rg.seed_group if key_used == "seed_group" else rg.seed_global
            if val is not None:
                rep_vals.add(int(val))
        if not rep_vals:
            _warn(f"[WARN] per_group: Keine Replicate-IDs (key={key_used}) für {dataset_id}/{case_id} -> union-Fallback.")
            _fallback_union_for_is(dataset_id, case_id, reason=f"no_replicate_ids:{key_used}")
            continue

        # Warnung, falls Lösungen Gruppen referenzieren, die nicht in runs.jsonl existieren
        known_group_ids = {rg.group_id for rg in rgs_is if rg.group_id}
        orphan = [
            s
            for s in data.solutions
            if s.dataset_id == dataset_id and s.case_id == case_id and s.group_id and s.group_id not in known_group_ids
        ]
        if orphan:
            _warn(f"[WARN] per_group: solutions.jsonl enthält group_id ohne passendes runs.jsonl (z.B. {orphan[0].group_id}) für {dataset_id}/{case_id}.")

        meta: CaseMeta | None = data.case_meta.get((dataset_id, case_id))
        size_class = meta.size_class if meta is not None else None
        severity = meta.severity if meta is not None else None
        is_id = make_is_id(dataset_id, case_id)

        for rep in sorted(rep_vals):
            replicate_id = str(rep)

            # Pro Verfahren: run_groups dieser Replicate sammeln
            rgs_by_method: dict[str, list[RunGroupRecord]] = {}
            for rg in rgs_is:
                val = rg.seed_group if key_used == "seed_group" else rg.seed_global
                if val is None or int(val) != int(rep):
                    continue
                rgs_by_method.setdefault(rg.verfahren, []).append(rg)

            A_v: dict[str, list[Point]] = {}
            raw_feasible_counts: dict[str, int] = {}
            deduped_counts: dict[str, int] = {}

            runtime_by_method: dict[str, float | None] = {}
            ttff_by_method: dict[str, float | None] = {}
            feas_by_method: dict[str, float | None] = {}
            n_runs_by_method: dict[str, int] = {}
            n_success_by_method: dict[str, int] = {}

            for v in methods_list:
                rgs_v = rgs_by_method.get(v, [])
                if not rgs_v:
                    A_v[v] = []
                    raw_feasible_counts[v] = 0
                    deduped_counts[v] = 0
                    runtime_by_method[v] = None
                    ttff_by_method[v] = None
                    feas_by_method[v] = None
                    n_runs_by_method[v] = 0
                    n_success_by_method[v] = 0
                    continue

                if len(rgs_v) > 1:
                    rgs_v = sorted(rgs_v, key=lambda x: str(x.group_id))
                    _warn(
                        f"[WARN] per_group: Mehrere Run-Groups für {dataset_id}/{case_id}/{v} replicate_id={replicate_id} (n={len(rgs_v)}) -> Lösungen werden vereinigt, Laufzeit/TTFF per Median."
                    )

                # Lösungen aus allen passenden group_id vereinigen
                sols_v: list[SolutionRecord] = []
                for rg in rgs_v:
                    sols_v.extend(sol_by_group.get((dataset_id, case_id, v, rg.group_id), []))

                feasible_pts = [Point(F=s.F, producers=frozenset({v})) for s in sols_v if s.feasible]
                raw_feasible_counts[v] = len(feasible_pts)
                feasible_pts = dedupe_points(feasible_pts, eps=eps)
                deduped_counts[v] = len(feasible_pts)
                A_v[v] = non_dominated(feasible_pts, eps=eps)

                successes: list[bool] = []
                runtimes: list[float] = []
                ttffs: list[float] = []

                sol_elapsed_feas = [s.elapsed_seconds for s in sols_v if s.feasible and s.elapsed_seconds is not None]
                sol_elapsed_any = [s.elapsed_seconds for s in sols_v if s.elapsed_seconds is not None]

                for rg in rgs_v:
                    n_sol = len([s for s in sols_v if s.feasible and s.group_id == rg.group_id])
                    success = bool((rg.solutions_found or 0) > 0 or n_sol > 0)
                    successes.append(success)

                    rt = rg.wall_seconds_used
                    if (rt is None or rt <= 0) and sol_elapsed_any:
                        rt = float(max(sol_elapsed_any))
                    if rt is not None:
                        runtimes.append(float(rt))

                    t1 = rg.time_to_first_feasible_seconds
                    if (t1 is None or t1 <= 0) and sol_elapsed_feas and success:
                        t1 = float(min(sol_elapsed_feas))
                    if t1 is not None and success:
                        ttffs.append(float(t1))

                n_runs_by_method[v] = len(rgs_v)
                n_success_by_method[v] = sum(1 for x in successes if x)
                feas_by_method[v] = (n_success_by_method[v] / n_runs_by_method[v]) if n_runs_by_method[v] > 0 else None
                runtime_by_method[v] = _median(runtimes) if runtimes else None
                ttff_by_method[v] = _median(ttffs) if ttffs else None

            # P* pro Replicate
            union_pts: list[Point] = []
            for v in methods_list:
                union_pts.extend(A_v.get(v, []))
            union_pts = dedupe_points(union_pts, eps=eps)
            P_star = non_dominated(union_pts, eps=eps)

            inc, uniq = contribution_inclusive_unique(P_star, methods=methods_list)

            # Coverage Matrix pro Replicate
            for a in methods_list:
                for b in methods_list:
                    c = coverage(A_v.get(a, []), A_v.get(b, []), eps=eps)
                    coverage_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "case_id": case_id,
                            "is_id": is_id,
                            "size_class": size_class,
                            "severity": severity,
                            "replicate_id": replicate_id,
                            "replicate_key_used": key_used,
                            "method_a": a,
                            "method_b": b,
                            "coverage": c,
                            "n_A": len(A_v.get(a, [])),
                            "n_B": len(A_v.get(b, [])),
                        }
                    )

            # P*-Punkte (optional)
            if export_pstar_points:
                for p in sorted(P_star, key=lambda x: (x.F, ",".join(sorted(x.producers)))):
                    pstar_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "case_id": case_id,
                            "is_id": is_id,
                            "replicate_id": replicate_id,
                            "replicate_key_used": key_used,
                            "f_stab": p.F[0],
                            "f_ot": p.F[1],
                            "f_pref": p.F[2],
                            "f_fair": p.F[3],
                            "producers": ",".join(sorted(p.producers)),
                        }
                    )

            # Per-IS+Replicate+Verfahren Metriken
            for v in methods_list:
                rt = runtime_by_method.get(v)
                t1 = ttff_by_method.get(v)
                per_instance_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "case_id": case_id,
                        "is_id": is_id,
                        "size_class": size_class,
                        "severity": severity,
                        "replicate_id": replicate_id,
                        "replicate_key_used": key_used,
                        "method": v,
                        "n_feasible_raw": raw_feasible_counts.get(v, 0),
                        "n_feasible_deduped": deduped_counts.get(v, 0),
                        "nd_size": len(A_v.get(v, [])),
                        "p_star_size": len(P_star),
                        "contrib_inclusive": inc.get(v, 0.0),
                        "contrib_unique": uniq.get(v, 0.0),
                        "runtime_seconds": rt,
                        "ttff_seconds": t1,
                        # Format wie in den Union-Exports/Plots:
                        "runtime_median_seconds": rt,
                        "ttff_median_seconds": t1,
                        "feasibility_rate": feas_by_method.get(v),
                        "n_runs": n_runs_by_method.get(v, 0),
                        "n_success": n_success_by_method.get(v, 0),
                        "I": meta.I if meta else None,
                        "T": meta.T if meta else None,
                        "S": meta.S if meta else None,
                        "seed_instance": meta.seeds_instance if meta else None,
                        "seed_scenario": meta.seeds_scenario if meta else None,
                    }
                )

    return EvaluationResult(
        per_instance_rows=per_instance_rows,
        coverage_rows=coverage_rows,
        pstar_rows=pstar_rows,
        winrate_rows=winrate_rows,
        winrate_is_rows=winrate_is_rows,
    )
