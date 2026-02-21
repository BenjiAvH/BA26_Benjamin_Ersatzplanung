from __future__ import annotations

import argparse
from datetime import datetime, timezone
import subprocess
from pathlib import Path
from typing import Any

from etappe03_evaluation.io import LoadedData, load_input
from etappe03_evaluation.metrics import evaluate, METHODS_DEFAULT_ORDER
from etappe03_evaluation.aggregate import aggregate_per_instance_metrics, aggregate_coverage
from etappe03_evaluation.export import (
    fmt_median_iqr,
    pivot_coverage_median,
    write_csv,
    write_json,
    write_latex_table,
)
from etappe03_evaluation.plots import (
    plot_boxplots,
    plot_coverage_boxplots,
    plot_pareto_scatter_and_parallel,
    plot_quality_vs_time,
    plot_winrate_heatmap,
)


def _git_commit_hash() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out or None
    except Exception:
        return None


def _parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _filter_loaded_data(
    data: LoadedData,
    *,
    size_filter: set[str] | None,
    severity_filter: set[str] | None,
    seed_global: int | None,
) -> LoadedData:
    # Filter über IS-Metadaten
    allowed_is: set[tuple[str, str]] | None = None
    if size_filter is not None or severity_filter is not None:
        allowed_is = set()
        for (dataset_id, case_id), meta in data.case_meta.items():
            if size_filter is not None and (meta.size_class or "") not in size_filter:
                continue
            if severity_filter is not None and (meta.severity or "") not in severity_filter:
                continue
            allowed_is.add((dataset_id, case_id))

    solutions = data.solutions
    run_groups = data.run_groups
    subrun_ends = data.subrun_ends

    if allowed_is is not None:
        solutions = [s for s in solutions if (s.dataset_id, s.case_id) in allowed_is]
        run_groups = [g for g in run_groups if (g.dataset_id, g.case_id) in allowed_is]
        subrun_ends = [sr for sr in subrun_ends if (sr.dataset_id, sr.case_id) in allowed_is]

    if seed_global is not None:
        run_groups_seed = [g for g in run_groups if g.seed_global == seed_global]
        allowed_groups = {(g.dataset_id, g.case_id, g.verfahren, g.group_id) for g in run_groups_seed}
        solutions = [s for s in solutions if (s.dataset_id, s.case_id, s.verfahren, str(s.group_id or "")) in allowed_groups]
        run_groups = run_groups_seed
        subrun_ends = [sr for sr in subrun_ends if (sr.dataset_id, sr.case_id, sr.verfahren, sr.group_id) in allowed_groups]

    return LoadedData(
        case_meta=data.case_meta,
        solutions=solutions,
        run_groups=run_groups,
        subrun_ends=subrun_ends,
        omegas_v1=data.omegas_v1,
        warnings=data.warnings,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Etappe 03 (Evaluation): Berechnet mengenbasierte Pareto-Metriken (ND/P*/Coverage/Contribution) "
            "sowie sekundäre Laufzeitmetriken und erzeugt Exporte (CSV/LaTeX/Plots; vgl. schriftliche Arbeit, Kap. 5.3)."
        )
    )
    parser.add_argument("--input", type=Path, default=Path("logs"), help="Input-Root (z.B. fullrun_out/ oder logs/).")
    parser.add_argument("--out", type=Path, default=Path("evaluation_out"), help="Output-Ordner (default: evaluation_out/).")
    parser.add_argument("--methods", type=str, default="V1,V2,V3,V4", help="Komma-Liste: V1,V2,V3,V4 (default: alle).")
    parser.add_argument("--group-by", type=str, default="size,severity", help="Komma-Liste: size,severity (default).")
    parser.add_argument("--size", type=str, help="Filter size_class (Komma-Liste), z.B. small,medium,large.")
    parser.add_argument("--severity", type=str, help="Filter severity (Komma-Liste), z.B. leicht,mittel,schwer.")
    parser.add_argument("--seed", type=int, help="Filter seed_global aus runs.jsonl (falls vorhanden).")
    parser.add_argument("--eps", type=float, default=1e-9, help="Toleranz eps für Dominanz/Dedupe (numerische Robustheit; default: 1e-9).")
    parser.add_argument(
        "--set-mode",
        type=str,
        default="union",
        choices=["union", "per_group"],
        help="Bildung der Approximationsmengen A_v: union (default) oder per_group (pro Replicate).",
    )
    parser.add_argument(
        "--replicate-key",
        type=str,
        default="auto",
        choices=["auto", "seed_group", "seed_global"],
        help="Schlüssel für replicate_id im per_group-Modus (default: auto).",
    )
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"], help="Plot-Format (pdf|png).")
    parser.add_argument("--max-is-plots", type=int, default=12, help="Max. Anzahl IS für Scatter/Parallel-Plots (default: 12).")
    parser.add_argument("--export-pstar", action="store_true", help="Exportiere P*-Punkte nach pstar_points.csv (optional).")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True, help="Plots erzeugen (default: an).")
    parser.add_argument("--latex", action=argparse.BooleanOptionalAction, default=True, help="LaTeX-Tabellen erzeugen (default: an).")
    parser.add_argument(
        "--qt-x",
        type=str,
        default="runtime",
        choices=["runtime", "ttff"],
        help="Qualität-vs-Zeit Plot: x-Achse runtime oder ttff (default: runtime).",
    )
    parser.add_argument(
        "--qt-y",
        type=str,
        default="nd_size",
        choices=["nd_size", "contrib_inclusive", "contrib_unique"],
        help="Qualität-vs-Zeit Plot: y-Achse (default: nd_size).",
    )
    args = parser.parse_args(argv)

    def _json_safe(v: Any) -> Any:
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (list, tuple)):
            return [_json_safe(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _json_safe(x) for k, x in v.items()}
        return v

    methods = [m.strip().upper() for m in _parse_csv_list(args.methods)] or list(METHODS_DEFAULT_ORDER)
    group_by_raw = [g.strip().lower() for g in _parse_csv_list(args.group_by)]
    group_by: list[str] = []
    for g in group_by_raw:
        if g in {"size", "size_class"}:
            group_by.append("size_class")
        elif g in {"severity", "sev"}:
            group_by.append("severity")

    size_filter = set(_parse_csv_list(args.size)) if args.size else None
    severity_filter = set(_parse_csv_list(args.severity)) if args.severity else None

    data = load_input(Path(args.input), methods=set(methods), seed_global=args.seed)
    data = _filter_loaded_data(data, size_filter=size_filter, severity_filter=severity_filter, seed_global=args.seed)

    # Metadaten (deterministisch)
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "cli_args": _json_safe(vars(args)),
        "eps": float(args.eps),
        "input_root": str(Path(args.input)),
        "set_mode": str(args.set_mode),
        "replicate_key": str(args.replicate_key),
    }
    out_root = Path(args.out)
    write_json(out_root / "metadata.json", meta)

    # Warnings aus Loader (einmalig)
    if data.warnings:
        (out_root / "warnings.txt").write_text("\n".join(data.warnings) + "\n", encoding="utf-8")

    # Evaluation (primär: feasible-only)
    result = evaluate(
        data,
        eps=float(args.eps),
        methods=methods,
        export_pstar_points=bool(args.export_pstar),
        set_mode=str(args.set_mode),
        replicate_key=str(args.replicate_key),
    )

    # Aggregation (Median + IQR)
    numeric_fields = [
        "nd_size",
        "contrib_inclusive",
        "contrib_unique",
        "runtime_median_seconds",
        "ttff_median_seconds",
        "feasibility_rate",
    ]
    aggregated_metrics = aggregate_per_instance_metrics(result.per_instance_rows, group_by=group_by, numeric_fields=numeric_fields, global_label="all")
    coverage_aggregated = aggregate_coverage(result.coverage_rows, group_by=group_by, global_label="all")

    # --- Exporte: CSV --------------------------------------------------------
    set_mode = str(args.set_mode).strip().lower()
    if set_mode == "union":
        per_instance_fields = [
            "dataset_id",
            "case_id",
            "is_id",
            "size_class",
            "severity",
            "method",
            "n_feasible_raw",
            "n_feasible_deduped",
            "nd_size",
            "p_star_size",
            "contrib_inclusive",
            "contrib_unique",
            "runtime_median_seconds",
            "ttff_median_seconds",
            "feasibility_rate",
            "n_runs",
            "n_success",
            "I",
            "T",
            "S",
            "seed_instance",
            "seed_scenario",
        ]
        write_csv(out_root / "per_instance_metrics.csv", result.per_instance_rows, fieldnames=per_instance_fields)

        coverage_fields = [
            "dataset_id",
            "case_id",
            "is_id",
            "size_class",
            "severity",
            "method_a",
            "method_b",
            "coverage",
            "n_A",
            "n_B",
        ]
        write_csv(out_root / "coverage_matrix.csv", result.coverage_rows, fieldnames=coverage_fields)
    else:
        per_group_fields = [
            "dataset_id",
            "case_id",
            "is_id",
            "size_class",
            "severity",
            "replicate_id",
            "replicate_key_used",
            "replicate_fallback_reason",
            "method",
            "n_feasible_raw",
            "n_feasible_deduped",
            "nd_size",
            "p_star_size",
            "contrib_inclusive",
            "contrib_unique",
            "runtime_seconds",
            "ttff_seconds",
            "feasibility_rate",
            "n_runs",
            "n_success",
            "I",
            "T",
            "S",
            "seed_instance",
            "seed_scenario",
        ]
        write_csv(out_root / "per_group_metrics.csv", result.per_instance_rows, fieldnames=per_group_fields)

        coverage_fields_pg = [
            "dataset_id",
            "case_id",
            "is_id",
            "size_class",
            "severity",
            "replicate_id",
            "replicate_key_used",
            "replicate_fallback_reason",
            "method_a",
            "method_b",
            "coverage",
            "n_A",
            "n_B",
        ]
        write_csv(out_root / "coverage_matrix_per_group.csv", result.coverage_rows, fieldnames=coverage_fields_pg)

    agg_fields = [*group_by, "method", "n_is"]
    for nf in numeric_fields:
        agg_fields.extend([f"{nf}_median", f"{nf}_q1", f"{nf}_q3", f"{nf}_iqr"])
    write_csv(out_root / "aggregated_metrics.csv", aggregated_metrics, fieldnames=agg_fields)

    cov_agg_fields = [*group_by, "method_a", "method_b", "n_is", "coverage_median", "coverage_q1", "coverage_q3", "coverage_iqr"]
    write_csv(out_root / "coverage_matrix_aggregated.csv", coverage_aggregated, fieldnames=cov_agg_fields)

    if result.winrate_rows:
        winrate_fields = ["omega", "method", "wins", "n_is", "win_rate"]
        # omega als String serialisieren (CSV freundlich)
        winrate_rows = [{**r, "omega": str(r.get("omega"))} for r in result.winrate_rows]
        write_csv(out_root / "winrate_heatmap.csv", winrate_rows, fieldnames=winrate_fields)

    if result.winrate_is_rows:
        winrate_is_fields = ["dataset_id", "case_id", "is_id", "size_class", "severity", "omega", "winner", "best_z"]
        winrate_is_rows = [{**r, "omega": str(r.get("omega"))} for r in result.winrate_is_rows]
        write_csv(out_root / "winrate_per_is.csv", winrate_is_rows, fieldnames=winrate_is_fields)

    if result.pstar_rows:
        if set_mode == "union":
            pstar_fields = ["dataset_id", "case_id", "is_id", "f_stab", "f_ot", "f_pref", "f_fair", "producers"]
            write_csv(out_root / "pstar_points.csv", result.pstar_rows, fieldnames=pstar_fields)
        else:
            pstar_fields_pg = [
                "dataset_id",
                "case_id",
                "is_id",
                "replicate_id",
                "replicate_key_used",
                "replicate_fallback_reason",
                "f_stab",
                "f_ot",
                "f_pref",
                "f_fair",
                "producers",
            ]
            write_csv(out_root / "pstar_points_per_group.csv", result.pstar_rows, fieldnames=pstar_fields_pg)

    # MILP Debug: Subrun-Ende (Status/Gap/Bound falls vorhanden)
    if data.subrun_ends:
        milp_fields = [
            "dataset_id",
            "case_id",
            "verfahren",
            "group_id",
            "subrun_id",
            "status",
            "wall_seconds_used",
            "solutions_found",
            "time_to_first_feasible_seconds",
            "solver_status",
            "termination_reason",
            "mip_gap",
            "best_bound",
            "best_obj_scalar",
        ]
        milp_rows: list[dict[str, Any]] = []
        for sr in data.subrun_ends:
            milp_rows.append(
                {
                    "dataset_id": sr.dataset_id,
                    "case_id": sr.case_id,
                    "verfahren": sr.verfahren,
                    "group_id": sr.group_id,
                    "subrun_id": sr.subrun_id,
                    "status": sr.status,
                    "wall_seconds_used": sr.wall_seconds_used,
                    "solutions_found": sr.solutions_found,
                    "time_to_first_feasible_seconds": sr.time_to_first_feasible_seconds,
                    "solver_status": sr.solver_status,
                    "termination_reason": sr.termination_reason,
                    "mip_gap": sr.mip_gap,
                    "best_bound": sr.best_bound,
                    "best_obj_scalar": sr.best_obj_scalar,
                }
            )
        write_csv(out_root / "milp_subruns.csv", milp_rows, fieldnames=milp_fields)

    # Infeasible Solutions Debug (falls vorhanden)
    infeasible = [s for s in data.solutions if not s.feasible]
    if infeasible:
        inf_fields = ["dataset_id", "case_id", "verfahren", "group_id", "subrun_id", "solution_id", "F", "elapsed_seconds"]
        inf_rows = [
            {
                "dataset_id": s.dataset_id,
                "case_id": s.case_id,
                "verfahren": s.verfahren,
                "group_id": s.group_id,
                "subrun_id": s.subrun_id,
                "solution_id": s.solution_id,
                "F": str(s.F),
                "elapsed_seconds": s.elapsed_seconds,
            }
            for s in infeasible
        ]
        write_csv(out_root / "debug_infeasible_solutions.csv", inf_rows, fieldnames=inf_fields)

    # --- Exporte: LaTeX (booktabs) ------------------------------------------
    if args.latex:
        # Globales Aggregat (size=all, severity=all) als kompakte Tabelle
        rows_global = [r for r in aggregated_metrics if all(r.get(f) == "all" for f in group_by)]
        rows_global = sorted(rows_global, key=lambda r: str(r.get("method", "")))
        if rows_global:
            cols = ["Verfahren", "ND-Size", "Contrib (inkl.)", "Contrib (unique)", "Laufzeit (s)", "TTFF (s)", "Feasibility"]
            tex_rows: list[list[str]] = []
            for r in rows_global:
                tex_rows.append(
                    [
                        str(r.get("method", "")),
                        fmt_median_iqr(r, "nd_size", digits=2),
                        fmt_median_iqr(r, "contrib_inclusive", digits=3),
                        fmt_median_iqr(r, "contrib_unique", digits=3),
                        fmt_median_iqr(r, "runtime_median_seconds", digits=2),
                        fmt_median_iqr(r, "ttff_median_seconds", digits=3),
                        fmt_median_iqr(r, "feasibility_rate", digits=3),
                    ]
                )
            write_latex_table(
                out_root / "tables" / "aggregated_metrics__global.tex",
                caption="Aggregierte Metriken (Median [Q1,Q3]) — global über alle Instanz--Szenarien.",
                label="tab:agg_metrics_global",
                columns=cols,
                rows=tex_rows,
            )

        # Coverage-Matrix (Median) global
        if coverage_aggregated:
            table = pivot_coverage_median(coverage_aggregated, size_class="all", severity="all", methods=methods)
            if table and len(table) > 1:
                columns = table[0]
                rows = table[1:]
                write_latex_table(
                    out_root / "tables" / "coverage_matrix__global.tex",
                    caption="Coverage-Matrix (Median) — global über alle Instanz--Szenarien.",
                    label="tab:coverage_global",
                    columns=columns,
                    rows=rows,
                    col_align="l" + "r" * (len(columns) - 1),
                )

        # Win-Rate (falls verfügbar)
        if result.winrate_rows:
            omegas = sorted({r.get("omega") for r in result.winrate_rows if isinstance(r.get("omega"), tuple)})
            if omegas:
                cols = ["ω"] + methods
                lookup = {(r.get("omega"), str(r.get("method"))): r.get("win_rate") for r in result.winrate_rows}
                tex_rows = []
                for o in omegas:
                    row = [str(o)]
                    for m in methods:
                        val = lookup.get((o, m))
                        try:
                            row.append(f"{float(val):.3f}" if val is not None else "")
                        except Exception:
                            row.append("")
                    tex_rows.append(row)
                write_latex_table(
                    out_root / "tables" / "winrate__global.tex",
                    caption="Win-Rate über ω (aus V1-Logs) — Anteil der IS, in denen ein Verfahren unter Z_ω gewinnt.",
                    label="tab:winrate_global",
                    columns=cols,
                    rows=tex_rows,
                    col_align="l" + "r" * (len(cols) - 1),
                )

    # --- Plots ---------------------------------------------------------------
    if args.plots:
        fig_dir = out_root / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_boxplots(fig_dir / "boxplots", per_instance_rows=result.per_instance_rows, methods=methods, fmt=args.format)
        plot_coverage_boxplots(fig_dir / "boxplots_coverage", coverage_rows=result.coverage_rows, methods=methods, fmt=args.format)

        # Qualität vs Zeit (Scatter), getrennt nach size_class & severity
        qt_x = str(args.qt_x).strip().lower()
        qt_y = str(args.qt_y).strip().lower()
        if set_mode == "per_group":
            x_field = "runtime_seconds" if qt_x == "runtime" else "ttff_seconds"
        else:
            x_field = "runtime_median_seconds" if qt_x == "runtime" else "ttff_median_seconds"
        y_field = qt_y
        plot_quality_vs_time(fig_dir, rows=result.per_instance_rows, methods=methods, fmt=args.format, x_field=x_field, y_field=y_field)

        if result.winrate_rows:
            winrate_rows_plot = [{**r, "omega": r.get("omega")} for r in result.winrate_rows]
            plot_winrate_heatmap(fig_dir / "winrate_heatmap", winrate_rows=winrate_rows_plot, methods=methods, fmt=args.format)
        plot_pareto_scatter_and_parallel(
            fig_dir,
            data=data,
            methods=methods,
            eps=float(args.eps),
            fmt=args.format,
            max_is=int(args.max_is_plots),
        )

    return 0
