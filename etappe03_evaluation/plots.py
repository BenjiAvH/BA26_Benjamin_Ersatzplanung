from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from etappe03_evaluation.io import LoadedData, make_is_id
from etappe03_evaluation.pareto import Point, dedupe_points, non_dominated


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "Matplotlib ist nicht verfügbar. Bitte Abhängigkeit installieren oder --no-plots verwenden."
        ) from e
    return plt


def _methods_sorted(methods: Iterable[str]) -> list[str]:
    order = ["V1", "V2", "V3", "V4"]
    ms = {str(m).strip().upper() for m in methods if str(m).strip()}
    out = [m for m in order if m in ms]
    out.extend(sorted(ms - set(out)))
    return out


def _compute_sets_for_is(
    data: LoadedData,
    *,
    dataset_id: str,
    case_id: str,
    methods: list[str],
    eps: float,
) -> tuple[dict[str, list[Point]], list[Point], list[tuple[float, float, float, float]]]:
    # pro Verfahren: feasible -> dedupe -> ND
    sol = [s for s in data.solutions if s.dataset_id == dataset_id and s.case_id == case_id]
    A_v: dict[str, list[Point]] = {}
    all_feasible: list[tuple[float, float, float, float]] = []
    for v in methods:
        pts = [Point(F=s.F, producers=frozenset({v})) for s in sol if s.verfahren == v and s.feasible]
        for p in pts:
            all_feasible.append(p.F)
        pts = dedupe_points(pts, eps=eps)
        A_v[v] = non_dominated(pts, eps=eps)

    # P*
    union_pts: list[Point] = []
    for v in methods:
        union_pts.extend(A_v.get(v, []))
    union_pts = dedupe_points(union_pts, eps=eps)
    P_star = non_dominated(union_pts, eps=eps)
    return A_v, P_star, all_feasible


def plot_winrate_heatmap(
    out_path: Path,
    *,
    winrate_rows: list[dict[str, Any]],
    methods: list[str],
    fmt: str,
) -> None:
    plt = _require_matplotlib()

    omegas = sorted({r.get("omega") for r in winrate_rows if isinstance(r.get("omega"), tuple)})
    methods = _methods_sorted(methods)
    if not omegas or not methods:
        return

    # Matrix: len(omegas) x len(methods)
    mat = [[0.0 for _ in methods] for _ in omegas]
    lookup: dict[tuple[tuple[float, float, float, float], str], float] = {}
    for r in winrate_rows:
        omega = r.get("omega")
        method = str(r.get("method", "")).strip().upper()
        if not isinstance(omega, tuple) or method not in methods:
            continue
        val = r.get("win_rate")
        try:
            lookup[(omega, method)] = float(val) if val is not None else 0.0
        except Exception:
            lookup[(omega, method)] = 0.0

    for i, omega in enumerate(omegas):
        for j, m in enumerate(methods):
            mat[i][j] = lookup.get((omega, m), 0.0)

    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(methods)), max(4.0, 0.4 * len(omegas))))
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(list(range(len(methods))), labels=methods)
    omega_labels = [f"({o[0]:g},{o[1]:g},{o[2]:g},{o[3]:g})" for o in omegas]
    ax.set_yticks(list(range(len(omegas))), labels=omega_labels)
    ax.set_title("Win-Rate über ω (V1-ω-Sets)")
    fig.colorbar(im, ax=ax, label="Win-Rate")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(f".{fmt}"))
    plt.close(fig)


def plot_boxplots(
    out_path: Path,
    *,
    per_instance_rows: list[dict[str, Any]],
    methods: list[str],
    fmt: str,
) -> None:
    plt = _require_matplotlib()
    methods = _methods_sorted(methods)

    # Metriken (primary: Contribution; secondary: runtime/ttff/feasibility)
    metrics = [
        ("contrib_inclusive", "Contribution (inkl.)"),
        ("contrib_unique", "Contribution (unique)"),
        ("runtime_median_seconds", "Laufzeit (s, Median)"),
        ("ttff_median_seconds", "TTFF (s, Median)"),
        ("feasibility_rate", "Feasibility-Rate"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.5), sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (key, title) in zip(axes, metrics):
        data_by_method: list[list[float]] = []
        labels: list[str] = []
        for m in methods:
            vals = [r.get(key) for r in per_instance_rows if str(r.get("method", "")).upper() == m and r.get(key) is not None]
            vals_f: list[float] = []
            for v in vals:
                try:
                    vals_f.append(float(v))
                except Exception:
                    continue
            if not vals_f:
                continue
            data_by_method.append(vals_f)
            labels.append(m)

        if not data_by_method:
            ax.set_title(title)
            ax.text(0.5, 0.5, "keine Daten", ha="center", va="center")
            ax.set_xticks([])
            continue

        ax.boxplot(data_by_method, labels=labels, showfliers=False)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Boxplots: primäre & sekundäre Metriken (pro IS)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(f".{fmt}"))
    plt.close(fig)


def plot_quality_vs_time(
    out_dir: Path,
    *,
    rows: list[dict[str, Any]],
    methods: list[str],
    fmt: str,
    x_field: str,
    y_field: str,
) -> None:
    """
    Scatter-Plot "Qualität vs Zeit" (Kap. 5.3: Metriken & Vergleichsauswertung; Laufzeit--Qualitäts-Darstellungen).

    Erwartete Felder in rows:
      - Gruppierung: size_class, severity
      - x_field: z.B. runtime_median_seconds | ttff_median_seconds | runtime_seconds | ttff_seconds
      - y_field: nd_size | contrib_inclusive | contrib_unique

    Robustheit:
      - Wenn Felder fehlen oder keine Punkte vorhanden sind: kein Crash, keine Datei.
    """

    plt = _require_matplotlib()
    methods = _methods_sorted(methods)

    # Gruppieren nach (size_class, severity)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        sc = str(r.get("size_class") or "na")
        sev = str(r.get("severity") or "na")
        buckets.setdefault((sc, sev), []).append(r)

    out_dir.mkdir(parents=True, exist_ok=True)

    for (sc, sev), grp in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        any_points = False
        fig, ax = plt.subplots(figsize=(6.5, 4.5))

        for m in methods:
            xs: list[float] = []
            ys: list[float] = []
            for r in grp:
                if str(r.get("method", "")).strip().upper() != m:
                    continue
                x_raw = r.get(x_field)
                y_raw = r.get(y_field)
                if x_raw is None or y_raw is None:
                    continue
                try:
                    x = float(x_raw)
                    y = float(y_raw)
                except Exception:
                    continue
                xs.append(x)
                ys.append(y)

            if xs and ys:
                any_points = True
                ax.scatter(xs, ys, label=m, alpha=0.85, s=26)

        if not any_points:
            plt.close(fig)
            continue

        ax.set_xlabel(x_field)
        ax.set_ylabel(y_field)
        ax.set_title(f"Qualität vs Zeit ({sc}, {sev})")
        ax.grid(alpha=0.25)
        ax.legend(title="Verfahren", loc="best", frameon=True)
        fig.tight_layout()

        group_label = f"sc-{sc}__sev-{sev}"
        out_path = out_dir / f"quality_vs_time__{x_field}__{y_field}__{group_label}.{fmt}"
        fig.savefig(out_path)
        plt.close(fig)


def plot_coverage_boxplots(
    out_path: Path,
    *,
    coverage_rows: list[dict[str, Any]],
    methods: list[str],
    fmt: str,
) -> None:
    """
    Boxplots der Coverage-Werte C(A,B) je geordnetem Verfahrenpaar (A->B).

    Hinweis:
      Coverage ist paarweise definiert; daher wird direkt pro Paar geplottet
      (kein "neues" Aggregat/Metric).
    """

    plt = _require_matplotlib()
    methods = _methods_sorted(methods)

    pairs: list[tuple[str, str]] = [(a, b) for a in methods for b in methods if a != b]
    data_by_pair: list[list[float]] = []
    labels: list[str] = []
    for a, b in pairs:
        vals = [
            r.get("coverage")
            for r in coverage_rows
            if str(r.get("method_a", "")).upper() == a and str(r.get("method_b", "")).upper() == b and r.get("coverage") is not None
        ]
        vals_f: list[float] = []
        for v in vals:
            try:
                vals_f.append(float(v))
            except Exception:
                continue
        if not vals_f:
            continue
        data_by_pair.append(vals_f)
        labels.append(f"{a}->{b}")

    fig_w = max(8.0, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    if not data_by_pair:
        ax.text(0.5, 0.5, "keine Coverage-Daten", ha="center", va="center")
        ax.set_xticks([])
        ax.set_title("Coverage-Boxplots")
    else:
        ax.boxplot(data_by_pair, labels=labels, showfliers=False)
        ax.set_title("Coverage-Boxplots: C(A,B) pro Verfahrenpaar (A->B)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(f".{fmt}"))
    plt.close(fig)


def plot_pareto_scatter_and_parallel(
    out_dir: Path,
    *,
    data: LoadedData,
    methods: list[str],
    eps: float,
    fmt: str,
    max_is: int = 12,
) -> None:
    plt = _require_matplotlib()
    methods = _methods_sorted(methods)

    is_keys = sorted({(s.dataset_id, s.case_id) for s in data.solutions})
    is_keys = is_keys[: max(0, int(max_is))]

    pairs = [
        (0, 1, "f_stab", "f_ot"),
        (0, 2, "f_stab", "f_pref"),
        (1, 2, "f_ot", "f_pref"),
        (0, 3, "f_stab", "f_fair"),
    ]

    colors = {"V1": "#1f77b4", "V2": "#ff7f0e", "V3": "#2ca02c", "V4": "#d62728"}

    for dataset_id, case_id in is_keys:
        is_id = make_is_id(dataset_id, case_id)
        A_v, P_star, all_feasible = _compute_sets_for_is(data, dataset_id=dataset_id, case_id=case_id, methods=methods, eps=eps)

        # Scatterplots (je Zielpaar)
        for i, j, xi, yj in pairs:
            fig, ax = plt.subplots(figsize=(6.0, 4.5))
            for m in methods:
                pts = A_v.get(m, [])
                xs = [p.F[i] for p in pts]
                ys = [p.F[j] for p in pts]
                if xs and ys:
                    ax.scatter(xs, ys, s=18, alpha=0.75, label=m, color=colors.get(m))

            # P* Overlay (schwarz)
            xs_p = [p.F[i] for p in P_star]
            ys_p = [p.F[j] for p in P_star]
            if xs_p and ys_p:
                ax.scatter(xs_p, ys_p, s=22, alpha=0.9, label="P*", color="black", marker="x")

            ax.set_xlabel(xi)
            ax.set_ylabel(yj)
            ax.set_title(f"Pareto-Scatter: {dataset_id}/{case_id}")
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            out_path = out_dir / f"pareto_scatter__{is_id}__{xi}__{yj}.{fmt}"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path)
            plt.close(fig)

        # Parallel Coordinates (über 4 Ziele) – auf P* (reduziert, lesbarer)
        if not P_star:
            continue

        mins = (
            min(f[0] for f in all_feasible) if all_feasible else 0.0,
            min(f[1] for f in all_feasible) if all_feasible else 0.0,
            min(f[2] for f in all_feasible) if all_feasible else 0.0,
            min(f[3] for f in all_feasible) if all_feasible else 0.0,
        )
        maxs = (
            max(f[0] for f in all_feasible) if all_feasible else 1.0,
            max(f[1] for f in all_feasible) if all_feasible else 1.0,
            max(f[2] for f in all_feasible) if all_feasible else 1.0,
            max(f[3] for f in all_feasible) if all_feasible else 1.0,
        )

        def _norm(val: float, mn: float, mx: float) -> float:
            if mx == mn:
                return 0.0
            return (val - mn) / (mx - mn)

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        xs = [0, 1, 2, 3]
        labels = ["f_stab", "f_ot", "f_pref", "f_fair"]
        for p in P_star:
            # Farbe: erster Producer (deterministisch), sonst schwarz
            prod = sorted(p.producers)[0] if p.producers else "P*"
            c = colors.get(prod, "#000000")
            ys = [
                _norm(p.F[0], mins[0], maxs[0]),
                _norm(p.F[1], mins[1], maxs[1]),
                _norm(p.F[2], mins[2], maxs[2]),
                _norm(p.F[3], mins[3], maxs[3]),
            ]
            ax.plot(xs, ys, color=c, alpha=0.35, linewidth=1.0)

        ax.set_xticks(xs, labels=labels)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("normierte Ziele (pro IS)")
        ax.set_title(f"Parallel Coordinates (P*): {dataset_id}/{case_id}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = out_dir / f"parallel_coords__{is_id}.{fmt}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
