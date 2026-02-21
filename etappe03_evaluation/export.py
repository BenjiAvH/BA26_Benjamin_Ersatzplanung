from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("$", "\\$")
    )


def _fmt_float(x: Any, *, digits: int = 3) -> str:
    if x is None or x == "":
        return ""
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def fmt_median_iqr(row: dict[str, Any], base: str, *, digits: int = 3) -> str:
    med = _fmt_float(row.get(f"{base}_median"), digits=digits)
    q1 = _fmt_float(row.get(f"{base}_q1"), digits=digits)
    q3 = _fmt_float(row.get(f"{base}_q3"), digits=digits)
    if med == "" and q1 == "" and q3 == "":
        return ""
    return f"{med} [{q1}, {q3}]"


def write_latex_table(
    path: Path,
    *,
    caption: str,
    label: str,
    columns: list[str],
    rows: list[list[str]],
    col_align: str | None = None,
) -> None:
    """
    Minimaler booktabs-Tabellenexport (ohne zusätzliche Abhängigkeiten).
    """

    if col_align is None:
        col_align = "l" + "r" * (len(columns) - 1)

    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_align}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(_latex_escape(c) for c in columns) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(" & ".join(_latex_escape(str(x)) for x in r) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def pivot_coverage_median(
    coverage_agg_rows: list[dict[str, Any]],
    *,
    size_class: str = "all",
    severity: str = "all",
    methods: list[str],
) -> list[list[str]]:
    """
    Coverage-Matrix (Median) als 2D-Array für LaTeX/Plots.
    """

    lookup: dict[tuple[str, str], Any] = {}
    for r in coverage_agg_rows:
        if r.get("size_class") != size_class or r.get("severity") != severity:
            continue
        a = str(r.get("method_a"))
        b = str(r.get("method_b"))
        lookup[(a, b)] = r.get("coverage_median")

    table: list[list[str]] = []
    header = ["A\\B"] + methods
    table.append(header)
    for a in methods:
        row = [a]
        for b in methods:
            row.append(_fmt_float(lookup.get((a, b)), digits=3))
        table.append(row)
    return table

