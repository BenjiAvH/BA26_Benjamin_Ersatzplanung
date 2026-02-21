from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def _quantile(values: list[float], q: float) -> float | None:
    xs = [float(v) for v in values if v is not None]
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


def median_q1_q3(values: list[float]) -> tuple[float | None, float | None, float | None]:
    return (_quantile(values, 0.5), _quantile(values, 0.25), _quantile(values, 0.75))


def aggregate_per_instance_metrics(
    per_instance_rows: list[dict[str, Any]],
    *,
    group_by: list[str],
    numeric_fields: list[str],
    global_label: str = "all",
    include_global: bool = True,
) -> list[dict[str, Any]]:
    """
    Aggregiert per Verfahren über IS (Median + Q1/Q3).

    group_by:
      - z.B. ["size_class", "severity"]
      - globale Aggregation wird zusätzlich als (group_by=all) ergänzt.
    """

    def _group_key(row: dict[str, Any]) -> tuple:
        return tuple(row.get(f) for f in group_by) + (row.get("method"),)

    buckets: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in per_instance_rows:
        buckets[_group_key(row)].append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(buckets, key=lambda k: tuple("" if v is None else str(v) for v in k)):
        rows = buckets[key]
        method = rows[0].get("method")
        base: dict[str, Any] = {"method": method, "n_is": len({r.get("is_id") for r in rows})}
        for idx, field in enumerate(group_by):
            base[field] = key[idx]

        for nf in numeric_fields:
            vals = [r.get(nf) for r in rows if r.get(nf) is not None]
            vals_f = [float(v) for v in vals]
            med, q1, q3 = median_q1_q3(vals_f)
            base[f"{nf}_median"] = med
            base[f"{nf}_q1"] = q1
            base[f"{nf}_q3"] = q3
            base[f"{nf}_iqr"] = (q3 - q1) if (q1 is not None and q3 is not None) else None
        out.append(base)

    # Globale Aggregation über alle IS
    if include_global and per_instance_rows:
        rows_global = []
        for row in per_instance_rows:
            r2 = dict(row)
            for f in group_by:
                r2[f] = global_label
            rows_global.append(r2)
        out.extend(
            aggregate_per_instance_metrics(
                rows_global,
                group_by=group_by,
                numeric_fields=numeric_fields,
                global_label=global_label,
                include_global=False,
            )
        )

    # Duplikate durch Rekursion entfernen (deterministisch: letztes gewinnt nicht nötig)
    seen: set[tuple] = set()
    deduped: list[dict[str, Any]] = []
    for row in out:
        key = tuple(row.get(f) for f in group_by) + (row.get("method"),)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def aggregate_coverage(
    coverage_rows: list[dict[str, Any]],
    *,
    group_by: list[str],
    global_label: str = "all",
    include_global: bool = True,
) -> list[dict[str, Any]]:
    """
    Aggregiert Coverage-Werte (Median + Q1/Q3) pro (method_a, method_b) und Gruppe.
    """

    def _group_key(row: dict[str, Any]) -> tuple:
        return tuple(row.get(f) for f in group_by) + (row.get("method_a"), row.get("method_b"))

    buckets: dict[tuple, list[float]] = defaultdict(list)
    n_is_bucket: dict[tuple, set[str]] = defaultdict(set)
    for row in coverage_rows:
        c = row.get("coverage")
        if c is None:
            continue
        try:
            c_f = float(c)
        except Exception:
            continue
        key = _group_key(row)
        buckets[key].append(c_f)
        n_is_bucket[key].add(str(row.get("is_id")))

    out: list[dict[str, Any]] = []
    for key in sorted(buckets, key=lambda k: tuple("" if v is None else str(v) for v in k)):
        vals = buckets[key]
        med, q1, q3 = median_q1_q3(vals)
        row_out: dict[str, Any] = {
            "method_a": key[len(group_by) + 0],
            "method_b": key[len(group_by) + 1],
            "n_is": len(n_is_bucket.get(key, set())),
            "coverage_median": med,
            "coverage_q1": q1,
            "coverage_q3": q3,
            "coverage_iqr": (q3 - q1) if (q1 is not None and q3 is not None) else None,
        }
        for idx, field in enumerate(group_by):
            row_out[field] = key[idx]
        out.append(row_out)

    # Global
    if include_global and coverage_rows:
        rows_global = []
        for row in coverage_rows:
            r2 = dict(row)
            for f in group_by:
                r2[f] = global_label
            rows_global.append(r2)
        out.extend(aggregate_coverage(rows_global, group_by=group_by, global_label=global_label, include_global=False))

    # Deduplicate
    seen: set[tuple] = set()
    deduped: list[dict[str, Any]] = []
    for row in out:
        key = tuple(row.get(f) for f in group_by) + (row.get("method_a"), row.get("method_b"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped
