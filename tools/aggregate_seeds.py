from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_float(v: str | None) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Aggregiert Multi-Seed Fullrun-Ergebnisse.\n"
            "Quelle pro Seed-Run: <run_dir>/stage3_evaluation/aggregated_metrics.csv (global rows: group_by=all).\n"
            "Outputs: <root>/_aggregate/all_seeds_long.csv und agg_mean_std.csv"
        )
    )
    p.add_argument("--root", type=Path, default=Path("fullrun_out"), help="BASE_OUT_ROOT (default: fullrun_out/).")
    p.add_argument("--pattern", type=str, default="seed_*", help="Glob-Pattern für Run-Subdirs (default: seed_*).")
    args = p.parse_args(argv)

    root = Path(args.root).expanduser()
    run_dirs = sorted([p for p in root.glob(str(args.pattern)) if p.is_dir()])
    if not run_dirs:
        print(f"[FEHLER] Keine Run-Ordner gefunden unter {root} (pattern={args.pattern!r}).", file=sys.stderr)
        return 2

    long_rows: list[dict[str, Any]] = []
    values_by_key: dict[tuple[str, str], list[float]] = {}
    errors: list[str] = []

    for run_dir in run_dirs:
        run_name = run_dir.name
        meta_path = run_dir / "metadata.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = _read_json(meta_path)
            except Exception as e:
                errors.append(f"[WARN] Konnte metadata.json nicht lesen: {meta_path} ({type(e).__name__}: {e})")
                meta = {}

        run_seed = meta.get("run_seed")
        if run_seed is None:
            cli_args = meta.get("cli_args")
            if isinstance(cli_args, dict):
                run_seed = cli_args.get("seed")

        metrics_path = run_dir / "stage3_evaluation" / "aggregated_metrics.csv"
        if not metrics_path.exists():
            errors.append(f"[WARN] Fehlt: {metrics_path}")
            continue

        try:
            with metrics_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                if not fieldnames or "method" not in fieldnames:
                    errors.append(f"[WARN] Unerwartetes CSV-Schema (method fehlt): {metrics_path}")
                    continue
                method_idx = fieldnames.index("method")
                group_by_fields = fieldnames[:method_idx]

                rows = list(reader)
        except Exception as e:
            errors.append(f"[WARN] Konnte CSV nicht lesen: {metrics_path} ({type(e).__name__}: {e})")
            continue

        global_rows = []
        for r in rows:
            if all(str(r.get(f, "")).strip().lower() == "all" for f in group_by_fields):
                global_rows.append(r)
        if not global_rows:
            errors.append(f"[WARN] Keine global rows (group_by=all) gefunden: {metrics_path}")
            continue

        skip = set(group_by_fields) | {"method"}
        for r in global_rows:
            method = str(r.get("method", "")).strip().upper()
            if not method:
                continue
            for k, v in r.items():
                if k in skip:
                    continue
                fv = _parse_float(v)
                if fv is None:
                    continue
                long_rows.append(
                    {
                        "run_dir": str(run_dir),
                        "run_name": run_name,
                        "run_seed": run_seed,
                        "method": method,
                        "metric": str(k),
                        "value": fv,
                    }
                )
                values_by_key.setdefault((method, str(k)), []).append(float(fv))

    if not long_rows:
        print("[FEHLER] Keine Metriken gefunden (siehe Warnungen).", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        return 2

    out_dir = root / "_aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_long = out_dir / "all_seeds_long.csv"
    out_agg = out_dir / "agg_mean_std.csv"

    long_fieldnames = ["run_dir", "run_name", "run_seed", "method", "metric", "value"]
    with out_long.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=long_fieldnames)
        w.writeheader()
        for row in long_rows:
            w.writerow(row)

    agg_rows: list[dict[str, Any]] = []
    for (method, metric), xs in sorted(values_by_key.items()):
        n = len(xs)
        row_out: dict[str, Any] = {
            "method": method,
            "metric": metric,
            "n": n,
            "mean": statistics.mean(xs) if xs else None,
            "std": statistics.stdev(xs) if len(xs) >= 2 else None,
            "min": min(xs) if xs else None,
            "max": max(xs) if xs else None,
        }
        agg_rows.append(row_out)

    agg_fieldnames = ["method", "metric", "n", "mean", "std", "min", "max"]
    with out_agg.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fieldnames)
        w.writeheader()
        for row in agg_rows:
            w.writerow(row)

    for e in errors:
        print(e, file=sys.stderr)
    print(f"[OK] Wrote: {out_long}")
    print(f"[OK] Wrote: {out_agg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
