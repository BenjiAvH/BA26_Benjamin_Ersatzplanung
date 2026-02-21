from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _import_meta() -> dict[str, Any]:
    meta: dict[str, Any] = {"ok_import": False}
    try:
        import pyscipopt  # type: ignore
        from pyscipopt import Model  # type: ignore
    except Exception as e:
        meta["error"] = f"{type(e).__name__}: {e}"
        return meta

    meta["ok_import"] = True
    meta["pyscipopt_version"] = getattr(pyscipopt, "__version__", None)

    try:
        m = Model("version_probe")
        m.hideOutput()
        if hasattr(m, "getVersion"):
            meta["scip_version"] = str(m.getVersion())
    except Exception as e:
        meta["scip_version_error"] = f"{type(e).__name__}: {e}"

    return meta


def _solve_smoke() -> dict[str, Any]:
    from pyscipopt import Model  # type: ignore

    out: dict[str, Any] = {}
    m = Model("smoke")
    m.hideOutput()

    x = m.addVar("x", vtype="INTEGER", lb=0)
    m.addCons(x >= 1)
    m.setObjective(x, "minimize")
    m.optimize()

    out["status"] = str(m.getStatus())
    try:
        out["x"] = float(m.getVal(x))
    except Exception:
        out["x"] = None

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Etappe 02 (V1/V2): Kurztest für SCIP/PySCIPOpt (Import + Mini-MILP).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Gibt Ergebnisse als JSON aus (maschinenlesbar).",
    )
    args = parser.parse_args()

    meta = _import_meta()
    if not meta.get("ok_import", False):
        payload = {"meta": meta}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print("SCIP/PySCIPOpt nicht nutzbar.")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    solve = _solve_smoke()
    payload = {"meta": meta, "solve": solve}

    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print("SCIP/PySCIPOpt: OK")
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    x_val = solve.get("x")
    ok_value = (x_val is not None) and abs(float(x_val) - 1.0) <= 1e-9
    ok_status = str(solve.get("status", "")).lower() in {"optimal", "timelimit", "time_limit"}
    return 0 if (ok_status and ok_value) else 2


if __name__ == "__main__":
    raise SystemExit(main())
