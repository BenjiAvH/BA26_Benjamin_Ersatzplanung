from __future__ import annotations

import argparse
import json
from pathlib import Path

from etappe01_simulation.simulator.validate import validate_case


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Etappe 01: Validiert ein Case-JSON gegen harte Checks (optional: strict inkl. Post-Disturbance-Feasibility)."
    )
    parser.add_argument("case_json", type=Path, help="Pfad zu einem Case-JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strenger Modus: verlangt ok_strict (inkl. post_disturbance_feasibility_ok).",
    )
    args = parser.parse_args()

    case = json.loads(args.case_json.read_text(encoding="utf-8"))
    validation = validate_case(case)

    key = "ok_strict" if args.strict else "ok"
    ok = bool(validation.get(key, False))
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
