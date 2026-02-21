from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from typing import Any

from tools.validation import ValidationReport, read_json, require_path_exists


ALL_METHODS = ["V1", "V2", "V3", "V4"]
LEGACY_SEED = 20260217


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit_hash(*, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd) if cwd else None, text=True).strip() or None
    except Exception:
        return None


def _json_safe(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    return v


def _load_json(path: Path) -> Any:
    report = ValidationReport()
    data = read_json(path, report=report, level="full", required="light")
    report.raise_if_errors()
    return data


def _parse_csv_list(value: str) -> list[str]:
    return [v.strip() for v in str(value or "").split(",") if v.strip()]


def derive_seed(run_seed: int, method_id: str) -> int:
    h = hashlib.sha256(f"{int(run_seed)}:{str(method_id)}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def set_global_seed(seed: int) -> None:
    # Wirkt nur im aktuellen Prozess; Etappen-Subprozesse erhalten Seeds über CLI-Parameter (z.B. --seed).
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed) & 0xFFFFFFFF)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def _format_cmd(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(x) for x in cmd])
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _get_python_version(python_exe: str) -> tuple[int, int, int] | None:
    try:
        out = subprocess.check_output(
            [
                str(python_exe),
                "-c",
                "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')",
            ],
            text=True,
        ).strip()
    except Exception:
        return None

    parts = [p.strip() for p in out.split(".")]
    if len(parts) != 3:
        return None
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return None


def _log_print(msg: str, *, log_f: Any | None, lock: threading.Lock | None = None) -> None:
    if lock is not None:
        with lock:
            print(msg, flush=True)
            if log_f is not None:
                log_f.write(msg + "\n")
                log_f.flush()
        return
    print(msg, flush=True)
    if log_f is not None:
        log_f.write(msg + "\n")
        log_f.flush()


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    log_f: Any | None,
    dry_run: bool,
    extra_env: dict[str, str] | None = None,
    prefix: str | None = None,
    lock: threading.Lock | None = None,
) -> int:
    pfx = str(prefix or "")
    _log_print(f"{pfx}$ {_format_cmd(cmd)}", log_f=log_f, lock=lock)
    if dry_run:
        return 0

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})

    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        out_line = f"{pfx}{line}" if pfx else line
        if lock is not None:
            with lock:
                sys.stdout.write(out_line)
                sys.stdout.flush()
                if log_f is not None:
                    log_f.write(out_line)
                    log_f.flush()
        else:
            sys.stdout.write(out_line)
            sys.stdout.flush()
            if log_f is not None:
                log_f.write(out_line)
                log_f.flush()

    rc = int(proc.wait())
    if rc != 0:
        _log_print(f"{pfx}[FEHLER] Exitcode {rc} (Details: fullrun.log).", log_f=log_f, lock=lock)
    return rc


def _detect_entrypoints(repo_root: Path) -> dict[str, str]:
    stage1 = repo_root / "etappe01_simulation" / "scripts" / "generate_dataset.py"
    stage2 = repo_root / "etappe02_modelle" / "scripts" / "run_case.py"
    stage3 = repo_root / "etappe03_evaluation" / "__main__.py"

    report = ValidationReport()
    require_path_exists(stage1, report=report, level="full", kind="file", what=str(stage1))
    require_path_exists(stage2, report=report, level="full", kind="file", what=str(stage2))
    require_path_exists(stage3, report=report, level="full", kind="file", what=str(stage3))
    if report.errors:
        raise SystemExit(
            "Konnte nicht alle benötigten Entry-Points finden:\n- "
            + "\n- ".join(report.errors)
            + "\n(Erwartet: etappe01_simulation/scripts/generate_dataset.py, "
            "etappe02_modelle/scripts/run_case.py, etappe03_evaluation/__main__.py)"
        )

    return {
        "stage1_module": "etappe01_simulation.scripts.generate_dataset",
        "stage2_module": "etappe02_modelle.scripts.run_case",
        "stage3_module": "etappe03_evaluation",
    }


def _load_solver_plan(plan_path: Path, *, repo_root: Path) -> dict[str, Path]:
    report = ValidationReport()
    data = read_json(plan_path, report=report, level="full", must_be_object=True, what=str(plan_path))
    if report.errors:
        report.raise_if_errors()
    assert isinstance(data, dict)

    procedures = data.get("procedures")
    if not isinstance(procedures, list) or not procedures:
        report.add_error(f"Solver-Config ungültig: procedures fehlt/leer ({plan_path})")
        report.raise_if_errors()

    out: dict[str, Path] = {}
    for idx, item in enumerate(procedures):
        if not isinstance(item, dict):
            report.add_error(f"Solver-Config ungültig: procedures[{idx}] ist kein Objekt ({plan_path})")
            continue
        verfahren = str(item.get("verfahren", "")).strip().upper()
        cfg = item.get("config")
        if not verfahren or cfg is None:
            report.add_error(f"Solver-Config ungültig: procedures[{idx}] braucht verfahren+config ({plan_path})")
            continue

        cfg_path = Path(str(cfg))
        if not cfg_path.is_absolute():
            cfg_path = (repo_root / cfg_path).resolve()
        out[verfahren] = cfg_path

    for v, p in out.items():
        require_path_exists(
            p,
            report=report,
            level="full",
            kind="file",
            what=f"Config-Datei für {v} existiert nicht: {p} (aus {plan_path})",
        )
    report.raise_if_errors()
    return out


def _load_cases_from_manifest(manifest_path: Path, *, stage1_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    report = ValidationReport()
    data = read_json(manifest_path, report=report, level="full", must_be_object=True, what=str(manifest_path))
    report.raise_if_errors()
    assert isinstance(data, dict)

    dataset_id = str(data.get("dataset_id", "")).strip()
    cases = data.get("cases")
    if not dataset_id or not isinstance(cases, list):
        report.add_error(f"manifest.json Schema unerwartet: {manifest_path}")
        report.raise_if_errors()

    out: list[dict[str, Any]] = []
    for entry in cases:
        if not isinstance(entry, dict):
            continue
        case_id = str(entry.get("case_id", "")).strip()
        file_raw = entry.get("file")
        if not case_id:
            continue

        strict_ok = bool(entry.get("validation_ok_strict", False))
        ok = bool(entry.get("validation_ok", False))
        status = str(entry.get("status", "")).strip()

        candidates: list[Path] = []
        if isinstance(file_raw, str) and file_raw.strip():
            p = Path(file_raw)
            candidates.append(p)
            if not p.is_absolute():
                candidates.append((Path.cwd() / p).resolve())
        candidates.append((stage1_dir / f"{case_id}.json").resolve())

        case_path: Path | None = None
        for c in candidates:
            try:
                if c.exists():
                    case_path = c
                    break
            except Exception:
                continue
        out.append(
            {
                "case_id": case_id,
                "case_path": case_path,
                "validation_ok_strict": strict_ok,
                "validation_ok": ok,
                "status": status,
            }
        )

    if not out:
        report.add_error(f"Keine Cases im Manifest gefunden: {manifest_path}")
        report.raise_if_errors()
    return dataset_id, out


def _load_cases_fallback(stage1_dir: Path) -> tuple[str, list[tuple[str, Path]]]:
    report = ValidationReport()
    files = sorted(p for p in stage1_dir.glob("*.json") if p.name != "manifest.json")
    if not files:
        report.add_error(f"Keine Case-JSONs gefunden in: {stage1_dir}")
        report.raise_if_errors()

    # Dataset-ID aus erstem Case lesen (Fallback, falls manifest.json fehlt).
    first = _load_json(files[0])
    dataset_id = str(first.get("dataset_id", "")).strip() if isinstance(first, dict) else ""
    if not dataset_id:
        report.add_error(f"Konnte dataset_id nicht aus Case lesen: {files[0]}")
        report.raise_if_errors()

    cases: list[tuple[str, Path]] = []
    for p in files:
        try:
            raw = _load_json(p)
        except SystemExit:
            continue
        ok_strict = bool(raw.get("validation", {}).get("ok_strict", False)) if isinstance(raw, dict) else False
        if not ok_strict:
            continue
        cases.append((p.stem, p.resolve()))
    if not cases:
        report.add_error(f"Keine validen (ok_strict) Case-JSONs gefunden in: {stage1_dir}")
        report.raise_if_errors()
    return dataset_id, cases


def _filter_manifest_cases_for_stage2(
    dataset_id: str,
    entries: list[dict[str, Any]],
    *,
    allow_nonstrict: bool,
    log_f: Any | None,
) -> list[tuple[str, Path]]:
    report = ValidationReport()
    out: list[tuple[str, Path]] = []
    for entry in entries:
        case_id = str(entry.get("case_id", "")).strip()
        if not case_id:
            continue

        strict_ok = bool(entry.get("validation_ok_strict", False))
        ok = bool(entry.get("validation_ok", False))
        case_path = entry.get("case_path")
        case_path_p = case_path if isinstance(case_path, Path) else (Path(str(case_path)) if case_path else None)

        if not allow_nonstrict:
            if not strict_ok:
                _log_print(f"[skip/validation_strict] {dataset_id}/{case_id}", log_f=log_f)
                continue
        else:
            if not ok:
                _log_print(f"[skip/validation_ok] {dataset_id}/{case_id}", log_f=log_f)
                continue

        if case_path_p is None:
            report.add_error(f"Case-Datei nicht gefunden für case_id={case_id} (Dataset: {dataset_id}).")
            report.raise_if_errors()
        assert case_path_p is not None
        exists_report = ValidationReport()
        require_path_exists(case_path_p, report=exists_report, level="full", kind="file", what=str(case_path_p))
        if exists_report.errors:
            report.add_error(f"Case-Datei nicht gefunden für case_id={case_id} (Dataset: {dataset_id}).")
            report.raise_if_errors()
        out.append((case_id, case_path_p))

    if not out:
        report.add_error(f"Keine gültigen Cases für Etappe 2 gefunden (Dataset: {dataset_id}).")
        report.raise_if_errors()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fullrun-Orchestrator: Etappe 01 (Instanzen/Szenarien) -> Etappe 02 (V1–V4) -> Etappe 03 (Evaluation).\n"
            "Hinweis: Dieses Skript orchestriert ausschließlich und ruft die bestehenden CLIs der Etappen auf "
            "(fachliche Logik siehe jeweils im Code; konzeptioneller Bezug: schriftliche Arbeit (separat versioniert), Kap. 5)."
        )
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help=(
            "Python-Interpreter für alle Subprozesse (Etappe 01–03). "
            "Default: aktueller Interpreter (sys.executable)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("fullrun_out"),
        help="Output-Root (enthält stage1_dataset/, stage2_runs/, stage3_evaluation/). Default: fullrun_out/.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional: Run-spezifischer Subfolder unter --out (z.B. seed_1001).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Fortsetzen (Resume): vorhandene Stage-/Case-Artefakte wiederverwenden und Schritte überspringen.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Nur Kommandos ausgeben; keine Ausführung.")

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run-Seed für Etappe 02 (pro Verfahren deterministisch abgeleitet). Default: (legacy) 20260217.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("etappe01_simulation/configs/praxisnah_v1.json"),
        help="Config für Etappe 01 (JSON; Datengrundlage & Szenarien).",
    )
    parser.add_argument(
        "--solver-config",
        type=Path,
        default=Path("etappe02_modelle/configs/run_dataset.final.json"),
        help="Run-Plan für Etappe 02 (JSON; Zuordnung Verfahren -> Config).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="V1,V2,V3,V4",
        help="all oder Komma-Liste: V1,V2,V3,V4 (default: alle).",
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=300.0,
        help="Wall-Clock-Gesamtbudget pro (Case, Verfahren) in Sekunden. Default: 300.",
    )

    parser.add_argument("--eval-group-by", type=str, default="size,severity", help="Etappe 03: group-by (default: size,severity).")
    parser.add_argument("--eval-format", type=str, default="pdf", choices=["pdf", "png"], help="Etappe 03: Plot-Format (pdf|png).")
    parser.add_argument(
        "--eval-set-mode",
        type=str,
        default="union",
        choices=["union", "per_group"],
        help="Etappe 03: set-mode (union|per_group). Default: union.",
    )
    args = parser.parse_args(argv)
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    def _flag_used(flag: str) -> bool:
        return any(a == flag or str(a).startswith(flag + "=") for a in raw_argv)

    legacy_mode = not (_flag_used("--seed") or _flag_used("--run-name") or _flag_used("--methods"))

    repo_root = Path(__file__).resolve().parent
    entrypoints = _detect_entrypoints(repo_root)

    python_exe = str(args.python_exe).strip()
    preflight = ValidationReport()
    if not python_exe:
        preflight.add_error("--python-exe ist leer.")
        preflight.raise_if_errors()
    python_exe_resolved = shutil.which(python_exe) or python_exe

    min_py = (3, 10, 0)
    if not args.dry_run:
        py_ver = _get_python_version(python_exe_resolved)
        if py_ver is None:
            raise SystemExit(
                "Konnte --python-exe nicht ausführen, um die Version zu prüfen: "
                f"{python_exe_resolved}\n"
                "Tipp: Nutze z.B. das Conda-Env aus environment.yml:\n"
                "  conda env create -f environment.yml\n"
                "  conda activate ba_pipeline\n"
                "  python fullrun.py ..."
            )
        if py_ver < min_py:
            raise SystemExit(
                "Python-Version zu alt für diese Pipeline.\n"
                f"Gefunden: {py_ver[0]}.{py_ver[1]}.{py_ver[2]} ({python_exe_resolved})\n"
                f"Benötigt: >= {min_py[0]}.{min_py[1]}.{min_py[2]} (Projekt-Stand; Etappe 2/3).\n"
                "Empfohlen (reproduzierbar): Conda-Env aus environment.yml:\n"
                "  conda env create -f environment.yml\n"
                "  conda activate ba_pipeline\n"
                "  python fullrun.py ...\n"
                "Alternativ: Starte mit einem neueren Python, z.B. python3.12."
            )

    base_out_root = Path(args.out).expanduser()
    run_name = str(args.run_name).strip() if args.run_name is not None else None
    if run_name == "":
        preflight.add_error("--run-name ist leer.")
        run_name = None
    if run_name is None and args.seed is not None:
        run_name = f"seed_{int(args.seed)}"
    if run_name is not None:
        p_run_name = Path(run_name)
        if len(p_run_name.parts) != 1 or p_run_name.name != run_name:
            preflight.add_error(f"--run-name muss ein einzelner Ordnername sein (ohne Pfadtrenner): {run_name!r}")
    out_root = base_out_root if run_name is None else (base_out_root / run_name)
    stage1_dir = (out_root / "stage1_dataset").resolve()
    stage2_dir = (out_root / "stage2_runs").resolve()
    stage3_dir = (out_root / "stage3_evaluation").resolve()
    fullrun_log_path = (out_root / "fullrun.log").resolve()
    fullrun_meta_path = (out_root / "metadata.json").resolve()

    methods_raw = str(args.methods or "").strip()
    if methods_raw.lower() == "all" or methods_raw == "":
        methods = list(ALL_METHODS)
    else:
        methods = [m.strip().upper() for m in _parse_csv_list(methods_raw)]
    if not methods:
        preflight.add_error("--methods ist leer.")
    allowed = set(ALL_METHODS)
    unknown = [m for m in methods if m not in allowed]
    if unknown:
        preflight.add_error(f"Unbekannte Verfahren in --methods: {unknown} (erwartet: {','.join(ALL_METHODS)})")

    dataset_cfg_path = Path(args.dataset_config)
    if not dataset_cfg_path.is_absolute():
        dataset_cfg_path = (repo_root / dataset_cfg_path).resolve()
    require_path_exists(
        dataset_cfg_path,
        report=preflight,
        level="full",
        kind="file",
        what=f"Dataset-Config existiert nicht: {dataset_cfg_path}",
    )

    solver_cfg_path = Path(args.solver_config)
    if not solver_cfg_path.is_absolute():
        solver_cfg_path = (repo_root / solver_cfg_path).resolve()
    require_path_exists(
        solver_cfg_path,
        report=preflight,
        level="full",
        kind="file",
        what=f"Solver-Config existiert nicht: {solver_cfg_path}",
    )
    preflight.raise_if_errors()

    dataset_cfg = _load_json(dataset_cfg_path)
    dataset_id_from_cfg = str(dataset_cfg.get("dataset_id", "")).strip() if isinstance(dataset_cfg, dict) else ""

    run_seed_cli = args.seed
    seed_mode = run_seed_cli is not None
    run_seed = int(run_seed_cli) if run_seed_cli is not None else int(LEGACY_SEED)
    stage1_seed = int(LEGACY_SEED)
    if run_seed_cli is None:
        args.seed = int(LEGACY_SEED)

    method_seeds: dict[str, int] = {}
    if seed_mode:
        for m in methods:
            method_seeds[m] = int(derive_seed(int(run_seed), m))

    solver_cfg_sha256: str | None = None
    try:
        solver_cfg_sha256 = hashlib.sha256(solver_cfg_path.read_bytes()).hexdigest()
    except Exception:
        solver_cfg_sha256 = None

    if args.dry_run:
        _log_print("=== Fullrun (DRY-RUN) ===", log_f=None)
        _log_print(f"Projekt: {repo_root}", log_f=None)
        _log_print(f"Out:  {out_root}", log_f=None)
        _log_print(f"Stages: {stage1_dir} | {stage2_dir} | {stage3_dir}", log_f=None)
        _log_print(f"Python (Subprozesse): {python_exe_resolved}", log_f=None)
        _log_print(f"Entry-Points: {entrypoints}", log_f=None)
        _log_print("", log_f=None)

        # Etappe 01
        _log_print("=== Stage 1/3: Dataset-Generierung ===", log_f=None)
        cmd_stage1 = [
            python_exe_resolved,
            "-u",
            "-m",
            entrypoints["stage1_module"],
            "--config",
            str(dataset_cfg_path),
            "--out",
            str(stage1_dir),
            "--seed",
            str(int(stage1_seed)),
        ]
        run_cmd(cmd_stage1, cwd=repo_root, log_f=None, dry_run=True)

        # Cases nur zeigen, wenn Manifest bereits existiert.
        manifest_path = stage1_dir / "manifest.json"
        dataset_id = dataset_id_from_cfg or "<unbekannt>"
        cases: list[tuple[str, Path]] = []
        if manifest_path.exists():
            dataset_id, manifest_entries = _load_cases_from_manifest(manifest_path, stage1_dir=stage1_dir)
            cases = _filter_manifest_cases_for_stage2(
                dataset_id,
                manifest_entries,
                allow_nonstrict=False,
                log_f=None,
            )

        stage2_dataset_dir = stage2_dir / dataset_id

        _log_print("", log_f=None)
        _log_print("=== Stage 2/3: Läufe (V1–V4) ===", log_f=None)
        _log_print(f"Solver-Config: {solver_cfg_path}", log_f=None)
        if cases:
            _log_print(f"Cases: {len(cases)} (aus {manifest_path})", log_f=None)
        else:
            _log_print("Cases: (noch unbekannt; wird nach Stage 1 aus manifest.json bestimmt)", log_f=None)

        proc_cfg_by_method = _load_solver_plan(solver_cfg_path, repo_root=repo_root)
        if cases:
            for i, (case_id, case_path) in enumerate(cases, start=1):
                _log_print(f"Case {i}/{len(cases)}: {case_id}", log_f=None)
                for m in methods:
                    cfg = proc_cfg_by_method.get(m)
                    if cfg is None:
                        raise SystemExit(f"Solver-Config enthält keine Config für {m}: {solver_cfg_path}")
                    cmd = [
                        python_exe_resolved,
                        "-u",
                        "-m",
                        entrypoints["stage2_module"],
                        "--case",
                        str(case_path),
                        "--verfahren",
                        m,
                        "--config",
                        str(cfg),
                        "--out",
                        str(stage2_dataset_dir),
                        "--seed",
                        str(int(method_seeds.get(m, int(run_seed)))),
                        "--budget",
                        str(float(args.budget_seconds)),
                    ]
                    if args.resume:
                        cmd.append("--resume")
                    run_cmd(cmd, cwd=repo_root, log_f=None, dry_run=True)
        else:
            _log_print(f"(Beispiel) Stage-2 Output-Root wäre: {stage2_dataset_dir}", log_f=None)

        _log_print("", log_f=None)
        _log_print("=== Stage 3/3: Evaluation ===", log_f=None)
        cmd_stage3 = [
            python_exe_resolved,
            "-u",
            "-m",
            entrypoints["stage3_module"],
            "--input",
            str(out_root),
            "--out",
            str(stage3_dir),
            "--methods",
            ",".join(methods),
            "--group-by",
            str(args.eval_group_by),
            "--format",
            str(args.eval_format),
            "--set-mode",
            str(args.eval_set_mode),
        ]
        run_cmd(cmd_stage3, cwd=repo_root, log_f=None, dry_run=True)
        return 0

    # --- Run -----------------------------------------------------------------
    out_root.mkdir(parents=True, exist_ok=True)
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)
    stage3_dir.mkdir(parents=True, exist_ok=True)

    cli_args = dict(vars(args))
    if legacy_mode:
        cli_args.pop("run_name", None)

    meta: dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "git_commit": _git_commit_hash(cwd=repo_root),
        "argv": list(sys.argv if argv is None else ["fullrun.py", *argv]),
        "cli_args": _json_safe(cli_args),
        "python_subprocess": {
            "python_exe": str(python_exe_resolved),
            "python_version": ".".join(str(x) for x in (_get_python_version(python_exe_resolved) or (None, None, None))),
        },
        "repo_root": str(repo_root),
        "out_root": str(out_root.resolve()),
        "paths": {
            "stage1_dataset": str(stage1_dir),
            "stage2_runs": str(stage2_dir),
            "stage3_evaluation": str(stage3_dir),
            "fullrun_log": str(fullrun_log_path),
        },
        "entrypoints": dict(entrypoints),
    }
    if not legacy_mode:
        meta["seed_mode"] = bool(seed_mode)
        meta["run_seed"] = int(run_seed_cli) if run_seed_cli is not None else None
        meta["run_seed_cli"] = int(run_seed_cli) if run_seed_cli is not None else None
        meta["run_seed_effective"] = int(run_seed)
        meta["run_name"] = run_name
        meta["methods"] = list(methods)
        meta["method_seeds"] = dict(method_seeds)
    fullrun_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with fullrun_log_path.open("w", encoding="utf-8") as log_f:
        io_lock = threading.Lock()

        _log_print("=== Fullrun ===", log_f=log_f, lock=io_lock)
        _log_print(f"Start (UTC): {meta['timestamp_utc']}", log_f=log_f, lock=io_lock)
        _log_print(f"Git-Commit: {meta.get('git_commit')}", log_f=log_f, lock=io_lock)
        _log_print(f"Out: {out_root.resolve()}", log_f=log_f, lock=io_lock)
        _log_print(f"Stages: {stage1_dir} | {stage2_dir} | {stage3_dir}", log_f=log_f, lock=io_lock)
        _log_print(f"Python (Subprozesse): {python_exe_resolved}", log_f=log_f, lock=io_lock)
        _log_print(f"Entry-Points: {entrypoints}", log_f=log_f, lock=io_lock)
        _log_print("", log_f=log_f, lock=io_lock)

        # --- Etappe 01 --------------------------------------------------------
        manifest_path = stage1_dir / "manifest.json"
        _log_print("=== Stage 1/3: Dataset-Generierung ===", log_f=log_f, lock=io_lock)
        if args.resume and manifest_path.exists():
            _log_print(f"[skip/resume] Stage 1 übersprungen (Manifest existiert): {manifest_path}", log_f=log_f, lock=io_lock)
        else:
            cmd_stage1 = [
                python_exe_resolved,
                "-u",
                "-m",
                entrypoints["stage1_module"],
                "--config",
                str(dataset_cfg_path),
                "--out",
                str(stage1_dir),
                "--seed",
                str(int(stage1_seed)),
            ]
            rc = run_cmd(cmd_stage1, cwd=repo_root, log_f=log_f, dry_run=False, lock=io_lock)
            if rc != 0:
                return rc

        if manifest_path.exists():
            dataset_id, manifest_entries = _load_cases_from_manifest(manifest_path, stage1_dir=stage1_dir)
            cases = _filter_manifest_cases_for_stage2(
                dataset_id,
                manifest_entries,
                allow_nonstrict=False,
                log_f=log_f,
            )
        else:
            dataset_id, cases = _load_cases_fallback(stage1_dir)

        if dataset_id_from_cfg and dataset_id and dataset_id_from_cfg != dataset_id:
            _log_print(
                f"[WARN] dataset_id mismatch: Config={dataset_id_from_cfg} vs. Manifest/Cases={dataset_id}",
                log_f=log_f,
                lock=io_lock,
            )

        meta["dataset_id"] = dataset_id
        meta["paths"]["stage1_manifest"] = str(manifest_path) if manifest_path.exists() else None
        fullrun_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # --- Etappe 02 --------------------------------------------------------
        _log_print("", log_f=log_f, lock=io_lock)
        _log_print("=== Stage 2/3: Läufe (V1–V4) ===", log_f=log_f, lock=io_lock)
        proc_cfg_by_method = _load_solver_plan(solver_cfg_path, repo_root=repo_root)
        stage2_dataset_dir = (stage2_dir / dataset_id).resolve()
        stage2_dataset_dir.mkdir(parents=True, exist_ok=True)

        total = len(cases)
        _log_print(f"[INFO] Stage 2: cases={total}, methods={','.join(methods)}", log_f=log_f, lock=io_lock)

        def _run_one_case(case_idx: int, case_id: str, case_path: Path) -> int:
            _log_print(f"Case {case_idx}/{total}: {case_id}", log_f=log_f, lock=io_lock)

            marker = stage2_dataset_dir / f".done__{case_id}"
            if args.resume and marker.exists():
                marker_ok = False
                try:
                    marker_data = json.loads(marker.read_text(encoding="utf-8"))
                except Exception:
                    marker_data = None

                if isinstance(marker_data, dict):
                    marker_methods_raw = marker_data.get("methods")
                    if isinstance(marker_methods_raw, list):
                        marker_methods = [str(x).strip().upper() for x in marker_methods_raw]
                    else:
                        marker_methods = None

                    marker_budget = marker_data.get("budget_seconds")
                    marker_budget_ok = False
                    try:
                        marker_budget_ok = abs(float(marker_budget) - float(args.budget_seconds)) <= 1e-9
                    except Exception:
                        marker_budget_ok = False

                    marker_seed_raw = marker_data.get("run_seed_effective")
                    if marker_seed_raw is None:
                        marker_seed_raw = marker_data.get("seed_global")
                    marker_seed_ok = False
                    try:
                        marker_seed_ok = int(marker_seed_raw) == int(run_seed)
                    except Exception:
                        marker_seed_ok = False

                    marker_solver_hash = marker_data.get("solver_config_sha256")
                    marker_solver_ok = True
                    if "solver_config_sha256" in marker_data:
                        marker_solver_ok = (solver_cfg_sha256 is not None) and (str(marker_solver_hash) == str(solver_cfg_sha256))

                    marker_method_seeds_raw = marker_data.get("method_seeds")
                    marker_method_seeds_ok = True
                    if seed_mode:
                        if not isinstance(marker_method_seeds_raw, dict):
                            marker_method_seeds_ok = False
                        else:
                            try:
                                marker_method_seeds = {str(k).strip().upper(): int(v) for k, v in marker_method_seeds_raw.items()}
                            except Exception:
                                marker_method_seeds = {}
                                marker_method_seeds_ok = False
                            if marker_method_seeds != dict(method_seeds):
                                marker_method_seeds_ok = False
                    else:
                        if marker_method_seeds_raw is not None:
                            marker_method_seeds_ok = False

                    marker_ok = (
                        marker_methods == list(methods)
                        and marker_budget_ok
                        and marker_seed_ok
                        and marker_solver_ok
                        and marker_method_seeds_ok
                    )

                if marker_ok:
                    _log_print(f"[skip/resume] {case_id} (Marker ok: {marker.name})", log_f=log_f, lock=io_lock)
                    return 0
                _log_print(f"[resume] Marker mismatch/invalid: {case_id} (ignoriere {marker.name})", log_f=log_f, lock=io_lock)
            for m in methods:
                cfg = proc_cfg_by_method.get(m)
                if cfg is None:
                    raise SystemExit(f"Solver-Config enthält keine Config für {m}: {solver_cfg_path}")

                seed_stage2 = int(method_seeds.get(m, int(run_seed)))

                _log_print(f"[{case_id}] - {m}: start", log_f=log_f, lock=io_lock)
                cmd = [
                    python_exe_resolved,
                    "-u",
                    "-m",
                    entrypoints["stage2_module"],
                    "--case",
                    str(case_path),
                    "--verfahren",
                    m,
                    "--config",
                    str(cfg),
                    "--out",
                    str(stage2_dataset_dir),
                    "--seed",
                    str(int(seed_stage2)),
                    "--budget",
                    str(float(args.budget_seconds)),
                ]
                if args.resume:
                    cmd.append("--resume")

                rc = run_cmd(cmd, cwd=repo_root, log_f=log_f, dry_run=False, prefix=None, lock=io_lock)
                if rc != 0:
                    return rc
                _log_print(f"[{case_id}] - {m}: done", log_f=log_f, lock=io_lock)

            if legacy_mode:
                marker_payload = {
                    "case_id": case_id,
                    "dataset_id": dataset_id,
                    "methods": list(methods),
                    "seed_global": int(args.seed),
                    "budget_seconds": float(args.budget_seconds),
                    "created_at_utc": _utc_now_iso(),
                }
            else:
                marker_payload = {
                    "case_id": case_id,
                    "dataset_id": dataset_id,
                    "methods": list(methods),
                    "seed_mode": bool(seed_mode),
                    "run_seed_cli": int(run_seed_cli) if run_seed_cli is not None else None,
                    "run_seed_effective": int(run_seed),
                    "seed_global": int(run_seed),
                    "solver_config_sha256": solver_cfg_sha256,
                    "budget_seconds": float(args.budget_seconds),
                    "created_at_utc": _utc_now_iso(),
                }
                if seed_mode:
                    marker_payload["method_seeds"] = dict(method_seeds)
            marker.write_text(json.dumps(marker_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return 0

        for idx, (case_id, case_path) in enumerate(cases, start=1):
            rc = _run_one_case(idx, case_id, case_path)
            if rc != 0:
                return rc

        # --- Etappe 03 --------------------------------------------------------
        _log_print("", log_f=log_f, lock=io_lock)
        _log_print("=== Stage 3/3: Evaluation ===", log_f=log_f, lock=io_lock)

        done_file = stage3_dir / ("per_instance_metrics.csv" if args.eval_set_mode == "union" else "per_group_metrics.csv")
        if args.resume and done_file.exists():
            _log_print(f"[skip/resume] Stage 3 übersprungen (Output existiert): {done_file}", log_f=log_f, lock=io_lock)
        else:
            cmd_stage3 = [
                python_exe_resolved,
                "-u",
                "-m",
                entrypoints["stage3_module"],
                "--input",
                str(out_root.resolve()),
                "--out",
                str(stage3_dir),
                "--methods",
                ",".join(methods),
                "--group-by",
                str(args.eval_group_by),
                "--format",
                str(args.eval_format),
                "--set-mode",
                str(args.eval_set_mode),
            ]
            rc = run_cmd(cmd_stage3, cwd=repo_root, log_f=log_f, dry_run=False, lock=io_lock)
            if rc != 0:
                return rc

        _log_print("", log_f=log_f, lock=io_lock)
        _log_print("=== Fullrun fertig ===", log_f=log_f, lock=io_lock)
        _log_print(f"Log: {fullrun_log_path}", log_f=log_f, lock=io_lock)
        _log_print(f"Metadata: {fullrun_meta_path}", log_f=log_f, lock=io_lock)
        _log_print(f"Stage 1: {stage1_dir}", log_f=log_f, lock=io_lock)
        _log_print(f"Stage 2: {stage2_dir / dataset_id}", log_f=log_f, lock=io_lock)
        _log_print(f"Stage 3: {stage3_dir}", log_f=log_f, lock=io_lock)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
