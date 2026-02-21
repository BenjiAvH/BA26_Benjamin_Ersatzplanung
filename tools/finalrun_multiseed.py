from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any


ALL_METHODS: tuple[str, str, str, str] = ("V1", "V2", "V3", "V4")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_commit_hash(*, cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True).strip() or None
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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_repo_path(path: Path, *, repo_root: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _is_repo_root(path: Path) -> bool:
    p = Path(path)
    return (
        (p / "fullrun.py").is_file()
        and (p / "etappe01_simulation").is_dir()
        and (p / "etappe02_modelle").is_dir()
        and (p / "etappe03_evaluation").is_dir()
    )


def _find_repo_root(start: Path) -> Path | None:
    p = Path(start).resolve()
    for cand in [p, *p.parents]:
        if _is_repo_root(cand):
            return cand
    return None


def _detect_repo_root(*, script_path: Path) -> Path:
    """
    Robuste Ermittlung des Repository-Wurzelverzeichnisses:
      (a) git rev-parse --show-toplevel (falls verfügbar)
      (b) Fallback: Elternverzeichnisse hochlaufen, bis Marker gefunden sind
      (c) Falls nicht gefunden: klare Fehlermeldung
    """

    # (a) git (zuerst im Script-Verzeichnis, dann in CWD)
    for git_cwd in [Path(script_path).resolve().parent, Path.cwd().resolve()]:
        try:
            top = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=str(git_cwd),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            top = ""
        if top:
            cand = Path(top).expanduser().resolve()
            if _is_repo_root(cand):
                return cand

    # (b) Marker-Fallback (Script-Location, dann CWD)
    for start in [Path(script_path).resolve().parent, Path.cwd().resolve()]:
        found = _find_repo_root(start)
        if found is not None:
            return found

    # (c) Abbruch mit Hinweis
    raise SystemExit(
        "Konnte Repo-Root nicht automatisch bestimmen.\n"
        f"- Script: {Path(script_path).resolve()}\n"
        f"- CWD:    {Path.cwd().resolve()}\n"
        "Erwartete Repo-Struktur im Root:\n"
        "  - fullrun.py\n"
        "  - etappe01_simulation/\n"
        "  - etappe02_modelle/\n"
        "  - etappe03_evaluation/\n"
        "Behebung:\n"
        "  - Führe das Script aus dem Repository aus (z.B. zuerst 'cd <repo_root>').\n"
        "  - Stelle sicher, dass tools/finalrun_multiseed.py im Repo liegt und die Ordnerstruktur nicht verändert wurde."
    )


def _format_cmd(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(x) for x in cmd])
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _prepend_pythonpath(env: dict[str, str], repo_root: Path) -> dict[str, str]:
    out = dict(env)
    existing = out.get("PYTHONPATH", "")
    root = str(repo_root)
    if not existing:
        out["PYTHONPATH"] = root
        return out
    parts = [p for p in existing.split(os.pathsep) if p]
    if root not in parts:
        out["PYTHONPATH"] = os.pathsep.join([root, *parts])
    return out


def _parse_csv_ints(value: str) -> list[int]:
    raw = [v.strip() for v in str(value or "").split(",") if v.strip()]
    if not raw:
        raise ValueError("Liste ist leer.")
    out: list[int] = []
    for item in raw:
        try:
            out.append(int(item))
        except Exception as e:
            raise ValueError(f"Ungültiger Integer in Liste: {item!r}") from e
    return out


def _parse_methods(value: str) -> list[str]:
    s = str(value or "").strip()
    if not s or s.lower() == "all":
        return list(ALL_METHODS)
    items = [v.strip().upper() for v in s.split(",") if v.strip()]
    if not items:
        raise ValueError("Liste ist leer.")
    unknown = [m for m in items if m not in set(ALL_METHODS)]
    if unknown:
        raise ValueError(f"Unbekannte Verfahren: {unknown} (erwartet: {','.join(ALL_METHODS)} oder 'all').")
    # Dedup (stable)
    seen: set[str] = set()
    out: list[str] = []
    for m in items:
        if m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


def _manifest_has_seed_global(manifest_path: Path, *, expected_seed: int) -> tuple[bool, str | None]:
    try:
        data = _read_json(manifest_path)
    except Exception as e:
        return False, f"manifest.json konnte nicht gelesen werden: {manifest_path} ({type(e).__name__}: {e})"
    if not isinstance(data, dict):
        return False, f"manifest.json ungültig (Top-Level muss Objekt sein): {manifest_path}"
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        return False, f"manifest.json ungültig (cases fehlt/leer): {manifest_path}"

    mismatches: list[str] = []
    for entry in cases:
        if not isinstance(entry, dict):
            continue
        case_id = str(entry.get("case_id", "")).strip() or "<unknown>"
        seeds = entry.get("seeds")
        if not isinstance(seeds, dict):
            mismatches.append(f"{case_id}: seeds fehlt/ungültig")
            continue
        g = seeds.get("global")
        try:
            g_int = int(g)
        except Exception:
            mismatches.append(f"{case_id}: seeds.global fehlt/ungültig ({g!r})")
            continue
        if g_int != int(expected_seed):
            mismatches.append(f"{case_id}: seeds.global={g_int} (erwartet {int(expected_seed)})")

    if mismatches:
        preview = "\n- ".join(mismatches[:8])
        more = "" if len(mismatches) <= 8 else f"\n- ... (+{len(mismatches) - 8} weitere)"
        return False, f"Dataset-Seed passt nicht zu {manifest_path}:\n- {preview}{more}"
    return True, None


def _move_to_backup(path: Path, *, backup_dir: Path, label: str, run_id: str) -> Path | None:
    if not path.exists():
        return None
    _ensure_dir(backup_dir)
    base = backup_dir / f"{label}__bak__{run_id}"
    dst = base
    k = 1
    while dst.exists():
        k += 1
        dst = Path(f"{base}__{k}")
    shutil.move(str(path), str(dst))
    return dst


@dataclass(frozen=True)
class CmdResult:
    rc: int
    cmd: list[str]
    log_path: Path


def _run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    extra_env: dict[str, str] | None,
) -> CmdResult:
    _ensure_dir(log_path.parent)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env = _prepend_pythonpath(env, cwd)
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})

    header = "\n".join(
        [
            "=== finalrun_multiseed subprocess ===",
            f"timestamp_utc: {_utc_now_iso()}",
            f"cwd: {cwd}",
            f"cmd: {_format_cmd(cmd)}",
            "",
        ]
    ).encode("utf-8", errors="replace")

    print(f"$ {_format_cmd(cmd)}", flush=True)
    print(f"  log: {log_path}", flush=True)

    with log_path.open("wb") as log_f:
        log_f.write(header)
        log_f.flush()

        cp = subprocess.run(
            [str(x) for x in cmd],
            cwd=str(cwd),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )

    rc = int(cp.returncode)
    if rc != 0:
        msg = f"[FEHLER] Exitcode {rc} (Details: {log_path})"
        print(msg, file=sys.stderr, flush=True)
    return CmdResult(rc=rc, cmd=cmd, log_path=log_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Finalrun Driver (Multi-Seed): Stage1 einmal (Dataset), Stage2 mehrfach (Seeds), Stage3 zweimal (union + per_group).\n"
            "Wichtig: Nutzt bestehende CLI-Entry-Points via 'python -m ...' und verändert keine Etappen-Module."
        )
    )
    p.add_argument("--out-root", type=Path, default=Path("fullrun_out"), help="Output-Root (default: fullrun_out).")
    p.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("etappe01_simulation/configs/praxisnah_v1.json"),
        help="Dataset-Config (default: etappe01_simulation/configs/praxisnah_v1.json).",
    )
    p.add_argument("--dataset-seed", type=int, default=20260219, help="Seed für Stage1 Dataset (default: 20260219).")
    p.add_argument(
        "--plan",
        type=Path,
        default=Path("etappe02_modelle/configs/run_dataset.final.json"),
        help="Stage2 Run-Plan (default: etappe02_modelle/configs/run_dataset.final.json).",
    )
    p.add_argument("--budget", type=int, default=600, help="Stage2 Budget pro (Case, Verfahren) in Sekunden (default: 600).")
    p.add_argument("--run-seeds", type=str, default="101,202,303", help="CSV-Liste globaler Seeds für Stage2 (default: 101,202,303).")
    p.add_argument("--methods", type=str, default="V1,V2,V3,V4", help="CSV-Liste für Stage3 (default: V1,V2,V3,V4).")
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stage2 Resume aktivieren (default: an). Überspringt Gruppen mit group_end(status=ok).",
    )
    p.add_argument("--no-plots", action="store_true", help="Stage3: keine Plots erzeugen.")
    p.add_argument("--no-latex", action="store_true", help="Stage3: keine LaTeX-Tabellen erzeugen.")
    p.add_argument("--skip-stage1", action="store_true", help="Stage1 überspringen (erfordert existierendes stage1_dataset/manifest.json).")
    p.add_argument("--skip-stage2", action="store_true", help="Stage2 überspringen.")
    p.add_argument("--skip-stage3", action="store_true", help="Stage3 überspringen.")
    p.add_argument(
        "--package-thesis-run",
        action="store_true",
        help="Erzeugt results/thesis_run/ via tools/thesis_artifacts.py all (inkl. stage1 manifest + stage2 zip/none).",
    )
    p.add_argument(
        "--thesis-run-dir",
        type=Path,
        default=Path("results/thesis_run"),
        help="Zielordner für Thesis-Artefakte (default: results/thesis_run).",
    )
    p.add_argument("--stage2-zip", type=str, default="zip", choices=["none", "zip"], help="Thesis-Export Stage2: none|zip (default: zip).")
    p.add_argument(
        "--eval-replicate-key",
        type=str,
        default="seed_global",
        choices=["auto", "seed_group", "seed_global"],
        help="Stage3 per_group: replicate-key (default: seed_global).",
    )
    p.add_argument(
        "--eval-set-mode-union",
        type=str,
        default="union",
        choices=["union", "per_group"],
        help="Stage3 (1): set-mode (default: union).",
    )
    p.add_argument(
        "--eval-set-mode-per-group",
        type=str,
        default="per_group",
        choices=["union", "per_group"],
        help="Stage3 (2): set-mode (default: per_group).",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Stage2: bei Seed-Fehlern fortsetzen (default: strict -> Abbruch).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    script_path = Path(__file__).resolve()
    repo_root = _detect_repo_root(script_path=script_path)
    out_root = Path(args.out_root).expanduser().resolve()

    run_id = _new_run_id()
    stage1_dir = out_root / "stage1_dataset"
    stage2_dir = out_root / "stage2_runs"
    stage3_union_dir = out_root / "stage3_evaluation_union"
    stage3_pg_dir = out_root / "stage3_evaluation_per_group"
    meta_path = out_root / "metadata.json"
    logs_dir = out_root / "driver_logs"
    backups_dir = logs_dir / "backups"

    dataset_cfg_path = _resolve_repo_path(Path(args.dataset_config), repo_root=repo_root)
    plan_path = _resolve_repo_path(Path(args.plan), repo_root=repo_root)
    thesis_run_dir = _resolve_repo_path(Path(args.thesis_run_dir), repo_root=repo_root)

    # Vorprüfung: CLI-Werte früh parsen und validieren
    try:
        run_seeds = _parse_csv_ints(str(args.run_seeds))
        methods = _parse_methods(str(args.methods))
    except ValueError as e:
        print(f"[FEHLER] CLI-Argument ungültig: {e}", file=sys.stderr)
        return 2

    if int(args.budget) <= 0:
        print("[FEHLER] --budget muss > 0 sein.", file=sys.stderr)
        return 2

    # Ausgabestruktur vorbereiten (Metadaten/Logs auch bei frühem Abbruch)
    try:
        _ensure_dir(out_root)
        _ensure_dir(logs_dir)
    except Exception as e:
        print(f"[FEHLER] Konnte Output-Ordner nicht erstellen: {out_root} ({type(e).__name__}: {e})", file=sys.stderr)
        return 2

    # Vorhandene metadata.json sichern (nicht still überschreiben)
    prev_meta: dict[str, Any] | None = None
    if meta_path.exists():
        try:
            prev_meta_raw = _read_json(meta_path)
            prev_meta = prev_meta_raw if isinstance(prev_meta_raw, dict) else None
        except Exception:
            prev_meta = None
        try:
            shutil.copy2(meta_path, logs_dir / f"metadata_prev__{run_id}.json")
        except Exception:
            pass

    # Schutzprüfung: verhindert das Vermischen mit anderen Treibern/Läufen im selben out_root
    if prev_meta is not None:
        prev_driver = prev_meta.get("driver")
        if prev_driver is not None and str(prev_driver) not in {"finalrun_multiseed"}:
            print(
                "[FEHLER] out_root enthält bereits metadata.json eines anderen Drivers.\n"
                f"- out_root: {out_root}\n"
                f"- driver: {prev_driver!r}\n"
                "Bitte wähle ein anderes --out-root oder lösche den Output-Ordner.",
                file=sys.stderr,
            )
            return 2

        # Critical parameters must match when resuming the same out_root.
        critical_mismatches: list[str] = []
        prev_dataset_seed = prev_meta.get("dataset_seed")
        if prev_dataset_seed is not None and int(prev_dataset_seed) != int(args.dataset_seed):
            critical_mismatches.append(f"dataset_seed: {prev_dataset_seed} != {int(args.dataset_seed)}")
        prev_budget = prev_meta.get("budget_seconds")
        if prev_budget is not None and int(prev_budget) != int(args.budget):
            critical_mismatches.append(f"budget_seconds: {prev_budget} != {int(args.budget)}")
        prev_dataset_cfg = prev_meta.get("dataset_config")
        if prev_dataset_cfg is not None and str(Path(str(prev_dataset_cfg))) != str(dataset_cfg_path):
            critical_mismatches.append(f"dataset_config: {prev_dataset_cfg} != {dataset_cfg_path}")
        prev_plan = prev_meta.get("plan")
        if prev_plan is not None and str(Path(str(prev_plan))) != str(plan_path):
            critical_mismatches.append(f"plan: {prev_plan} != {plan_path}")

        if critical_mismatches:
            print(
                "[FEHLER] out_root scheint zu einem anderen Run zu gehören (kritische Parameter weichen ab):\n- "
                + "\n- ".join(critical_mismatches)
                + "\nBitte wähle ein anderes --out-root oder räume den Ordner auf.",
                file=sys.stderr,
            )
            return 2

    meta: dict[str, Any] = {
        "driver": "finalrun_multiseed",
        "driver_version": 1,
        "run_id": run_id,
        "status": "running",
        "start_utc": _utc_now_iso(),
        "end_utc": None,
        "exit_code": None,
        "git_commit": _git_commit_hash(cwd=repo_root),
        "argv": list(sys.argv if argv is None else ["tools/finalrun_multiseed.py", *argv]),
        "repo_root": str(repo_root),
        "out_root": str(out_root),
        "paths": {
            "stage1_dataset": str(stage1_dir),
            "stage2_runs": str(stage2_dir),
            "stage3_evaluation_union": str(stage3_union_dir),
            "stage3_evaluation_per_group": str(stage3_pg_dir),
            "driver_logs": str(logs_dir),
        },
        "dataset_config": str(dataset_cfg_path),
        "dataset_seed": int(args.dataset_seed),
        "plan": str(plan_path),
        "budget_seconds": int(args.budget),
        "run_seeds": list(run_seeds),
        "methods": list(methods),
        "cli_args": _json_safe(vars(args)),
        "stages": {
            "stage1": {"status": "pending"},
            "stage2": {"status": "pending", "seeds": []},
            "stage3_union": {"status": "pending"},
            "stage3_per_group": {"status": "pending"},
            "thesis_artifacts": {"status": "pending" if bool(args.package_thesis_run) else "skipped"},
        },
        "logs": {},
    }
    _write_json(meta_path, meta)

    exit_code = 0
    had_stage2_failures = False
    try:
        # --- Existenzprüfungen der Eingaben (nachdem Metadaten geschrieben sind)
        if not dataset_cfg_path.exists():
            raise SystemExit(f"--dataset-config nicht gefunden: {dataset_cfg_path}")
        if not dataset_cfg_path.is_file():
            raise SystemExit(f"--dataset-config ist keine Datei: {dataset_cfg_path}")
        if not plan_path.exists():
            raise SystemExit(f"--plan nicht gefunden: {plan_path}")
        if not plan_path.is_file():
            raise SystemExit(f"--plan ist keine Datei: {plan_path}")

        # --- Etappe 01: Datensatz (genau einmal) ------------------------------
        manifest_path = stage1_dir / "manifest.json"
        if bool(args.skip_stage1):
            if not manifest_path.exists():
                raise SystemExit(f"--skip-stage1 gesetzt, aber Manifest fehlt: {manifest_path}")
            ok, err = _manifest_has_seed_global(manifest_path, expected_seed=int(args.dataset_seed))
            if not ok:
                raise SystemExit(err or f"Manifest-Check fehlgeschlagen: {manifest_path}")
            meta["stages"]["stage1"]["status"] = "skipped"
        else:
            if manifest_path.exists():
                ok, err = _manifest_has_seed_global(manifest_path, expected_seed=int(args.dataset_seed))
                if not ok:
                    raise SystemExit(err or f"Manifest-Check fehlgeschlagen: {manifest_path}")
                meta["stages"]["stage1"]["status"] = "skipped_existing"
            else:
                _ensure_dir(stage1_dir)
                log_stage1 = logs_dir / f"{run_id}__stage1.log"
                meta["logs"]["stage1"] = str(log_stage1)
                _write_json(meta_path, meta)

                cmd_stage1 = [
                    sys.executable,
                    "-u",
                    "-m",
                    "etappe01_simulation.scripts.generate_dataset",
                    "--config",
                    str(dataset_cfg_path),
                    "--out",
                    str(stage1_dir),
                    "--seed",
                    str(int(args.dataset_seed)),
                ]
                r = _run_cmd(cmd_stage1, cwd=repo_root, log_path=log_stage1, extra_env=None)
                if r.rc != 0:
                    meta["stages"]["stage1"]["status"] = "failed"
                    raise SystemExit(f"Stage 1 fehlgeschlagen (exit={r.rc}). Log: {r.log_path}")

                if not manifest_path.exists():
                    meta["stages"]["stage1"]["status"] = "failed"
                    raise SystemExit(f"Stage 1 fertig, aber Manifest fehlt: {manifest_path}")

                ok, err = _manifest_has_seed_global(manifest_path, expected_seed=int(args.dataset_seed))
                if not ok:
                    meta["stages"]["stage1"]["status"] = "failed"
                    raise SystemExit(err or f"Manifest-Check fehlgeschlagen: {manifest_path}")
                meta["stages"]["stage1"]["status"] = "ok"

        meta["paths"]["stage1_manifest"] = str(manifest_path)
        _write_json(meta_path, meta)

        # --- Etappe 02: Runs (Multi-Seed) -----------------------------------
        if bool(args.skip_stage2):
            meta["stages"]["stage2"]["status"] = "skipped"
            _write_json(meta_path, meta)
        else:
            _ensure_dir(stage2_dir)
            seed_rows: list[dict[str, Any]] = []
            for seed in run_seeds:
                log_stage2 = logs_dir / f"{run_id}__stage2_seed_{int(seed)}.log"
                cmd_stage2 = [
                    sys.executable,
                    "-u",
                    "-m",
                    "etappe02_modelle.scripts.run_dataset",
                    "--dataset",
                    str(manifest_path),
                    "--plan",
                    str(plan_path),
                    "--out",
                    str(stage2_dir),
                    "--seed",
                    str(int(seed)),
                    "--budget",
                    str(float(args.budget)),
                ]
                if bool(args.resume):
                    cmd_stage2.append("--resume")

                r = _run_cmd(cmd_stage2, cwd=repo_root, log_path=log_stage2, extra_env=None)
                row = {"seed": int(seed), "exit_code": int(r.rc), "log": str(r.log_path)}
                seed_rows.append(row)
                meta["stages"]["stage2"]["seeds"] = seed_rows
                _write_json(meta_path, meta)

                if r.rc != 0:
                    had_stage2_failures = True
                    if not bool(args.continue_on_error):
                        meta["stages"]["stage2"]["status"] = "failed"
                        raise SystemExit(f"Stage 2 fehlgeschlagen für seed={seed} (exit={r.rc}). Log: {r.log_path}")

            meta["stages"]["stage2"]["status"] = "ok" if not had_stage2_failures else "partial"
            _write_json(meta_path, meta)

        # --- Etappe 03: Evaluation (union + per_group) ----------------------
        if bool(args.skip_stage3):
            meta["stages"]["stage3_union"]["status"] = "skipped"
            meta["stages"]["stage3_per_group"]["status"] = "skipped"
            _write_json(meta_path, meta)
        else:
            methods_str = ",".join(methods)

            # No silent overwrite: move old outputs to backups/
            _move_to_backup(stage3_union_dir, backup_dir=backups_dir, label="stage3_evaluation_union", run_id=run_id)
            _move_to_backup(stage3_pg_dir, backup_dir=backups_dir, label="stage3_evaluation_per_group", run_id=run_id)

            # union
            log_stage3_u = logs_dir / f"{run_id}__stage3_union.log"
            meta["logs"]["stage3_union"] = str(log_stage3_u)
            _write_json(meta_path, meta)
            cmd_stage3_u = [
                sys.executable,
                "-u",
                "-m",
                "etappe03_evaluation",
                "--input",
                str(out_root),
                "--out",
                str(stage3_union_dir),
                "--methods",
                methods_str,
                "--set-mode",
                str(args.eval_set_mode_union),
            ]
            if str(args.eval_set_mode_union).strip().lower() == "per_group":
                cmd_stage3_u.extend(["--replicate-key", str(args.eval_replicate_key)])
            if bool(args.no_plots):
                cmd_stage3_u.append("--no-plots")
            if bool(args.no_latex):
                cmd_stage3_u.append("--no-latex")
            r_u = _run_cmd(cmd_stage3_u, cwd=repo_root, log_path=log_stage3_u, extra_env=None)
            if r_u.rc != 0:
                meta["stages"]["stage3_union"]["status"] = "failed"
                raise SystemExit(f"Stage 3 (union) fehlgeschlagen (exit={r_u.rc}). Log: {r_u.log_path}")
            meta["stages"]["stage3_union"]["status"] = "ok"
            _write_json(meta_path, meta)

            # per_group
            log_stage3_pg = logs_dir / f"{run_id}__stage3_per_group.log"
            meta["logs"]["stage3_per_group"] = str(log_stage3_pg)
            _write_json(meta_path, meta)
            cmd_stage3_pg = [
                sys.executable,
                "-u",
                "-m",
                "etappe03_evaluation",
                "--input",
                str(out_root),
                "--out",
                str(stage3_pg_dir),
                "--methods",
                methods_str,
                "--set-mode",
                str(args.eval_set_mode_per_group),
            ]
            if str(args.eval_set_mode_per_group).strip().lower() == "per_group":
                cmd_stage3_pg.extend(["--replicate-key", str(args.eval_replicate_key)])
            if bool(args.no_plots):
                cmd_stage3_pg.append("--no-plots")
            if bool(args.no_latex):
                cmd_stage3_pg.append("--no-latex")
            r_pg = _run_cmd(cmd_stage3_pg, cwd=repo_root, log_path=log_stage3_pg, extra_env=None)
            if r_pg.rc != 0:
                meta["stages"]["stage3_per_group"]["status"] = "failed"
                raise SystemExit(f"Stage 3 (per_group) fehlgeschlagen (exit={r_pg.rc}). Log: {r_pg.log_path}")
            meta["stages"]["stage3_per_group"]["status"] = "ok"
            _write_json(meta_path, meta)

        # --- Thesis artifacts (optional) ------------------------------------
        if bool(args.package_thesis_run):
            stage3_alias_dir = out_root / "stage3_evaluation"
            if stage3_union_dir.exists() and stage3_union_dir.is_dir():
                _move_to_backup(stage3_alias_dir, backup_dir=backups_dir, label="stage3_evaluation_alias", run_id=run_id)
                shutil.copytree(stage3_union_dir, stage3_alias_dir)
            else:
                raise SystemExit(f"Thesis-Export benötigt Stage3-Union Output: {stage3_union_dir}")

            log_thesis = logs_dir / f"{run_id}__thesis_artifacts.log"
            meta["logs"]["thesis_artifacts"] = str(log_thesis)
            _write_json(meta_path, meta)
            cmd_thesis = [
                sys.executable,
                "-u",
                str(repo_root / "tools" / "thesis_artifacts.py"),
                "all",
                "--src",
                str(out_root),
                "--dst",
                str(thesis_run_dir),
                "--clean",
                "--include-stage1-manifest",
                "--stage2",
                str(args.stage2_zip),
            ]
            r_t = _run_cmd(cmd_thesis, cwd=repo_root, log_path=log_thesis, extra_env=None)
            if r_t.rc != 0:
                meta["stages"]["thesis_artifacts"]["status"] = "failed"
                raise SystemExit(f"Thesis-Export fehlgeschlagen (exit={r_t.rc}). Log: {r_t.log_path}")

            # Optional: per_group-Ausgabe zusätzlich in thesis_run/ ablegen (ohne thesis_artifacts.py umzubauen)
            if stage3_pg_dir.exists() and stage3_pg_dir.is_dir():
                dst_pg = thesis_run_dir / "stage3_evaluation_per_group"
                shutil.copytree(stage3_pg_dir, dst_pg, dirs_exist_ok=True)

                # Nachkopierte Artefakte ebenfalls sanitizen (all hat bereits sanitize ausgeführt).
                log_thesis_san = logs_dir / f"{run_id}__thesis_artifacts_sanitize_after_per_group.log"
                meta["logs"]["thesis_artifacts_sanitize_after_per_group"] = str(log_thesis_san)
                _write_json(meta_path, meta)
                cmd_thesis_san = [
                    sys.executable,
                    "-u",
                    str(repo_root / "tools" / "thesis_artifacts.py"),
                    "sanitize",
                    "--dst",
                    str(thesis_run_dir),
                ]
                r_san = _run_cmd(cmd_thesis_san, cwd=repo_root, log_path=log_thesis_san, extra_env=None)
                if r_san.rc != 0:
                    meta["stages"]["thesis_artifacts"]["status"] = "failed"
                    raise SystemExit(f"Thesis-Export sanitize fehlgeschlagen (exit={r_san.rc}). Log: {r_san.log_path}")

            # Danach: Scan (Schutzprüfung) – immer ausführen; muss 0 Treffer liefern, sonst Abbruch.
            log_thesis_scan = logs_dir / f"{run_id}__thesis_artifacts_scan.log"
            meta["logs"]["thesis_artifacts_scan"] = str(log_thesis_scan)
            _write_json(meta_path, meta)
            cmd_thesis_scan = [
                sys.executable,
                "-u",
                str(repo_root / "tools" / "thesis_artifacts.py"),
                "scan",
                "--dst",
                str(thesis_run_dir),
            ]
            r_scan = _run_cmd(cmd_thesis_scan, cwd=repo_root, log_path=log_thesis_scan, extra_env=None)
            if r_scan.rc != 0:
                meta["stages"]["thesis_artifacts"]["status"] = "failed"
                raise SystemExit(f"Thesis-Export scan fehlgeschlagen (exit={r_scan.rc}). Log: {r_scan.log_path}")

            meta["stages"]["thesis_artifacts"]["status"] = "ok"
            _write_json(meta_path, meta)

        exit_code = 0 if not had_stage2_failures else 2
        meta["status"] = "success" if exit_code == 0 else "partial"
        meta["exit_code"] = int(exit_code)
        return int(exit_code)

    except SystemExit as e:
        exit_code = 2
        msg = str(e)
        if msg:
            print(f"[FEHLER] {msg}", file=sys.stderr)
        meta["status"] = "failed"
        meta["exit_code"] = int(exit_code)
        meta["error"] = msg
        return int(exit_code)
    except Exception as e:
        exit_code = 2
        print(f"[FEHLER] Unbehandelter Fehler: {type(e).__name__}: {e}", file=sys.stderr)
        meta["status"] = "failed"
        meta["exit_code"] = int(exit_code)
        meta["error"] = {"type": type(e).__name__, "message": str(e)}
        return int(exit_code)
    finally:
        meta["end_utc"] = _utc_now_iso()
        meta["exit_code"] = int(exit_code)
        _write_json(meta_path, meta)


if __name__ == "__main__":
    raise SystemExit(main())
