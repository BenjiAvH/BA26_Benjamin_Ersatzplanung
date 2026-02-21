"""
tools/test.py — Ende-zu-Ende-Kurztest (Vorabprüfung)

Ziel
- Vor einem vollständigen Lauf wird schnell und reproduzierbar geprüft, ob die Pipeline in den Etappen 01–03
  grundsätzlich ausführbar ist:
  * Importe und JSON-Konfigurationen sind lesbar.
  * SCIP/PySCIPOpt ist verfügbar (für MILP-Verfahren V1/V2).
  * Etappe 01: Datensatzgenerierung funktioniert; `manifest.json` ist plausibel.
  * Etappe 02: `run_case` (V1–V4) läuft auf genau einem Fall ohne unerwarteten Abbruch.
  * Etappe 02: Pfadkollisionstest für `run_dataset` (das Arbeitsverzeichnis darf nicht fälschlich bevorzugt werden).
  * Treiber-Skript `tools/finalrun_multiseed.py` inkl. Etappe 03 (Vereinigung und pro Gruppe) sowie Paketierung für den
    schriftlichen Teil der Bachelorarbeit.

Ausführung im Repository-Hauptverzeichnis
    python3 tools/test.py

Optionen
- Siehe `python3 tools/test.py -h` (Seed-Werte, Budgets in Sekunden, Verfahren).

Ausgaben
- Alle Dateien werden unter `<repo_hauptverzeichnis>/_preflight/<LAUF_ID>/` abgelegt (Protokolle und Kurztest-Ausgaben).
  Vorhandene Ergebnisordner werden dabei nicht verändert.

Hinweis
- Die Vorabprüfung ist absichtlich defensiv: Für Etappe 02 werden auch Zeitlimit oder Unlösbarkeit als unkritisch
  akzeptiert, solange kein unerwarteter Abbruch auftritt (kein Status `error`, keine unbehandelte Ausnahme).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


# --------------------------- Hilfsfunktionen --------------------------------


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def is_repo_root(p: Path) -> bool:
    return (
        (p / "fullrun.py").is_file()
        and (p / "etappe01_simulation").is_dir()
        and (p / "etappe02_modelle").is_dir()
        and (p / "etappe03_evaluation").is_dir()
    )


def detect_repo_root() -> Path:
    # 1) `git rev-parse --show-toplevel`
    for start in [Path(__file__).resolve().parent, Path.cwd().resolve()]:
        try:
            top = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=str(start),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            top = ""
        if top:
            cand = Path(top).resolve()
            if is_repo_root(cand):
                return cand

    # 2) Elternverzeichnisse aufwärts prüfen
    for start in [Path(__file__).resolve().parent, Path.cwd().resolve()]:
        p = start
        for cand in [p, *p.parents]:
            if is_repo_root(cand):
                return cand.resolve()

    raise SystemExit(
        "Repository-Hauptverzeichnis nicht gefunden. Bitte aus dem Repository starten (da wo fullrun.py liegt)."
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_text(p: Path, s: str) -> None:
    ensure_dir(p.parent)
    p.write_text(s, encoding="utf-8")


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, data: Any) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepend_pythonpath(env: dict[str, str], repo_root: Path) -> dict[str, str]:
    out = dict(env)
    root = str(repo_root)
    existing = out.get("PYTHONPATH", "")
    if not existing:
        out["PYTHONPATH"] = root
        return out
    parts = [p for p in existing.split(os.pathsep) if p]
    if root not in parts:
        out["PYTHONPATH"] = os.pathsep.join([root, *parts])
    return out


@dataclass
class CmdResult:
    rc: int
    cmd: list[str]
    cwd: Path
    log_path: Path


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    repo_root: Path,
    log_path: Path,
    env_extra: dict[str, str] | None = None,
) -> CmdResult:
    ensure_dir(log_path.parent)
    env = prepend_pythonpath(os.environ.copy(), repo_root)
    if env_extra:
        env.update(env_extra)

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"Arbeitsverzeichnis: {cwd}\n")
        f.write("Befehl: " + " ".join(map(str, cmd)) + "\n\n")
        f.flush()
        cp = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
            text=False,  # Ausgabe wird bereits in eine Datei umgeleitet
        )
    return CmdResult(rc=int(cp.returncode), cmd=cmd, cwd=cwd, log_path=log_path)


def print_hr(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARNUNG] {msg}")


def fail(msg: str) -> None:
    print(f"[FEHLER] {msg}")


def load_json_safe(path: Path) -> tuple[bool, Any | None, str | None]:
    try:
        return True, read_json(path), None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def find_first_case(stage1_dir: Path) -> Path | None:
    # Bevorzuge sc-small, ansonsten erste *.json (außer manifest.json)
    candidates = sorted([p for p in stage1_dir.glob("*.json") if p.name != "manifest.json"])
    for p in candidates:
        if "sc-small" in p.name:
            return p
    return candidates[0] if candidates else None


def parse_group_end_status(runs_jsonl: Path) -> str | None:
    """
    runs.jsonl ist "flach": `RunEvent.to_dict()` legt Felder aus einem Unterobjekt direkt auf der obersten Ebene ab.
    Daher steht `group_end.status` in `rec["status"]` (und nicht verschachtelt in einem Unterobjekt).
    """
    if not runs_jsonl.exists():
        return None
    last: str | None = None
    try:
        for line in runs_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("event") == "group_end" and "status" in rec:
                last = str(rec.get("status"))
    except Exception:
        return None
    return last


# ------------------------------ Prüfungen -----------------------------------


@dataclass
class Step:
    name: str
    fn: Callable[[], bool]


class Preflight:
    def __init__(self, *, repo_root: Path, run_dir: Path, args: argparse.Namespace):
        self.repo_root = repo_root
        self.run_dir = run_dir
        self.args = args

        self.logs_dir = run_dir / "logs"
        self.out_root = run_dir / "smoke_out"  # Struktur analog zu finalrun_multiseed
        self.stage1_dir = self.out_root / "stage1_dataset"
        self.stage2_single_dir = self.out_root / "stage2_single"
        self.stage2_cwdtest_dir = self.out_root / "stage2_cwdtest"
        self.plan_smoke_path = self.out_root / "plan_smoke.json"
        self.thesis_run_dir = run_dir / "thesis_run_smoke"

        self.dataset_cfg = (repo_root / "etappe01_simulation" / "configs" / "praxisnah_v1.json").resolve()
        self.plan_full = (repo_root / "etappe02_modelle" / "configs" / "run_dataset.final.json").resolve()

        self.cfgs = {
            "V1": (repo_root / "etappe02_modelle" / "configs" / "v1_ws_milp.final.json").resolve(),
            "V2": (repo_root / "etappe02_modelle" / "configs" / "v2_eps_milp.final.json").resolve(),
            "V3": (repo_root / "etappe02_modelle" / "configs" / "v3_vnd_vns.final.json").resolve(),
            "V4": (repo_root / "etappe02_modelle" / "configs" / "v4_upgh.final.json").resolve(),
        }

        ensure_dir(self.logs_dir)
        ensure_dir(self.out_root)

        self.scip_ok: bool = False

    def step_00_environment(self) -> bool:
        print_hr("00) Umgebung und Grundlagen")
        ok(f"Python: {sys.version.split()[0]} ({sys.executable})")
        ok(f"Betriebssystem: {os.name} / Plattform={sys.platform}")
        ok(f"Repository-Hauptverzeichnis: {self.repo_root}")
        ok(f"Laufordner: {self.run_dir}")

        # Git-Informationen (optional)
        try:
            h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(self.repo_root), text=True, stderr=subprocess.DEVNULL).strip()
            ok(f"git HEAD: {h}")
        except Exception:
            warn("git HEAD nicht lesbar (git fehlt oder das Repository wird nicht als git erkannt).")

        # JSON-Plausibilitätsprüfung
        paths = [self.dataset_cfg, self.plan_full, *self.cfgs.values()]
        all_ok = True
        for p in paths:
            if not p.exists():
                fail(f"Fehlt: {p}")
                all_ok = False
                continue
            j_ok, _, err = load_json_safe(p)
            if not j_ok:
                fail(f"Ungültiges JSON: {p} ({err})")
                all_ok = False
        if all_ok:
            ok("Alle relevanten JSON-Konfigurationen lassen sich einlesen.")
        return all_ok

    def step_01_imports(self) -> bool:
        print_hr("01) Python-Importe")
        sys.path.insert(0, str(self.repo_root))
        try:
            import etappe01_simulation  # noqa: F401
            import etappe02_modelle  # noqa: F401
            import etappe03_evaluation  # noqa: F401
        except Exception as e:
            fail(f"Import-Fehler: {type(e).__name__}: {e}")
            return False
        ok("Importe: etappe01_simulation / etappe02_modelle / etappe03_evaluation erfolgreich")
        return True

    def step_02_check_scip(self) -> bool:
        print_hr("02) SCIP/PySCIPOpt-Prüfung (für V1/V2)")
        log = self.logs_dir / "check_scip.log"
        r = run_cmd(
            [sys.executable, "-u", "-m", "etappe02_modelle.scripts.check_scip"],
            cwd=self.repo_root,
            repo_root=self.repo_root,
            log_path=log,
        )
        if r.rc != 0:
            warn(f"check_scip Rückgabecode={r.rc} (Details: {log})")
            self.scip_ok = False
            # Nicht als Fehler werten, da V3/V4 ohne SCIP laufen sollen.
            return True

        txt = log.read_text(encoding="utf-8", errors="ignore")
        if "OK" in txt.upper():
            ok(f"SCIP/PySCIPOpt: OK (Protokoll: {log})")
            self.scip_ok = True
            return True

        warn(f"check_scip wurde ausgeführt, aber 'OK' nicht gefunden (Protokoll: {log})")
        self.scip_ok = False
        return True

    def step_03_stage1_generate_dataset(self) -> bool:
        print_hr("03) Etappe 01 — Datensatz generieren und Manifest prüfen")
        # Ausgabeordner neu erstellen
        if self.stage1_dir.exists():
            shutil.rmtree(self.stage1_dir)
        ensure_dir(self.stage1_dir)

        log = self.logs_dir / "stage1_generate_dataset.log"
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "etappe01_simulation.scripts.generate_dataset",
            "--config",
            str(self.dataset_cfg),
            "--out",
            str(self.stage1_dir),
            "--seed",
            str(int(self.args.dataset_seed)),
            "--overwrite",
        ]
        r = run_cmd(cmd, cwd=self.repo_root, repo_root=self.repo_root, log_path=log)
        if r.rc != 0:
            fail(f"Etappe 01 Rückgabecode={r.rc} (Protokoll: {log})")
            return False

        manifest = self.stage1_dir / "manifest.json"
        if not manifest.exists():
            fail(f"Manifest fehlt: {manifest}")
            return False

        data = read_json(manifest)
        cases = data.get("cases", [])
        if not isinstance(cases, list) or not cases:
            fail("manifest.json: cases ist leer/ungültig.")
            return False

        strict_bad = [c for c in cases if isinstance(c, dict) and not bool(c.get("validation_ok_strict", False))]
        status_bad = [c for c in cases if isinstance(c, dict) and str(c.get("status")) not in {"written", "exists"}]

        ok(f"Fälle im Manifest: {len(cases)}")
        if strict_bad:
            # Das ist ungewöhnlich; Etappe 02 würde diese Fälle überspringen.
            warn(f"{len(strict_bad)} Fälle haben validation_ok_strict=False (Etappe 02 würde sie überspringen).")
            preview = [str(c.get("case_id")) for c in strict_bad[:6]]
            warn(f"Beispiele: {preview}")
        if status_bad:
            warn(f"{len(status_bad)} Fälle haben einen Status außerhalb der erwarteten Werte.")
            preview = [(str(c.get("case_id")), str(c.get("status"))) for c in status_bad[:6]]
            warn(f"Beispiele: {preview}")

        # Vollständiges Manifest sichern und anschließend ein Mini-Manifest für die schnelle Etappe 02 und den Treiberlauf erstellen
        manifest_full = self.stage1_dir / "manifest_full.json"
        shutil.copy2(manifest, manifest_full)

        # Einen plausiblen Fall auswählen
        good = [c for c in cases if isinstance(c, dict) and bool(c.get("validation_ok_strict", False))]
        if not good:
            fail("Kein Fall mit validation_ok_strict=True im Manifest gefunden.")
            return False
        one = good[0]
        mini = {"dataset_id": data.get("dataset_id"), "cases": [one]}
        write_json(manifest, mini)
        ok(f"Manifest für den Kurztest auf 1 Fall reduziert: {one.get('case_id')} (Sicherung: {manifest_full.name})")

        return True

    def step_04_make_smoke_plan(self) -> bool:
        print_hr("04) Kurztest-Plan erstellen (nur schnelle Verfahren)")
        ok_full, plan, err = load_json_safe(self.plan_full)
        if not ok_full or not isinstance(plan, dict):
            fail(f"Laufplan nicht lesbar: {self.plan_full} ({err})")
            return False

        procedures = plan.get("procedures")
        if not isinstance(procedures, list) or not procedures:
            fail("Laufplan: procedures fehlt oder ist leer.")
            return False

        keep = [m.strip().upper() for m in self.args.smoke_methods.split(",") if m.strip()]
        keep_set = set(keep)
        new_procs: list[dict[str, Any]] = []
        for item in procedures:
            if not isinstance(item, dict):
                continue
            v = str(item.get("verfahren", "")).upper()
            if v in keep_set:
                new_procs.append(item)

        if not new_procs:
            fail(f"Kurztest-Plan wäre leer (Verfahren={keep}).")
            return False

        smoke = dict(plan)
        smoke["procedures"] = new_procs
        write_json(self.plan_smoke_path, smoke)
        ok(f"Kurztest-Plan geschrieben: {self.plan_smoke_path} (Verfahren={keep})")
        return True

    def step_05_stage2_single_case_all_methods(self) -> bool:
        print_hr("05) Etappe 02 — run_case auf 1 Fall (V1–V4) / Abbruchprüfung")
        case_path = find_first_case(self.stage1_dir)
        if case_path is None or not case_path.exists():
            fail(f"Keine Fall-JSON-Datei in {self.stage1_dir} gefunden.")
            return False
        ok(f"Fall: {case_path.name}")

        # Ausgabeordner bereinigen
        if self.stage2_single_dir.exists():
            shutil.rmtree(self.stage2_single_dir)
        ensure_dir(self.stage2_single_dir)

        methods_to_try: list[str] = ["V3", "V4"]
        if not self.args.skip_milp and self.scip_ok:
            methods_to_try = ["V1", "V2", "V3", "V4"]
        elif not self.args.skip_milp and not self.scip_ok:
            warn("SCIP/PySCIPOpt nicht verfügbar -> V1/V2 werden übersprungen (V3/V4 werden geprüft).")

        all_ok = True
        for m in methods_to_try:
            budget = float(self.args.budget_case_milp if m in {"V1", "V2"} else self.args.budget_case_heur)
            log = self.logs_dir / f"stage2_run_case_{m}.log"
            cmd = [
                sys.executable, "-u", "-m", "etappe02_modelle.scripts.run_case",
                "--case", str(case_path),
                "--verfahren", m,
                "--config", str(self.cfgs[m]),
                "--out", str(self.stage2_single_dir),
                "--seed", str(int(self.args.stage2_seed)),
                "--budget", str(budget),
            ]
            r = run_cmd(cmd, cwd=self.repo_root, repo_root=self.repo_root, log_path=log)

            # Rückgabecode 0 (normal) oder 2 (Zeitlimit/Unlösbarkeit) akzeptieren, sofern kein unerwarteter Abbruch vorliegt.
            if r.rc not in {0, 2}:
                fail(f"{m}: Rückgabecode={r.rc} (Protokoll: {log})")
                all_ok = False
                continue

            runs = self.stage2_single_dir / case_path.stem / m / "runs.jsonl"
            status = parse_group_end_status(runs)
            if status is None:
                fail(f"{m}: runs.jsonl fehlt oder ist nicht auswertbar ({runs})")
                all_ok = False
                continue
            if status == "error":
                fail(f"{m}: Status am Gruppenende ist 'error' (Protokoll: {log})")
                all_ok = False
                continue

            ok(f"{m}: Rückgabecode={r.rc}, Status am Gruppenende={status} (Protokoll: {log.name})")

        return all_ok

    def step_06_run_dataset_cwd_collision_test(self) -> bool:
        print_hr("06) run_dataset: Arbeitsverzeichnis-Kollisionstest (darf Arbeitsverzeichnis nicht bevorzugen)")
        # `run_dataset` aus einem temporären Arbeitsverzeichnis starten, das einen absichtlich ungültigen Konfigurationspfad enthält.
        # Wenn die Pfadauflösung fälschlich das Arbeitsverzeichnis bevorzugt, muss der Aufruf hart fehlschlagen (rc nicht in {0,2}).
        fake_root = Path(tempfile.mkdtemp(prefix="ba_cwd_collision_"))
        try:
            # Fake-Pfad anlegen: etappe02_modelle/configs/v3_vnd_vns.final.json mit absichtlich ungültigem JSON
            rel = Path("etappe02_modelle") / "configs" / "v3_vnd_vns.final.json"
            fake_cfg = fake_root / rel
            ensure_dir(fake_cfg.parent)
            write_text(fake_cfg, "{ dies ist kein gültiges json !!!")

            manifest = self.stage1_dir / "manifest.json"  # reduziert (1 Fall)
            if not manifest.exists():
                fail(f"manifest fehlt: {manifest}")
                return False

            if self.stage2_cwdtest_dir.exists():
                shutil.rmtree(self.stage2_cwdtest_dir)
            ensure_dir(self.stage2_cwdtest_dir)

            log = self.logs_dir / "stage2_run_dataset_cwd_collision.log"
            cmd = [
                sys.executable, "-u", "-m", "etappe02_modelle.scripts.run_dataset",
                "--dataset", str(manifest.resolve()),
                "--plan", str(self.plan_smoke_path.resolve()),
                "--out", str(self.stage2_cwdtest_dir.resolve()),
                "--seed", str(int(self.args.stage2_seed)),
                "--budget", str(float(self.args.budget_dataset)),
                "--resume",
            ]

            r = run_cmd(cmd, cwd=fake_root, repo_root=self.repo_root, log_path=log)

            # Bestanden, wenn der Lauf normal endet (rc 0 oder 2). Nicht bestanden, wenn ein unerwarteter Abbruch auftritt (sonstiger rc).
            if r.rc in {0, 2}:
                ok(f"run_dataset im temporären Arbeitsverzeichnis wurde erfolgreich ausgeführt (Rückgabecode={r.rc}); Pfadauflösung bevorzugt Basisverzeichnisse (Protokoll: {log.name})")
                return True

            fail(f"run_dataset im temporären Arbeitsverzeichnis unerwartet abgebrochen (Rückgabecode={r.rc}); Verdacht: Arbeitsverzeichnis wurde bevorzugt (Protokoll: {log})")
            return False
        finally:
            try:
                shutil.rmtree(fake_root)
            except Exception:
                pass

    def step_07_driver_end_to_end(self) -> bool:
        print_hr("07) Ende-zu-Ende-Treiberlauf (finalrun_multiseed) und Paketierung für die Bachelorarbeit")
        # Vorhandenen Datensatz aus Etappe 01 verwenden (bereits erzeugt) und Etappe 01 im Treiber überspringen.
        driver = self.repo_root / "tools" / "finalrun_multiseed.py"
        if not driver.exists():
            fail(f"Treiber-Skript fehlt: {driver}")
            return False

        # Etappe-02/Etappe-03-Ausgaben innerhalb von out_root bereinigen, um Verwechslungen zu vermeiden
        for sub in ["stage2_runs", "stage3_evaluation_union", "stage3_evaluation_per_group", "driver_logs", "metadata.json", "stage3_evaluation"]:
            p = self.out_root / sub
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

        if self.thesis_run_dir.exists():
            shutil.rmtree(self.thesis_run_dir)

        log = self.logs_dir / "driver_smoke.log"
        cmd = [
            sys.executable, "-u", str(driver),
            "--out-root", str(self.out_root),
            "--dataset-config", str(self.dataset_cfg),
            "--dataset-seed", str(int(self.args.dataset_seed)),
            "--plan", str(self.plan_smoke_path),
            "--budget", str(int(self.args.budget_driver)),
            "--run-seeds", str(int(self.args.stage2_seed)),
            "--methods", str(self.args.smoke_methods),
            "--no-plots",
            "--no-latex",
            "--skip-stage1",
            "--package-thesis-run",
            "--thesis-run-dir", str(self.thesis_run_dir),
            "--stage2-zip", str(self.args.stage2_zip),
        ]
        r = run_cmd(cmd, cwd=self.repo_root, repo_root=self.repo_root, log_path=log)
        if r.rc not in {0, 2}:
            fail(f"Treiber-Skript Rückgabecode={r.rc} (Protokoll: {log})")
            return False

        meta = self.out_root / "metadata.json"
        if not meta.exists():
            fail(f"metadata.json fehlt: {meta}")
            return False

        # Zentrale Artefakte prüfen
        stage3_union = self.out_root / "stage3_evaluation_union"
        stage3_pg = self.out_root / "stage3_evaluation_per_group"
        if not stage3_union.exists():
            fail(f"Etappe 03 (Vereinigung) fehlt: {stage3_union}")
            return False
        if not stage3_pg.exists():
            fail(f"Etappe 03 (pro Gruppe) fehlt: {stage3_pg}")
            return False

        # Paketierung: Auswertung pro Gruppe muss vorhanden sein (dies war zuvor die kritische Fehlerquelle)
        dst_pg = self.thesis_run_dir / "stage3_evaluation_per_group"
        if not dst_pg.exists():
            fail(f"Paket für die Bachelorarbeit: Auswertung pro Gruppe fehlt: {dst_pg}")
            return False

        ok(f"Treiber-Skript erfolgreich (Rückgabecode={r.rc}). Etappe 03 (Vereinigung und pro Gruppe) vorhanden; Paket für die Bachelorarbeit vollständig.")
        return True

    def step_08_driver_resume(self) -> bool:
        print_hr("08) Fortsetzen- und Idempotenzprüfung (Treiber, zweiter Lauf)")
        driver = self.repo_root / "tools" / "finalrun_multiseed.py"
        log = self.logs_dir / "driver_smoke_rerun.log"
        cmd = [
            sys.executable, "-u", str(driver),
            "--out-root", str(self.out_root),
            "--dataset-config", str(self.dataset_cfg),
            "--dataset-seed", str(int(self.args.dataset_seed)),
            "--plan", str(self.plan_smoke_path),
            "--budget", str(int(self.args.budget_driver)),
            "--run-seeds", str(int(self.args.stage2_seed)),
            "--methods", str(self.args.smoke_methods),
            "--no-plots",
            "--no-latex",
            "--skip-stage1",
            "--package-thesis-run",
            "--thesis-run-dir", str(self.thesis_run_dir),
            "--stage2-zip", str(self.args.stage2_zip),
        ]
        r = run_cmd(cmd, cwd=self.repo_root, repo_root=self.repo_root, log_path=log)
        if r.rc not in {0, 2}:
            fail(f"Treiber-Skript (zweiter Lauf) Rückgabecode={r.rc} (Protokoll: {log})")
            return False
        ok(f"Treiber-Skript (zweiter Lauf) erfolgreich (Rückgabecode={r.rc}). (Protokoll: {log.name})")
        return True

    def run_all(self) -> int:
        steps: list[Step] = [
            Step("00_umgebung", self.step_00_environment),
            Step("01_importe", self.step_01_imports),
            Step("02_scip_pruefung", self.step_02_check_scip),
            Step("03_etappe01_datensatz", self.step_03_stage1_generate_dataset),
            Step("04_kurztest_plan", self.step_04_make_smoke_plan),
            Step("05_etappe02_run_case", self.step_05_stage2_single_case_all_methods),
            Step("06_etappe02_arbeitsverzeichnis_kollision", self.step_06_run_dataset_cwd_collision_test),
            Step("07_treiber_ende_zu_ende", self.step_07_driver_end_to_end),
            Step("08_treiber_fortsetzen", self.step_08_driver_resume),
        ]

        results: list[tuple[str, bool]] = []
        all_ok = True

        for s in steps:
            try:
                passed = bool(s.fn())
            except KeyboardInterrupt:
                raise
            except Exception as e:
                passed = False
                print_hr(f"AUSNAHME in {s.name}")
                fail(f"{type(e).__name__}: {e}")
            results.append((s.name, passed))
            if not passed:
                all_ok = False
                if self.args.fail_fast:
                    break

        # Zusammenfassung
        print_hr("ZUSAMMENFASSUNG")
        for name, passed in results:
            print(f"{'BESTANDEN' if passed else 'NICHT BESTANDEN'}  {name}")
        summary_path = self.run_dir / "summary.json"
        write_json(summary_path, {"results": [{"step": n, "passed": p} for n, p in results]})
        ok(f"Zusammenfassung geschrieben: {summary_path}")
        ok(f"Protokolle: {self.logs_dir}")
        ok(f"Kurztest-Ausgaben: {self.out_root}")
        ok(f"Paket für die Bachelorarbeit (Kurztest): {self.thesis_run_dir}")

        if all_ok:
            ok("Vorabprüfung: Alle Schritte bestanden. Der vollständige Lauf kann gestartet werden.")
            return 0
        fail("Vorabprüfung: Mindestens ein Schritt ist nicht bestanden. Details stehen in den Protokollen im Laufordner.")
        return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vorabprüfung und Kurztest der BA-Pipeline")
    p.add_argument("--dataset-seed", type=int, default=12345, help="Seed für Etappe 01 (Datensatz) (Standard: 12345).")
    p.add_argument("--stage2-seed", type=int, default=101, help="Seed für Etappe 02 (Standard: 101).")

    p.add_argument("--smoke-methods", type=str, default="V3,V4", help="Verfahren für den Kurztest im Treiber (Standard: V3,V4).")

    p.add_argument("--budget-case-heur", type=float, default=3.0, help="Budget (Sekunden) pro run_case für V3/V4.")
    p.add_argument("--budget-case-milp", type=float, default=5.0, help="Budget (Sekunden) pro run_case für V1/V2.")
    p.add_argument("--budget-dataset", type=float, default=5.0, help="Budget (Sekunden) pro (Fall, Verfahren) im run_dataset-Arbeitsverzeichnis-Test.")
    p.add_argument("--budget-driver", type=int, default=5, help="Budget (Sekunden) im Treiber pro (Fall, Verfahren).")

    p.add_argument("--stage2-zip", type=str, default="zip", choices=["none", "zip"], help="Komprimierung der Etappe-02-Ausgaben im Paket der Bachelorarbeit.")
    p.add_argument("--skip-milp", action="store_true", help="MILP-Verfahren V1/V2 überspringen.")
    p.add_argument("--fail-fast", action="store_true", help="Beim ersten nicht bestandenen Schritt abbrechen.")
    p.add_argument("--run-id", type=str, default="", help="Optional: eigene Lauf-ID (überschreibt die automatisch erzeugte ID).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = detect_repo_root()

    run_id = args.run_id.strip() or utc_run_id()
    run_dir = repo_root / "_preflight" / run_id
    ensure_dir(run_dir)

    pre = Preflight(repo_root=repo_root, run_dir=run_dir, args=args)
    return pre.run_all()


if __name__ == "__main__":
    raise SystemExit(main())
