"""
Artefakt-Export (Paketierung + Pfad-Bereinigung) für die Bachelorarbeit.

Dieses Tool wird nach einem Fullrun ausgeführt:
- Es kopiert/packt Artefakte aus einem bestehenden Fullrun-Ordner (z.B. fullrun_out/).
- Es neutralisiert Pfadangaben in kopierten Textartefakten (z.B. Nutzerverzeichnisse), ohne Ergebniswerte neu zu berechnen.

Schutzprüfung: Es werden ausschließlich Standardbibliothek-Module verwendet (argparse, pathlib, json, shutil, zipfile, re, sys).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import sys
import zipfile


TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".tex", ".md", ".txt"}
SANITIZE_SUFFIXES = {".json", ".csv", ".tex", ".md", ".txt"}  # Bereinigung nur für Textdateien; ZIP-Inhalte separat.


_SANITIZE_RULES: list[tuple[str, re.Pattern[str], str]] = [
    # Windows: C:\Users\<name>\... oder C:\\Users\\<name>\\...
    ("win_users_bs", re.compile(r"(?i)([A-Z]:\\+Users\\+)([^\\/\s\"'<>|]+)"), "<USER_HOME>"),
    # Windows (selten): C:/Users/<name>/...
    ("win_users_fs", re.compile(r"(?i)([A-Z]:/+Users/+)([^/\\\s\"'<>|]+)"), "<USER_HOME>"),
    # Linux: /home/<name>/...
    ("posix_home", re.compile(r"(/home/)([^/\s]+)"), "<USER_HOME>"),
    # macOS: /Users/<name>/...
    ("posix_users", re.compile(r"(/Users/)([^/\s]+)"), "<USER_HOME>"),
    # Cloud-Speicher-Pfade im Text neutralisieren.
    ("onedrive", re.compile(r"(?i)OneDrive"), "<CLOUD_DRIVE>"),
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_text_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text_utf8(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _sanitize_text(text: str) -> tuple[str, int]:
    changed = 0
    out = text
    for _, rx, repl in _SANITIZE_RULES:
        out, n = rx.subn(repl, out)
        changed += int(n)
    return out, changed


def _iter_text_files(root: Path, *, suffixes: set[str]) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in suffixes:
            continue
        out.append(p)
    return sorted(out)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    _ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def _zip_dir(src_dir: Path, zip_path: Path) -> dict[str, int]:
    """
    Packt src_dir deterministisch als ZIP.
    Textdateien (TEXT_SUFFIXES) werden dabei **bereinigt** in die ZIP geschrieben,
    ohne die Quelldateien zu verändern.
    """
    if not src_dir.exists():
        raise SystemExit(f"Stage2-Ordner nicht gefunden: {src_dir}")
    if not src_dir.is_dir():
        raise SystemExit(f"Stage2-Pfad ist kein Ordner: {src_dir}")

    _ensure_dir(zip_path.parent)
    if zip_path.exists():
        zip_path.unlink()

    stats = {"files_total": 0, "files_text_sanitized": 0, "replacements": 0}

    try:
        compression = zipfile.ZIP_DEFLATED
    except Exception:
        compression = zipfile.ZIP_STORED

    with zipfile.ZipFile(zip_path, mode="w", compression=compression) as zf:
        for p in sorted(src_dir.rglob("*")):
            if not p.is_file():
                continue
            stats["files_total"] += 1
            arcname = p.relative_to(src_dir).as_posix()
            suffix = p.suffix.lower()
            if suffix in TEXT_SUFFIXES:
                data = p.read_bytes()
                try:
                    text = data.decode("utf-8")
                except Exception:
                    zf.write(p, arcname)
                    continue
                sanitized, n = _sanitize_text(text)
                stats["replacements"] += int(n)
                stats["files_text_sanitized"] += 1
                zf.writestr(arcname, sanitized.encode("utf-8"))
            else:
                zf.write(p, arcname)

    return stats


def cmd_package(args: argparse.Namespace) -> int:
    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise SystemExit(f"--src nicht gefunden: {src}")
    if not src.is_dir():
        raise SystemExit(f"--src ist kein Ordner: {src}")

    if bool(args.clean) and dst.exists():
        shutil.rmtree(dst)

    _ensure_dir(dst)

    # Root-Metadaten aus Fullrun (falls vorhanden)
    copied_meta = _copy_if_exists(src / "metadata.json", dst / "metadata.json")
    if copied_meta:
        print(f"[ok] metadata.json -> {dst / 'metadata.json'}")
    else:
        print("[warn] metadata.json nicht gefunden (wird übersprungen).")

    # Etappe 01 Manifest (optional)
    if bool(args.include_stage1_manifest):
        m_src = src / "stage1_dataset" / "manifest.json"
        m_dst = dst / "stage1_dataset" / "manifest.json"
        if _copy_if_exists(m_src, m_dst):
            print(f"[ok] stage1_dataset/manifest.json -> {m_dst}")
        else:
            print("[warn] stage1_dataset/manifest.json nicht gefunden (wird übersprungen).")

    # Etappe 03 (minimal): CSV + Tabellen + Abbildungen
    stage3_src = src / "stage3_evaluation"
    if not stage3_src.exists() or not stage3_src.is_dir():
        raise SystemExit(f"Stage3-Ordner nicht gefunden: {stage3_src}")

    stage3_dst = dst / "stage3_evaluation"
    _ensure_dir(stage3_dst)

    csv_files = sorted(stage3_src.glob("*.csv"))
    for p in csv_files:
        shutil.copy2(p, stage3_dst / p.name)
    print(f"[ok] stage3 CSV: {len(csv_files)} Dateien")

    tables_src = stage3_src / "tables"
    if tables_src.exists() and tables_src.is_dir():
        shutil.copytree(tables_src, stage3_dst / "tables", dirs_exist_ok=True)
        print("[ok] stage3 tables/ kopiert")
    else:
        print("[warn] stage3 tables/ nicht gefunden (wird übersprungen).")

    figures_src = stage3_src / "figures"
    if figures_src.exists() and figures_src.is_dir():
        shutil.copytree(figures_src, stage3_dst / "figures", dirs_exist_ok=True)
        print("[ok] stage3 figures/ kopiert")
    else:
        print("[warn] stage3 figures/ nicht gefunden (wird übersprungen).")

    # Etappe 02 (optional als ZIP)
    stage2_mode = str(args.stage2 or "none").strip().lower()
    if stage2_mode not in {"none", "zip"}:
        raise SystemExit("--stage2 muss 'none' oder 'zip' sein.")
    if stage2_mode == "zip":
        stats = _zip_dir(src / "stage2_runs", dst / "stage2_runs.zip")
        print(
            "[ok] stage2_runs.zip erstellt "
            f"(files_total={stats['files_total']}, text_sanitized={stats['files_text_sanitized']}, replacements={stats['replacements']})"
        )

    return 0


def cmd_sanitize(args: argparse.Namespace) -> int:
    dst = Path(args.dst)
    if not dst.exists() or not dst.is_dir():
        raise SystemExit(f"--dst nicht gefunden oder kein Ordner: {dst}")

    changed_files: list[tuple[Path, int]] = []
    for p in _iter_text_files(dst, suffixes=SANITIZE_SUFFIXES):
        try:
            original = _read_text_utf8(p)
        except Exception:
            # Sollte bei unseren Artefakten selten passieren; wir überspringen bewusst.
            continue
        sanitized, n = _sanitize_text(original)
        if sanitized != original:
            _write_text_utf8(p, sanitized)
            changed_files.append((p, int(n)))

    if not changed_files:
        print("[ok] sanitize: keine Änderungen.")
        return 0

    total_repl = sum(n for _, n in changed_files)
    print(f"[ok] sanitize: {len(changed_files)} Datei(en) geändert, replacements={total_repl}")
    for p, n in changed_files:
        print(f"  - {p} (replacements={n})")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """
    Prüft, ob nach der Bereinigung noch unbereinigte Pfade/Strings vorhanden sind.
    Verwendet die gleichen Regex-Pattern wie _SANITIZE_RULES (Schutzprüfung).
    """
    dst = Path(args.dst)
    if not dst.exists() or not dst.is_dir():
        raise SystemExit(f"--dst nicht gefunden oder kein Ordner: {dst}")

    hits: list[tuple[Path, str, int]] = []
    for p in _iter_text_files(dst, suffixes=SANITIZE_SUFFIXES):
        try:
            text = _read_text_utf8(p)
        except Exception:
            continue
        for name, rx, _ in _SANITIZE_RULES:
            if rx.search(text):
                try:
                    n = len(rx.findall(text))
                except Exception:
                    n = 1
                hits.append((p, str(name), int(n)))

    if not hits:
        print("[ok] scan: keine Treffer (keine unbereinigten Pfade gefunden).")
        return 0

    total = sum(n for _, _, n in hits)
    print(f"[FAIL] scan: Treffer gefunden (files={len(hits)}, matches={total}).", file=sys.stderr)
    for p, name, n in hits[:50]:
        print(f"  - {p} (rule={name}, matches={n})", file=sys.stderr)
    if len(hits) > 50:
        print(f"  - ... (+{len(hits) - 50} weitere Dateien)", file=sys.stderr)
    return 2


def cmd_all(args: argparse.Namespace) -> int:
    rc = cmd_package(args)
    if rc != 0:
        return int(rc)
    rc = cmd_sanitize(args)
    return int(rc)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Paketierung + Pfad-Bereinigung für Thesis-Artefakte (results/thesis_run/).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_pkg = sub.add_parser("package", help="Kopiert/packt minimale Artefakte nach results/thesis_run/.")
    p_pkg.add_argument("--src", type=Path, default=Path("fullrun_out"), help="Quelle (Fullrun-Output-Ordner).")
    p_pkg.add_argument("--dst", type=Path, default=Path("results/thesis_run"), help="Zielordner (wird versioniert).")
    p_pkg.add_argument("--clean", action="store_true", help="Löscht dst vor dem Kopieren.")
    p_pkg.add_argument("--include-stage1-manifest", action="store_true", help="Kopiert stage1_dataset/manifest.json mit.")
    p_pkg.add_argument("--stage2", type=str, default="none", choices=["none", "zip"], help="Stage-2 Artefakte: none|zip.")
    p_pkg.set_defaults(func=cmd_package)

    p_san = sub.add_parser("sanitize", help="Bereinigt persönliche Pfade in Textdateien unter dst.")
    p_san.add_argument("--dst", type=Path, default=Path("results/thesis_run"), help="Zielordner (wird in-place bearbeitet).")
    p_san.set_defaults(func=cmd_sanitize)

    p_scan = sub.add_parser("scan", help="Scannt dst auf unbereinigte Pfade/Strings (Schutzprüfung).")
    p_scan.add_argument("--dst", type=Path, default=Path("results/thesis_run"), help="Zielordner (wird gelesen).")
    p_scan.set_defaults(func=cmd_scan)

    p_all = sub.add_parser("all", help="Führt package + sanitize aus.")
    p_all.add_argument("--src", type=Path, default=Path("fullrun_out"), help="Quelle (Fullrun-Output-Ordner).")
    p_all.add_argument("--dst", type=Path, default=Path("results/thesis_run"), help="Zielordner (wird versioniert).")
    p_all.add_argument("--clean", action="store_true", help="Löscht dst vor dem Kopieren.")
    p_all.add_argument("--include-stage1-manifest", action="store_true", help="Kopiert stage1_dataset/manifest.json mit.")
    p_all.add_argument("--stage2", type=str, default="none", choices=["none", "zip"], help="Stage2-Artefakte: none|zip.")
    p_all.set_defaults(func=cmd_all)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    func = getattr(args, "func", None)
    if func is None:
        raise SystemExit("Kein Subcommand gewählt.")
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
