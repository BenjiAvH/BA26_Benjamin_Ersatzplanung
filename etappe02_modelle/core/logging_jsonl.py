from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping


def utc_now_iso() -> str:
    # ISO-8601, UTC, ohne Mikrosekunden (stabil/lesbar).
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _git_commit_and_dirty(cwd: Path) -> tuple[str | None, bool | None]:
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
        )
        if top.returncode != 0:
            return None, None
        repo_root = top.stdout.strip()

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if commit.returncode != 0:
            return None, None

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if status.returncode != 0:
            return commit.stdout.strip() or None, None
        dirty = bool(status.stdout.strip())
        return commit.stdout.strip() or None, dirty
    except FileNotFoundError:
        return None, None


def env_snapshot(*, cwd: Path | None = None, solver: str | None = None) -> dict[str, Any]:
    """
    Environment-Metadaten (Kap. 5.1: Forschungsdesign & Evaluationsstrategie; Reproduzierbarkeit).
    """

    if cwd is None:
        cwd = Path.cwd()

    git_commit, git_dirty = _git_commit_and_dirty(cwd)
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "os": os.name,
        "platform": platform.platform(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "solver": solver,
    }


@dataclass
class JsonlWriter:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(dict(record), ensure_ascii=False) + "\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
