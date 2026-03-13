from __future__ import annotations

import json
import platform
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunPaths:
    run_dir: Path
    figures_dir: Path
    arrays_dir: Path
    logs_dir: Path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def init_run_dir(output_root: Path, tag: str) -> RunPaths:
    run_dir = output_root / f"{utc_timestamp()}_{tag}"
    figures_dir = run_dir / "figures"
    arrays_dir = run_dir / "arrays"
    logs_dir = run_dir / "logs"

    figures_dir.mkdir(parents=True, exist_ok=False)
    arrays_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True, exist_ok=False)

    return RunPaths(run_dir=run_dir, figures_dir=figures_dir, arrays_dir=arrays_dir, logs_dir=logs_dir)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT, text=True)
            .strip()
        )
    except Exception:
        return "not-a-git-repo"


def build_repro_metadata(config_path: Path) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "config_path": str(config_path.resolve()),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
