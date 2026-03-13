#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path

PACKAGES = [
    "numpy",
    "scipy",
    "matplotlib",
    "yaml",
    "pydantic",
    "pandas",
    "h5py",
    "fitsio",
    "emcee",
    "camb",
    "velocileptors",
]


def main() -> int:
    versions = {
        "python": platform.python_version(),
        "executable": sys.executable,
    }

    failed = []
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            failed.append((name, str(exc)))

    print(json.dumps(versions, indent=2, sort_keys=True))

    if failed:
        print("\nFAILED IMPORTS:")
        for name, err in failed:
            print(f"- {name}: {err}")
        return 1

    output = Path("results/logs/import_versions.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(versions, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
