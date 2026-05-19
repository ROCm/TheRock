#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
===============================================================================
AORTA triage smoke test (manual execution only)

This script is NOT part of automated CI runs.

Runs ``aorta triage run`` against a recipe file and checks milestone-style
plumbing: dispatcher exit code 0, a single ``matrix.md`` under ``--output-dir``,
and no cell-level ``error`` rows in ``matrix.json``.

Pre-requisites
--------------

1. ``aorta`` on ``PATH`` (``pip install`` of the public ``aorta`` package, or an
   editable install from a local checkout).

2. **``AORTA_RECIPE_PATH``** — absolute path to a triage recipe YAML.

3. **``AORTA_MITIGATIONS_FILE``** (optional) — path to a mitigations/environments
   sidecar JSON passed through to ``--mitigations-file`` when set.

4. **GPU + ROCm host** with docker if the recipe uses docker-backed environments.
   Registry credentials for non-public images are out of band; TheRock CI does not
   provide them today.

Usage (remove ``pytestmark`` skip locally only; do not commit)::

    export AORTA_RECIPE_PATH=/path/to/smoke-recipe.yaml
    export AORTA_MITIGATIONS_FILE=/path/to/sidecar.json   # if required by recipe
    export HIP_VISIBLE_DEVICES=0
    pytest build_tools/github_actions/test_executable_scripts/test_aorta_triage.py \\
        -k test_aorta_triage_smoke -s

===============================================================================
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skip(
    "Manual execution only — requires GPU, ROCm, and AORTA_RECIPE_PATH"
)

logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent


def _recipe_path() -> Path:
    raw = os.getenv("AORTA_RECIPE_PATH")
    if not raw:
        pytest.skip("AORTA_RECIPE_PATH is not set (path to triage recipe YAML)")
    recipe = Path(raw)
    if not recipe.is_file():
        pytest.fail(f"Recipe not found: {recipe}")
    return recipe


def _mitigations_file() -> Path | None:
    raw = os.getenv("AORTA_MITIGATIONS_FILE")
    if not raw:
        return None
    sidecar = Path(raw)
    if not sidecar.is_file():
        pytest.fail(f"AORTA_MITIGATIONS_FILE not found: {sidecar}")
    return sidecar


def _assert_no_error_cells(run_dir: Path) -> None:
    matrix_json_paths = sorted(run_dir.rglob("matrix.json"))
    if len(matrix_json_paths) != 1:
        pytest.fail(
            f"Expected exactly one matrix.json under {run_dir}, "
            f"found {len(matrix_json_paths)}: {matrix_json_paths}"
        )
    matrix_doc = json.loads(matrix_json_paths[0].read_text(encoding="utf-8"))
    error_cells: list[str] = []
    for cell in matrix_doc.get("cells", []):
        if cell.get("error") is not None:
            error_cells.append(f"{cell.get('name')}: {cell['error']}")
        elif cell.get("confound") == "error":
            error_cells.append(f"{cell.get('name')}: confound=error")
    if error_cells:
        pytest.fail("Matrix cells in error state:\n" + "\n".join(error_cells))


def test_aorta_triage_smoke(tmp_path: Path) -> None:
    if shutil.which("aorta") is None:
        pytest.skip("aorta CLI not on PATH")

    recipe = _recipe_path()
    cmd = [
        "aorta",
        "triage",
        "run",
        "--recipe",
        str(recipe),
        "--output-dir",
        str(tmp_path),
    ]
    sidecar = _mitigations_file()
    if sidecar is not None:
        cmd.extend(["--mitigations-file", str(sidecar)])

    logging.info("++ Exec [%s]$ %s", THEROCK_DIR, shlex.join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=THEROCK_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"aorta triage run failed (exit {proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    matrix_md_paths = sorted(tmp_path.rglob("matrix.md"))
    if len(matrix_md_paths) != 1:
        pytest.fail(
            f"Expected exactly one matrix.md under {tmp_path}, "
            f"found {len(matrix_md_paths)}: {matrix_md_paths}"
        )

    run_dir = matrix_md_paths[0].parent
    logging.info("matrix.md: %s", matrix_md_paths[0])
    _assert_no_error_cells(run_dir)
