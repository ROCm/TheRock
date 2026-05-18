#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
===============================================================================
AORTA Triage Smoke Test (Manual Execution Only)

This script is NOT part of automated CI runs.

Exercises ``aorta triage run`` end-to-end against the ``recom_repro`` smoke
recipe from ``aorta-internal`` (``recipes/recom_repro_smoke.yaml``). Milestone
1 pass criteria per ROCm/aorta-internal#26:

* dispatcher exit code 0
* ``matrix.md`` produced under ``--output-dir``
* no matrix cells in the ``error`` state (cell-level failure)

Pre-requisites
--------------

1. **Dual editable install** (``aorta-internal/docs/triage_matrix_quickstart.md``
   section 1)::

       pip install -e <path>/aorta -e <path>/aorta-internal

   Per ``aorta-internal/CLAUDE.md`` rule #5: do **not** submodule public
   ``aorta`` into ``aorta-internal``. Use a sibling clone (or a pinned
   ``rocm-aorta`` PyPI package once published on internal PyPI).

2. **``AORTA_INTERNAL_DIR``** must point at the ``aorta-internal`` checkout
   so this test can resolve ``recipes/recom_repro_smoke.yaml``.

3. **GPU + ROCm host** with docker available. The smoke recipe runs the
   ``recom_repro`` workload inside docker (``nan-repro`` environment).

4. **Private image auth.** Step A uses a known-good image already wired into
   the recipe (``rocm/pytorch-private:nan-repro``). These are private AMD
   images; TheRock CI will not pull them without explicit registry credentials.

5. **Digest pinning (follow-up).** ``CLAUDE.md`` rule #7 requires digest-pinned
   gate images. This milestone does not assert on image digests; that lands in
   the follow-up ticket after milestone 1.

Expected runtime: ~3-5 minutes on MI350X once ``nan-repro`` is pulled; first
pull adds ~5-10 minutes.

Manual validation (document in your PR / run notes, not in CI)
---------------------------------------------------------------

* **Step A** -- run against the recipe's known-good ``rocm/pytorch-private:nan-repro``.
* **Step B** -- re-run against the latest ROCm image available in TheRock at run
  time. A Step B failure where Step A passed is reportable data; do not "fix"
  it inside this script.

Usage (after temporarily removing ``pytestmark`` skip, or passing
``--runxfail`` is insufficient -- un-skip locally only, do not commit)::

    export AORTA_INTERNAL_DIR=/path/to/aorta-internal
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
    "Manual execution only - requires GPU, ROCm, private docker image auth, "
    "AORTA_INTERNAL_DIR sibling install"
)

logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

RECIPE_REL = Path("recipes") / "recom_repro_smoke.yaml"


def _resolve_recipe() -> Path:
    aorta_internal = os.getenv("AORTA_INTERNAL_DIR")
    if not aorta_internal:
        pytest.skip("AORTA_INTERNAL_DIR is not set (path to aorta-internal checkout)")
    recipe = Path(aorta_internal) / RECIPE_REL
    if not recipe.is_file():
        pytest.fail(f"Smoke recipe not found: {recipe}")
    return recipe


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

    for trial_path in sorted(run_dir.rglob("trial_*.json")):
        trial = json.loads(trial_path.read_text(encoding="utf-8"))
        exit_status = trial.get("exit_status")
        if exit_status in (None, "ok"):
            continue
        # Trial-level failures are acceptable for smoke (bug may not fire at
        # 50 steps); only cell-level "could not start" counts as milestone error.
        pass


def test_aorta_triage_smoke(tmp_path: Path) -> None:
    if shutil.which("aorta") is None:
        pytest.skip("aorta CLI not on PATH (pip install -e aorta -e aorta-internal)")

    recipe = _resolve_recipe()
    cmd = [
        "aorta",
        "triage",
        "run",
        "--recipe",
        str(recipe),
        "--output-dir",
        str(tmp_path),
    ]

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
