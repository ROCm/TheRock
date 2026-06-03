#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""simulator_runner.py

Wraps an existing per-component test driver (e.g. ``test_rocrand.py``) so the
tests execute against the rocjitsu simulator instead of a physical AMD GPU.

The simulator runs entirely on CPU. We arrange three things before delegating
to the existing test driver:

1. The rocjitsu KFD interposer is preloaded so the real HIP/HSA stack talks to
   the simulated driver.
2. The rocjitsu JSON config and FlatBuffers schema are pointed at via env vars
   that the interposer reads at process start.
3. A ``GTEST_FILTER`` is composed from a preset (allow patterns) and a per-
   component skip-list (deny patterns), both defined in
   ``simulator_runner_filters.yaml``.

Required env (set by the workflow):

  THEROCK_BIN_DIR
      Path to the ``bin/`` directory of the populated ROCm dist (i.e.
      ``<rocm_root>/bin``). This mirrors the convention used by the rest of
      TheRock's per-component test drivers (e.g. ``test_rocdecode.py``,
      ``test_runner.py``) which compute the ROCm root as
      ``Path(THEROCK_BIN_DIR).parent``. Rocjitsu's interposer, config and
      schema are looked up under that ROCm root:
      ``<root>/lib/librocjitsu_kmd.so``,
      ``<root>/share/rocjitsu/configs/...``,
      ``<root>/share/rocjitsu/schemas/...``.

Usage:
  python simulator_runner.py --component rocrand --filter-preset basic
"""

import argparse
import logging
import os
import shlex
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "ERROR: PyYAML is required for simulator_runner.py. "
        "Install with `pip install pyyaml`.",
        file=sys.stderr,
    )
    sys.exit(2)


SCRIPT_DIR = Path(__file__).resolve().parent
FILTERS_PATH = SCRIPT_DIR / "simulator_runner_filters.yaml"

# Map --component to the existing per-component driver script in this directory.
COMPONENT_DRIVERS = {
    "rocrand": "test_rocrand.py",
    "hiprand": "test_hiprand.py",
}

# Per-test wall-clock cap (seconds) for ctest under the simulator. Bounds any
# single test that wanders into a slow path so a stall fails fast instead of
# eating the whole workflow step budget. Honored by ctest via the
# CTEST_TEST_TIMEOUT env var. 10 min is generous for a single create/destroy
# cycle under PDES while still cutting the 60-min step timeout we hit in
# Rocjitsu_003.txt down to a clear per-test failure.
DEFAULT_CTEST_TEST_TIMEOUT_SECONDS = 600


def _resolve_rocjitsu_paths(bin_dir: Path) -> dict[str, Path]:
    """Locate the rocjitsu interposer, config and schema in a populated dist.

    ``bin_dir`` is the ``bin/`` directory of the unified ROCm install (the
    value of ``THEROCK_BIN_DIR`` in TheRock's test harness). The rocjitsu
    artifact installs side-by-side with the rest of ROCm, so we resolve
    everything relative to the ROCm root (``bin_dir.parent``)::

        <root>/lib/librocjitsu_kmd.so
        <root>/share/rocjitsu/configs/amdgpu_cdna4_kmd.json
        <root>/share/rocjitsu/schemas/simulation_config.fbs
    """
    rocm_root = bin_dir.parent
    paths = {
        "preload": rocm_root / "lib" / "librocjitsu_kmd.so",
        "config": rocm_root
        / "share"
        / "rocjitsu"
        / "configs"
        / "amdgpu_cdna4_kmd.json",
        "schema": rocm_root
        / "share"
        / "rocjitsu"
        / "schemas"
        / "simulation_config.fbs",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "rocjitsu install is incomplete under "
            f"ROCm root {rocm_root} (THEROCK_BIN_DIR={bin_dir}). Missing:\n  "
            + "\n  ".join(missing)
            + "\nMake sure the build enabled "
            "-DTHEROCK_ENABLE_EMULATION=ON -DTHEROCK_ENABLE_ROCJITSU=ON "
            "and the rocjitsu artifact was unpacked into the dist."
        )
    return paths


def _load_filters(component: str, preset: str) -> tuple[list[str], list[str]]:
    """Return (allow_patterns, skip_patterns) for ``component`` and ``preset``."""
    if not FILTERS_PATH.exists():
        raise FileNotFoundError(f"Simulator filter config not found: {FILTERS_PATH}")
    with FILTERS_PATH.open() as fh:
        cfg = yaml.safe_load(fh) or {}
    comp_cfg = cfg.get(component)
    if not comp_cfg:
        raise KeyError(
            f"No entry for component '{component}' in {FILTERS_PATH}. "
            f"Known components: {sorted(cfg.keys())}"
        )
    presets = comp_cfg.get("presets") or {}
    if preset not in presets:
        raise KeyError(
            f"Unknown preset '{preset}' for component '{component}'. "
            f"Known presets: {sorted(presets.keys())}"
        )
    allow = list(presets[preset] or [])
    skip = list(comp_cfg.get("skip") or [])
    return allow, skip


def _compose_gtest_filter(allow: list[str], skip: list[str]) -> str:
    """Build a GTest filter string of the form ``allow:-skip``.

    See https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests
    for the syntax.
    """
    allow_part = ":".join(allow) if allow else "*"
    if not skip:
        return allow_part
    skip_part = ":".join(skip)
    return f"{allow_part}:-{skip_part}"


def _build_env(bin_dir: Path, gtest_filter: str) -> dict[str, str]:
    paths = _resolve_rocjitsu_paths(bin_dir)
    env = os.environ.copy()
    # Compose LD_PRELOAD so we don't drop an existing preload set by the user.
    existing_preload = env.get("LD_PRELOAD", "").strip()
    preload_value = (
        f"{paths['preload']}:{existing_preload}"
        if existing_preload
        else str(paths["preload"])
    )
    env["LD_PRELOAD"] = preload_value
    env["RJ_CONFIG"] = str(paths["config"])
    env["RJ_SCHEMA"] = str(paths["schema"])
    # SDMA is needed by the HIP runtime path the interposer emulates; mirror
    # the value used by rocjitsu's own ctest suite.
    env["HSA_ENABLE_SDMA"] = env.get("HSA_ENABLE_SDMA", "1")
    env["GTEST_FILTER"] = gtest_filter
    # Bound any individual ctest case so a stalled test fails fast instead of
    # consuming the whole workflow step budget. Respects an explicit caller
    # override (e.g. nightly runs of the `full` preset may want longer).
    env["CTEST_TEST_TIMEOUT"] = env.get(
        "CTEST_TEST_TIMEOUT", str(DEFAULT_CTEST_TEST_TIMEOUT_SECONDS)
    )
    return env


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a TheRock per-component test driver under the rocjitsu "
            "simulator (CPU-only)."
        )
    )
    parser.add_argument(
        "--component",
        required=True,
        choices=sorted(COMPONENT_DRIVERS.keys()),
        help="Component whose existing test driver should be wrapped.",
    )
    parser.add_argument(
        "--filter-preset",
        default="basic",
        help=(
            "GTest filter preset defined in simulator_runner_filters.yaml. "
            "Typically one of: basic, quick, full."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    bin_dir_env = os.environ.get("THEROCK_BIN_DIR")
    if not bin_dir_env:
        print(
            "ERROR: THEROCK_BIN_DIR is not set. The simulator runner needs "
            "the path to the populated ROCm `bin/` dir "
            "(e.g. build/dist/rocm/bin).",
            file=sys.stderr,
        )
        return 2
    bin_dir = Path(bin_dir_env).resolve()

    try:
        allow, skip = _load_filters(args.component, args.filter_preset)
        env = _build_env(bin_dir, _compose_gtest_filter(allow, skip))
    except (FileNotFoundError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    gtest_filter = env["GTEST_FILTER"]

    driver_script = SCRIPT_DIR / COMPONENT_DRIVERS[args.component]
    if not driver_script.exists():
        print(f"ERROR: driver script not found: {driver_script}", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(driver_script)]
    logging.info(
        "[simulator_runner] component=%s preset=%s", args.component, args.filter_preset
    )
    logging.info("[simulator_runner] THEROCK_BIN_DIR=%s", bin_dir)
    logging.info("[simulator_runner] LD_PRELOAD=%s", env["LD_PRELOAD"])
    logging.info("[simulator_runner] RJ_CONFIG=%s", env["RJ_CONFIG"])
    logging.info("[simulator_runner] RJ_SCHEMA=%s", env["RJ_SCHEMA"])
    logging.info("[simulator_runner] GTEST_FILTER=%s", gtest_filter)
    logging.info("[simulator_runner] CTEST_TEST_TIMEOUT=%s", env["CTEST_TEST_TIMEOUT"])
    logging.info("[simulator_runner] exec: %s", shlex.join(cmd))

    # execvpe replaces this process so the driver inherits PID 1 in container
    # use cases and signals propagate cleanly.
    try:
        os.execvpe(cmd[0], cmd, env)
    except OSError as e:
        print(f"ERROR: failed to exec driver: {e}", file=sys.stderr)
        return 2
    return 0  # unreachable


if __name__ == "__main__":
    sys.exit(main())
