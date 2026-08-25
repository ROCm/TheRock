#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""simulator_runner.py

Wraps an existing per-component test driver (e.g. ``test_rocrand.py``) so the
tests execute against the rocjitsu simulator instead of a physical AMD GPU.

The simulator runs entirely on CPU. We arrange two things before delegating
to the existing test driver, then enforce two post-run guards:

1. The existing per-component driver is launched through the ``rocjitsu`` CLI
   (``rocjitsu --config <config.json> -- <driver>``). The CLI writes the
   interposer config-discovery file, sets ``LD_PRELOAD`` to the KFD interposer
   and execs the driver; ``ctest`` and the test binaries it spawns inherit
   both, so the real HIP/HSA stack talks to the simulated driver. This mirrors
   how rocjitsu's own corpus CI wraps ``pytest``.
2. A ``GTEST_FILTER`` is composed from a preset (allow patterns) and a per-
   component skip-list (deny patterns), both defined in
   ``simulator_runner_filters.yaml``. The preset also supplies a
   ``ctest_regex`` to narrow which ctest binaries run, and a ``min_gtests``
   lower bound for the post-run guards.

Post-run guards (Rocjitsu_005 lesson):

A. **No silent empty-set passes.** GoogleTest exits 0 when ``GTEST_FILTER``
   matches nothing in a binary. ctest then counts the binary as Passed, and
   the workflow shows green even though zero simulator coverage ran. After
   the driver returns we scan
   ``<bin_dir>/<ctest_dir>/Testing/Temporary/LastTest.log`` for the literal
   ``did not match any test; no tests were run`` line in any binary that
   matched ``ctest_regex`` (i.e. was in scope for this preset) and fail the
   run if any are found.
B. **Minimum coverage floor.** We sum the gtest counts reported by every
   in-scope binary ("[==========] N tests from M test suites ran.") and fail
   the run if the total is below the preset's ``min_gtests``. Set
   ``SIMULATOR_RUNNER_SKIP_GUARDS=1`` to bypass A+B for debugging.

Required env (set by the workflow):

  THEROCK_BIN_DIR
      Path to the ``bin/`` directory of the populated ROCm dist (i.e.
      ``<rocm_root>/bin``). This mirrors the convention used by the rest of
      TheRock's per-component test drivers (e.g. ``test_rocdecode.py``,
      ``test_runner.py``) which compute the ROCm root as
      ``Path(THEROCK_BIN_DIR).parent``. Rocjitsu's CLI, interposer and config
      are looked up under that ROCm root:
      ``<root>/bin/rocjitsu``,
      ``<root>/lib/librocjitsu_kmd.so``,
      ``<root>/share/rocjitsu/configs/...``.

Usage:
  python simulator_runner.py --component rocrand --filter-preset basic
"""

import argparse
import logging
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
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

# Default ctest -R regex when a preset does not declare one. ".*" matches
# every ctest binary, preserving today's behavior for the `quick` and `full`
# presets where we genuinely want every binary to run.
DEFAULT_CTEST_REGEX = ".*"

# Default ctest -E (exclude) regex. Empty means "exclude nothing". A preset (or
# shard) may set `ctest_exclude_regex` to drop specific binaries from the -R set
# - needed for suite variants that carry their own CMake-baked GTEST_FILTER, so
# the component skip: list cannot reach them (e.g. drop the cpp_basic suite whose
# ffm variant has a 7200s ctest timeout, or the philox/xorwow poisson hangs).
DEFAULT_CTEST_EXCLUDE_REGEX = ""

# Default min_gtests floor when a preset does not declare one. >=1 means
# "at least one gtest must actually execute", which is the bare minimum
# needed to catch the empty-set silent-pass failure mode.
DEFAULT_MIN_GTESTS = 1

# Sentinel substring GoogleTest prints when GTEST_FILTER matches nothing.
# Stable across gtest versions (see googletest/src/gtest.cc).
EMPTY_FILTER_SENTINEL = "did not match any test; no tests were run"

# Map AMDGPU_FAMILIES value -> the rocjitsu config JSON shipped under
# <root>/share/rocjitsu/configs/. rocjitsu's install rule (rocm-systems
# emulation/rocjitsu/CMakeLists.txt) copies the whole `configs/` directory
# wholesale, so every config is present in each dist; we only pick the right
# one at runtime based on the family the caller is testing.
#
# Config variants mirror rocjitsu's own corpus CI
# (rocm-systems .github/workflows/rocjitsu-corpus-tests.yml): the plain
# (non-`_kmd`) configs for CDNA, the per-board `_kmd` configs for RDNA, and the
# dedicated config for gfx1250. Keep this in sync with the families enabled for
# the `rocrand-simulator` job in fetch_test_configurations.py.
FAMILY_TO_ROCJITSU_CONFIG = {
    "gfx94X-dcgpu": "amdgpu_cdna3.json",
    "gfx950-dcgpu": "amdgpu_cdna4.json",
    "gfx110X-all": "amdgpu_rdna3_gfx1100_w7900_kmd.json",
    "gfx120X-all": "amdgpu_rdna4_gfx1201_r9700_kmd.json",
    "gfx125X-dcgpu": "amdgpu_gfx1250.json",
}


@dataclass(frozen=True)
class PresetConfig:
    """Resolved view of one preset entry from simulator_runner_filters.yaml."""

    allow: list[str]
    skip: list[str]
    ctest_regex: str
    ctest_exclude_regex: str
    min_gtests: int


@dataclass(frozen=True)
class ComponentConfig:
    """Resolved view of one component entry from simulator_runner_filters.yaml."""

    ctest_dir: str
    skip: list[str]


def _resolve_rocjitsu_paths(bin_dir: Path, amdgpu_family: str) -> dict[str, Path]:
    """Locate the rocjitsu CLI, interposer and config in a populated dist.

    ``bin_dir`` is the ``bin/`` directory of the unified ROCm install (the
    value of ``THEROCK_BIN_DIR`` in TheRock's test harness). The rocjitsu
    artifact installs side-by-side with the rest of ROCm, so we resolve
    everything relative to the ROCm root (``bin_dir.parent``)::

        <root>/bin/rocjitsu
        <root>/lib/librocjitsu_kmd.so
        <root>/share/rocjitsu/configs/<config>.json

    The exact config filename is picked from ``amdgpu_family`` via
    ``FAMILY_TO_ROCJITSU_CONFIG``. Unknown families raise ``KeyError`` rather
    than silently falling back, so a misconfigured caller cannot quietly
    exercise the wrong config. The interposer lib is not preloaded directly;
    the ``rocjitsu`` CLI discovers and preloads it (via ``bin/../lib``), but we
    still validate it is present so failures are reported up front.
    """
    try:
        config_name = FAMILY_TO_ROCJITSU_CONFIG[amdgpu_family]
    except KeyError as e:
        raise KeyError(
            f"AMDGPU_FAMILIES={amdgpu_family!r} has no rocjitsu config "
            f"mapping. Supported families: "
            f"{sorted(FAMILY_TO_ROCJITSU_CONFIG.keys())}."
        ) from e
    rocm_root = bin_dir.parent
    paths = {
        "cli": rocm_root / "bin" / "rocjitsu",
        "interposer": rocm_root / "lib" / "librocjitsu_kmd.so",
        "config": rocm_root / "share" / "rocjitsu" / "configs" / config_name,
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


def _load_config(
    component: str,
    preset: str,
    shard_index: int = 1,
    total_shards: int = 1,
) -> tuple[ComponentConfig, PresetConfig]:
    """Load and resolve component + preset config from the YAML file.

    Supports both the modern mapping form (``preset: {allow: [...],
    ctest_regex: ..., min_gtests: ...}``) and the legacy flat-list form
    (``preset: [pattern1, pattern2]``) for back-compat with older entries.

    A mapping-form preset may also define a ``shards`` list. When the job runs
    with ``total_shards > 1``, the preset resolves to ``shards[shard_index-1]``
    so each shard runs its own explicitly-curated scope (each shard is a
    separate GHA job with its own timeout). Keys a shard omits fall back to the
    preset-level values. With ``total_shards == 1`` (or no ``shards`` declared),
    resolution is unchanged.
    """
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

    ctest_dir = comp_cfg.get("ctest_dir")
    if not ctest_dir:
        raise KeyError(
            f"Component '{component}' in {FILTERS_PATH} is missing required "
            f"key 'ctest_dir' (the dir under THEROCK_BIN_DIR holding "
            f"CTestTestfile.cmake, e.g. 'rocRAND')."
        )

    presets = comp_cfg.get("presets") or {}
    if preset not in presets:
        raise KeyError(
            f"Unknown preset '{preset}' for component '{component}'. "
            f"Known presets: {sorted(presets.keys())}"
        )

    skip = list(comp_cfg.get("skip") or [])

    raw_preset = presets[preset]
    if isinstance(raw_preset, list):
        allow = list(raw_preset)
        ctest_regex = DEFAULT_CTEST_REGEX
        ctest_exclude_regex = DEFAULT_CTEST_EXCLUDE_REGEX
        min_gtests = DEFAULT_MIN_GTESTS
    elif isinstance(raw_preset, dict):
        allow = list(raw_preset.get("allow") or [])
        ctest_regex = raw_preset.get("ctest_regex") or DEFAULT_CTEST_REGEX
        ctest_exclude_regex = (
            raw_preset.get("ctest_exclude_regex") or DEFAULT_CTEST_EXCLUDE_REGEX
        )
        min_gtests = int(raw_preset.get("min_gtests") or DEFAULT_MIN_GTESTS)
    else:
        raise TypeError(
            f"Preset '{preset}' for component '{component}' has unsupported "
            f"type {type(raw_preset).__name__}; expected list or mapping."
        )

    # Per-shard scope selection. A mapping-form preset may declare a `shards`
    # list; when TOTAL_SHARDS>1 each shard runs its own curated scope so shards
    # cover disjoint, explicit test sets rather than duplicating the whole
    # preset. A shard inherits any key it omits from the preset-level values.
    if total_shards > 1 and isinstance(raw_preset, dict):
        shards = raw_preset.get("shards")
        if shards:
            if not isinstance(shards, list):
                raise TypeError(
                    f"Preset '{preset}' 'shards' must be a list; got "
                    f"{type(shards).__name__}."
                )
            if len(shards) != total_shards:
                raise ValueError(
                    f"Preset '{preset}' declares {len(shards)} shard scope(s) "
                    f"but TOTAL_SHARDS={total_shards}; they must match."
                )
            if not 1 <= shard_index <= total_shards:
                raise ValueError(
                    f"SHARD_INDEX={shard_index} out of range for "
                    f"TOTAL_SHARDS={total_shards} (expected 1..{total_shards})."
                )
            shard_cfg = shards[shard_index - 1]
            if not isinstance(shard_cfg, dict):
                raise TypeError(
                    f"Preset '{preset}' shard {shard_index} must be a mapping; "
                    f"got {type(shard_cfg).__name__}."
                )
            if shard_cfg.get("allow"):
                allow = list(shard_cfg["allow"])
            ctest_regex = shard_cfg.get("ctest_regex") or ctest_regex
            ctest_exclude_regex = (
                shard_cfg.get("ctest_exclude_regex") or ctest_exclude_regex
            )
            min_gtests = int(shard_cfg.get("min_gtests") or min_gtests)

    return (
        ComponentConfig(ctest_dir=ctest_dir, skip=skip),
        PresetConfig(
            allow=allow,
            skip=skip,
            ctest_regex=ctest_regex,
            ctest_exclude_regex=ctest_exclude_regex,
            min_gtests=min_gtests,
        ),
    )


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


def _build_env(
    gtest_filter: str,
    ctest_dir: Path,
    ctest_regex: str,
    ctest_exclude_regex: str = "",
) -> dict[str, str]:
    # The rocjitsu CLI owns LD_PRELOAD and the config-discovery file; we only
    # set the test-harness knobs here. See main() for the CLI wrapper command.
    env = os.environ.copy()
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
    # Knobs picked up by the wrapped driver (e.g. test_rocrand.py). All are
    # opt-in: drivers fall back to their existing on-device behavior when the
    # vars are unset, so the real-GPU lane is unaffected.
    env["SIMULATOR_CTEST_DIR"] = str(ctest_dir)
    env["SIMULATOR_CTEST_INCLUDE_REGEX"] = ctest_regex
    # ctest -E exclude regex (drops binaries from the include set). Only set when
    # non-empty so the common "no exclusions" case leaves the driver's behavior
    # untouched.
    if ctest_exclude_regex:
        env["SIMULATOR_CTEST_EXCLUDE_REGEX"] = ctest_exclude_regex
    # Guard 4: under the deterministic simulator, retrying a failing case
    # cannot turn it green - it only hides real bugs. Tell the driver to drop
    # --repeat. Drivers that don't honor this still work; the guard above is
    # the primary defense.
    env["SIMULATOR_NO_RETRY"] = env.get("SIMULATOR_NO_RETRY", "1")
    return env


# --- Post-run guard parsing ---------------------------------------------------

# Header line ctest writes per binary in Testing/Temporary/LastTest.log:
#   "10/52 Testing: test_rocrand_host"
# (The first number is the run order, second is the total. We only need the
# binary name.)
_TESTING_HEADER_RE = re.compile(r"^\s*(?:\d+/\d+)\s+Testing:\s+(?P<name>\S+)\s*$")

# Footer-ish summary line GoogleTest prints when one or more tests actually
# ran:
#   "[==========] 3 tests from 2 test suites ran. (55 ms total)"
# The "0 tests from 0 test suites ran" form ALSO matches this regex; the
# guard differentiates by ALSO checking for EMPTY_FILTER_SENTINEL on the
# same binary.
_RAN_LINE_RE = re.compile(
    r"^\[==========\]\s+(?P<n>\d+)\s+tests?\s+from\s+\d+\s+test\s+suites?\s+ran"
)


@dataclass(frozen=True)
class BinaryResult:
    """One ctest binary's run summary parsed from LastTest.log."""

    name: str
    gtests_run: int
    empty_filter: bool


def _parse_last_test_log(log_path: Path) -> list[BinaryResult]:
    """Split LastTest.log into per-binary BinaryResult records.

    Tolerates noise: anything between the "Testing: <name>" header and the
    next header is attributed to <name>. Returns an empty list if the file
    is missing or unreadable - callers decide whether that is fatal.
    """
    if not log_path.is_file():
        return []
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return []

    results: list[BinaryResult] = []
    current_name: str | None = None
    current_gtests = 0
    current_empty = False

    def _flush() -> None:
        if current_name is not None:
            results.append(
                BinaryResult(
                    name=current_name,
                    gtests_run=current_gtests,
                    empty_filter=current_empty,
                )
            )

    for line in text.splitlines():
        header_m = _TESTING_HEADER_RE.match(line)
        if header_m:
            _flush()
            current_name = header_m.group("name")
            current_gtests = 0
            current_empty = False
            continue
        if current_name is None:
            continue
        if EMPTY_FILTER_SENTINEL in line:
            current_empty = True
            continue
        ran_m = _RAN_LINE_RE.match(line)
        if ran_m:
            # Multiple "ran." lines per binary should not happen (gtest_main
            # prints it once); if it does, take the max so a retry doesn't
            # zero us out.
            current_gtests = max(current_gtests, int(ran_m.group("n")))
            continue
    _flush()
    return results


@dataclass(frozen=True)
class GuardOutcome:
    """Result of running the post-run guards over one component's LastTest.log."""

    ok: bool
    in_scope_count: int
    in_scope_total_gtests: int
    empty_filter_binaries: list[str]
    messages: list[str]


def _run_guards(
    log_path: Path,
    ctest_regex: str,
    min_gtests: int,
) -> GuardOutcome:
    """Apply Guard A (no empty-set passes) and Guard B (min coverage floor).

    Returns a GuardOutcome whose `messages` always includes a one-line human
    summary suitable for the workflow step log; `ok` is False iff either
    guard tripped.
    """
    results = _parse_last_test_log(log_path)
    if not results:
        return GuardOutcome(
            ok=False,
            in_scope_count=0,
            in_scope_total_gtests=0,
            empty_filter_binaries=[],
            messages=[
                f"::error::Could not parse ctest LastTest.log at {log_path}. "
                f"This usually means ctest never ran any binaries; check the "
                f"driver output above."
            ],
        )

    try:
        in_scope_re = re.compile(ctest_regex)
    except re.error as e:
        return GuardOutcome(
            ok=False,
            in_scope_count=0,
            in_scope_total_gtests=0,
            empty_filter_binaries=[],
            messages=[
                f"::error::Invalid ctest_regex {ctest_regex!r} in preset "
                f"config: {e}"
            ],
        )

    in_scope = [r for r in results if in_scope_re.search(r.name)]
    in_scope_count = len(in_scope)
    in_scope_total = sum(r.gtests_run for r in in_scope)
    empty_in_scope = [r.name for r in in_scope if r.empty_filter]

    messages: list[str] = []
    ok = True

    if in_scope_count == 0:
        ok = False
        messages.append(
            f"::error::Preset's ctest_regex {ctest_regex!r} matched zero of "
            f"the {len(results)} binaries ctest ran. Check the regex against "
            f"the binary names in LastTest.log."
        )

    if empty_in_scope:
        ok = False
        messages.append(
            f"::error::{len(empty_in_scope)} in-scope binary(ies) reported "
            f"'{EMPTY_FILTER_SENTINEL}' - GTEST_FILTER matched nothing in "
            f"them, so they passed without running any tests. Offenders: "
            + ", ".join(empty_in_scope)
        )

    if in_scope_total < min_gtests:
        ok = False
        messages.append(
            f"::error::Coverage floor not met: in-scope binaries ran "
            f"{in_scope_total} gtest case(s), preset requires at least "
            f"{min_gtests}. Either the filter is too narrow for this preset "
            f"or the simulator skipped tests silently."
        )

    summary = (
        f"[simulator_runner] guards: in_scope_binaries={in_scope_count}/"
        f"{len(results)} gtests_ran={in_scope_total} "
        f"min_required={min_gtests} empty_filter_in_scope={len(empty_in_scope)}"
    )
    messages.insert(0, summary)
    return GuardOutcome(
        ok=ok,
        in_scope_count=in_scope_count,
        in_scope_total_gtests=in_scope_total,
        empty_filter_binaries=empty_in_scope,
        messages=messages,
    )


# --- CLI / main ---------------------------------------------------------------


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

    amdgpu_family = (os.environ.get("AMDGPU_FAMILIES") or "").strip()
    if not amdgpu_family:
        print(
            "ERROR: AMDGPU_FAMILIES is not set. The simulator runner needs it "
            "to pick the rocjitsu config under "
            "<rocm_root>/share/rocjitsu/configs/. Supported values: "
            f"{sorted(FAMILY_TO_ROCJITSU_CONFIG.keys())}.",
            file=sys.stderr,
        )
        return 2

    # Shard identity is injected by the CI matrix (test_component.yml). Each
    # shard is its own GHA job with its own timeout; the preset's `shards` list
    # (if any) decides what each shard actually runs. Default to a single shard
    # so local/non-sharded invocations are unaffected.
    def _env_int(name: str) -> int:
        try:
            return int(os.environ.get(name, "1"))
        except ValueError:
            return 1

    shard_index = _env_int("SHARD_INDEX")
    total_shards = _env_int("TOTAL_SHARDS")

    try:
        comp_cfg, preset_cfg = _load_config(
            args.component, args.filter_preset, shard_index, total_shards
        )
        ctest_dir = bin_dir / comp_cfg.ctest_dir
        paths = _resolve_rocjitsu_paths(bin_dir, amdgpu_family)
        env = _build_env(
            _compose_gtest_filter(preset_cfg.allow, preset_cfg.skip),
            ctest_dir,
            preset_cfg.ctest_regex,
            preset_cfg.ctest_exclude_regex,
        )
    except (FileNotFoundError, KeyError, TypeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    gtest_filter = env["GTEST_FILTER"]

    driver_script = SCRIPT_DIR / COMPONENT_DRIVERS[args.component]
    if not driver_script.exists():
        print(f"ERROR: driver script not found: {driver_script}", file=sys.stderr)
        return 2

    # Launch the driver through the rocjitsu CLI: it writes the interposer
    # config-discovery file, sets LD_PRELOAD and execs the driver. ctest and the
    # test binaries it spawns inherit both. Mirrors rocjitsu's corpus CI, which
    # wraps `pytest` the same way.
    cmd = [
        str(paths["cli"]),
        "--config",
        str(paths["config"]),
        "--",
        sys.executable,
        str(driver_script),
    ]
    logging.info(
        "[simulator_runner] component=%s preset=%s", args.component, args.filter_preset
    )
    logging.info(
        "[simulator_runner] SHARD_INDEX=%d TOTAL_SHARDS=%d", shard_index, total_shards
    )
    logging.info("[simulator_runner] THEROCK_BIN_DIR=%s", bin_dir)
    logging.info("[simulator_runner] AMDGPU_FAMILIES=%s", amdgpu_family)
    logging.info("[simulator_runner] rocjitsu_cli=%s", paths["cli"])
    logging.info("[simulator_runner] rocjitsu_config=%s", paths["config"])
    logging.info("[simulator_runner] GTEST_FILTER=%s", gtest_filter)
    logging.info("[simulator_runner] CTEST_TEST_TIMEOUT=%s", env["CTEST_TEST_TIMEOUT"])
    logging.info(
        "[simulator_runner] SIMULATOR_CTEST_DIR=%s", env["SIMULATOR_CTEST_DIR"]
    )
    logging.info(
        "[simulator_runner] SIMULATOR_CTEST_INCLUDE_REGEX=%s",
        env["SIMULATOR_CTEST_INCLUDE_REGEX"],
    )
    logging.info(
        "[simulator_runner] SIMULATOR_CTEST_EXCLUDE_REGEX=%s",
        env.get("SIMULATOR_CTEST_EXCLUDE_REGEX", ""),
    )
    logging.info("[simulator_runner] SIMULATOR_NO_RETRY=%s", env["SIMULATOR_NO_RETRY"])
    logging.info(
        "[simulator_runner] min_gtests=%d (preset=%s)",
        preset_cfg.min_gtests,
        args.filter_preset,
    )
    logging.info("[simulator_runner] exec: %s", shlex.join(cmd))

    try:
        completed = subprocess.run(cmd, env=env, check=False)
    except OSError as e:
        print(f"ERROR: failed to exec driver: {e}", file=sys.stderr)
        return 2
    driver_rc = completed.returncode
    logging.info("[simulator_runner] driver exited with rc=%d", driver_rc)

    if os.environ.get("SIMULATOR_RUNNER_SKIP_GUARDS") == "1":
        logging.info(
            "[simulator_runner] SIMULATOR_RUNNER_SKIP_GUARDS=1; skipping "
            "post-run guards. Returning driver rc=%d.",
            driver_rc,
        )
        return driver_rc

    log_path = ctest_dir / "Testing" / "Temporary" / "LastTest.log"
    guard = _run_guards(log_path, preset_cfg.ctest_regex, preset_cfg.min_gtests)
    for msg in guard.messages:
        # Emit to stderr so ::error:: lines surface in the GitHub Actions
        # step annotation feed even when stdout is captured/redirected.
        print(msg, file=sys.stderr)

    if driver_rc != 0:
        # A real test failure should win over a guard failure - surface it
        # as-is so on-call sees the actual test error first.
        return driver_rc
    if not guard.ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
