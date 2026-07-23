# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared helpers for running ROCm tests under the mirage GPU emulator.

These utilities let a test script execute an existing test binary on top of
``rocjitsu`` -- a CPU-based AMD GPU emulator -- through the ``mirage`` CLI. That
means integration tests can run on GPU-less ``rocjitsu-cpu`` runners instead of
requiring real accelerators.

To add a new emulated test:

  1. Fetch the ``mirage`` and ``rocjitsu`` artifacts (``--mirage --rocjitsu``)
     alongside the component under test in ``fetch_test_configurations.py``.
  2. Write a small test script that builds the native command (e.g.
     ``["./rocrtst64"]``) and calls :func:`build_mirage_run_command`.
  3. Add a matrix entry with ``"emulation": True`` so the job is routed to the
     ``rocjitsu-cpu`` node (see ``fetch_test_configurations.py``).

``mirage`` ships builtin profiles named after the GPU they emulate; every
builtin targets the ``rocjitsu`` software emulator. See
``rocm-systems/emulation/mirage/docs/cli.md`` for the full CLI reference.
"""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Default emulator backend. ``rocjitsu`` performs ISA-level execution of AMD GPU
# code objects on the CPU, so no accelerator is required on the runner.
DEFAULT_EMULATOR = "rocjitsu"

# Map TheRock AMDGPU family prefixes to the matching builtin mirage profile.
# Each builtin profile pins a single agent and targets the rocjitsu emulator:
#   mi300x -> MI300X (gfx942 / CDNA3)
#   mi350x -> MI350X (gfx950 / CDNA4)
#   mi450x -> MI450X (gfx1250)
# Extend this mapping as rocjitsu gains support for more agents.
_MIRAGE_PROFILE_BY_FAMILY_PREFIX = {
    "gfx942": "mi300x",
    "gfx94": "mi300x",  # gfx94X family label (gfx940/gfx941/gfx942)
    "gfx950": "mi350x",
    "gfx1250": "mi450x",
    "gfx125": "mi450x",  # gfx125X family label
}


def select_mirage_profile(amdgpu_family):
    """Return the builtin mirage profile for an AMDGPU family, or ``None``.

    ``None`` means rocjitsu has no emulated agent for this family, so the caller
    should skip the emulated test rather than run it against a mismatched agent.
    """
    if not amdgpu_family:
        return None
    family = amdgpu_family.lower()
    # Check longest prefixes first so that, e.g., "gfx1250" wins over "gfx125"
    # and "gfx950" wins over "gfx94".
    for prefix in sorted(_MIRAGE_PROFILE_BY_FAMILY_PREFIX, key=len, reverse=True):
        if family.startswith(prefix):
            return _MIRAGE_PROFILE_BY_FAMILY_PREFIX[prefix]
    return None


def find_mirage_binary(bin_dir=None):
    """Locate the ``mirage`` executable in ``bin_dir`` or on ``PATH``.

    The mirage artifact installs the binary as ``bin/mirage`` (see
    ``emulation/artifact-mirage.toml``), which lands in ``THEROCK_BIN_DIR``.
    """
    if bin_dir:
        candidate = Path(bin_dir) / "mirage"
        if candidate.exists():
            return str(candidate.resolve())
    found = shutil.which("mirage")
    if found:
        return found
    raise FileNotFoundError(
        "Could not find the 'mirage' binary. Ensure the mirage artifact is "
        "installed (fetch with --mirage) and available in THEROCK_BIN_DIR or "
        "on PATH."
    )


def build_mirage_run_command(
    inner_cmd,
    *,
    profile,
    emulator=DEFAULT_EMULATOR,
    passthrough_env=None,
    mirage_bin=None,
    bin_dir=None,
):
    """Wrap a native command with ``mirage run`` so it executes under emulation.

    Args:
        inner_cmd: The native command to run, as a list (e.g. ``["./rocrtst64"]``).
        profile: mirage builtin profile name (e.g. ``"mi450x"``).
        emulator: Emulator backend to force (defaults to ``rocjitsu``).
        passthrough_env: Optional mapping of environment variables to forward
            into the emulated process via ``--env KEY=VALUE`` (e.g. GTest
            sharding/filter variables). Entries with a ``None`` value are
            skipped.
        mirage_bin: Explicit path to the mirage binary. If omitted, it is
            resolved from ``bin_dir`` then ``PATH``.
        bin_dir: Directory to search for the mirage binary when ``mirage_bin``
            is not given.

    Returns:
        The full ``mirage run`` argv as a list of strings. ``mirage run``
        propagates the exit code of the wrapped command, so callers can rely on
        ``subprocess.run(..., check=True)``.
    """
    mirage = mirage_bin or find_mirage_binary(bin_dir)
    cmd = [mirage, "run", "--profile", profile, "--emulator", emulator]
    for key, value in (passthrough_env or {}).items():
        if value is not None:
            cmd += ["--env", f"{key}={value}"]
    cmd.append("--")
    cmd.extend(str(part) for part in inner_cmd)
    return cmd
