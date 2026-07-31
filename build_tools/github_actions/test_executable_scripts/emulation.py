# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared helpers for component tests running under a GPU emulator.

Emulated test jobs are derived by
``build_tools/github_actions/fetch_test_configurations.py`` from an ``emulate``
field on a test matrix entry. Those jobs run on the CPU cluster with no GPU
attached, and their whole ``test_script`` is wrapped in a mirage session::

    <rocm>/bin/mirage run --profile mi350x --emulator rocjitsu \\
        --env TEST_EMULATOR=rocjitsu --env ... -- python test_runner.py

so a test script is already inside the emulated environment by the time it
starts, and never has to invoke mirage itself.

Which tests an emulated job runs is decided by the matrix entry
(``emulate_test_type``) rather than by the script, so most scripts need
nothing from this module. It exists for the ones that have to *know* they are
emulated -- like ``test_emulation_smoke.py``, whose whole job is to check the
emulated agent's identity.

``TEST_EMULATOR``
    Emulator backend name, e.g. ``rocjitsu``. Empty on hardware, so
    :func:`is_emulated` is the only switch a script needs.
``TEST_EMULATOR_PROFILE``
    mirage builtin profile pinning the emulated GPU, e.g. ``mi350x`` /
    ``mi450x``. See ``rocm-systems/emulation/mirage/builtin/src/profiles.rs``.
"""

import os
import sys
from collections.abc import Mapping
from pathlib import Path

# `os.environ` is an `os._Environ`, not a `dict`, so annotate the parameter as
# the read-only mapping these helpers actually need.
Env = Mapping[str, str]


def emulator_name(env: Env = os.environ) -> str:
    """Emulator backend for this job, or "" when running on hardware."""
    return (env.get("TEST_EMULATOR") or "").strip()


def emulator_profile(env: Env = os.environ) -> str:
    """mirage profile for this job, or "" when running on hardware."""
    return (env.get("TEST_EMULATOR_PROFILE") or "").strip()


def is_emulated(env: Env = os.environ) -> bool:
    """True when this job's tests run under an emulator."""
    return bool(emulator_name(env))


def rocm_path(env: Env = os.environ) -> Path:
    """Root of the ROCm install the artifacts were unpacked into.

    Mirrors test_runner.py: THEROCK_BIN_DIR points at ``<rocm>/bin``.
    """
    bin_dir = env.get("THEROCK_BIN_DIR")
    if not bin_dir:
        raise RuntimeError("THEROCK_BIN_DIR is not set")
    return Path(bin_dir).resolve().parent


def log_emulator_banner(env: Env = os.environ) -> None:
    """Print what is being emulated, for readability in CI logs."""
    print(f"# TEST_EMULATOR: {emulator_name(env) or '<none>'}")
    print(f"# TEST_EMULATOR_PROFILE: {emulator_profile(env) or '<none>'}")
    print(f"# AMDGPU_FAMILIES: {env.get('AMDGPU_FAMILIES', '<unset>')}")
    sys.stdout.flush()
