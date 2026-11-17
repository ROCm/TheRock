# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Running component tests against an emulated GPU.

Some GPU targets have no test hardware in CI. A test component can opt into an
*emulated variant* that runs on the CPU cluster, with the mirage CLI driving
the rocjitsu software GPU emulator instead of a real device::

    "rocrtst": {..., "emulate": "rocjitsu"}

Everything specific to emulation lives here, so fetch_test_configurations.py
stays a list of components. See docs/development/adding_tests.md.
"""

import os
import re
import sys
from collections.abc import Mapping
from copy import deepcopy

# AMDGPU family prefix -> mirage builtin profile, matched longest key first.
# See rocm-systems/emulation/mirage/builtin/src/profiles.rs.
MIRAGE_PROFILE_BY_FAMILY_PREFIX = {
    "gfx94": "mi300x",
    "gfx950": "mi350x",
    "gfx125": "mi450x",
}

# Profiles we schedule jobs for. gfx94X maps above so local runs can resolve it,
# but it has ample CI hardware.
EMULATED_PROFILES = frozenset({"mi350x", "mi450x"})

# Emulated jobs are far slower than the same tests on hardware, so scale the
# component's declared timeout -- but cap it, since a component that needs more
# than an hour needs a cheaper `emulate_test_type`, not a longer leash.
TIMEOUT_MULTIPLIER = 10
MAX_TIMEOUT_MINUTES = 60

# Artifacts an emulated job needs on top of its component's own.
FETCH_ARTIFACT_ARGS = ("--mirage", "--rocjitsu")

# Matrix keys that configure emulation. Consumed here, never emitted: leaking
# them would make `component.emulate` truthy on hardware jobs too.
MATRIX_KEYS = (
    "emulate",
    "emulate_only",
    "emulate_test_type",
    "emulate_env",
)

# `mirage host` starts the workload with env_clear(), so anything the workflow
# put in the environment has to cross the session boundary by name. Cross-check
# against test_component.yml's `env:` blocks, not the test scripts: variables
# read by the ROCm runtime are invisible to a grep of the scripts.
FORWARDED_ENV = (
    "THEROCK_BIN_DIR",
    "OUTPUT_ARTIFACTS_DIR",
    "AMDGPU_FAMILIES",
    "AMDGPU_TARGETS",
    "TEST_TYPE",
    "TEST_COMPONENT",
    "SHARD_INDEX",
    "TOTAL_SHARDS",
    # Thread limits test_component.yml derives from the pod's CPU budget.
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    # Read by the ROCm runtime; set for every other job in the matrix.
    "ROCM_KPACK_DEBUG",
    # setup-python's interpreter links against a libpython under this path, so
    # without it `python` inside the session dies with exit 127 before running
    # anything. Safe for rocjitsu, which injects LD_PRELOAD and never sets this;
    # an emulator that does set it (mirage's hotswap) would be overridden here.
    "LD_LIBRARY_PATH",
)

# What an `emulate_env` NAME=VALUE pair may contain. Narrow on purpose: the
# wrapped command crosses the workflow and a shell before mirage sees it.
_ENV_LITERAL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_.:/,+-]+")


# ---------------------------------------------------------------------------
# Matrix side: used by fetch_test_configurations.py in the configure job.
# ---------------------------------------------------------------------------


def get_mirage_profile(amdgpu_families: str | None) -> str | None:
    """Return the mirage builtin profile emulating `amdgpu_families`, if any."""
    if not amdgpu_families:
        return None
    family = amdgpu_families.lower()
    for prefix in sorted(MIRAGE_PROFILE_BY_FAMILY_PREFIX, key=len, reverse=True):
        if family.startswith(prefix):
            return MIRAGE_PROFILE_BY_FAMILY_PREFIX[prefix]
    return None


def get_emulated_profile(amdgpu_families: str | None, platform: str) -> str | None:
    """Return the profile to schedule emulated jobs for, or None for hardware."""
    if platform != "linux":
        # The mirage and rocjitsu artifacts are Linux only.
        return None
    profile = get_mirage_profile(amdgpu_families)
    return profile if profile in EMULATED_PROFILES else None


def wrap_in_mirage_run(
    test_script: str,
    emulator: str,
    profile: str,
    emulate_env: dict[str, str] | None = None,
) -> str:
    """Wrap `test_script` so the whole thing runs inside a mirage session.

    Emulator and profile are baked in as literals so the command reproduces a
    run on its own. The result contains no quote characters, matching the rest
    of the matrix; no path in it may contain spaces.
    """
    # $THEROCK_BIN_DIR, not a fixed path: test_component.yml unpacks artifacts
    # to ./build, the reproduction container to ./therock-build.
    parts = [
        "$THEROCK_BIN_DIR/mirage",
        "run",
        "--profile",
        profile,
        "--emulator",
        emulator,
        f"--env TEST_EMULATOR={emulator}",
        f"--env TEST_EMULATOR_PROFILE={profile}",
    ]
    # Sorted so the emitted command is stable across configure runs.
    for name, value in sorted((emulate_env or {}).items()):
        if not _ENV_LITERAL_RE.fullmatch(f"{name}={value}"):
            raise ValueError(
                f"emulate_env entry {name}={value!r} is not a quote-free, "
                "space-free literal; mirage receives this as a bare shell word"
            )
        parts.append(f"--env {name}={value}")
    # `${NAME:+...}` so a variable the workflow left unset stays unset, rather
    # than being set to "" -- which is not the same thing to libgomp.
    parts += [f"${{{name}:+--env {name}=${name}}}" for name in FORWARDED_ENV]
    parts += ["--", test_script]
    return " ".join(parts)


def build_emulated_job(job_config: dict, emulator: str, profile: str) -> dict:
    """Derive the emulated variant of an already-expanded job config.

    Deriving from the expanded config means the variant inherits every earlier
    decision -- exclude_family, test labels, project selection, container
    image -- for free.
    """
    emulated = deepcopy(job_config)
    base_job_name = job_config["job_name"]

    emulated["job_name"] = f"{base_job_name} (emulated {profile})"
    # test_runner.py maps TEST_COMPONENT to a test directory, and the decorated
    # job name is not one.
    emulated["test_component"] = job_config.get("test_component", base_job_name)
    emulated["emulator"] = emulator
    emulated["emulator_profile"] = profile
    # rocjitsu emulates the GPU in software, so this must not hold a GPU runner.
    emulated["linux_cpu_runner"] = True
    emulated["timeout_minutes"] = min(
        job_config["timeout_minutes"] * TIMEOUT_MULTIPLIER,
        MAX_TIMEOUT_MINUTES,
    )
    # install_rocm_from_artifacts.py treats --base-only as exclusive, so
    # "--base-only --mirage" would fetch neither. Dropping it loses nothing:
    # the extra-artifact branch pulls the base patterns too.
    base_args = [
        arg
        for arg in job_config.get("fetch_artifact_args", "").split()
        if arg != "--base-only"
    ]
    emulated["fetch_artifact_args"] = " ".join(base_args + list(FETCH_ARTIFACT_ARGS))
    # An emulated job already runs a reduced subset; sharding it costs more in
    # artifact fetches than it saves.
    emulated["total_shards"] = 1
    emulated["shard_arr"] = [1]
    # A pin, not a default: a nightly asking for "comprehensive" must not drag
    # the emulated variant into a category the emulator cannot finish.
    emulate_test_type = job_config.get("emulate_test_type")
    if emulate_test_type:
        emulated["test_type"] = emulate_test_type
    emulated["test_script"] = wrap_in_mirage_run(
        job_config["test_script"],
        emulator,
        profile,
        job_config.get("emulate_env"),
    )
    # test_artifacts.yml checks these *before* linux_cpu_runner when picking a
    # runner, so leaving either would route this job onto GPU hardware.
    emulated.pop("multi_gpu", None)
    emulated.pop("multi_gpu_runner", None)
    emulated.pop("is_benchmark", None)
    for key in MATRIX_KEYS:
        emulated.pop(key, None)
    return emulated


# ---------------------------------------------------------------------------
# Run-time side: used by test scripts already inside the mirage session.
# ---------------------------------------------------------------------------

# `os.environ` is an `os._Environ`, not a `dict`.
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


def log_emulator_banner(env: Env = os.environ) -> None:
    """Print what is being emulated, for readability in CI logs."""
    print(f"# TEST_EMULATOR: {emulator_name(env) or '<none>'}")
    print(f"# TEST_EMULATOR_PROFILE: {emulator_profile(env) or '<none>'}")
    print(f"# AMDGPU_FAMILIES: {env.get('AMDGPU_FAMILIES', '<unset>')}")
    print(f"# AMDGPU_TARGETS: {env.get('AMDGPU_TARGETS', '<unset>')}")
    sys.stdout.flush()
