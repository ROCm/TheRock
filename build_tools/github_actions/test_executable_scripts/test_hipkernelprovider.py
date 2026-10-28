# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Environment-setup wrapper around test_runner.py for hipkernelprovider.

Makes the `rocke` Python package importable, then delegates to test_runner.py
so test selection stays on the standard CTest category labels. Two independent
sources are staged into the test artifact: the wheels, which this installs, and
the co-located rocke package, which this puts on PYTHONPATH. Without at least
one, the rocKE CTest entries fail with ModuleNotFoundError.
"""

import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_RUNNER = SCRIPT_DIR / "test_runner.py"


def fail(message) -> NoReturn:
    """Report a setup failure as one line rather than a traceback.

    Exit 1 overlaps with ctest's error code and test_runner.py's own setup
    failures, but all of them mean the job failed; the ERROR line identifies
    the wrapper as the source.
    """
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def run_forwarding_exit_code(cmd) -> NoReturn:
    """Run cmd and exit with a status describing how it ended.

    A child killed by signal N gives a returncode of -N, which sys.exit() would
    mangle into 256-N; report the shell's 128+N instead.
    """
    try:
        returncode = subprocess.run(cmd).returncode
    except OSError as e:
        fail(f"could not execute {shlex.join(cmd)}: {e}")
    if returncode < 0:
        signal_number = -returncode
        print(
            f"ERROR: {Path(cmd[-1]).name} was terminated by signal {signal_number}.",
            file=sys.stderr,
        )
        sys.exit(128 + signal_number)
    sys.exit(returncode)


logging.basicConfig(level=logging.INFO)

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
if not THEROCK_BIN_DIR:
    fail("THEROCK_BIN_DIR environment variable is required but not set.")

test_dir = Path(THEROCK_BIN_DIR) / "hip_kernel_provider"
wheel_dir = test_dir / "wheels"
# Path.glob, not glob.glob: the latter would treat a glob metacharacter in
# THEROCK_BIN_DIR as part of the pattern and silently match nothing.
wheels = sorted(str(p) for p in wheel_dir.glob("*.whl"))
platform_wheels = [w for w in wheels if Path(w).name.startswith("rocke-")]
if wheels:
    logging.info(f"Installing rocke wheels: {wheels}")
    # uv, not `python -m pip`: setup_venv.py builds the CI venv with `uv venv`,
    # which does not seed pip. The platform wheel owns the sdk-core extra;
    # rocke_library and other staged wheels keep their dependency-free install.
    other_wheels = [w for w in wheels if w not in platform_wheels]
    install_commands = []
    if platform_wheels:
        # Refresh the locally built platform wheel without replacing an already
        # compatible shared SDK. AMD precedes PyPI so uv's first-index strategy
        # does not select PyPI's rocm-sdk-core placeholder instead of the SDK.
        install_commands.append(
            [
                "uv",
                "pip",
                "install",
                "--reinstall-package",
                "rocke",
                "--index",
                "https://nightly.repo.amd.com/rocm/whl-next/",
                "--default-index",
                "https://pypi.org/simple/",
                "--prerelease",
                "allow",
                "--python",
                sys.executable,
            ]
            + [f"{w}[sdk-core]" for w in platform_wheels]
        )
    if other_wheels:
        # ROCKE_WHEEL_VERSION is not bumped per build, so a reused venv would
        # otherwise keep the old wheel.
        install_commands.append(
            [
                "uv",
                "pip",
                "install",
                "--no-deps",
                "--reinstall",
                "--python",
                sys.executable,
            ]
            + other_wheels
        )
    for install_cmd in install_commands:
        logging.info(f"++ Exec $ {shlex.join(install_cmd)}")
        try:
            subprocess.run(install_cmd, check=True)
        except OSError as e:
            fail(
                f"could not run uv ({e}); it installs the rocKE wheels (venv has no pip)."
            )
        except subprocess.CalledProcessError as e:
            fail(f"installing the rocKE wheels failed with exit code {e.returncode}.")
else:
    logging.info(f"No rocKE wheels staged in {wheel_dir}, skipping install")

# Fallback to the co-located rocke package for any rocKE CTest entry that does
# not pin its own PYTHONPATH. The wheels and the package are staged under
# separate CMake options, so either can be absent independently; the entries
# themselves are gated separately again and may run with neither.
if (test_dir / "rocke").is_dir():
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(test_dir), existing]))
    logging.info(f"PYTHONPATH += {test_dir}")
elif not wheels:
    logging.warning(
        f"Neither rocKE wheels nor a rocke package found under {test_dir}. "
        "Any rocKE CTest entry that runs will fail to import rocke."
    )

# Native installed tests live outside the interpreter prefix. Resolve and load
# the selected compiler in the installed Python environment, then forward it to
# CTest without replacing a caller-supplied nonempty override.
if platform_wheels or (test_dir / "rocke").is_dir():
    probe = """
import json
from dataclasses import asdict
from rocke.runtime.comgr import _resolve_lib, loaded_compiler_info
_resolve_lib()
info = loaded_compiler_info()
if info is None or info.llvm_version is None or not info.requested_comgr:
    raise RuntimeError("rocKE compiler could not be loaded and queried")
print(json.dumps(asdict(info)))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe], check=True, capture_output=True, text=True
        )
    except OSError as e:
        fail(f"could not execute the rocKE compiler probe: {e}")
    except subprocess.CalledProcessError as e:
        fail(f"rocKE compiler setup failed: {e.stderr or e}")
    try:
        info = json.loads(result.stdout)
        selected = info["requested_comgr"]
        if not isinstance(selected, str) or not selected or not info["llvm_version"]:
            raise ValueError("missing compiler path or version")
    except (ValueError, KeyError, TypeError) as e:
        fail(f"invalid rocKE compiler probe output: {e}")
    logging.info("Selected rocKE compiler: %s", info)
    if not os.environ.get("ROCKE_COMGR_LIB"):
        os.environ["ROCKE_COMGR_LIB"] = selected

# Subprocess rather than import: test_runner.py validates the environment and
# calls sys.exit() at module scope.
run_forwarding_exit_code([sys.executable, str(TEST_RUNNER)])
