# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Checks that the emulated GPU comes up and identifies itself correctly.

The job's `test_script` is already wrapped in a mirage session, so running a
ROCm tool here exercises the whole stack: mirage -> profile -> rocjitsu ->
emulated KFD/HSA -> ROCr runtime from the artifacts -> the tool.

Not a duplicate of the "Driver / GPU sanity check" step, which is skipped here:
that reports on the host's amdgpu driver, and an emulated job runs on a CPU
runner that has none. This device exists only inside the mirage session.

TEST_TYPE selects how much to run; see docs/development/test_filtering.md.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import emulation

logging.basicConfig(level=logging.INFO)

THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR", "./build/bin"))

# Test categories in increasing depth. The ffm-* tiers are the same depths
# under a different name.
CATEGORY_DEPTH = {
    "quick": 1,
    "standard": 2,
    "comprehensive": 3,
    "full": 4,
}

# Tools to run, with the shallowest category that includes each. Both
# enumerate the emulated agent; rocminfo walks more of the runtime to do it.
CHECKS = (
    ("quick", "rocm_agent_enumerator"),
    ("standard", "rocminfo"),
)

# A tool that has not exited by now is wedged, not slow. Bounded here rather
# than by the step timeout so the log still shows how far it got.
CHECK_TIMEOUT_SECONDS = 10 * 60


def get_test_depth(test_type: str) -> int:
    """Depth for `test_type`, defaulting to the shallowest."""
    category = test_type.lower().removeprefix("ffm-")
    depth = CATEGORY_DEPTH.get(category)
    if depth is None:
        logging.warning(f"Unknown TEST_TYPE '{test_type}', running quick checks")
        return CATEGORY_DEPTH["quick"]
    return depth


def get_expected_targets() -> set[str]:
    """gfx targets this job was built and scheduled for.

    From the job's own inputs rather than a table of what each profile
    emulates: the assertion worth making is that the emulated device matches
    the artifacts under test.
    """
    return {
        target.strip()
        for target in os.getenv("AMDGPU_TARGETS", "").split(",")
        if target.strip()
    }


def run_check(tool: str) -> str:
    """Run `tool` inside the mirage session and return its stdout."""
    cmd = [str(THEROCK_BIN_DIR / tool)]
    logging.info(f"++ Exec $ {' '.join(cmd)}")
    # Already inside the mirage session, so this is a plain invocation.
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
        timeout=CHECK_TIMEOUT_SECONDS,
    )
    # Always echo it; the emulated run's output is the point of the job.
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.stdout


def main() -> int:
    if not emulation.is_emulated():
        print(
            "ERROR: test_emulation.py only runs in an emulated job "
            "(TEST_EMULATOR is unset).",
            file=sys.stderr,
        )
        return 1

    emulation.log_emulator_banner()

    depth = get_test_depth(os.getenv("TEST_TYPE") or "quick")
    reported_targets = set()
    for category, tool in CHECKS:
        if CATEGORY_DEPTH[category] > depth:
            logging.info(f"Skipping {tool}, which needs TEST_TYPE >= {category}")
            continue
        output = run_check(tool)
        reported_targets |= {w for w in output.split() if w.startswith("gfx")}

    if not reported_targets:
        print("ERROR: the emulated stack reported no gfx agent", file=sys.stderr)
        return 1

    expected_targets = get_expected_targets()
    if not expected_targets:
        # Warn rather than fail: the checks above already proved the stack
        # comes up, and there is nothing to compare against.
        logging.warning("AMDGPU_TARGETS is unset, not checking the agent identity")
    elif not expected_targets & reported_targets:
        print(
            f"ERROR: expected an agent for {sorted(expected_targets)}, but the "
            f"emulator reported {sorted(reported_targets)}",
            file=sys.stderr,
        )
        return 1

    print(f"# Emulated agent(s) present: {sorted(reported_targets)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
