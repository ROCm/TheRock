# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke test for the mirage/rocjitsu emulator.

Runs `rocminfo` inside a mirage session and checks that the emulated agent it
reports matches the profile the job asked for. That single command exercises
the whole emulation stack end to end:

  mirage CLI -> builtin profile -> rocjitsu emulator -> emulated KFD/HSA ->
  ROCr runtime from the artifacts -> rocminfo

When this fails, every other emulated job is expected to fail too, so it is
deliberately the cheapest and loudest check we have.

Environment variables:
  TEST_EMULATOR          Emulator backend (set by test_component.yml).
  TEST_EMULATOR_PROFILE  mirage builtin profile, e.g. "mi350x" / "mi450x".
  THEROCK_BIN_DIR        ROCm bin directory from the fetched artifacts.
"""

import logging
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emulation

logging.basicConfig(level=logging.INFO)

# gfx target each mirage builtin profile is expected to present to the guest,
# from the builtin agent definitions in
# rocm-systems/emulation/mirage/builtin/src/agents.rs. This is the assertion
# the smoke test exists to make, so it is stated here rather than read back
# from anything the emulated stack produces.
EXPECTED_GFX_BY_PROFILE = {
    "mi300x": "gfx942",
    "mi350x": "gfx950",
    "mi450x": "gfx1250",
}

# rocminfo has to come up, enumerate the emulated agent and exit. Anything
# beyond this means the stack is wedged, and a hang that runs out the step
# timeout would otherwise print nothing at all.
ROCMINFO_TIMEOUT_SECONDS = 15 * 60


def check_rocminfo_output(output: str, profile: str) -> list[str]:
    """Validate rocminfo output for `profile`; returns a list of problems.

    Kept separate from process handling so it can be unit tested without an
    emulator.
    """
    problems = []

    if "Agent " not in output:
        problems.append("rocminfo reported no agents at all")

    gfx_names = set(re.findall(r"gfx[0-9a-z]+", output))
    if not gfx_names:
        problems.append("rocminfo reported no gfx agent name")

    expected = EXPECTED_GFX_BY_PROFILE.get(profile)
    if expected and not any(name.startswith(expected) for name in gfx_names):
        problems.append(
            f"expected a {expected} agent for mirage profile '{profile}', "
            f"but rocminfo reported {sorted(gfx_names) or 'none'}"
        )
    return problems


def main() -> int:
    if not emulation.is_emulated():
        print(
            "ERROR: test_emulation_smoke.py only runs in an emulated job "
            "(TEST_EMULATOR is unset).",
            file=sys.stderr,
        )
        return 1

    # This process is already inside the mirage session, so `rocminfo` reports
    # the emulated agent rather than whatever the host has (or has not) got.
    emulation.log_emulator_banner()

    profile = emulation.emulator_profile()
    rocminfo = emulation.rocm_path() / "bin" / "rocminfo"
    if not rocminfo.is_file():
        print(f"ERROR: rocminfo not found at {rocminfo}", file=sys.stderr)
        return 1

    # Already inside the mirage session, so this is a plain invocation.
    cmd = [str(rocminfo)]
    logging.info(f"++ Exec $ {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
            timeout=ROCMINFO_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as timeout:
        # Bounded here rather than left to the step timeout so the log still
        # shows how far the emulator got before it wedged.
        print(timeout.stdout or "")
        print(timeout.stderr or "", file=sys.stderr)
        print(
            f"ERROR: rocminfo did not exit within {ROCMINFO_TIMEOUT_SECONDS}s "
            f"under mirage profile '{profile}'.",
            file=sys.stderr,
        )
        return 1

    # Always echo the emulated run's output; it is the whole point of the job.
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(
            f"ERROR: rocminfo exited {result.returncode} under mirage profile "
            f"'{profile}'.",
            file=sys.stderr,
        )
        return result.returncode

    problems = check_rocminfo_output(result.stdout, profile)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    print(f"# rocminfo ran successfully on an emulated {profile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
