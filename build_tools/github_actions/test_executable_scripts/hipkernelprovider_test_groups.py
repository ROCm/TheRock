# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""hip-kernel-provider's CI test groups, one job each (fetch_test_configurations.py).

Each engine group selects the tests whose CTest labels match its regex. Every
engine's external integration suites carry the engine name as a label (set by
add_external_integration_test_target() in rocm-libraries). The "unit" group
runs everything no engine group claims, so a test missing here still runs.

run_group() runs test_runner.py's main() with its ctest command narrowed to one
group, so tier, GPU and sharding handling stay test_runner.py's.
"""

import sys

# Group name -> CTest label regex. Regexes stay within the subset that Python re
# and CMake string(REGEX) (ctest -L/-LE) read the same way.
ENGINE_GROUPS = {
    "hip-mlops": "^HIP_MLOPS_ENGINE$",
    "asm-sdpa": "^ASM_SDPA_ENGINE$",
    "flydsl": "^hipkernel_flydsl_",
    "rocke": "^hipkernel_Gfx",
}
UNIT_GROUP = "unit"
GROUPS = (UNIT_GROUP, *ENGINE_GROUPS)


def add_group_filter(cmd: list[str], group: str) -> list[str]:
    """Return a copy of a ctest command narrowed to one group's tests.

    An engine group adds `-L <regex>`; ctest ANDs repeated -L, so the tier and
    GPU labels still apply. The unit group excludes every engine group. ctest
    also ANDs repeated -LE, so the exclusion joins any existing -LE pattern.
    """
    if group not in GROUPS:
        raise ValueError(f"unknown hipkernelprovider test group {group!r}")
    cmd = list(cmd)
    if group != UNIT_GROUP:
        return cmd + ["-L", ENGINE_GROUPS[group]]

    exclude = "|".join(ENGINE_GROUPS.values())
    if "-LE" in cmd:
        i = cmd.index("-LE") + 1
        cmd[i] = f"{cmd[i]}|{exclude}"
        return cmd
    return cmd + ["-LE", exclude]


def run_group(group: str) -> int:
    """Run test_runner.py's main() with its ctest command narrowed to `group`."""
    import test_runner

    build = test_runner.build_ctest_command

    def build_for_group(*args, **kwargs):
        return add_group_filter(build(*args, **kwargs), group)

    test_runner.build_ctest_command = build_for_group
    print(f"# hipkernelprovider test group: {group}")
    return test_runner.main()


if __name__ == "__main__":
    sys.exit(run_group(sys.argv[1]))
