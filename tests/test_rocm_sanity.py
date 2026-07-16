# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
from pytest_check import check
import logging
import os
import platform
import pytest
import re
import shlex
import subprocess
import sys

THIS_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR")).resolve()

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")

# Importing is_asan from amdgpu_family_matrix.py
sys.path.append(str(THIS_DIR.parent / "build_tools" / "github_actions"))
from amdgpu_family_matrix import is_asan


def is_windows():
    return "windows" == platform.system().lower()


def get_asan_preload_env():
    """Returns env with LD_PRELOAD for ASAN builds, None otherwise."""
    if not is_asan():
        return None
    machine = platform.machine()
    if machine in ("x86_64", "AMD64"):
        arch = "x86_64"
    elif machine == "aarch64":
        arch = "aarch64"
    else:
        return None
    clangxx = THEROCK_BIN_DIR / ".." / "lib" / "llvm" / "bin" / "clang++"
    if not clangxx.exists():
        return None
    result = subprocess.run(
        [str(clangxx), f"--print-file-name=libclang_rt.asan-{arch}.so"],
        capture_output=True,
        text=True,
    )
    asan_lib = result.stdout.strip()
    if not asan_lib or not Path(asan_lib).exists():
        return None
    env = os.environ.copy()
    env["LD_PRELOAD"] = asan_lib
    return env


def run_command(command: list[str], cwd=None, env=None):
    logger.info(f"++ Run [{cwd}]$ {shlex.join(command)}")
    process = subprocess.run(
        command, capture_output=True, cwd=cwd, shell=is_windows(), text=True, env=env
    )
    if process.returncode != 0:
        logger.error(f"Command failed!")
        logger.error("command stdout:")
        for line in process.stdout.splitlines():
            logger.error(line)
        logger.error("command stderr:")
        for line in process.stderr.splitlines():
            logger.error(line)
        raise Exception(f"Command failed: `{shlex.join(command)}`, see output above")
    return process


@pytest.fixture(scope="session")
def rocm_info_output():
    try:
        return str(run_command([f"{THEROCK_BIN_DIR}/rocminfo"]).stdout)
    except Exception as e:
        logger.info(str(e))
        return None


class TestROCmSanity:
    @pytest.mark.skipif(is_windows(), reason="rocminfo is not supported on Windows")
    # TODO(#3312): Re-enable once rocminfo test is fixed for ASAN builds
    @pytest.mark.skipif(
        is_asan(), reason="rocminfo test fails with ASAN build, see TheRock#3312"
    )
    @pytest.mark.parametrize(
        "to_search",
        [
            (r"Device\s*Type:\s*GPU"),
            (r"Name:\s*gfx"),
            (r"Vendor\s*Name:\s*AMD"),
        ],
        ids=[
            "rocminfo - GPU Device Type Search",
            "rocminfo - GFX Name Search",
            "rocminfo - AMD Vendor Name Search",
        ],
    )
    def test_rocm_output(self, rocm_info_output, to_search):
        if not rocm_info_output:
            pytest.fail("Command rocminfo failed to run")
        check.is_not_none(
            re.search(to_search, rocm_info_output),
            f"Failed to search for {to_search} in rocminfo output",
        )

    # TODO(#3313): Re-enable once hipcc test is fixed for ASAN builds
    @pytest.mark.skipif(
        is_asan(), reason="hipcc test fails with ASAN build, see TheRock#3313"
    )
    # TODO(#4755): Re-enable test for windows once offload-arch.exe is fixed
    @pytest.mark.skipif(
        is_windows(),
        reason="Windows offload-arch.exe is not retrieving correct data, ignoring test",
    )
    def test_hip_printf(self):
        platform_executable_suffix = ".exe" if is_windows() else ""

        # Look up offload arch, e.g. gfx1100, for explicit `--offload-arch`.
        # See https://github.com/ROCm/llvm-project/issues/302:
        #   * If this is omitted on Linux, hipcc uses rocm_agent_enumerator.
        #   * If this is omitted on Windows, hipcc uses a default (e.g. gfx906).
        # We include it on both platforms for consistency.
        offload_arch_executable_file = f"offload-arch{platform_executable_suffix}"
        offload_arch_path = (
            THEROCK_BIN_DIR
            / ".."
            / "lib"
            / "llvm"
            / "bin"
            / offload_arch_executable_file
        ).resolve()
        process = run_command([str(offload_arch_path)])

        # Extract the arch from the command output, working around
        # https://github.com/ROCm/TheRock/issues/1118. We only expect the output
        # to contain 'gfx####` text but some ROCm releases contained stray
        # "HIP Library Path" logging first.
        # **Note**: this partly defaults the purpose of the sanity check, since
        # that should really be a test failure. However, per discussion on
        # https://github.com/ROCm/TheRock/pull/3257 we found that system
        # installs of ROCm (DLLs in system32) take precedence over user
        # installs (PATH env var) under certain conditions. Hopefully a
        # different unit test elsewhere in ROCm catches that more directly.
        offload_arch = None
        for line in process.stdout.splitlines():
            if "gfx" in line:
                offload_arch = line
                break
        assert (
            offload_arch is not None
        ), f"Expected offload-arch to return gfx####, got:\n{process.stdout}"

        # Compiling .cpp file using hipcc
        hipcc_check_executable_file = f"hipcc_check{platform_executable_suffix}"
        run_command(
            [
                f"{THEROCK_BIN_DIR}/hipcc",
                str(THIS_DIR / "hipcc_check.cpp"),
                "-Xlinker",
                f"-rpath={THEROCK_BIN_DIR}/../lib/",
                f"--offload-arch={offload_arch}",
                "-o",
                hipcc_check_executable_file,
            ],
            cwd=str(THEROCK_BIN_DIR),
        )

        # Running and checking the executable
        platform_executable_prefix = "./" if not is_windows() else ""
        hipcc_check_executable = f"{platform_executable_prefix}hipcc_check"
        process = run_command(
            [hipcc_check_executable],
            cwd=str(THEROCK_BIN_DIR),
            env=get_asan_preload_env(),
        )
        check.equal(process.returncode, 0)
        check.greater(
            os.path.getsize(str(THEROCK_BIN_DIR / hipcc_check_executable_file)), 0
        )

    @pytest.mark.skipif(
        is_windows(),
        reason="rocm_agent_enumerator is not supported on Windows",
    )
    def test_rocm_agent_enumerator(self):
        process = run_command([f"{THEROCK_BIN_DIR}/rocm_agent_enumerator"])
        output = process.stdout
        return_code = process.returncode
        check.equal(return_code, 0)
        check.is_true(output)
