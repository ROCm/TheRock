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
import time

THIS_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR")).resolve()

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")

# Importing is_asan from github_actions_api.py
sys.path.append(str(THIS_DIR.parent / "build_tools" / "github_actions"))
from github_actions_api import is_asan


def is_windows():
    return "windows" == platform.system().lower()


def run_command(command: list[str], cwd=None, capture=True):
    logger.info(f"++ Run [{cwd}]$ {shlex.join(command)}")
    process = subprocess.run(
        command,
        capture_output=capture,
        cwd=cwd,
        shell=is_windows(),
        text=True,
    )
    if process.returncode != 0:
        if capture:
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
    def test_hip_simple(self):
        """Minimal kernel without printf to isolate gfx1150 hang (TheRock#3199)."""
        platform_executable_suffix = ".exe" if is_windows() else ""
        offload_arch_path = (
            THEROCK_BIN_DIR
            / ".."
            / "lib"
            / "llvm"
            / "bin"
            / f"offload-arch{platform_executable_suffix}"
        ).resolve()
        process = run_command([str(offload_arch_path)])
        offload_arch = None
        for line in process.stdout.splitlines():
            if "gfx" in line:
                offload_arch = line
                break
        assert (
            offload_arch is not None
        ), f"Expected offload-arch to return gfx####, got:\n{process.stdout}"

        executable = f"hip_simple_check{platform_executable_suffix}"
        run_command(
            [
                f"{THEROCK_BIN_DIR}/hipcc",
                str(THIS_DIR / "hip_simple_check.cpp"),
                "-Xlinker",
                f"-rpath={THEROCK_BIN_DIR}/../lib/",
                f"--offload-arch={offload_arch}",
                "-o",
                executable,
            ],
            cwd=str(THEROCK_BIN_DIR),
        )

        # Dump dmesg before running to capture pre-existing GPU firmware
        # errors that may cause all subsequent kernel launches to stall.
        try:
            dmesg_before = subprocess.run(
                ["dmesg", "--level=err,warn", "-T"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            logger.info("=== dmesg (err+warn) before hip_simple_check ===")
            for line in dmesg_before.stdout.splitlines()[-50:]:
                logger.info(line)
        except Exception as e:
            logger.warning(f"dmesg failed: {e}")

        # Run with a 30s grace period; if it hangs, attach gdb to get
        # userspace backtraces for all threads (TheRock#3199).
        # The container has --cap-add SYS_PTRACE for this purpose.
        platform_executable_prefix = "./" if not is_windows() else ""
        exe = f"{platform_executable_prefix}{executable}"
        logger.info(f"++ Run [{THEROCK_BIN_DIR}]$ {exe}")
        proc = subprocess.Popen(
            [exe],
            cwd=str(THEROCK_BIN_DIR),
        )

        hang_timeout = 30
        try:
            proc.wait(timeout=hang_timeout)
        except subprocess.TimeoutExpired:
            logger.error(f"hip_simple_check hung for {hang_timeout}s — attaching gdb")

            # Dump dmesg again to see if new GPU errors appeared during hang
            try:
                dmesg_after = subprocess.run(
                    ["dmesg", "-T"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                logger.info("=== dmesg (last 100 lines) after hang ===")
                for line in dmesg_after.stdout.splitlines()[-100:]:
                    logger.info(line)
            except Exception as e:
                logger.warning(f"dmesg failed: {e}")

            # Install gdb (container runs as root, image is Ubuntu 24.04)
            try:
                subprocess.run(
                    ["apt-get", "install", "-y", "-qq", "gdb"],
                    capture_output=True,
                    timeout=60,
                )
            except Exception as e:
                logger.warning(f"Failed to install gdb: {e}")

            # Attach gdb and dump backtraces for all threads
            gdb_cmd = [
                "gdb",
                "-batch",
                "-ex",
                "thread apply all bt full",
                "-ex",
                "info registers",
                "-ex",
                "detach",
                "-p",
                str(proc.pid),
            ]
            try:
                gdb_proc = subprocess.run(
                    gdb_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                logger.info("=== gdb backtrace output ===")
                for line in gdb_proc.stdout.splitlines():
                    logger.info(line)
                if gdb_proc.stderr:
                    logger.info("=== gdb stderr ===")
                    for line in gdb_proc.stderr.splitlines():
                        logger.info(line)
            except FileNotFoundError:
                logger.error("gdb not found, falling back to /proc")
                # Fallback: read /proc/PID/maps for address info
                try:
                    maps = Path(f"/proc/{proc.pid}/maps").read_text()
                    logger.info(f"=== /proc/{proc.pid}/maps ===")
                    for line in maps.splitlines():
                        if (
                            "libhsa" in line
                            or "libamdhip" in line
                            or "hip_simple" in line
                        ):
                            logger.info(line)
                except Exception as e:
                    logger.warning(f"Cannot read /proc maps: {e}")
            except Exception as e:
                logger.error(f"gdb failed: {e}")

            proc.kill()
            proc.wait()

        check.equal(proc.returncode, 0)

    # TODO(#3313): Re-enable once hipcc test is fixed for ASAN builds
    @pytest.mark.skipif(
        is_asan(), reason="hipcc test fails with ASAN build, see TheRock#3313"
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

        # Running and checking the executable.
        # capture=False so HIP debug output streams directly to the log
        # if the process hangs (see TheRock#3199).
        platform_executable_prefix = "./" if not is_windows() else ""
        hipcc_check_executable = f"{platform_executable_prefix}hipcc_check"
        process = run_command(
            [hipcc_check_executable], cwd=str(THEROCK_BIN_DIR), capture=False
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

    @pytest.mark.skipif(is_windows(), reason="amdsmitst is not supported on Windows")
    # TODO(#2789): Remove skip once amdsmi supports gfx1151
    @pytest.mark.skipif(
        AMDGPU_FAMILIES == "gfx1151", reason="Linux gfx1151 does not support amdsmi yet"
    )
    def test_amdsmi_suite(self):
        amdsmi_test_bin = (
            THEROCK_BIN_DIR.parent / "share" / "amd_smi" / "tests" / "amdsmitst"
        ).resolve()

        assert (
            amdsmi_test_bin.exists()
        ), f"amdsmitst not found at expected location: {amdsmi_test_bin}"
        assert os.access(
            amdsmi_test_bin, os.X_OK
        ), f"amdsmitst is not executable: {amdsmi_test_bin}"

        include_tests = [
            "amdsmitstReadOnly.*",
            "amdsmitstReadWrite.FanReadWrite",
            "amdsmitstReadWrite.TestOverdriveReadWrite",
            "amdsmitstReadWrite.TestPciReadWrite",
            "amdsmitstReadWrite.TestPowerReadWrite",
            "amdsmitstReadWrite.TestPerfCntrReadWrite",
            "amdsmitstReadWrite.TestEvtNotifReadWrite",
            "AmdSmiDynamicMetricTest.*",
        ]

        exclude_tests = [
            "amdsmitstReadOnly.TempRead",
            "amdsmitstReadOnly.TestFrequenciesRead",
            "amdsmitstReadWrite.TestPowerReadWrite",
        ]

        TESTS_TO_IGNORE = {
            "gfx90a": {
                # TODO(#2963): Re-enable once amdsmi tests are fixed for gfx90X-dcgpu
                "linux": [
                    "amdsmitstReadOnly.TestSysInfoRead",
                    "amdsmitstReadOnly.TestIdInfoRead",
                    "amdsmitstReadWrite.TestPciReadWrite",
                ]
            },
            "gfx110X-all": {
                # TODO(#2963): Re-enable once amdsmi tests are fixed for gfx110X-all
                "linux": [
                    "amdsmitstReadWrite.FanReadWrite",
                ]
            },
        }

        platform_key = "windows" if is_windows() else "linux"
        if (
            AMDGPU_FAMILIES in TESTS_TO_IGNORE
            and platform_key in TESTS_TO_IGNORE[AMDGPU_FAMILIES]
        ):
            ignored_tests = TESTS_TO_IGNORE[AMDGPU_FAMILIES][platform_key]
            exclude_tests.extend(ignored_tests)

        gtest_filter = f"{':'.join(include_tests)}:-{':'.join(exclude_tests)}"
        cmd = [str(amdsmi_test_bin), f"--gtest_filter={gtest_filter}"]

        process = run_command(cmd, cwd=str(amdsmi_test_bin.parent))

        combined = (process.stdout or "") + "\n" + (process.stderr or "")
        for line in combined.splitlines():
            if "[==========]" in line:
                print(f"[amdsmitst-summary] {line}")

        check.equal(process.returncode, 0)
