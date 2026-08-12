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


def _opencl_env():
    """Point the system OpenCL ICD loader at this build's vendor runtime.

    clinfo loads the vendor (amdocl64) through the system ICD loader
    (libOpenCL / OpenCL.dll); TheRock does not ship the loader. Setting
    OCL_ICD_FILENAMES makes the test exercise this build's runtime without
    relying on a system-wide /etc or registry ICD registration (absent in the
    CI container).
    """
    env = os.environ.copy()
    if is_windows():
        # Bare-metal runner: the driver registers amdocl64 system-wide, so point
        # at the build's vendor only if present; otherwise fall back to the loader.
        vendor = THEROCK_BIN_DIR / "amdocl64.dll"
        if vendor.exists():
            env["OCL_ICD_FILENAMES"] = str(vendor)
        return env
    # Linux runs in a container with no system ICD registration, so the vendor
    # is required; fail loud rather than letting the search asserts fail opaquely.
    lib = THEROCK_BIN_DIR.parent / "lib"
    vendor = lib / "opencl" / "libamdocl64.so"
    if not vendor.exists():
        raise FileNotFoundError(f"amdocl64 vendor runtime not found at {vendor}")
    env["OCL_ICD_FILENAMES"] = str(vendor)
    # The vendor's deps (libamd_comgr, libhsa-runtime64, sysdeps) are spread
    # across the install's lib dirs; make them resolvable regardless of how the
    # artifacts flatten, so the loader can dlopen the vendor.
    candidates = (lib, lib / "llvm" / "lib", lib / "rocm_sysdeps" / "lib")
    ld_path = [str(d) for d in candidates if d.is_dir()]
    if env.get("LD_LIBRARY_PATH"):
        ld_path.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_path)
    # Surface the loader's reason (e.g. a failed dlopen) if enumeration fails.
    env["OCL_ICD_DEBUG"] = "1"
    return env


def _diag(label, cmd, env=None):
    """Run a diagnostic command and log its combined output (never raises)."""
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    logger.error(f"--- {label} ---")
    for line in (r.stdout + r.stderr).splitlines() or ["(no output)"]:
        logger.error(line)


def _log_opencl_diagnostics(env):
    """On clinfo failure, gather loader + ICD-registration info to root-cause -1001."""
    if is_windows():
        return
    clinfo = f"{THEROCK_BIN_DIR}/clinfo"
    vendor = env.get("OCL_ICD_FILENAMES", "")
    root = THEROCK_BIN_DIR.parent
    logger.error(f"OCL_ICD_FILENAMES={vendor}")
    _diag("ldd clinfo", ["ldd", clinfo])
    # Identify the loader behind libOpenCL.so.1 (ocl-icd vs Khronos).
    _diag(
        "loader identity",
        [
            "bash",
            "-lc",
            f"L=$(ldd {clinfo} | awk '/libOpenCL/{{print $3}}'); readlink -f $L; "
            "dpkg -S $(readlink -f $L) 2>/dev/null; "
            "strings $L 2>/dev/null | grep -iE 'ocl-icd|khronos|OCL_ICD' | head",
        ],
    )
    # Decisive: canonical /etc/OpenCL/vendors registration with an absolute path,
    # no OCL_ICD_* env, AMD_LOG_LEVEL=4 so CLR logs whether it initializes.
    reg = {k: v for k, v in env.items() if not k.startswith("OCL_ICD")}
    reg["AMD_LOG_LEVEL"] = "4"
    _diag(
        "clinfo via /etc/OpenCL/vendors (AMD_LOG_LEVEL=4)",
        [
            "bash",
            "-lc",
            f"mkdir -p /etc/OpenCL/vendors && echo '{vendor}' > /etc/OpenCL/vendors/amdocl64.icd && {clinfo}",
        ],
        env=reg,
    )


@pytest.fixture(scope="session")
def clinfo_output():
    clinfo = f"{THEROCK_BIN_DIR}/clinfo" + (".exe" if is_windows() else "")
    env = _opencl_env()
    try:
        return str(run_command([clinfo], env=env).stdout)
    except Exception as e:
        logger.info(str(e))
        _log_opencl_diagnostics(env)
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

    # clinfo enumerates the GPU through the system OpenCL ICD loader ->
    # amdocl64, exercising the OpenCL runtime path that rocminfo does not.
    # Runs on Windows too, where OpenCL (PAL backend) is the enumeration path
    # that works (rocminfo is HSA-only and unsupported there).
    @pytest.mark.skipif(
        is_asan(),
        reason="runtime GPU enumeration is flaky under ASAN, see TheRock#3312",
    )
    @pytest.mark.parametrize(
        "to_search",
        [
            (r"Platform\s*Name:\s*AMD Accelerated Parallel Processing"),
            (r"Device\s*Type:\s*CL_DEVICE_TYPE_GPU"),
            (r"Name:\s*gfx"),
        ],
        ids=[
            "clinfo - AMD Platform Search",
            "clinfo - GPU Device Type Search",
            "clinfo - GFX Name Search",
        ],
    )
    def test_clinfo_output(self, clinfo_output, to_search):
        if not clinfo_output:
            pytest.fail("Command clinfo failed to run")
        check.is_not_none(
            re.search(to_search, clinfo_output),
            f"Failed to search for {to_search} in clinfo output",
        )

    # TODO(#4755): Re-enable test for windows once offload-arch.exe is fixed
    @pytest.mark.skipif(
        is_windows(),
        reason="Windows offload-arch.exe is not retrieving correct data, ignoring test",
    )
    def test_hip_printf(self):
        platform_executable_suffix = ".exe" if is_windows() else ""

        # Look up offload arch, e.g. gfx1100, for explicit `--offload-arch`.
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

        # Compiling .cpp file using amdclang++
        rocm_path = (THEROCK_BIN_DIR / "..").resolve()
        hip_check_executable_file = f"hip_check{platform_executable_suffix}"
        run_command(
            [
                f"{THEROCK_BIN_DIR}/amdclang++",
                f"--hip-path={rocm_path}",
                f"--hip-device-lib-path={rocm_path}/lib/llvm/amdgcn/bitcode",
                "-x",
                "hip",
                str(THIS_DIR / "hip_check.cpp"),
                "-Xlinker",
                f"-rpath={THEROCK_BIN_DIR}/../lib/",
                f"--offload-arch={offload_arch}",
                "-o",
                hip_check_executable_file,
            ],
            cwd=str(THEROCK_BIN_DIR),
        )

        # Running and checking the executable
        platform_executable_prefix = "./" if not is_windows() else ""
        hip_check_executable = f"{platform_executable_prefix}hip_check"
        process = run_command([hip_check_executable], cwd=str(THEROCK_BIN_DIR))
        check.equal(process.returncode, 0)
        check.greater(
            os.path.getsize(str(THEROCK_BIN_DIR / hip_check_executable_file)), 0
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
