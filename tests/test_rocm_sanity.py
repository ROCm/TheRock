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
import shutil
import subprocess
import sys
import tempfile

THIS_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR")).resolve()

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")

# Importing is_asan from amdgpu_family_matrix.py
sys.path.append(str(THIS_DIR.parent / "build_tools" / "github_actions"))
from amdgpu_family_matrix import is_asan


def is_windows():
    return "windows" == platform.system().lower()


def run_command(
    command: list[str], cwd=None, env: dict[str, str] | None = None, capture=True
):
    logger.info(f"++ Run [{cwd}]$ {shlex.join(command)}")
    process = subprocess.run(
        command, capture_output=capture, cwd=cwd, text=True, env=env
    )
    if process.returncode != 0:
        logger.error(f"Command failed!")
        if capture:
            logger.error("command stdout:")
            for line in process.stdout.splitlines():
                logger.error(line)
            logger.error("command stderr:")
            for line in process.stderr.splitlines():
                logger.error(line)
        raise Exception(f"Command failed: `{shlex.join(command)}`, see output above")
    return process


def get_offload_arch() -> str:
    platform_executable_suffix = ".exe" if is_windows() else ""

    # Look up offload arch, e.g. gfx1100, for explicit `--offload-arch`.
    offload_arch_executable_file = f"offload-arch{platform_executable_suffix}"
    offload_arch_path = (
        THEROCK_BIN_DIR / ".." / "lib" / "llvm" / "bin" / offload_arch_executable_file
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
    return offload_arch


def compile_hip(source: str, executable_file: str, offload_arch: str):
    platform_executable_suffix = ".exe" if is_windows() else ""
    # Compiling .cpp file using amdclang++
    # On Linux, bin/amdclang++ is a symlink to lib/llvm/bin/amdclang++.
    # On Windows, the symlink is not created, so use lib/llvm/bin/ directly.
    rocm_path = (THEROCK_BIN_DIR / "..").resolve()
    if is_windows():
        amdclangxx_path = str(
            (
                THEROCK_BIN_DIR
                / ".."
                / "lib"
                / "llvm"
                / "bin"
                / f"amdclang++{platform_executable_suffix}"
            ).resolve()
        )
    else:
        amdclangxx_path = f"{THEROCK_BIN_DIR}/amdclang++"
    command = [
        amdclangxx_path,
        f"--hip-path={rocm_path}",
        f"--hip-device-lib-path={rocm_path}/lib/llvm/amdgcn/bitcode",
        "-x",
        "hip",
        str(THIS_DIR / source),
        f"--offload-arch={offload_arch}",
        "-o",
        executable_file,
    ]
    if not is_windows():
        # Windows lld-link does not support rpath.
        command.extend(["-Xlinker", f"-rpath={THEROCK_BIN_DIR}/../lib/"])
    run_command(command, cwd=str(THEROCK_BIN_DIR))


def dump_kmsg(label: str, last_n: int = 50):
    """Log the tail of the kernel log; /dev/kmsg bypasses dmesg_restrict."""
    try:
        import fcntl

        with open("/dev/kmsg", "r", errors="replace") as f:
            fd = f.fileno()
            fcntl.fcntl(
                fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK
            )
            lines = []
            try:
                while line := f.readline():
                    lines.append(line.rstrip())
            except (BlockingIOError, OSError):
                pass
        if lines:
            logger.info(f"=== {label} (last {last_n} from /dev/kmsg) ===")
            for line in lines[-last_n:]:
                logger.info(line)
            return
    except (ImportError, OSError):
        pass
    try:
        result = subprocess.run(
            ["dmesg", "-T"], capture_output=True, text=True, timeout=10
        )
        logger.info(f"=== {label} (last {last_n} from dmesg) ===")
        for line in result.stdout.splitlines()[-last_n:]:
            logger.info(line)
    except Exception as e:
        logger.warning(f"{label}: dmesg failed: {e}")


@pytest.fixture(scope="session")
def rocm_info_output():
    try:
        return str(run_command([f"{THEROCK_BIN_DIR}/rocminfo"]).stdout)
    except Exception as e:
        logger.info(str(e))
        return None


@pytest.fixture(scope="session")
def clinfo_output() -> str:
    env = os.environ.copy()
    if is_windows():
        vendor = THEROCK_BIN_DIR / "amdocl64.dll"
        if not vendor.is_file():
            raise FileNotFoundError(f"OpenCL vendor runtime not found: {vendor}")
        env["PATH"] = str(THEROCK_BIN_DIR) + os.pathsep + env.get("PATH", "")
        # The system ICD loader can fall back to a registered driver. Load the
        # built vendor directly via clinfo's application-local OpenCL.dll.
        # amdocl64 exports the OpenCL entry points used by clinfo.
        with tempfile.TemporaryDirectory(prefix="therock-clinfo-") as directory:
            clinfo = Path(directory) / "clinfo.exe"
            shutil.copy2(THEROCK_BIN_DIR / "clinfo.exe", clinfo)
            shutil.copy2(vendor, Path(directory) / "OpenCL.dll")
            # Keep Comgr beside the vendor so CLR cannot pick up an older
            # driver-installed DLL from System32 before searching PATH.
            shutil.copy2(
                THEROCK_BIN_DIR / "amd_comgr.dll",
                Path(directory) / "amd_comgr.dll",
            )
            process = run_command([str(clinfo)], cwd=directory, env=env)
            # clinfo can exit successfully even when device initialization fails.
            if process.stderr:
                logger.info("clinfo stderr:\n%s", process.stderr)
            return process.stdout

    lib_dir = THEROCK_BIN_DIR.parent / "lib"
    vendor = lib_dir / "opencl" / "libamdocl64.so"
    if not vendor.is_file():
        raise FileNotFoundError(f"OpenCL vendor runtime not found: {vendor}")

    # Support both the distro ocl-icd loader and the Khronos loader without
    # depending on system-wide ICD registration.
    env["OCL_ICD_VENDORS"] = str(vendor)
    env["OCL_ICD_FILENAMES"] = str(vendor)
    library_dirs = (lib_dir, lib_dir / "llvm" / "lib", lib_dir / "rocm_sysdeps" / "lib")
    library_path = [str(path) for path in library_dirs if path.is_dir()]
    if env.get("LD_LIBRARY_PATH"):
        library_path.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(library_path)

    return run_command([str(THEROCK_BIN_DIR / "clinfo")], env=env).stdout


class TestROCmSanity:
    @pytest.mark.skipif(
        is_asan(),
        reason="runtime GPU enumeration is flaky under ASAN, see TheRock#3312",
    )
    @pytest.mark.skipif(
        AMDGPU_FAMILIES and "gfx125X-dcgpu" in AMDGPU_FAMILIES,
        reason="clinfo test is disabled for gfx125X-dcgpu due to hangs",
    )
    @pytest.mark.parametrize(
        "to_search",
        [
            r"Platform\s*Name:\s*AMD Accelerated Parallel Processing",
            r"Device\s*Type:\s*CL_DEVICE_TYPE_GPU",
            r"Name:\s*gfx",
        ],
        ids=[
            "clinfo - AMD Platform Search",
            "clinfo - GPU Device Type Search",
            "clinfo - GFX Name Search",
        ],
    )
    def test_clinfo_output(self, clinfo_output: str, to_search: str):
        check.is_not_none(
            re.search(to_search, clinfo_output),
            f"Failed to search for {to_search} in clinfo output:\n{clinfo_output}",
        )

    def test_hipify_version(self):
        executable = "hipify-clang.exe" if is_windows() else "hipify-clang"
        # Exercise the installed tool's library search paths, not the test runner's.
        env = os.environ.copy()
        env.pop("LD_LIBRARY_PATH", None)
        env.pop("LD_PRELOAD", None)
        process = subprocess.run(
            [str(THEROCK_BIN_DIR / executable), "--version"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        assert "LLVM version" in process.stdout

    @pytest.mark.skipif(is_windows(), reason="rocminfo is not supported on Windows")
    # TODO(#3312): Re-enable once rocminfo test is fixed for ASAN builds
    @pytest.mark.skipif(
        is_asan(), reason="rocminfo test fails with ASAN build, see TheRock#3312"
    )
    # TODO(#7659): Re-enable once rocminfo is fixed for gfx125X-dcgpu
    @pytest.mark.skipif(
        AMDGPU_FAMILIES and "gfx125X-dcgpu" in AMDGPU_FAMILIES,
        reason="rocminfo test is disabled for gfx125X-dcgpu due to kernel bug, see #7659",
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

    @pytest.mark.skipif(is_windows(), reason="gfx1150 hang debugging, Linux only")
    def test_hip_simple(self):
        """Minimal kernel without printf to isolate gfx1150 hang (TheRock#3199)."""
        executable_file = "hip_simple_check"
        compile_hip("hip_simple_check.cpp", executable_file, get_offload_arch())

        dump_kmsg("kernel log before hip_simple_check")
        # The binary has a 20s SIGALRM watchdog that dumps a backtrace and exits 2.
        exe = str(THEROCK_BIN_DIR / executable_file)
        logger.info(f"++ Run [{THEROCK_BIN_DIR}]$ {exe}")
        proc = subprocess.Popen([exe], cwd=str(THEROCK_BIN_DIR))
        hang_timeout = 30
        try:
            proc.wait(timeout=hang_timeout)
        except subprocess.TimeoutExpired:
            logger.error(f"hip_simple_check hung for {hang_timeout}s, killing")
            proc.kill()
            proc.wait()
        if proc.returncode != 0:
            dump_kmsg("kernel log after hip_simple_check", last_n=100)
        check.equal(proc.returncode, 0)

    # TODO(#7458): Re-enable once gfx1250 binary translator supports this kernel code pattern
    @pytest.mark.skipif(
        AMDGPU_FAMILIES and "gfx125X-dcgpu" in AMDGPU_FAMILIES,
        reason="gfx1250 binary translator does not yet support this kernel code pattern, see #7458",
    )
    def test_hip_vector_add(self):
        platform_executable_suffix = ".exe" if is_windows() else ""
        offload_arch = get_offload_arch()
        hip_check_executable_file = f"hip_check{platform_executable_suffix}"
        compile_hip("hip_check.cpp", hip_check_executable_file, offload_arch)

        # Stream output so HIP debug logs are visible if it hangs (TheRock#3199).
        hip_check_executable = str(THEROCK_BIN_DIR / hip_check_executable_file)
        process = run_command(
            [hip_check_executable], cwd=str(THEROCK_BIN_DIR), capture=False
        )
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
