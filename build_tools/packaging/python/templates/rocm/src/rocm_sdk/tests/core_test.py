# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Installation package tests for the core package."""

import importlib
import locale
import mmap
from pathlib import Path
import platform
import struct
import subprocess
import sys
import unittest

from .. import _dist_info as di
from . import utils

import rocm_sdk

# Keep in sync with PATCHELF_PAD_MARKER in
# build_tools/_therock_utils/py_packaging.py. See that file for why the pad
# exists and why we want the absence of this marker in shipped binaries.
PATCHELF_PAD_MARKER_BYTES = b"__therock_patchelf_pad__"
ELF_MAGIC = b"\x7fELF"
# ELF program header flag bits.
PF_X = 0x1
PF_W = 0x2
PF_R = 0x4
PT_LOAD = 1

utils.assert_is_physical_package(rocm_sdk)

core_mod_name = di.ALL_PACKAGES["core"].get_py_package_name()
core_mod = importlib.import_module(core_mod_name)
utils.assert_is_physical_package(core_mod)

so_paths = utils.get_module_shared_libraries(core_mod)
is_windows = platform.system() == "Windows"

# Console script tests are templated across tuples of
#   (script_name, cl, expected_text, required)
# For example:
#   ("hipcc", ["--help"], "clang LLVM compiler", True)
#   This will run `hipcc --help` and check the output for "clang LLVM compiler".
#   If the script does not exist, the test case will fail.
COMMON_CONSOLE_SCRIPT_TESTS = [
    ("amdclang", ["--help"], "clang LLVM compiler", True),
    ("amdclang++", ["--help"], "clang LLVM compiler", True),
    ("amdclang-cpp", ["--help"], "clang LLVM compiler", True),
    ("amdclang-cl", ["-help"], "clang LLVM compiler", True),
    ("amdflang", ["--help"], "LLVM compiler", True),
    ("amdlld", ["-flavor", "ld.lld", "--help"], "USAGE:", True),
    ("hipcc", ["--help"], "clang LLVM compiler", True),
    ("hipconfig", [], "HIP version:", True),
    ("hipify-clang", ["--help"], "USAGE:", True),
]

LINUX_CONSOLE_SCRIPT_TESTS = [
    ("amd-smi", [], "AMD-SMI", True),
    ("rocm_agent_enumerator", [], "", True),
    ("rocminfo", [], "", True),
    ("rocm-smi", [], "Management", True),
    ("hipify-perl", ["--help"], "USAGE:", True),
]

WINDOWS_CONSOLE_SCRIPT_TESTS = [
    ("hipInfo", [], "", True),
]

CONSOLE_SCRIPT_TESTS = COMMON_CONSOLE_SCRIPT_TESTS + (
    WINDOWS_CONSOLE_SCRIPT_TESTS if is_windows else LINUX_CONSOLE_SCRIPT_TESTS
)


def _file_contains_bytes(path: Path, needle: bytes) -> bool:
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size < len(needle):
        return False
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return mm.find(needle) != -1


def _is_elf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == ELF_MAGIC
    except OSError:
        return False


def _parse_elf64_pt_loads(path: Path):
    """Return (e_entry, [(p_flags, p_vaddr, p_memsz), ...]) for PT_LOAD segments.

    Only supports 64-bit little-endian ELFs. Returns None for anything else
    (32-bit, big-endian, non-ELF). ROCm only ships 64-bit LE binaries.
    """
    with open(path, "rb") as f:
        ehdr = f.read(64)
        if len(ehdr) < 64 or not ehdr.startswith(ELF_MAGIC):
            return None
        if ehdr[4] != 2 or ehdr[5] != 1:
            return None
        e_entry, e_phoff = struct.unpack_from("<QQ", ehdr, 0x18)
        e_phentsize, e_phnum = struct.unpack_from("<HH", ehdr, 0x36)
        if e_phentsize < 56 or e_phnum == 0:
            return None
        f.seek(e_phoff)
        phdrs = f.read(e_phentsize * e_phnum)
    segments = []
    for i in range(e_phnum):
        base = i * e_phentsize
        p_type = struct.unpack_from("<I", phdrs, base)[0]
        if p_type != PT_LOAD:
            continue
        p_flags = struct.unpack_from("<I", phdrs, base + 0x04)[0]
        p_vaddr = struct.unpack_from("<Q", phdrs, base + 0x10)[0]
        p_memsz = struct.unpack_from("<Q", phdrs, base + 0x28)[0]
        segments.append((p_flags, p_vaddr, p_memsz))
    return e_entry, segments


class ROCmCoreTest(unittest.TestCase):
    def testInstallationLayout(self):
        """The `rocm_sdk` and core module must be siblings on disk."""
        sdk_path = Path(rocm_sdk.__file__)
        self.assertEqual(
            sdk_path.name,
            "__init__.py",
            msg="Expected `rocm_sdk` module to be a non-namespace package",
        )
        core_path = Path(core_mod.__file__)
        self.assertEqual(
            core_path.name,
            "__init__.py",
            msg="Expected core module to be a non-namespace package",
        )
        self.assertEqual(
            sdk_path.parent.parent,
            core_path.parent.parent,
            msg="Paths are not siblings",
        )

    def testSharedLibrariesLoad(self):
        self.assertTrue(
            so_paths, msg="Expected core package to contain shared libraries"
        )

        for so_path in so_paths:
            if "amd_smi" in str(so_path) or "goamdsmi" in str(so_path):
                # TODO: Library preloads for amdsmi need to be implement.
                # Though this is not needed for the amd-smi client.
                continue
            if "clang_rt" in str(so_path):
                # clang_rt and sanitizer libraries are not all intended to be
                # loadable arbitrarily.
                continue
            if "libLLVMOffload" in str(so_path):
                # recent addition from upstream, issue tracked in
                # https://github.com/ROCm/TheRock/issues/2537
                continue
            if "lib/roctracer" in str(so_path) or "share/roctracer" in str(so_path):
                # Internal roctracer libraries are meant to be pre-loaded
                # explicitly and cannot necessarily be loaded standalone.
                continue
            if (
                "lib/rocprofiler-sdk/" in str(so_path)
                or "libexec/rocprofiler-sdk/" in str(so_path)
                or "libpyrocpd" in str(so_path)
                or "libpyroctx" in str(so_path)
            ):
                # Internal rocprofiler-sdk libraries are meant to be pre-loaded
                # explicitly and cannot necessarily be loaded standalone.
                continue
            if "libtest_linking_lib" in str(so_path):
                # rocprim unit tests, not actual library files
                continue
            if "opencl" in str(so_path):
                # We use OpenCL ICD from distro rather than TheRock
                # and we do not build it
                continue
            with self.subTest(msg="Check shared library loads", so_path=so_path):
                # Load each in an isolated process because not all libraries in the tree
                # are designed to load into the same process (i.e. LLVM runtime libs,
                # etc).
                command = "import ctypes; import sys; ctypes.CDLL(sys.argv[1])"
                cmd = [sys.executable, "-c", command, str(so_path)]
                subprocess.check_call(cmd)

    def testConsoleScripts(self):
        for script_name, cl, expected_text, required in CONSOLE_SCRIPT_TESTS:
            script_path = utils.find_console_script(script_name)
            if not required and script_path is None:
                continue
            with self.subTest(msg=f"Check console-script {script_name}"):
                self.assertIsNotNone(
                    script_path,
                    msg=f"Console script {script_path} does not exist",
                )
                encoding = locale.getpreferredencoding()
                output_text = subprocess.check_output(
                    [script_path] + cl,
                    stderr=subprocess.STDOUT,
                ).decode(encoding)
                if expected_text not in output_text:
                    self.fail(
                        f"Expected '{expected_text}' in console-script {script_name} outuput:\n"
                        f"{output_text}"
                    )

    def testPreloadLibraries(self):
        target_family = di.determine_target_family()

        for lib_entry in di.ALL_LIBRARIES.values():
            # Only test for packages we have installed.
            if lib_entry.package.has_py_package(target_family):
                with self.subTest(
                    msg="Check rocm_sdk.preload_libraries",
                    shortname=lib_entry.shortname,
                ):
                    rocm_sdk.preload_libraries(lib_entry.shortname)

    @unittest.skipIf(is_windows, "Patchelf RPATH pad only applies on ELF")
    def testPatchelfPadStripped(self):
        """No shipped file should still contain the patchelf RPATH pad marker.

        TheRock links binaries with a ~1KB filler RPATH entry whose job is to
        reserve .dynstr space for patchelf to overwrite in place at packaging
        time. Finding the marker in an installed file means py_packaging
        didn't rewrite that binary's RPATH (likely a file-type detection
        miss), so the ~1KB pad leaked into the shipped artifact. Harmless at
        runtime but a packaging bug worth failing CI on.

        Context: https://github.com/ROCm/TheRock/issues/4271
        """
        core_root = Path(core_mod.__file__).parent
        offenders = []
        for f in core_root.rglob("*"):
            if not f.is_file() or f.is_symlink():
                continue
            if _file_contains_bytes(f, PATCHELF_PAD_MARKER_BYTES):
                offenders.append(str(f.relative_to(core_root)))
        self.assertFalse(
            offenders,
            msg=(
                "Files still contain the patchelf RPATH pad marker "
                f"{PATCHELF_PAD_MARKER_BYTES.decode()!r}; py_packaging did not "
                "rewrite their RPATHs:\n  " + "\n  ".join(sorted(offenders))
            ),
        )

    @unittest.skipIf(is_windows, "ELF layout check only applies on Linux")
    def testExecutableElfLayoutIntact(self):
        """Every shipped ELF executable's first PT_LOAD must be non-writable.

        The RHEL 8.10 / EL 4.18 execve() crash reproduced in issue #4271 is
        triggered by ELFs whose first PT_LOAD segment is read-write at a
        non-canonical base address (e.g. 0x3ff000 instead of 0x400000). That
        layout is the telltale signature of patchelf having grown .dynstr and
        prepended a new PT_LOAD. We assert the invariant directly instead of
        trying to enumerate every kernel that cares: the first PT_LOAD of a
        freshly linked executable is always R or RX, never RW.

        Shared objects (`.so`) are excluded because the kernel bug only
        affects executables loaded via execve(); ld.so handles the layout
        correctly for DT_NEEDED loads.
        """
        core_root = Path(core_mod.__file__).parent
        bin_dir = core_root / "bin"
        if not bin_dir.is_dir():
            self.skipTest("Core package has no bin/ directory")

        offenders = []
        checked = 0
        for f in bin_dir.rglob("*"):
            if not f.is_file() or f.is_symlink():
                continue
            if not _is_elf(f):
                continue
            parsed = _parse_elf64_pt_loads(f)
            if parsed is None:
                continue
            _, segments = parsed
            if not segments:
                continue
            checked += 1
            first_flags, first_vaddr, _ = segments[0]
            if first_flags & PF_W:
                offenders.append(
                    f"{f.relative_to(core_root)}: first PT_LOAD is writable "
                    f"(flags=0x{first_flags:x}, vaddr=0x{first_vaddr:x})"
                )

        self.assertGreater(checked, 0, msg=f"Found no ELF executables under {bin_dir}")
        self.assertFalse(
            offenders,
            msg=(
                "ELF layout looks patchelf-mutated (issue #4271); expected "
                "first PT_LOAD to be read-only on executables:\n  "
                + "\n  ".join(sorted(offenders))
            ),
        )
