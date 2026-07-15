# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test of the library trees."""

"""Installation package tests for the core package."""

import importlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import unittest

from .. import _dist_info as di
from . import utils

import rocm_sdk


class ROCmDevelTest(unittest.TestCase):
    def testInstallationLayout(self):
        """The `rocm_sdk` and devel module must be siblings on disk."""
        sdk_path = Path(rocm_sdk.__file__)
        self.assertEqual(
            sdk_path.name,
            "__init__.py",
            msg="Expected `rocm_sdk` module to be a non-namespace package",
        )
        import rocm_sdk_devel

        devel_path = Path(rocm_sdk_devel.__file__)
        self.assertEqual(
            devel_path.name,
            "__init__.py",
            msg=f"Expected `rocm_sdk_devel` module to be a non-namespace package",
        )
        self.assertEqual(
            sdk_path.parent.parent,
            devel_path.parent.parent,
            msg="Paths are not siblings",
        )

    def testCLIPathBin(self):
        cmd = [sys.executable, "-m", "rocm_sdk", "path", "--bin"]
        output = utils.run_command(cmd, capture=True).decode().strip()
        path = Path(output)
        self.assertTrue(path.exists(), msg=f"Expected bin path {path} to exist")

    def testCLIPathCMake(self):
        cmd = [sys.executable, "-m", "rocm_sdk", "path", "--cmake"]
        output = utils.run_command(cmd, capture=True).decode().strip()
        path = Path(output)
        self.assertTrue(path.exists(), msg=f"Expected cmake path {path} to exist")
        hip_file = path / "hip" / "hip-config.cmake"
        self.assertTrue(
            hip_file.exists(), msg=f"Expected hip config to exist {hip_file}"
        )

    def testCLIPathRoot(self):
        cmd = [sys.executable, "-m", "rocm_sdk", "path", "--root"]
        output = utils.run_command(cmd, capture=True).decode().strip()
        path = Path(output)
        self.assertTrue(path.exists(), msg=f"Expected root path {path} to exist")
        bin_path = path / "bin"
        self.assertTrue(bin_path.exists(), msg=f"Expected bin path {bin_path} to exist")

    def testCLIUsesDevelRootPath(self):
        root_path_output = (
            utils.run_command(
                [sys.executable, "-m", "rocm_sdk", "path", "--root"], capture=True
            )
            .decode()
            .strip()
        )
        root_path = Path(root_path_output)

        # CLI scripts by default run from _rocm_sdk_core.
        # When the devel package is installed they should run from _rocm_sdk_devel.
        rocmpath_output = (
            utils.run_command(["hipconfig", "--rocmpath"], capture=True)
            .decode()
            .strip()
        )
        rocmpath = Path(rocmpath_output)
        self.assertTrue(
            root_path.is_dir(), msg=f"Expected root path {root_path} to exist"
        )
        self.assertTrue(
            rocmpath.is_dir(),
            msg=f"Expected `hipconfig --rocmpath` directory {rocmpath} to exist",
        )
        # On Linux, RHEL-like venvs often have lib64 -> lib; root_path may spell
        # lib64 while rocmpath realpaths to lib. Use samefile for this reason.
        self.assertTrue(
            os.path.samefile(root_path, rocmpath),
            msg=(
                "Expected `hipconfig --rocmpath` and `rocm_sdk path --root` to refer to the "
                f"same directory; got {root_path} vs {rocmpath} "
                f"(resolved: {root_path.resolve()} vs {rocmpath.resolve()})"
            ),
        )

    @unittest.skipIf(
        platform.system() == "Windows", "root LLVM symlink only exists on Linux"
    )
    def testRootLLVMSymlinkExists(self):
        # We had a bug where the root llvm/ symlink, which is for backwards compat,
        # was not materialized. Verify it is.
        cmd = [sys.executable, "-m", "rocm_sdk", "path", "--root"]
        output = utils.run_command(cmd, capture=True).decode().strip()
        path = Path(output) / "llvm" / "bin" / "clang++"
        self.assertTrue(path.exists(), msg=f"Expected {path} to exist")

    def testSharedLibrariesLoad(self):
        # Make sure the devel package is expanded.
        cmd = [sys.executable, "-m", "rocm_sdk", "path", "--root"]
        _ = utils.run_command(cmd, capture=True).decode().strip()

        # Ensure that the platform package exists now.
        mod_name = di.ALL_PACKAGES["devel"].get_py_package_name(
            target_family=di.determine_target_family()
        )
        mod = importlib.import_module(mod_name)
        utils.assert_is_physical_package(mod)
        so_paths = utils.get_module_shared_libraries(mod)

        self.assertTrue(
            so_paths, msg="Expected core package to contain shared libraries"
        )

        for so_path in so_paths:
            if "clang_rt" in str(so_path):
                # clang_rt and sanitizer libraries are not all intended to be
                # loadable arbitrarily.
                continue
            if "libhipsolver_fortran" in str(so_path):
                # Currently fails to load unless libgfortran.so.5 exists on the system.
                # TODO(#3115): Decide if this test should be permanently
                #     disabled or fixed and then re-enabled somehow. This
                #     library may only be used by tests and we might not care
                #     about it failing to load standalone.
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
            if so_path.name.endswith(".abi3.so") or ".cpython-" in so_path.name:
                # Python C extensions use symbols resolved at import time,
                # not via dlopen — ctypes.CDLL fails across interpreter
                # versions (e.g. .abi3.so using PyType_FromMetaclass on <3.12).
                continue

            extra_setup = ""
            if (
                "hipdnn_plugins" in str(so_path) or "test_plugins" in str(so_path)
            ) and platform.system() == "Windows":
                # hipdnn plugins have dependencies on other libraries (e.g. miopen).
                # In a real-world scenario, hipdnn_backend loads these plugins, and
                # the dependencies are found because they reside in the same directory
                # (or are otherwise resolvable).
                # To simulate this loading behavior in the test:
                # - On Linux, RPATH ($ORIGIN/../../) handles dependency resolution.
                # - On Windows, we must manually add the library directory (calculated
                #   relative to the plugin) via add_dll_directory, as there is no RPATH equivalent.
                # We assume the plugin is at .../{lib|bin}/hipdnn_plugins/engines/plugin.so
                # and the dependencies are at .../{lib|bin}.
                lib_dir = str(so_path.parents[2]).replace("\\", "\\\\")
                extra_setup = f"import os; os.add_dll_directory('{lib_dir}') if hasattr(os, 'add_dll_directory') else None; "
                # The rocke-client engine also DT_NEEDEDs the kpack runtime, which
                # ships in the core wheel and is not under lib_dir; add its dir too.
                kpack_paths = rocm_sdk.find_libraries("rocm_kpack")
                if kpack_paths:
                    kpack_dir = str(kpack_paths[0].parent).replace("\\", "\\\\")
                    extra_setup += f"os.add_dll_directory('{kpack_dir}') if hasattr(os, 'add_dll_directory') else None; "

            with self.subTest(msg="Check shared library loads", so_path=so_path):
                # Load each in an isolated process because not all libraries in the tree
                # are designed to load into the same process (i.e. LLVM runtime libs,
                # etc).
                command = (
                    extra_setup + "import ctypes; import sys; ctypes.CDLL(sys.argv[1])"
                )

                subprocess.check_call([sys.executable, "-c", command, str(so_path)])

    def testLibrariesMirroredIntoDevel(self):
        """Every file in the libraries platform tree must also appear in the
        devel tree as the same (hardlinked) file.

        Host libraries are mirrored when the devel tree is expanded; per-ISA
        device payloads (.kpack archives, Tensile/MIOpen kernels, per-arch .so)
        are mirrored by `_devel._reconcile_device_links`. Walking libraries and
        checking each entry exists in devel is sufficient because devel is a
        superset - extra devel files (headers, cmake, soname aliases, other
        arches) are irrelevant. The only libraries entry with no devel
        counterpart by design is the `.devel_links/` manifest dir, which drives
        the reconcile rather than being a payload.
        """
        target_family = di.determine_target_family()
        try:
            libraries_mod = importlib.import_module(
                di.ALL_PACKAGES["libraries"].get_py_package_name(
                    target_family=target_family
                )
            )
        except ModuleNotFoundError:
            self.skipTest("rocm-sdk-libraries is not installed")

        # Expand the devel tree and reconcile device links before comparing.
        utils.run_command(
            [sys.executable, "-m", "rocm_sdk", "path", "--root"], capture=True
        )
        try:
            devel_mod = importlib.import_module(
                di.ALL_PACKAGES["devel"].get_py_package_name(
                    target_family=target_family
                )
            )
        except ModuleNotFoundError:
            self.skipTest("rocm[devel] is not installed")

        def _platform_dir(mod) -> Path:
            # In kpack-split mode `_rocm_sdk_libraries` is a namespace package
            # (no __init__.py) so the libraries and device wheels can share the
            # directory; __file__ is then None, so fall back to __path__.
            if mod.__file__ is not None:
                return Path(mod.__file__).parent
            return Path(next(iter(mod.__path__)))

        libraries_dir = _platform_dir(libraries_mod)
        devel_dir = _platform_dir(devel_mod)

        def _skip(rel: Path) -> bool:
            # Each platform package has its own independent __init__.py marker.
            if rel == Path("__init__.py"):
                return True
            if "__pycache__" in rel.parts:
                return True
            # Device-wheel manifest dir: drives the reconcile, not a payload.
            return rel.parts[0] == ".devel_links"

        missing = []
        not_hardlinked = []
        for libs_file in libraries_dir.rglob("*"):
            if not libs_file.is_file():
                continue
            rel = libs_file.relative_to(libraries_dir)
            if _skip(rel):
                continue
            devel_file = devel_dir / rel
            if not devel_file.is_file():
                missing.append(rel)
            elif not devel_file.samefile(libs_file):
                not_hardlinked.append(rel)

        def _fmt(items: list) -> str:
            shown = ", ".join(str(p) for p in items[:10])
            return shown + (f" (+{len(items) - 10} more)" if len(items) > 10 else "")

        self.assertFalse(
            missing,
            msg=f"{len(missing)} libraries file(s) missing from devel: {_fmt(missing)}",
        )
        self.assertFalse(
            not_hardlinked,
            msg=(
                f"{len(not_hardlinked)} libraries file(s) present in devel but not "
                f"hardlinked (different inode): {_fmt(not_hardlinked)}"
            ),
        )
