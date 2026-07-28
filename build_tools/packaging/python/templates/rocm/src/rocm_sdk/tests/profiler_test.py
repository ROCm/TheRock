# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Installation package tests for the rocm-profiler package.

These are installation (integration) tests: they import the physically
installed profiler package, verify the profiler-hub runtime dependency is
bundled, and execute a profiler console script end-to-end.
"""

import importlib
import locale
from pathlib import Path
import platform
import subprocess
from typing import NamedTuple
import unittest

from .. import _dist_info as di
from . import utils

import rocm_sdk

utils.assert_is_physical_package(rocm_sdk)

profiler_mod_name = di.ALL_PACKAGES["profiler"].get_py_package_name()
profiler_mod = importlib.import_module(profiler_mod_name)
utils.assert_is_physical_package(profiler_mod)

so_paths = utils.get_module_shared_libraries(profiler_mod)
is_windows = platform.system() == "Windows"


class ConsoleScriptTest(NamedTuple):
    """A single console-script invocation to exercise.

    Running each console script exercises the full trampoline in
    rocm_profiler._cli: LD_LIBRARY_PATH setup, locating the packaged binary,
    and os.execv into it. A missing NEEDED dependency (e.g.
    libprofiler-hub.so.0) surfaces here as a non-zero exit / loader error.

    Attributes:
        script_name: Console-script name as installed on PATH.
        cl: Command-line arguments to pass to the script.
        expected_text: Substring required in output ("" only checks exit code).
        required: Fail if the console script is absent (vs. skip).
    """

    script_name: str
    cl: list[str]
    expected_text: str
    required: bool


# Each --version invocation dlopens librocprof-sys and its NEEDED dependencies
# (e.g. libprofiler-hub), so a missing bundled library surfaces here as a load
# failure.
CONSOLE_SCRIPT_TESTS = [
    ConsoleScriptTest("rocprof-sys-sample", ["--version"], "rocprof-sys-sample", True),
    ConsoleScriptTest("rocprof-sys-run", ["--version"], "rocprof-sys-run", True),
    ConsoleScriptTest(
        "rocprof-sys-instrument", ["--version"], "rocprof-sys-instrument", True
    ),
]


@unittest.skipIf(is_windows, "rocm-profiler is not supported on Windows")
class ROCmProfilerTest(unittest.TestCase):
    def test_installation_layout(self):
        """The `rocm_sdk` and profiler module must be siblings on disk."""
        # A concrete __init__.py (vs. a namespace package with __file__ == None)
        # proves the package was installed as real files, so its bundled .so
        # files and the RPATHs computed against them physically exist.
        sdk_path = Path(rocm_sdk.__file__)
        self.assertEqual(
            sdk_path.name,
            "__init__.py",
            msg="Expected `rocm_sdk` module to be a non-namespace package",
        )
        profiler_path = Path(profiler_mod.__file__)
        self.assertEqual(
            profiler_path.name,
            "__init__.py",
            msg=f"Expected `{profiler_mod_name}` module to be a non-namespace package",
        )
        # Cross-package $ORIGIN RPATHs are baked at build time as a relative walk
        # between the two package dirs; they only resolve if both land in the
        # same site-packages. Equal parent.parent confirms that precondition.
        self.assertEqual(
            sdk_path.parent.parent,
            profiler_path.parent.parent,
            msg="Paths are not siblings",
        )

    def test_all_required_libraries_resolve(self):
        """Every NEEDED dependency of every profiler .so must resolve.

        A newly added library (libprofiler-hub.so.0) was once missing from the
        packaging manifest and never made it into the wheel, but any bundled
        dependency could go missing the same way. Rather than name one library,
        ask the dynamic loader to resolve each profiler shared object and assert
        nothing is reported as "not found". This covers
        in-package deps, cross-package deps reachable via the $ORIGIN RPATHs, and
        system libraries in one deterministic, GPU-free check (ldd resolves the
        link map without running the binary).
        """
        self.assertTrue(
            so_paths,
            msg="No shared libraries found in the installed profiler package",
        )

        unresolved: dict[str, list[str]] = {}
        for so_path in so_paths:
            result = subprocess.run(
                ["ldd", str(so_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding=locale.getpreferredencoding(),
            )
            missing = [
                line.strip()
                for line in result.stdout.splitlines()
                if "not found" in line
            ]
            if missing:
                unresolved[so_path.name] = missing

        self.assertFalse(
            unresolved,
            msg=(
                "Unresolved NEEDED dependencies in the installed profiler "
                f"package: {unresolved}"
            ),
        )

    def test_rocprof_sys_unversioned_library_aliases(self):
        """Unversioned aliases for profiler runtime deps must exist and resolve.

        populate_runtime_files() writes only the SONAME-matching versioned
        libraries (e.g. libprofiler-hub.so.0); ensure_profiler_library_symlinks()
        recreates the unversioned aliases. Mirror the source patterns
        ("librocprof-sys*.so.*", "libprofiler-hub*.so.*") and, for every versioned
        library found, assert the corresponding unversioned alias is present and
        points at a real target (whether materialized as a symlink or a plain
        file), so a build that drops any of them is caught.
        """
        lib_dir = Path(profiler_mod.__file__).parent / "lib"
        alias_patterns = ("librocprof-sys*.so.*", "libprofiler-hub*.so.*")

        versioned = [
            target for pattern in alias_patterns for target in lib_dir.glob(pattern)
        ]
        self.assertTrue(
            versioned,
            msg=(
                "No versioned profiler libraries matching "
                f"{alias_patterns} found under {lib_dir}"
            ),
        )

        for target in versioned:
            # ensure_profiler_library_symlinks() derives the alias by stripping
            # the trailing version suffix (Path.with_suffix("")), e.g.
            # libprofiler-hub.so.0 -> libprofiler-hub.so.
            link = target.with_suffix("")
            with self.subTest(alias=link.name):
                self.assertTrue(
                    link.exists(),
                    msg=f"Unversioned alias {link} is missing or dangles",
                )

    def test_rocprof_sys_console_scripts(self):
        """Test the console scripts are installed and executable."""
        for test in CONSOLE_SCRIPT_TESTS:
            script_path = utils.find_console_script(test.script_name)
            if not test.required and script_path is None:
                continue
            with self.subTest(msg=f"Check console-script {test.script_name}"):
                self.assertIsNotNone(
                    script_path,
                    msg=f"Console script {test.script_name} does not exist",
                )
                encoding = locale.getpreferredencoding()
                output_text = subprocess.check_output(
                    [script_path] + test.cl,
                    stderr=subprocess.STDOUT,
                ).decode(encoding)
                if test.expected_text and test.expected_text not in output_text:
                    self.fail(
                        f"Expected '{test.expected_text}' in console-script "
                        f"{test.script_name} output:\n{output_text}"
                    )
