# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Installation package tests for the rocm-profiler package.

These are installation (integration) tests: they import the physically
installed profiler package, verify the profiler-hub runtime dependency is
bundled, and execute a profiler console script end-to-end.
"""

import csv
import importlib
import locale
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import NamedTuple
import unittest

from .. import _dist_info as di
from . import utils

import rocm_sdk

# Only the package *name* is resolved at import time; importing the package
# itself (and probing its shared libraries) happens in setUpClass so this module
# stays safe to import on any platform, even where the profiler wheel is absent.
profiler_mod_name = di.ALL_PACKAGES["profiler"].get_py_package_name()
is_windows = platform.system() == "Windows"

# Bound each helper subprocess so a hung ldd / console script fails the test
# instead of stalling until the CI job-level timeout.
SUBPROCESS_TIMEOUT_SECONDS = 60


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
    ConsoleScriptTest("rocsys", ["--version"], "rocsys", True),
]

ROCPD_FIXTURE_SCRIPT = """
import sqlite3
import sys
from rocpd.schema import RocpdSchema

connection = sqlite3.connect(sys.argv[1])
RocpdSchema(
    uuid="_wheel_smoke", guid="00000000-0000-0000-0000-000000009535"
).write_schema(connection)

def insert(table, columns, values):
    placeholders = ", ".join(["?"] * len(values))
    connection.execute(
        f"INSERT INTO {table}_wheel_smoke ({columns}) VALUES ({placeholders})",
        values,
    )

insert("rocpd_info_node", "id, hash, machine_id, system_name, hostname, release, "
       "version, hardware_name, domain_name",
       (1, 1, "machine", "Linux", "host", "r", "v", "x86_64", "domain"))
insert("rocpd_info_process", "id, nid, ppid, pid, init, fini, start, end, command",
       (1, 1, 1, 1, 1000, 4000, 1000, 4000, "wheel_smoke"))
insert("rocpd_info_thread", "id, nid, ppid, pid, tid", (1, 1, 1, 1, 1))
insert("rocpd_string", "id, string", (1, "wheel_smoke"))
insert("rocpd_string", "id, string", (2, "HIP_API"))
for identifier, start, end in ((1, 1000, 2000), (2, 2000, 4000)):
    insert("rocpd_event", "id, category_id, correlation_id",
           (identifier, 2, identifier))
    insert("rocpd_region", "id, nid, pid, tid, start, end, name_id, event_id",
           (identifier, 1, 1, 1, start, end, 1, identifier))
connection.commit()
connection.close()
"""


@unittest.skipIf(is_windows, "rocm-profiler is not supported on Windows")
class ROCmProfilerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        utils.assert_is_physical_package(rocm_sdk)
        try:
            cls.profiler_mod = importlib.import_module(profiler_mod_name)
        except ImportError as e:
            raise unittest.SkipTest(f"rocm-profiler is not installed: {e}")
        utils.assert_is_physical_package(cls.profiler_mod)
        cls.so_paths = utils.get_module_shared_libraries(cls.profiler_mod)

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
        profiler_path = Path(self.profiler_mod.__file__)
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
            self.so_paths,
            msg="No shared libraries found in the installed profiler package",
        )
        self.assertIsNotNone(
            shutil.which("ldd"),
            msg="ldd not found on PATH; required to verify library dependencies",
        )

        unresolved: dict[str, list[str]] = {}
        for so_path in self.so_paths:
            result = subprocess.run(
                ["ldd", str(so_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding=locale.getpreferredencoding(),
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
            )
            # ldd exits 0 even when a NEEDED dependency is "not found", so a
            # non-zero exit means ldd itself failed (bad file, unreadable, etc.)
            # and the "not found" scan below cannot be trusted for this object.
            if result.returncode != 0:
                unresolved[so_path.name] = [
                    f"ldd failed (exit {result.returncode}): {result.stdout.strip()}"
                ]
                continue
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
        lib_dir = Path(self.profiler_mod.__file__).parent / "lib"
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
                    timeout=SUBPROCESS_TIMEOUT_SECONDS,
                ).decode(encoding)
                if test.expected_text and test.expected_text not in output_text:
                    self.fail(
                        f"Expected '{test.expected_text}' in console-script "
                        f"{test.script_name} output:\n{output_text}"
                    )

    def test_rocpd_csv_and_summary(self):
        """Convert non-empty data using the installed wheel and its interpreter."""
        core_name = di.ALL_PACKAGES["core"].get_py_package_name()
        core_module = importlib.import_module(core_name)
        utils.assert_is_physical_package(core_module)
        core_path = Path(core_module.__file__).parent
        python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        environment = dict(os.environ)
        environment.pop("PYTHONHOME", None)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONPATH"] = str(
            core_path / "lib" / python_version / "site-packages"
        )
        environment["PATH"] = (
            str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
        )
        rocpd = utils.find_console_script("rocpd")
        self.assertIsNotNone(rocpd, "rocpd console script is missing")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database = root / "input.db"
            commands = [
                [sys.executable, "-c", ROCPD_FIXTURE_SCRIPT, str(database)],
                [
                    rocpd,
                    "convert",
                    "-i",
                    str(database),
                    "-f",
                    "csv",
                    "-d",
                    str(root / "csv"),
                ],
                [
                    rocpd,
                    "summary",
                    "-i",
                    str(database),
                    "-f",
                    "csv",
                    "-d",
                    str(root / "summary"),
                ],
            ]
            for command in commands:
                result = subprocess.run(
                    command,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=SUBPROCESS_TIMEOUT_SECONDS,
                )
                self.assertEqual(result.returncode, 0, result.stdout)

            rows_by_output = {}
            for output in ("csv", "summary"):
                files = sorted((root / output).glob("*.csv"))
                self.assertTrue(files, f"No {output} files were generated")
                rows = []
                for path in files:
                    with path.open(newline="", encoding="utf-8") as stream:
                        rows.extend(
                            row
                            for row in csv.DictReader(stream)
                            if "wheel_smoke" in row.values()
                        )
                rows_by_output[output] = rows
            self.assertEqual(len(rows_by_output["csv"]), 2)
            self.assertEqual(len(rows_by_output["summary"]), 1)
            summary = {
                name.casefold(): value
                for name, value in rows_by_output["summary"][0].items()
            }
            self.assertEqual(int(summary["calls"]), 2)

    def test_rocprof_trace_decoder_python_api(self):
        """The packaged rocprof_trace_decoder API must be importable.

        Skips on ROCm builds whose rocprof-trace-decoder ships no Python API.
        """
        try:
            trace_decoder_mod = importlib.import_module("rocprof_trace_decoder")
        except ImportError as e:
            raise unittest.SkipTest(f"rocprof_trace_decoder is not installed: {e}")

        utils.assert_is_physical_package(trace_decoder_mod)
