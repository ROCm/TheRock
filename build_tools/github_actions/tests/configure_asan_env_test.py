#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for configure_asan_env.py"""

import os
import platform
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from configure_asan_env import (
    ASAN_RUNTIME_LIBS,
    LSAN_SUPPRESSIONS,
    STATIC_ASAN_ENV,
    AsanEnvironmentError,
    main,
    resolve_asan_env,
)

# The runtime is located by executing a clang, which the tests stub with a
# shell script. Windows has no equivalent shebang mechanism.
requires_posix = unittest.skipIf(
    os.name != "posix", "stub executables require a POSIX shell"
)

ARCH_RUNTIME = f"libclang_rt.asan-{platform.machine()}.so"


def _make_executable(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _make_clang_stub(
    path: Path,
    *,
    runtime: Optional[Path] = None,
    runtime_name: str = "libclang_rt.asan.so",
    symbolizer: Optional[Path] = None,
) -> None:
    """Writes a stub that answers clang's -print-*-name queries.

    Real clang echoes the requested name back when it cannot find the file, so
    the stub does the same for anything it was not told about.
    """
    lines = ["#!/bin/sh", 'arg="$1"']
    if runtime is not None:
        lines.append(
            f'if [ "$arg" = "-print-file-name={runtime_name}" ]; then '
            f'echo "{runtime}"; exit 0; fi'
        )
    if symbolizer is not None:
        lines.append(
            'if [ "$arg" = "-print-prog-name=llvm-symbolizer" ]; then '
            f'echo "{symbolizer}"; exit 0; fi'
        )
    lines.append('echo "${arg#*=}"')
    _make_executable(path, "\n".join(lines) + "\n")


def _make_artifacts(
    root: Path,
    *,
    clang=True,
    symbolizer=True,
    runtime=True,
    runtime_name="libclang_rt.asan.so",
) -> Path:
    """Builds a fake artifact tree; returns the artifacts dir."""
    artifacts = root / "build"
    runtime_path = artifacts / "lib" / "llvm" / "lib" / "clang" / "24" / "asan.so"
    if runtime:
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text("")
    if clang:
        _make_clang_stub(
            artifacts / "llvm" / "bin" / "clang",
            runtime=runtime_path if runtime else None,
            runtime_name=runtime_name,
        )
    if symbolizer:
        _make_executable(artifacts / "llvm" / "bin" / "llvm-symbolizer", "#!/bin/sh\n")
    return artifacts


def _make_python_install(root: Path, *, compiler="amdclang++", **kwargs) -> Path:
    """Builds a fake rocm-sdk-core console-script dir; returns the bin dir."""
    bin_dir = root / "venv" / "bin"
    runtime_path = root / "site-packages" / "rocm_sdk_core" / "lib" / ARCH_RUNTIME
    symbolizer_path = bin_dir / "llvm-symbolizer"

    if kwargs.get("runtime", True):
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text("")
    if kwargs.get("symbolizer", True):
        _make_executable(symbolizer_path, "#!/bin/sh\n")
    if kwargs.get("clang", True):
        _make_clang_stub(
            bin_dir / compiler,
            runtime=runtime_path if kwargs.get("runtime", True) else None,
            runtime_name=ARCH_RUNTIME,
            symbolizer=symbolizer_path if kwargs.get("symbolizer", True) else None,
        )
    bin_dir.mkdir(parents=True, exist_ok=True)
    return bin_dir


class _IsolatedPath:
    """Makes a directory the whole of PATH for the duration of the block.

    Replacing rather than prepending keeps these tests from finding a real
    amdclang++ on the developer machine or the runner image.
    """

    def __init__(self, directory: Path):
        self._directory = directory

    def __enter__(self):
        self._previous = os.environ.get("PATH", "")
        os.environ["PATH"] = str(self._directory)

    def __exit__(self, *exc):
        os.environ["PATH"] = self._previous


class TestArtifactTree(unittest.TestCase):
    @requires_posix
    def test_resolves_runtime_and_symbolizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _make_artifacts(Path(tmp))
            env, warnings = resolve_asan_env(artifacts)

            self.assertEqual(warnings, [])
            self.assertTrue(env["ASAN_RUNTIME_PATH"].endswith("asan.so"))
            self.assertTrue(env["ASAN_SYMBOLIZER_PATH"].endswith("llvm-symbolizer"))

    @requires_posix
    def test_falls_back_to_the_arch_suffixed_runtime_name(self):
        """#8077 reintroduced libclang_rt.asan-<arch>.so; both spellings ship."""
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _make_artifacts(Path(tmp), runtime_name=ARCH_RUNTIME)
            env, _ = resolve_asan_env(artifacts)

            self.assertTrue(env["ASAN_RUNTIME_PATH"].endswith("asan.so"))

    @requires_posix
    def test_static_values_are_always_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _make_artifacts(Path(tmp))
            env, _ = resolve_asan_env(artifacts)
            for key, value in STATIC_ASAN_ENV.items():
                self.assertEqual(env[key], value)

    def test_link_order_check_is_disabled(self):
        """Uninstrumented tools (amdclang++) abort without this."""
        self.assertIn("verify_asan_link_order=0", STATIC_ASAN_ENV["ASAN_OPTIONS"])

    @requires_posix
    def test_symbolizer_path_is_absolute_for_a_relative_artifacts_dir(self):
        """hip_check runs from build/bin, so a relative path resolves to nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            _make_artifacts(Path(tmp))
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                env, _ = resolve_asan_env(Path("./build"))
            finally:
                os.chdir(cwd)

            self.assertTrue(Path(env["ASAN_SYMBOLIZER_PATH"]).is_absolute())
            self.assertTrue(Path(env["ASAN_RUNTIME_PATH"]).is_absolute())

    @requires_posix
    def test_library_path_covers_lib_and_sysdeps(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _make_artifacts(Path(tmp))
            env, _ = resolve_asan_env(artifacts)

            self.assertIn("rocm_sysdeps", env["LD_LIBRARY_PATH"])

    @requires_posix
    def test_missing_symbolizer_warns_but_keeps_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _make_artifacts(Path(tmp), symbolizer=False)
            env, warnings = resolve_asan_env(artifacts)

            self.assertIn("ASAN_RUNTIME_PATH", env)
            self.assertNotIn("ASAN_SYMBOLIZER_PATH", env)
            self.assertTrue(any("unsymbolized" in w for w in warnings))


class TestFailFast(unittest.TestCase):
    """An unusable ASAN environment halts the job instead of running degraded."""

    def test_missing_clang_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _make_artifacts(Path(tmp), clang=False)
            with self.assertRaisesRegex(AsanEnvironmentError, "clang not found"):
                resolve_asan_env(artifacts)

    @requires_posix
    def test_missing_runtime_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _make_artifacts(Path(tmp), runtime=False)
            with self.assertRaisesRegex(AsanEnvironmentError, "not ASAN-instrumented"):
                resolve_asan_env(artifacts)

    def test_missing_compiler_on_path_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = _make_python_install(Path(tmp), clang=False)
            with _IsolatedPath(bin_dir):
                with self.assertRaisesRegex(AsanEnvironmentError, "rocm-sdk-core"):
                    resolve_asan_env()


class TestPythonInstall(unittest.TestCase):
    """rocm-sdk-core puts amdclang++ on PATH, so no artifact tree is needed."""

    @requires_posix
    def test_resolves_runtime_from_path_compiler(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = _make_python_install(Path(tmp))
            with _IsolatedPath(bin_dir):
                env, warnings = resolve_asan_env()

            self.assertEqual(warnings, [])
            self.assertTrue(env["ASAN_RUNTIME_PATH"].endswith(ARCH_RUNTIME))

    @requires_posix
    def test_falls_back_to_amdclang_when_no_amdclangpp(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = _make_python_install(Path(tmp), compiler="amdclang")
            with _IsolatedPath(bin_dir):
                env, _ = resolve_asan_env()

            self.assertTrue(env["ASAN_RUNTIME_PATH"].endswith(ARCH_RUNTIME))

    @requires_posix
    def test_resolves_symbolizer_via_print_prog_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = _make_python_install(Path(tmp))
            with _IsolatedPath(bin_dir):
                env, _ = resolve_asan_env()

            self.assertTrue(env["ASAN_SYMBOLIZER_PATH"].endswith("llvm-symbolizer"))

    @requires_posix
    def test_missing_symbolizer_warns_but_keeps_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = _make_python_install(Path(tmp), symbolizer=False)
            with _IsolatedPath(bin_dir):
                env, warnings = resolve_asan_env()

            self.assertIn("ASAN_RUNTIME_PATH", env)
            self.assertNotIn("ASAN_SYMBOLIZER_PATH", env)
            self.assertTrue(any("unsymbolized" in w for w in warnings))

    @requires_posix
    def test_library_path_is_left_alone(self):
        """Wheels carry their own RPATH; overriding it would shadow them."""
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = _make_python_install(Path(tmp))
            with _IsolatedPath(bin_dir):
                env, _ = resolve_asan_env()

            self.assertNotIn("LD_LIBRARY_PATH", env)


class TestLeakSuppressions(unittest.TestCase):
    def test_suppressions_file_ships_with_the_script(self):
        self.assertTrue(LSAN_SUPPRESSIONS.is_file())

    def test_leak_detection_is_not_disabled_wholesale(self):
        """Scoped suppressions instead of detect_leaks=0, so ROCm leaks surface."""
        self.assertNotIn("detect_leaks", STATIC_ASAN_ENV["ASAN_OPTIONS"])

    @requires_posix
    def test_lsan_options_points_at_the_suppressions_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _make_artifacts(Path(tmp))
            env, _ = resolve_asan_env(artifacts)

            self.assertEqual(
                env["LSAN_OPTIONS"], f"suppressions={LSAN_SUPPRESSIONS}"
            )

    def test_interpreter_allocator_is_suppressed(self):
        """_PyMem_RawMalloc is what rocm-sdk wheel tests actually report."""
        self.assertIn("leak:_PyMem_Raw", LSAN_SUPPRESSIONS.read_text())


class TestRuntimeNames(unittest.TestCase):
    def test_both_runtime_spellings_are_tried(self):
        self.assertIn("libclang_rt.asan.so", ASAN_RUNTIME_LIBS)
        self.assertIn(ARCH_RUNTIME, ASAN_RUNTIME_LIBS)


class TestMain(unittest.TestCase):
    def _run_main(self, argv, tmp: Path) -> dict[str, str]:
        env_file = tmp / "github_env"
        env_file.touch()
        previous = os.environ.get("GITHUB_ENV")
        os.environ["GITHUB_ENV"] = str(env_file)
        try:
            self.assertEqual(main(argv), 0)
        finally:
            if previous is None:
                del os.environ["GITHUB_ENV"]
            else:
                os.environ["GITHUB_ENV"] = previous
        return dict(
            line.split("=", 1) for line in env_file.read_text().splitlines() if line
        )

    @requires_posix
    def test_writes_key_value_lines_to_github_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _make_artifacts(Path(tmp))
            written = self._run_main(["--artifacts-dir", str(artifacts)], Path(tmp))

            self.assertEqual(written["ASAN_OPTIONS"], STATIC_ASAN_ENV["ASAN_OPTIONS"])
            self.assertIn("ASAN_RUNTIME_PATH", written)
            self.assertIn("ASAN_SYMBOLIZER_PATH", written)
            self.assertIn("LSAN_OPTIONS", written)

    @requires_posix
    def test_artifacts_dir_is_optional(self):
        """test_rocm_wheels.yml calls this with no arguments at all."""
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = _make_python_install(Path(tmp))
            with _IsolatedPath(bin_dir):
                written = self._run_main([], Path(tmp))

            self.assertIn("ASAN_RUNTIME_PATH", written)
            self.assertNotIn("LD_LIBRARY_PATH", written)


if __name__ == "__main__":
    unittest.main()
