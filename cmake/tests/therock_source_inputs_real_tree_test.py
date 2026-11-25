#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check that real ADDITIONAL_SOURCE_INPUTS declarations reach fingerprints.

The fixture tests in this directory cover the mechanism. This test covers real
declarations: for subprojects that read sources outside their own directory, it
commits a file inside that source in the real submodule, reconfigures, and checks
that the subproject's artifact fingerprints change while an unrelated artifact's
do not. Each commit is removed again afterwards.

Only target-neutral artifacts are used: with KPACK_SPLIT_ARTIFACTS on,
therock_artifacts.cmake does not write fingerprint files for target-specific
artifacts such as blas.

It runs on Linux and Windows; each platform checks the declarations it uses.
It needs all sources fetched and a configured build directory
(THEROCK_BUILD_DIR, default build/), and it reconfigures that build, so it only
runs when selected:

    python -m pytest -m manual cmake/tests/therock_source_inputs_real_tree_test.py
"""

import os
import subprocess
import sys
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from therock_fingerprint_test_utils import THEROCK_ROOT

pytestmark = pytest.mark.manual

BUILD_DIR = Path(os.environ.get("THEROCK_BUILD_DIR", THEROCK_ROOT / "build"))
# Third-party system dependencies: no subproject in rocm-systems or
# rocm-libraries contributes to them.
CONTROL_ARTIFACT = "sysdeps"
PROBE_NAME = "therock-fingerprint-probe.txt"
GIT_IDENTITY = (
    "-c",
    "user.name=Fingerprint test",
    "-c",
    "user.email=test@example.invalid",
)


def run(*args: str | Path) -> str:
    result = subprocess.run(
        [str(arg) for arg in args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {args}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def reconfigure() -> None:
    run("cmake", "-S", THEROCK_ROOT, "-B", BUILD_DIR)


def fingerprints() -> dict[str, str]:
    """Return {fingerprint file name: fingerprint} for every artifact component."""
    paths = sorted((BUILD_DIR / "artifacts").glob("*.fprint"))
    return {path.name: path.read_text(encoding="utf-8") for path in paths}


def of_artifact(values: dict[str, str], artifact: str) -> dict[str, str]:
    """Select the fingerprints of one artifact's components."""
    return {name: v for name, v in values.items() if name.startswith(artifact + "_")}


@contextmanager
def committed_probe(directory: Path) -> Iterator[None]:
    """Commit a new file in directory, then remove the commit on exit."""
    if subprocess.run(
        ["git", "-C", str(directory), "diff", "--quiet", "HEAD"]
    ).returncode:
        raise RuntimeError(
            f"The checkout containing {directory} has uncommitted changes"
        )
    probe = directory / PROBE_NAME
    probe.write_text("probe\n", encoding="utf-8")
    try:
        # --sparse: the monorepos may be sparse checkouts.
        run("git", "-C", directory, "add", "--sparse", "--", PROBE_NAME)
        run(
            "git",
            "-C",
            directory,
            *GIT_IDENTITY,
            "commit",
            "-q",
            "--no-verify",
            "-m",
            "Fingerprint probe",
        )
    except RuntimeError:
        run("git", "-C", directory, "reset", "-q", "--", PROBE_NAME)
        probe.unlink()
        raise
    try:
        yield
    finally:
        run("git", "-C", directory, "reset", "-q", "--keep", "HEAD~1")


class RealSourceInputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (BUILD_DIR / "CMakeCache.txt").is_file():
            raise RuntimeError(f"Configure TheRock in {BUILD_DIR} first")
        reconfigure()
        cls.baseline = fingerprints()

    @classmethod
    def tearDownClass(cls) -> None:
        # Leave the build matching the restored sources.
        reconfigure()

    def assert_input_reaches_artifact(self, source_input: str, artifact: str) -> None:
        # Fingerprints before the change come from the configure in setUpClass;
        # every test removes its commit, so the sources match that configure.
        before = of_artifact(self.baseline, artifact)
        control_before = of_artifact(self.baseline, CONTROL_ARTIFACT)
        self.assertTrue(before, f"{artifact} has no valid fingerprints")
        self.assertTrue(control_before, f"{CONTROL_ARTIFACT} has no valid fingerprints")
        with committed_probe(THEROCK_ROOT / source_input):
            reconfigure()
            changed = fingerprints()
        after = of_artifact(changed, artifact)
        control_after = of_artifact(changed, CONTROL_ARTIFACT)
        unchanged = sorted(name for name in before if after.get(name) == before[name])
        self.assertEqual(unchanged, [], f"{source_input} did not change {artifact}")
        self.assertEqual(control_after, control_before)

    def test_hip_common_sources_reach_the_hip_runtime(self) -> None:
        self.assert_input_reaches_artifact("rocm-systems/projects/hip", "core-hip")

    @unittest.skipIf(sys.platform == "win32", "amdsmi is not built on Windows")
    def test_shared_sha256_reaches_amdsmi(self) -> None:
        self.assert_input_reaches_artifact("rocm-systems/shared/sha256", "core-amdsmi")

    def test_spirv_translator_reaches_the_compiler(self) -> None:
        self.assert_input_reaches_artifact("compiler/spirv-llvm-translator", "amd-llvm")

    @unittest.skipUnless(sys.platform == "win32", "declared only on Windows")
    def test_windows_interop_reaches_the_hip_runtime(self) -> None:
        self.assert_input_reaches_artifact(
            "rocm-systems/shared/amdgpu-windows-interop", "core-hip"
        )


if __name__ == "__main__":
    unittest.main()
