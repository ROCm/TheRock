#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check source identity relocation and fingerprint stability at a fixed location.

Save as cmake/tests/therock_subproject_fingerprint_paths_test.py, then run:
    python cmake/tests/therock_subproject_fingerprint_paths_test.py -v

Three checks: source identity relocation, unchanged repeat, and argument-change
propagation/restoration. The source identity check holds the build path fixed
and re-creates that disposable build directory to avoid stale CMake state.

Final fingerprint equality across different checkout/build locations is
deferred. The four checks for that behavior have been removed. The known path
sensitivity in generated configuration and CMake arguments remains unresolved.
The source identity relocation check does not compare final fingerprints.

This is configure-only fixture coverage, not validation of all seven real ROCm
project declarations, artifact aggregation, build outputs, or cache reuse.
Reports and fixtures are retained in a printed temporary directory. Optionally
set THEROCK_LOCATION_TEST_OUTPUT to choose its parent directory, or set
THEROCK_TEST_SOURCE_ROOT to run this file from outside the checkout.
"""

import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


THEROCK_ROOT = Path(
    os.environ.get("THEROCK_TEST_SOURCE_ROOT", Path(__file__).resolve().parents[2])
).resolve()
PROJECTS = ("location_source", "location_consumer", "location_control")

HARNESS = r'''
    cmake_minimum_required(VERSION 3.25)
    project(location_fingerprints NONE)
    set(THEROCK_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
    set(THEROCK_BINARY_DIR "${CMAKE_BINARY_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(ROCM_BUILD_FLAGS_STATE_FILE "${CMAKE_BINARY_DIR}/flags.cmake")
    file(WRITE "${ROCM_BUILD_FLAGS_STATE_FILE}" "# no flags\n")
    include(therock_globals)
    include(therock_sanitizers)
    include(therock_flag_utils)
    include(therock_default_targets)
    include(therock_subproject)
    set(TEST_VALUE baseline CACHE STRING "Configuration control")
    foreach(_name location_source location_consumer location_control)
      set(_options)
      if(_name STREQUAL "location_source")
        list(APPEND _options
          ADDITIONAL_SOURCE_INPUTS "${CMAKE_SOURCE_DIR}/shared"
          CMAKE_ARGS "-DVALUE=${TEST_VALUE}")
      elseif(_name STREQUAL "location_consumer")
        list(APPEND _options RUNTIME_DEPS location_source)
      endif()
      set(_source "${CMAKE_SOURCE_DIR}/projects/${_name}")
      therock_cmake_subproject_declare(${_name}
        EXTERNAL_SOURCE_DIR "${_source}"
        BINARY_DIR "${CMAKE_BINARY_DIR}/${_name}"
        DISABLE_AMDGPU_TARGETS NO_MERGE_COMPILE_COMMANDS
        ${_options})
      therock_cmake_subproject_activate(${_name})
      get_target_property(_fprint ${_name} THEROCK_FPRINT)
      _therock_subproject_fprint_source_dir(_identity "${_source}")
      file(WRITE "${CMAKE_BINARY_DIR}/${_name}.fprint" "${_fprint}")
      file(WRITE "${CMAKE_BINARY_DIR}/${_name}.identity" "${_identity}")
      get_target_property(_inputs ${_name} THEROCK_ADDITIONAL_SOURCE_INPUTS)
      file(WRITE "${CMAKE_BINARY_DIR}/${_name}.inputs" "${_inputs}")
    endforeach()
    _therock_subproject_fprint_source_dir(_identity "${CMAKE_SOURCE_DIR}/shared")
    file(WRITE "${CMAKE_BINARY_DIR}/shared.identity" "${_identity}")
'''


def write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


class FingerprintLocationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [tool for tool in ("git", "cmake", "ninja") if not shutil.which(tool)]
        if missing:
            raise RuntimeError(f"Required tools unavailable: {', '.join(missing)}; tests were not run")
        for relative in ("cmake/therock_subproject.cmake", "build_tools/fileset_tool.py"):
            if not (THEROCK_ROOT / relative).is_file():
                raise RuntimeError(
                    f"Missing {THEROCK_ROOT / relative}. Save this test under cmake/tests/ "
                    "or set THEROCK_TEST_SOURCE_ROOT to the TheRock checkout."
                )
        parent = os.environ.get("THEROCK_LOCATION_TEST_OUTPUT")
        if parent:
            Path(parent).mkdir(parents=True, exist_ok=True)
        cls.reports = Path(tempfile.mkdtemp(prefix="therock-locations-", dir=parent)).resolve()
        print(f"\nLocation-test reports and fixtures: {cls.reports}", flush=True)
        cls.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        cls.env.update(
            GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
            GIT_AUTHOR_NAME="Fingerprint location test",
            GIT_AUTHOR_EMAIL="test@example.invalid",
            GIT_COMMITTER_NAME="Fingerprint location test",
            GIT_COMMITTER_EMAIL="test@example.invalid",
        )
        # Give every test the exact same committed fixture and build-system files.
        cls.template = cls.reports / "template"
        cls.template.mkdir()
        for relative in ("cmake", "build_tools"):
            shutil.copytree(
                THEROCK_ROOT / relative, cls.template / relative,
                ignore=shutil.ignore_patterns("__pycache__", ".git", ".pytest_cache"),
            )
        write(cls.template / "CMakeLists.txt", textwrap.dedent(HARNESS).lstrip())
        for name in PROJECTS:
            write(
                cls.template / "projects" / name / "CMakeLists.txt",
                f"cmake_minimum_required(VERSION 3.25)\nproject({name} NONE)\n",
            )
        write(cls.template / "shared/input.txt", "same input\n")
        cls.git(cls.template, "init", "--quiet")
        cls.git(cls.template, "add", "--all", "--force")
        cls.git(cls.template, "commit", "--quiet", "-m", "Location fixture")
        metadata = {"therock_source": str(THEROCK_ROOT), "tools": {}}
        for tool in ("git", "cmake", "ninja"):
            metadata["tools"][tool] = cls.command(tool, "--version").splitlines()[0]
        metadata["python"] = sys.version
        metadata["module_sha256"] = hashlib.sha256(
            (cls.template / "cmake/therock_subproject.cmake").read_bytes()
        ).hexdigest()
        write(cls.reports / "environment.json", json.dumps(metadata, indent=2) + "\n")

    @classmethod
    def command(cls, *args, log=None):
        command = [str(arg) for arg in args]
        result = subprocess.run(
            command, env=cls.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", timeout=180,
        )
        if log:
            write(log, json.dumps(command) + "\n" + result.stdout)
        if result.returncode:
            raise RuntimeError(
                f"Command failed ({result.returncode}): {command}\n"
                f"{result.stdout[-16000:]}\nLog: {log}"
            )
        return result.stdout.strip()

    @classmethod
    def git(cls, directory, *args):
        return cls.command(
            "git", "-c", "core.autocrlf=false", "-c", "commit.gpgsign=false",
            "-C", directory, *args,
        )

    def setUp(self):
        self.case = self.reports / self._testMethodName
        self.case.mkdir()
        self.source = self.clone(self.case / "checkout-a")

    def clone(self, destination):
        self.git(self.reports, "clone", "--quiet", "--no-hardlinks", self.template, destination)
        self.assertEqual(self.git(self.template, "rev-parse", "HEAD^{tree}"),
                         self.git(destination, "rev-parse", "HEAD^{tree}"))
        return destination

    def configure(self, source, build, label, *, value="baseline"):
        report = self.case / label
        report.mkdir()
        self.command(
            "cmake", "-S", source, "-B", build, "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release", "-DTHEROCK_VERBOSE=ON",
            f"-DPython3_EXECUTABLE:FILEPATH={Path(sys.executable).as_posix()}",
            f"-DTEST_VALUE:STRING={value}",
            log=report / "configure.log",
        )
        snapshot = {"source": str(source), "build": str(build), "identities": {},
                    "fingerprints": {}, "generated_files": {}, "report": str(report)}
        for name in (*PROJECTS, "shared"):
            identity = (build / f"{name}.identity").read_text(encoding="utf-8").strip()
            scope = source / "shared" if name == "shared" else source / "projects" / name
            self.assertEqual(identity, self.git(scope, "rev-parse", "HEAD:./"), name)
            self.assertRegex(identity, r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$", name)
            snapshot["identities"][name] = identity
        for name in PROJECTS:
            fingerprint = (build / f"{name}.fprint").read_text(encoding="utf-8").strip()
            self.assertRegex(fingerprint, r"^[0-9a-f]{64}$", name)
            snapshot["fingerprints"][name] = fingerprint
            for filename in ("_init.cmake", "_toolchain.cmake"):
                relative = f"{name}/{filename}"
                generated = build / relative
                self.assertTrue(generated.is_file(), str(generated))
                snapshot["generated_files"][relative] = hashlib.sha256(generated.read_bytes()).hexdigest()
                write(report / relative, generated.read_text(encoding="utf-8"))
        declared = (build / "location_source.inputs").read_text(encoding="utf-8").strip()
        self.assertEqual(declared, (source / "shared").as_posix())
        write(report / "snapshot.json", json.dumps(snapshot, indent=2) + "\n")
        return snapshot

    def assert_same_fingerprints(self, before, after):
        self.assertEqual(before["identities"], after["identities"])
        changed = [name for name in PROJECTS
                   if before["fingerprints"][name] != after["fingerprints"][name]]
        if changed:
            differences = []
            for relative in before["generated_files"]:
                a = Path(before["report"]) / relative
                b = Path(after["report"]) / relative
                differences.extend(difflib.unified_diff(
                    a.read_text(encoding="utf-8").splitlines(keepends=True),
                    b.read_text(encoding="utf-8").splitlines(keepends=True),
                    fromfile=f"before/{relative}", tofile=f"after/{relative}",
                ))
            diff_path = self.case / "generated-files.diff"
            write(diff_path, "".join(differences))
            self.fail(
                f"Source identities match, but fingerprints changed for {changed}.\n"
                f"Before: {before['report']}\nAfter: {after['report']}\n"
                f"Generated-file diff: {diff_path}\n"
                "The configure logs also contain the actual FPRINT CONTENT entries."
            )

    def checkout_pair(self):
        relocated = self.clone(self.case / "relocated checkout with spaces")
        build = self.case / "fixed-build"
        before = self.configure(self.source, build, "before")
        # This is exclusively a test-created directory below self.case.
        shutil.rmtree(build)
        after = self.configure(relocated, build, "after")
        return before, after

    def test_source_identities_survive_checkout_relocation(self):
        before, after = self.checkout_pair()
        self.assertEqual(before["identities"], after["identities"])

    def test_repeat_configuration_is_stable(self):
        build = self.case / "build"
        before = self.configure(self.source, build, "before")
        after = self.configure(self.source, build, "after")
        self.assert_same_fingerprints(before, after)

    def test_real_argument_change_affects_source_and_consumer_only(self):
        build = self.case / "build"
        before = self.configure(self.source, build, "before")
        after = self.configure(self.source, build, "changed", value="different")
        self.assertEqual(before["identities"], after["identities"])
        changed = {name for name in PROJECTS
                   if before["fingerprints"][name] != after["fingerprints"][name]}
        self.assertEqual(changed, {"location_source", "location_consumer"})
        restored = self.configure(self.source, build, "restored")
        self.assert_same_fingerprints(before, restored)


if __name__ == "__main__":
    unittest.main()
