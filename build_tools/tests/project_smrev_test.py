#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Source identity tests using temporary Git repositories and real CMake wiring."""

import importlib.util
import json
import os
import re
import shlex
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "project_smrev", ROOT / "build_tools/generate_project_smrev.py"
)
smrev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smrev)


def command(*args, cwd=None, ok=True, env=None):
    result = subprocess.run(
        [str(a) for a in args], cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if ok and result.returncode:
        raise AssertionError(f"{args}:\n{result.stdout}")
    return result


def init_git(path):
    path.mkdir(parents=True, exist_ok=True)
    command("git", "init", "-q", path)
    command("git", "-C", path, "config", "user.name", "Source identity test")
    command("git", "-C", path, "config", "user.email", "test@example.invalid")


def commit(repo, message="fixture", date=None):
    command("git", "-C", repo, "add", "-A")
    env = dict(os.environ)
    if date:
        env.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    command("git", "-C", repo, "commit", "-qm", message, env=env)


@unittest.skipUnless(shutil.which("git"), "Git is required")
class SourceIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / "repo"
        init_git(self.repo)
        self.write("projects/a/input.txt", "a\n")
        self.write("shared/common/input.txt", "shared\n")
        self.write("projects/b/input.txt", "b\n")
        self.write(".gitignore", "generated/\n")
        commit(self.repo, date="2025-01-01T00:00:00+00:00")
        self.profile = {
            "paths": ["projects/a", "shared/common"],
            "optional_paths": ["version.json"],
            "environment": ["THEROCK_SMREV_TEST_VALUE"],
            "versions": [],
        }
        self.marker = self.root / "build/.a.smrev"

    def write(self, relative, content):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def generate(self, **kwargs):
        return smrev.generate("a", self.repo / "projects/a", self.marker,
                              repository="fixture", profile=self.profile, **kwargs)

    def test_unrelated_commit_and_repeat_preserve_identity_and_mtime(self):
        before = self.generate()
        stamp = self.marker.stat().st_mtime_ns
        self.write("projects/b/input.txt", "changed\n")
        commit(self.repo)
        self.assertEqual(before, self.generate())
        self.assertEqual(stamp, self.marker.stat().st_mtime_ns)
        self.assertNotIn(smrev.git(self.repo, "rev-parse", "HEAD"), self.marker.read_text())

    def test_project_and_shared_changes_invalidate(self):
        previous = self.generate()
        for path in ("projects/a/input.txt", "shared/common/input.txt"):
            self.write(path, "edited\n")
            commit(self.repo)
            current = self.generate()
            self.assertNotEqual(previous, current)
            previous = current

    def test_rename_and_mode_changes_invalidate(self):
        first = self.generate()
        path = self.repo / "projects/a/input.txt"
        renamed = path.with_name("renamed.txt")
        path.rename(renamed)
        commit(self.repo)
        second = self.generate()
        self.assertNotEqual(first, second)
        renamed.chmod(0o755)
        commit(self.repo)
        self.assertNotEqual(second, self.generate())

    def test_optional_file_creation_and_removal(self):
        first = self.generate()
        path = self.write("version.json", "{}\n")
        commit(self.repo)
        self.assertNotEqual(first, self.generate())
        path.unlink()
        commit(self.repo)
        self.assertEqual(first, self.generate())

    def test_shared_file_input(self):
        self.profile["paths"] = ["projects/a", "shared/common/input.txt"]
        first = self.generate()
        self.write("shared/common/input.txt", "next\n")
        commit(self.repo)
        self.assertNotEqual(first, self.generate())

    def test_ignored_generated_files_do_not_enter_identity(self):
        first = self.generate()
        self.write("projects/a/generated/version.h", "producer commit label\n")
        self.assertEqual(first, self.generate())

    def test_dirty_and_staged_sources_remove_owned_marker(self):
        for staged in (False, True):
            with self.subTest(staged=staged):
                self.generate()
                self.write("projects/a/input.txt", "dirty\n")
                if staged:
                    command("git", "-C", self.repo, "add", "projects/a/input.txt")
                with self.assertRaisesRegex(smrev.SourceIdentityError, "tracked source"):
                    self.generate()
                self.assertFalse(self.marker.exists())
                command("git", "-C", self.repo, "restore", "--staged", "--worktree", ".")

    def test_untracked_selected_source_is_rejected(self):
        self.generate()
        self.write("projects/a/new.txt", "new\n")
        with self.assertRaisesRegex(smrev.SourceIdentityError, "untracked"):
            self.generate()
        self.assertFalse(self.marker.exists())

    def test_untracked_unrelated_source_is_not_an_input(self):
        first = self.generate()
        self.write("projects/b/new.txt", "new\n")
        self.assertEqual(first, self.generate())

    def test_missing_declared_source_removes_marker(self):
        self.generate()
        self.profile["paths"].append("shared/missing")
        with self.assertRaisesRegex(smrev.SourceIdentityError, "missing"):
            self.generate()
        self.assertFalse(self.marker.exists())

    def test_assume_unchanged_and_skip_worktree_are_rejected(self):
        for flag in ("assume-unchanged", "skip-worktree"):
            with self.subTest(flag=flag):
                self.generate()
                command("git", "-C", self.repo, "update-index", "--" + flag, "projects/a/input.txt")
                with self.assertRaisesRegex(smrev.SourceIdentityError, "unsupported"):
                    self.generate()
                command("git", "-C", self.repo, "update-index", "--no-" + flag, "projects/a/input.txt")

    def test_foreign_and_symlink_outputs_are_preserved(self):
        self.marker.parent.mkdir()
        self.marker.write_text("external producer marker")
        with self.assertRaisesRegex(smrev.SourceIdentityError, "foreign"):
            self.generate()
        self.assertEqual(self.marker.read_text(), "external producer marker")
        self.marker.unlink()
        victim = self.root / "victim"
        victim.write_text("keep")
        self.marker.symlink_to(victim)
        with self.assertRaisesRegex(smrev.SourceIdentityError, "symlink"):
            self.generate()
        self.assertEqual(victim.read_text(), "keep")

    def test_marker_cannot_be_written_in_source(self):
        self.marker = self.repo / ".a.smrev"
        with self.assertRaisesRegex(smrev.SourceIdentityError, "outside"):
            self.generate()
        self.assertFalse(self.marker.exists())

    def test_invalid_path_rejected(self):
        for path in ("../a", "/a", "a/../b", "a;command", "a//b", "a\nb"):
            with self.subTest(path=path), self.assertRaises(smrev.SourceIdentityError):
                smrev.source_path(path)

    def test_missing_declaration_removes_owned_marker(self):
        self.generate()
        with self.assertRaisesRegex(smrev.SourceIdentityError, "missing source-identity declaration"):
            smrev.generate("a", self.repo / "projects/a", self.marker,
                           repository="fixture", profile=None)
        self.assertFalse(self.marker.exists())

    def test_scope_must_cover_source_and_entry_must_match(self):
        self.generate()
        with self.assertRaisesRegex(smrev.SourceIdentityError, "does not cover"):
            smrev.generate("a", self.repo / "projects/a", self.marker,
                           repository="fixture", profile={"paths": ["projects/b"]})
        self.assertFalse(self.marker.exists())
        with self.assertRaisesRegex(smrev.SourceIdentityError, "layout"):
            self.generate(entry="projects/b")

    def test_version_query_declarations(self):
        expected = {
            "latest:pkg-*": {"kind": "latest", "pattern": "pkg-*"},
            "nearest:v*": {"kind": "nearest", "pattern": "v*"},
            "nearest-annotated:[0-9]*": {"kind": "nearest", "pattern": "[0-9]*", "tags": False},
            "exact": {"kind": "exact"},
            "timestamp:projects/a": {"kind": "timestamp", "path": "projects/a"},
        }
        for declaration, query in expected.items():
            self.assertEqual(smrev.parse_version_query(declaration), query)
        for bad in ("", "latest", "latest:", "exact:tag", "unknown:x", "timestamp:../a"):
            with self.subTest(bad=bad), self.assertRaises(smrev.SourceIdentityError):
                smrev.parse_version_query(bad)

    def test_environment_unset_empty_and_value_are_distinct(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("THEROCK_SMREV_TEST_VALUE", None)
            first = self.generate()
            os.environ["THEROCK_SMREV_TEST_VALUE"] = ""
            second = self.generate()
            os.environ["THEROCK_SMREV_TEST_VALUE"] = "release"
            third = self.generate()
        self.assertIsNone(first["environment"]["THEROCK_SMREV_TEST_VALUE"])
        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)

    def test_dynamic_environment_names(self):
        self.profile["environment_patterns"] = [r"THEROCK_CUDA[0-9]+_[0-9]+_ROOT"]
        first = self.generate()
        with patch.dict(os.environ, {"THEROCK_CUDA12_8_ROOT": "/sdk"}):
            self.assertNotEqual(first, self.generate())
        self.assertEqual(first, self.generate())

    def test_selected_version_tags_only(self):
        self.profile["versions"] = [{"kind": "latest", "pattern": "pkg-*"}]
        first = self.generate()
        command("git", "-C", self.repo, "tag", "other-9.0")
        self.assertEqual(first, self.generate())
        command("git", "-C", self.repo, "tag", "pkg-2.0")
        second = self.generate()
        self.assertNotEqual(first, second)
        command("git", "-C", self.repo, "tag", "pkg-1.0")
        self.assertEqual(second, self.generate())
        command("git", "-C", self.repo, "tag", "-d", "pkg-2.0")
        self.assertNotEqual(second, self.generate())

    def test_nearest_and_exact_tags(self):
        self.profile["versions"] = [{"kind": "nearest", "pattern": "v*"}, {"kind": "exact"}]
        first = self.generate()
        command("git", "-C", self.repo, "tag", "v1.0")
        second = self.generate()
        self.assertNotEqual(first, second)
        self.write("projects/b/input.txt", "unrelated\n")
        commit(self.repo)
        third = self.generate()
        self.assertEqual(second["version_inputs"][0], third["version_inputs"][0])
        self.assertNotEqual(second["version_inputs"][1], third["version_inputs"][1])

    def test_project_version_timestamp_and_shallow_rejection(self):
        self.profile["versions"] = [{"kind": "timestamp", "path": "projects/a"}]
        first = self.generate()
        self.write("projects/b/input.txt", "unrelated\n")
        commit(self.repo, date="2025-02-01T00:00:00+00:00")
        self.assertEqual(first, self.generate())
        clone = self.root / "shallow"
        command("git", "clone", "-q", "--depth=1", self.repo.as_uri(), clone)
        with self.assertRaisesRegex(smrev.SourceIdentityError, "complete Git history"):
            smrev.calculate(clone, self.profile, "fixture")

    def test_relocation_does_not_change_identity(self):
        first = self.generate()
        clone = self.root / "other-checkout"
        command("git", "clone", "-q", self.repo, clone)
        second = smrev.generate("a", clone / "projects/a", self.root / "other-build/.a.smrev",
                                repository="fixture", profile=self.profile)
        self.assertEqual(first, second)

    def test_check_rejects_stale_inputs_without_rewriting(self):
        self.generate()
        self.write("projects/a/input.txt", "next\n")
        commit(self.repo)
        with self.assertRaisesRegex(smrev.SourceIdentityError, "rerun cmake"):
            self.generate(check=True)
        self.assertFalse(self.marker.exists())

    def test_cli_request_checks_and_regeneration(self):
        # Requests retain configured declarations for build-time checks.
        for path in ("projects/amdsmi", "shared/cuid", "shared/sha256"):
            self.write(path + "/data.txt", "data\n")
        commit(self.repo)
        script = ROOT / "build_tools/generate_project_smrev.py"
        request = self.root / "request.json"
        args = (sys.executable, script, "--target", "amdsmi", "--source",
                self.repo / "projects/amdsmi", "--output", self.marker, "--request", request,
                "--repository", "rocm-systems", "--input", "shared/cuid", "--input", "shared/sha256")
        command(*args)
        before = self.marker.read_bytes()
        saved_request = json.loads(request.read_text())
        self.assertEqual(saved_request["entry"], "projects/amdsmi")
        self.assertEqual(saved_request["profile"]["paths"],
                         ["projects/amdsmi", "shared/cuid", "shared/sha256"])
        self.marker.unlink()
        command(*args)
        self.assertEqual(before, self.marker.read_bytes())
        command(sys.executable, script, "--check-request", request)
        result = command(*args, "--version-query", "unknown:tag", ok=False)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.marker.exists())
        command(*args)
        self.write("shared/cuid/data.txt", "changed\n")
        commit(self.repo)
        result = command(sys.executable, script, "--check-request", request, ok=False)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.marker.exists())

    def test_nested_checkout_presence_dirty_and_pin(self):
        nested = self.root / "nested-origin"
        init_git(nested)
        (nested / "data").write_text("one\n")
        commit(nested)
        command("git", "-C", self.repo, "-c", "protocol.file.allow=always",
                "submodule", "add", "-q", nested, "projects/a/vendor")
        commit(self.repo)
        initialized = self.generate()
        checkout = self.repo / "projects/a/vendor"
        (checkout / "data").write_text("dirty\n")
        with self.assertRaises(smrev.SourceIdentityError):
            self.generate()
        command("git", "-C", checkout, "restore", "data")
        command("git", "-C", self.repo, "submodule", "deinit", "-f", "projects/a/vendor")
        absent = self.generate()
        self.assertNotEqual(initialized, absent)
        self.assertEqual(absent["nested_sources"][0]["state"], "uninitialized")


def project_declarations():
    """Read only the literal FPRINT_PROJECT blocks for declaration coverage.

    This narrow test reader is not a CMake interpreter. Real CMake expansion and
    propagation are exercised separately by CMakeIntegrationTest.
    """
    helper = (ROOT / "cmake/therock_project_smrev.cmake").read_text()
    common = {name: shlex.split(body) for name, body in re.findall(
        r"set\((THEROCK_PROJECT_[A-Z_]+)\s+([^)]*)\)", helper)}
    defaults = {"THEROCK_" + name.upper().replace("-", "_") + "_SOURCE_DIR": value
                for name, value in re.findall(
                    r'therock_enable_external_source\("([^"]+)"\s+"([^"]+)"',
                    (ROOT / "CMakeLists.txt").read_text())}
    result = {}
    files = command("git", "-C", ROOT, "ls-files", "--", "*CMakeLists.txt").stdout.splitlines()
    for name in files:
        text = (ROOT / name).read_text()
        for match in re.finditer(r"therock_cmake_subproject_declare\(([^\s)]+)\s*\n", text):
            lines = text[match.end():].splitlines()
            if not lines or lines[0].strip() != "FPRINT_PROJECT_SOURCE":
                continue
            block = []
            for line in lines:
                item = line.strip()
                if item.startswith(("FPRINT_PROJECT_", '"', "${")):
                    block.append(item)
                else:
                    break
            fields = {}
            current = None
            for token in shlex.split("\n".join(block)):
                if token.startswith("FPRINT_PROJECT_"):
                    current = token.removeprefix("FPRINT_PROJECT_")
                    fields[current] = []
                elif token.startswith("${"):
                    fields[current].extend(common[token[2:-1]])
                else:
                    fields[current].append(token)
            # All reviewed declarations have a literal or rooted source path.
            source_match = re.search(r'(?m)^\s*EXTERNAL_SOURCE_DIR\s+"([^"\n]+)"', text[match.end():])
            if source_match is None:
                raise AssertionError(f"Missing source for {match[1]}")
            source = source_match[1]
            for variable, value in defaults.items():
                source = source.replace("${" + variable + "}", value)
            if source.startswith("${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/"):
                repository = "rocm-systems"
                entry = source.split("}/", 1)[1]
            elif source.startswith("${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/"):
                repository = "rocm-libraries"
                entry = source.split("}/", 1)[1]
            else:
                if source.startswith("${THEROCK_SOURCE_DIR}/"):
                    source = source.split("}/", 1)[1]
                else:
                    source = (Path(name).parent / source).as_posix()
                repository = next((r for r in (
                    "compiler/amd-llvm", "compiler/hipify", "base/half",
                    "base/rocm-cmake", "math-libs/libhipcxx"
                ) if source == r or source.startswith(r + "/")), None)
                if repository is None:
                    raise AssertionError(f"Unreviewed source expression: {source}")
                entry = str(Path(source).relative_to(repository))
            scope = fields.get("SCOPE", [entry])[0]
            profile = {
                "paths": [scope, *fields.get("INPUTS", [])],
                "optional_paths": fields.get("OPTIONAL_INPUTS", []),
                "environment": fields.get("ENVIRONMENT", []),
                "environment_patterns": fields.get("ENVIRONMENT_PATTERNS", []),
                "versions": [smrev.parse_version_query(q) for q in fields.get("VERSION_QUERIES", [])],
            }
            if match[1] in result:
                raise AssertionError(f"Duplicate declaration: {match[1]}")
            result[match[1]] = {"repository": repository, "entry": entry,
                                "profile": profile, "fields": fields, "block": "\n".join(block)}
    return result


class DeclarationTest(unittest.TestCase):
    def test_all_reviewed_declarations_resolve(self):
        declarations = project_declarations()
        self.assertEqual(len(declarations), 76)
        self.assertFalse((ROOT / "build_tools/project_smrev_inputs.json").exists())
        for target, declaration in declarations.items():
            with self.subTest(target=target):
                profile = declaration["profile"]
                for path in profile["paths"] + profile["optional_paths"]:
                    smrev.source_path(path)
                self.assertTrue(any(p == "." or declaration["entry"] == p or
                                    declaration["entry"].startswith(p + "/") for p in profile["paths"]))

    def test_required_shared_inputs_and_mutable_fetch_exclusions(self):
        declarations = project_declarations()
        for target, required in {
            "amdsmi": {"shared/cuid", "shared/sha256"},
            "hip-clr": {"projects/clr", "projects/hip", "projects/hrr"},
            "rocBLAS": {"shared/tensile"},
            "hipBLASLt": {"shared/origami", "shared/rocroller", "shared/stinkytofu"},
            "miopenprovider": {"dnn-providers/cmake"},
        }.items():
            self.assertLessEqual(required, set(declarations[target]["profile"]["paths"]))
        for target in ("mirage", "profiler-hub"):
            self.assertTrue(declarations[target]["fields"]["UNCACHEABLE_REASON"])

    @unittest.skipUnless(shutil.which("git"), "Git is required")
    def test_every_target_generates_and_repeats_with_its_declaration(self):
        declarations = project_declarations()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repositories = {}
            for declaration in declarations.values():
                name = declaration["repository"]
                if name not in repositories:
                    repositories[name] = root / name.replace("/", "_")
                    init_git(repositories[name])
                repo = repositories[name]
                for relative in [declaration["entry"], *declaration["profile"]["paths"]]:
                    path = repo / relative
                    path.mkdir(parents=True, exist_ok=True)
                    (path / "input.txt").write_text("fixture\n")
            for repo in repositories.values():
                commit(repo)
            for target, declaration in declarations.items():
                with self.subTest(target=target):
                    repo = repositories[declaration["repository"]]
                    output = root / "markers" / ("." + target + ".smrev")
                    args = (target, repo / declaration["entry"], output)
                    kwargs = {"repository": declaration["repository"],
                              "profile": declaration["profile"], "entry": declaration["entry"]}
                    first = smrev.generate(*args, **kwargs)
                    stamp = output.stat().st_mtime_ns
                    smrev.generate(*args, **kwargs, check=True)
                    self.assertEqual(first, smrev.generate(*args, **kwargs))
                    self.assertEqual(stamp, output.stat().st_mtime_ns)


@unittest.skipUnless(shutil.which("cmake") and shutil.which("ninja") and shutil.which("git"),
                     "CMake, Ninja and Git are required")
class CMakeIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / "sources"
        init_git(self.repo)
        for path in ("projects/amdsmi", "shared/cuid", "shared/sha256", "projects/rocm-core"):
            directory = self.repo / path
            directory.mkdir(parents=True)
            (directory / "input.txt").write_text("fixture\n")
        for project in ("amdsmi", "rocm-core"):
            (self.repo / "projects" / project / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.25)\nproject(fixture LANGUAGES NONE)\n'
                'install(FILES input.txt DESTINATION share)\n'
            )
        commit(self.repo)
        self.source = self.root / "superproject"
        self.source.mkdir()
        self.build = self.root / "build"
        (self.source / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.25)
project(smrev_fixture LANGUAGES NONE)
list(PREPEND CMAKE_MODULE_PATH "{ROOT}/cmake")
set(THEROCK_SOURCE_DIR "{ROOT}")
set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
set(THEROCK_ROCM_SYSTEMS_SOURCE_DIR "{self.repo}")
find_package(Python3 COMPONENTS Interpreter REQUIRED)
include(therock_globals)
include(therock_sanitizers)
include(therock_flag_utils)
include(therock_default_targets)
include(therock_subproject)
set(ROCM_BUILD_FLAGS_STATE_FILE "${{CMAKE_BINARY_DIR}}/flags.cmake")
file(CONFIGURE OUTPUT "${{ROCM_BUILD_FLAGS_STATE_FILE}}" CONTENT "# fixture flags\n")
foreach(name IN ITEMS amdsmi rocm-core)
  set(shared_inputs)
  if(name STREQUAL "amdsmi")
    set(shared_inputs shared/cuid shared/sha256)
  endif()
  therock_cmake_subproject_declare(${{name}}
    FPRINT_PROJECT_SOURCE
    FPRINT_PROJECT_INPUTS ${{shared_inputs}}
    EXTERNAL_SOURCE_DIR "{self.repo}/projects/${{name}}"
    BINARY_DIR "${{CMAKE_BINARY_DIR}}/${{name}}"
    NO_MERGE_COMPILE_COMMANDS DISABLE_AMDGPU_TARGETS)
  therock_cmake_subproject_activate(${{name}})
  get_target_property(fp ${{name}} THEROCK_FPRINT)
  file(WRITE "${{CMAKE_BINARY_DIR}}/${{name}}.fprint" "${{fp}}")
endforeach()
file(MAKE_DIRECTORY "${{CMAKE_CURRENT_SOURCE_DIR}}/consumer")
file(CONFIGURE OUTPUT "${{CMAKE_CURRENT_SOURCE_DIR}}/consumer/CMakeLists.txt"
  CONTENT "cmake_minimum_required(VERSION 3.25)\nproject(consumer NONE)\n")
therock_cmake_subproject_declare(consumer
  EXTERNAL_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/consumer"
  BINARY_DIR "${{CMAKE_BINARY_DIR}}/consumer"
  RUNTIME_DEPS amdsmi
  FPRINT_SOURCE_HASH DISABLE_AMDGPU_TARGETS NO_MERGE_COMPILE_COMMANDS)
therock_cmake_subproject_activate(consumer)
get_target_property(fp consumer THEROCK_FPRINT)
file(WRITE "${{CMAKE_BINARY_DIR}}/consumer.fprint" "${{fp}}")
''')

    def configure(self, *args, ok=True):
        return command("cmake", "-S", self.source, "-B", self.build, "-G", "Ninja",
                       "-DTHEROCK_PROJECT_SMREV=ON", *args, ok=ok)

    def fingerprints(self):
        return {name: (self.build / (name + ".fprint")).read_text()
                for name in ("amdsmi", "rocm-core", "consumer")}

    def test_actual_subproject_route_and_stability(self):
        self.configure()
        first = self.fingerprints()
        self.assertTrue(all(len(fp) == 64 for fp in first.values()))
        self.configure()
        self.assertEqual(first, self.fingerprints())
        (self.repo / "projects/amdsmi/input.txt").write_text("edited\n")
        commit(self.repo)
        self.configure()
        second = self.fingerprints()
        self.assertNotEqual(first["amdsmi"], second["amdsmi"])
        self.assertEqual(first["rocm-core"], second["rocm-core"])
        self.assertNotEqual(first["consumer"], second["consumer"])
        marker = self.build / "source-identities/.amdsmi.smrev"
        self.assertEqual(json.loads(marker.read_text())["schema"], smrev.SCHEMA)

    def test_child_stage_guard_and_no_change_build(self):
        self.configure()
        command("cmake", "--build", self.build, "--target", "amdsmi+stage")
        stamp = self.build / "amdsmi/stamp/configure.stamp"
        before = stamp.stat().st_mtime_ns
        command("cmake", "--build", self.build, "--target", "amdsmi+stage")
        self.assertEqual(before, stamp.stat().st_mtime_ns)
        self.assertEqual((self.build / "amdsmi/stage/share/input.txt").read_text(), "fixture\n")
        (self.repo / "projects/amdsmi/input.txt").write_text("dirty\n")
        result = command("cmake", "--build", self.build, "--target", "amdsmi+stage", ok=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.build / "amdsmi/stage/share/input.txt").read_text(), "fixture\n")

    def test_guard_rejects_dirty_source(self):
        self.configure()
        command("cmake", "--build", self.build, "--target", "therock-check-project-identities")
        (self.repo / "projects/amdsmi/input.txt").write_text("dirty\n")
        result = command("cmake", "--build", self.build, "--target", "therock-check-project-identities", ok=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tracked source changes", result.stdout)
        self.assertFalse((self.build / "source-identities/.amdsmi.smrev").exists())

    def test_disabling_restores_existing_reader(self):
        self.configure("-DTHEROCK_PROJECT_SMREV=OFF")
        self.assertFalse((self.build / "source-identities/.amdsmi.smrev").exists())
        first = self.fingerprints()
        (self.repo / "projects/amdsmi/input.txt").write_text("edited\n")
        commit(self.repo)
        self.configure("-DTHEROCK_PROJECT_SMREV=OFF")
        second = self.fingerprints()
        self.assertNotEqual(first["rocm-core"], second["rocm-core"])

    def test_missing_monorepo_policy_fails(self):
        path = self.source / "CMakeLists.txt"
        path.write_text(path.read_text() + f'''
add_custom_target(unreviewed)
therock_project_smrev(marker cacheable unreviewed "{self.repo}/projects/amdsmi")
''')
        result = self.configure(ok=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing project source-identity declaration", result.stdout)

    def test_unmodeled_fetch_has_no_valid_artifact_fingerprint(self):
        source = self.repo / "profilers/profiler-hub"
        source.mkdir(parents=True)
        (source / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.25)\nproject(fixture NONE)\n")
        commit(self.repo)
        path = self.source / "CMakeLists.txt"
        path.write_text(path.read_text() + f'''
therock_cmake_subproject_declare(profiler-hub
  FPRINT_PROJECT_SOURCE
  FPRINT_PROJECT_UNCACHEABLE_REASON "fixture: external input not modeled"
  EXTERNAL_SOURCE_DIR "{source}" BINARY_DIR "${{CMAKE_BINARY_DIR}}/hub"
  DISABLE_AMDGPU_TARGETS)
therock_cmake_subproject_activate(profiler-hub)
get_target_property(hub_fp profiler-hub THEROCK_FPRINT)
if(hub_fp)
  message(FATAL_ERROR "Unmodeled external fetch was marked cacheable")
endif()
''')
        self.configure()
        self.assertTrue((self.build / "source-identities/.profiler-hub.smrev").exists())

    def test_declared_values_and_conditions_reach_request(self):
        cmake = self.source / "CMakeLists.txt"
        text = cmake.read_text().replace(
            "    FPRINT_PROJECT_INPUTS ${shared_inputs}",
            """    FPRINT_PROJECT_INPUTS ${shared_inputs}
    FPRINT_PROJECT_OPTIONAL_INPUTS ${THEROCK_PROJECT_ROOT_INPUTS}
    FPRINT_PROJECT_ENVIRONMENT ${THEROCK_PROJECT_ROCM_VERSION_ENVIRONMENT}
      THEROCK_SMREV_TEST_VALUE
    FPRINT_PROJECT_ENVIRONMENT_PATTERNS "THEROCK_CUDA[0-9]+_[0-9]+_ROOT"
    FPRINT_PROJECT_VERSION_QUERIES "latest:fixture-*" "nearest-annotated:v*" exact""")
        text = text.replace("set(shared_inputs shared/cuid shared/sha256)",
                            """set(shared_inputs shared/cuid)
    if(INCLUDE_SHA256)
      list(APPEND shared_inputs shared/sha256)
    endif()""")
        cmake.write_text(text)
        self.configure("-DINCLUDE_SHA256=OFF")
        request = json.loads((self.build / "source-identities/amdsmi.request.json").read_text())
        profile = request["profile"]
        self.assertEqual(request["repository"], "rocm-systems")
        self.assertEqual(profile["paths"], ["projects/amdsmi", "shared/cuid"])
        self.assertIn("VERSION", profile["optional_paths"])
        self.assertIn("ROCM_BUILD_ID", profile["environment"])
        self.assertIn("THEROCK_SMREV_TEST_VALUE", profile["environment"])
        self.assertEqual(profile["environment_patterns"], ["THEROCK_CUDA[0-9]+_[0-9]+_ROOT"])
        self.assertEqual(profile["versions"], [
            {"kind": "latest", "pattern": "fixture-*"},
            {"kind": "nearest", "pattern": "v*", "tags": False}, {"kind": "exact"}])
        self.configure("-DINCLUDE_SHA256=ON")
        request = json.loads((self.build / "source-identities/amdsmi.request.json").read_text())
        self.assertIn("shared/sha256", request["profile"]["paths"])
        command("cmake", "--build", self.build, "--target", "therock-check-project-identities")

    def test_parent_scope_covers_nested_source(self):
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text().replace(
            "    FPRINT_PROJECT_SOURCE", "    FPRINT_PROJECT_SOURCE\n    FPRINT_PROJECT_SCOPE projects"))
        self.configure()
        marker = json.loads((self.build / "source-identities/.amdsmi.smrev").read_text())
        self.assertIn("projects", [p["path"] for p in marker["inputs"]])
        self.assertNotIn("projects/amdsmi", [p["path"] for p in marker["inputs"]])

    def test_conflicting_source_modes_fail(self):
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text().replace(
            "    FPRINT_PROJECT_SOURCE", "    FPRINT_PROJECT_SOURCE FPRINT_SOURCE_HASH"))
        result = self.configure(ok=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incompatible", result.stdout)

    def test_inputs_without_opt_in_fail(self):
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text().replace("    FPRINT_PROJECT_SOURCE\n", ""))
        result = self.configure(ok=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires FPRINT_PROJECT_SOURCE", result.stdout)

    def test_missing_scope_value_fails(self):
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text().replace(
            "    FPRINT_PROJECT_SOURCE", "    FPRINT_PROJECT_SOURCE FPRINT_PROJECT_SCOPE"))
        result = self.configure(ok=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FPRINT_PROJECT_SCOPE requires a value", result.stdout)

    def test_all_project_declarations_through_real_cmake(self):
        declarations = project_declarations()
        repositories = {}
        for declaration in declarations.values():
            name = declaration["repository"]
            if name not in repositories:
                repositories[name] = self.source / name
                init_git(repositories[name])
            for relative in [declaration["entry"], *declaration["profile"]["paths"]]:
                directory = repositories[name] / relative
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "input.txt").write_text("fixture\n")
        for repo in repositories.values():
            commit(repo)
        text = (self.source / "CMakeLists.txt").read_text().split("foreach(name IN ITEMS", 1)[0]
        text += f'''
set(THEROCK_SOURCE_DIR "{self.source}")
set(THEROCK_ROCM_SYSTEMS_SOURCE_DIR "{repositories['rocm-systems']}")
set(THEROCK_ROCM_LIBRARIES_SOURCE_DIR "{repositories['rocm-libraries']}")
'''
        for target, declaration in declarations.items():
            source = repositories[declaration["repository"]] / declaration["entry"]
            text += f'''
therock_cmake_subproject_declare({target}
  {declaration['block']}
  EXTERNAL_SOURCE_DIR "{source}" BINARY_DIR "${{CMAKE_BINARY_DIR}}/{target}"
  DISABLE_AMDGPU_TARGETS NO_MERGE_COMPILE_COMMANDS)
therock_project_smrev(marker cacheable {target} "{source}")
'''
        (self.source / "CMakeLists.txt").write_text(text)
        self.configure()
        self.assertEqual(len((self.build / "source-identities/requests.txt").read_text().splitlines()), 76)
        for target, declaration in declarations.items():
            with self.subTest(target=target):
                expected = smrev.calculate(repositories[declaration["repository"]],
                                           declaration["profile"], declaration["repository"])
                self.assertEqual((self.build / f"source-identities/.{target}.smrev").read_bytes(),
                                 smrev.encoded(expected))
        command("cmake", "--build", self.build, "--target", "therock-check-project-identities")


if __name__ == "__main__":
    unittest.main()
