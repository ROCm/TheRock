#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Source identity tests using temporary Git repositories and real CMake wiring."""

import importlib.util
import json
import os
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
        self.catalog = self.root / "catalog.json"
        self.marker = self.root / "build/.a.smrev"
        self.save_catalog()

    def save_catalog(self):
        self.catalog.write_text(json.dumps({
            "schema_version": 1,
            "targets": {"a": {"repository": "fixture", "entry": "projects/a", "profile": "a"}},
            "profiles": {"a": self.profile},
        }))

    def write(self, relative, content):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def generate(self, **kwargs):
        self.save_catalog()
        return smrev.generate("a", self.repo / "projects/a", self.marker,
                              catalog_path=self.catalog, **kwargs)

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

    def test_unknown_target_removes_owned_marker(self):
        self.generate()
        with self.assertRaisesRegex(smrev.SourceIdentityError, "no source-identity policy"):
            smrev.generate("absent", self.repo / "projects/a", self.marker, catalog_path=self.catalog)
        self.assertFalse(self.marker.exists())

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
                                catalog_path=self.catalog)
        self.assertEqual(first, second)

    def test_check_rejects_stale_inputs_without_rewriting(self):
        self.generate()
        self.write("projects/a/input.txt", "next\n")
        commit(self.repo)
        with self.assertRaisesRegex(smrev.SourceIdentityError, "rerun cmake"):
            self.generate(check=True)
        self.assertFalse(self.marker.exists())

    def test_cli_request_checks_and_regeneration(self):
        # Exercise the real checked-in AMD SMI policy through the CLI.
        for path in ("projects/amdsmi", "shared/cuid", "shared/sha256"):
            self.write(path + "/data.txt", "data\n")
        commit(self.repo)
        script = ROOT / "build_tools/generate_project_smrev.py"
        request = self.root / "request.json"
        args = (sys.executable, script, "--target", "amdsmi", "--source",
                self.repo / "projects/amdsmi", "--output", self.marker, "--request", request)
        command(*args)
        before = self.marker.read_bytes()
        self.marker.unlink()
        command(*args)
        self.assertEqual(before, self.marker.read_bytes())
        command(sys.executable, script, "--check-request", request)
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


class CatalogTest(unittest.TestCase):
    def test_all_reviewed_targets_and_profiles_resolve(self):
        data = json.loads(smrev.CATALOG.read_text())
        self.assertEqual(len(data["targets"]), 76)
        for target, policy in data["targets"].items():
            with self.subTest(target=target):
                profile = data["profiles"][policy["profile"]]
                self.assertTrue(profile["paths"])
                for path in profile["paths"] + profile.get("optional_paths", []):
                    smrev.source_path(path)
                self.assertTrue(any(path == "." or policy["entry"] == path or
                                    policy["entry"].startswith(path + "/") for path in profile["paths"]))

    def test_required_shared_inputs_and_mutable_fetch_exclusions(self):
        data = json.loads(smrev.CATALOG.read_text())
        for target, required in {
            "amdsmi": {"shared/cuid", "shared/sha256"},
            "hip-clr": {"projects/clr", "projects/hip", "projects/hrr"},
            "rocBLAS": {"shared/tensile"},
            "hipBLASLt": {"shared/origami", "shared/rocroller", "shared/stinkytofu"},
            "miopenprovider": {"dnn-providers/cmake"},
        }.items():
            self.assertLessEqual(required, set(data["profiles"][data["targets"][target]["profile"]]["paths"]))
        for target in ("mirage", "profiler-hub"):
            self.assertTrue(data["targets"][target]["uncacheable_reason"])

    @unittest.skipUnless(shutil.which("git"), "Git is required")
    def test_every_target_generates_and_repeats_with_its_catalog(self):
        # Tiny committed inputs exercise every policy/entry mapping. This is
        # policy coverage, not an audit of the real projects' complete builds.
        data = json.loads(smrev.CATALOG.read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repositories = {}
            for policy in data["targets"].values():
                name = policy["repository"]
                if name not in repositories:
                    repositories[name] = root / name.replace("/", "_")
                    init_git(repositories[name])
                repo = repositories[name]
                profile = data["profiles"][policy["profile"]]
                for relative in [policy["entry"], *profile["paths"]]:
                    path = repo / relative
                    path.mkdir(parents=True, exist_ok=True)
                    (path / "input.txt").write_text("fixture\n")
            for repo in repositories.values():
                commit(repo)
            for target, policy in data["targets"].items():
                with self.subTest(target=target):
                    repo = repositories[policy["repository"]]
                    output = root / "markers" / ("." + target + ".smrev")
                    first = smrev.generate(target, repo / policy["entry"], output)
                    stamp = output.stat().st_mtime_ns
                    smrev.generate(target, repo / policy["entry"], output, check=True)
                    self.assertEqual(first, smrev.generate(target, repo / policy["entry"], output))
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
  therock_cmake_subproject_declare(${{name}}
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
        self.assertIn("Missing project source-identity policy", result.stdout)

    def test_unmodeled_fetch_has_no_valid_artifact_fingerprint(self):
        source = self.repo / "profilers/profiler-hub"
        source.mkdir(parents=True)
        (source / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.25)\nproject(fixture NONE)\n")
        commit(self.repo)
        path = self.source / "CMakeLists.txt"
        path.write_text(path.read_text() + f'''
therock_cmake_subproject_declare(profiler-hub
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


if __name__ == "__main__":
    unittest.main()
