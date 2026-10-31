# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host/source fixtures only; no ROCm compiler or performance run."""

from contextlib import closing
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import shutil
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "normal_rocfft_aot", ROOT / "build_tools/normal_rocfft_aot.py"
)
product = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(product)


def environment():
    return dict(
        GITHUB_ACTIONS="true",
        GITHUB_REPOSITORY="ROCm/TheRock",
        NORMAL_VALIDATION="true",
        STAGE_NAME="math-libs",
        AMDGPU_FAMILIES="gfx103X-all",
        RELEASE_TYPE="ci",
        GITHUB_RUN_ID="fixture-run",
        GITHUB_RUN_ATTEMPT="1",
    )


def fixture_prepare(root, case):
    build = root / "build"
    output = build / "logs/normal-validation/rocfft-aot"
    output.mkdir(parents=True)
    product.save(output / "source-control.json", {"host_fixture": True})
    fixture_configure_edge(root, root / "source")
    toolchain = build / "math-libs/rocFFT/_toolchain.cmake"
    toolchain.write_text(
        "set(GPU_TARGETS "
        + ";".join(product.ARCHES)
        + ' CACHE STRING "From super-project" FORCE)\n'
    )
    with patch.object(product, "verify_source"):
        state = product.prepare_product(
            build, output, root / "source", case, environment()
        )
    return build, output, state


def fixture_configure_edge(root, libraries):
    fixture = json.loads(
        (
            ROOT / "build_tools/tests/fixtures/normal_rocfft_configure_command.json"
        ).read_text()
    )
    original = fixture["actual_rocfft_configure_command"]
    if (
        hashlib.sha256(original.encode()).hexdigest()
        != fixture["configure_command_sha256"]
    ):
        raise AssertionError("Retained command provenance changed")
    command = original.replace(
        fixture["workspace_prefix"], shlex.quote(root.as_posix())
    )
    # Only the source directory differs in older host fixtures' inert layout.
    command = command.replace(
        shlex.quote(root.as_posix()) + "/rocm-libraries/projects/rocfft",
        shlex.quote((libraries / "projects/rocfft").as_posix()),
    )
    build = root / "build"
    toolchain = build / "math-libs/rocFFT/_toolchain.cmake"
    toolchain.parent.mkdir(parents=True, exist_ok=True)
    toolchain.write_text(
        "set(GPU_TARGETS "
        + ";".join(product.ARCHES)
        + ' CACHE STRING "From super-project" FORCE)\n'
    )
    (build / "build.ninja").write_text(
        "rule configure_rocfft\n  command = "
        + command
        + "\nbuild math-libs/rocFFT/stamp/configure.stamp: configure_rocfft\n"
    )
    return command, toolchain


class ProductTests(unittest.TestCase):
    def test_actual_configure_edge_selects_toolchain_without_executing_or_configuring(
        self,
    ):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "workspace with spaces"
            libraries = root / "rocm-libraries"
            command, toolchain = fixture_configure_edge(root, libraries)
            nested = root / "build/math-libs/rocFFT/build"
            # An attractive wrong basename must never be selected as a fallback.
            wrong = toolchain.with_name("build_toolchain.cmake")
            wrong.write_text('set(GPU_TARGETS gfx942 CACHE STRING "wrong" FORCE)')
            before = {p.relative_to(root) for p in root.rglob("*")}
            receipt = product.generated_toolchain(root / "build", nested, libraries)
            self.assertEqual(receipt["command"], command)
            self.assertEqual(Path(receipt["toolchain"]), toolchain)
            self.assertEqual(receipt["toolchain_sha256"], product.digest(toolchain))
            self.assertFalse(nested.exists())
            self.assertEqual(before, {p.relative_to(root) for p in root.rglob("*")})
            # A real property/command change is followed, not another basename guess.
            renamed = toolchain.with_name("original_generated_toolchain.cmake")
            toolchain.rename(renamed)
            graph = root / "build/build.ninja"
            graph.write_text(
                graph.read_text().replace(
                    "/_toolchain.cmake", "/original_generated_toolchain.cmake"
                )
            )
            self.assertEqual(
                Path(
                    product.generated_toolchain(root / "build", nested, libraries)[
                        "toolchain"
                    ]
                ),
                renamed,
            )
            with self.assertRaises(subprocess.CalledProcessError):
                product.generated_toolchain(root / "missing-build", nested, libraries)
            graph.write_text(graph.read_text().replace("-B", "-Bwrong-", 1))
            with self.assertRaisesRegex(ValueError, "source/build paths"):
                product.generated_toolchain(root / "build", nested, libraries)

    def test_default_and_other_families_stages_do_not_select_product(self):
        self.assertFalse(product.selected(False, {}, "win32"))
        for name, value in [
            ("AMDGPU_FAMILIES", "gfx94X-all"),
            ("STAGE_NAME", "comm-libs"),
        ]:
            self.assertFalse(
                product.selected(True, {**environment(), name: value}, "linux")
            )
        self.assertTrue(product.selected(True, environment(), "linux"))
        for name, value in [
            ("GITHUB_REPOSITORY", "ROCm/rockrel"),
            ("NORMAL_VALIDATION", "false"),
            ("RELEASE_TYPE", "nightly"),
            ("ROCFFT_LAYER", "1"),
            ("HIPRTC_COMPILE_OPTIONS_APPEND", "-O0"),
            ("AMD_COMGR_EMIT_VERBOSE_LOGS", "1"),
        ]:
            with self.assertRaises(ValueError):
                product.selected(True, {**environment(), name: value}, "linux")
        with self.assertRaises(ValueError):
            product.selected(True, environment(), "win32")

    def test_fresh_controls_differ_only_in_policy_after_random_identifiers(self):
        states = []
        with tempfile.TemporaryDirectory() as raw:
            for case in ["baseline", "candidate"]:
                root = Path(raw) / case
                _, output, state = fixture_prepare(root, case)
                states.append(state)
                self.assertFalse(Path(state["path"]).exists())
                self.assertEqual(
                    Path(state["marker"]).read_bytes(),
                    (state["owner_id"] + "\n").encode(),
                )
                self.assertEqual(state["initial_comgr"]["file_count"], 0)
                self.assertTrue(state["initial_comgr"]["complete"])
                self.assertEqual(state["device"], state["original_parent_device"])
                self.assertEqual(state["runtime_environment"]["AMD_COMGR_CACHE"], "1")
                self.assertEqual(
                    state["runtime_environment"]["AMD_COMGR_CACHE_POLICY"], ""
                )
                with patch.object(product, "verify_source"), self.assertRaises(
                    ValueError
                ):
                    product.prepare_product(
                        Path(state["build"]),
                        output,
                        root / "source",
                        case,
                        environment(),
                    )
                Path(state["path"]).write_bytes(b"keep existing DB")
                original = Path(state["nested"]) / "library/src/rocfft_temp_cache.db"
                original.write_bytes(b"keep original default")
                with self.assertRaises(ValueError):
                    product.require_fresh_outputs(Path(state["nested"]))
                self.assertEqual(original.read_bytes(), b"keep original default")
            self.assertEqual(
                states[0]["runtime_environment"]["ROCFFT_AOT_BATCH_TRANSACTION"], "0"
            )
            self.assertEqual(
                states[1]["runtime_environment"]["ROCFFT_AOT_BATCH_TRANSACTION"], "all"
            )
            self.assertNotEqual(states[0]["owner_id"], states[1]["owner_id"])
            self.assertEqual(Path(states[0]["path"]).read_bytes(), b"keep existing DB")

    def test_stock_outputs_and_target_changes_refuse_without_deletion(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            nested = root / "nested"
            path = nested / "library/src/rocfft_kernel_cache_gfx1030.db"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"required output")
            with self.assertRaises(ValueError):
                product.require_fresh_outputs(nested)
            self.assertEqual(path.read_bytes(), b"required output")
            file = root / "toolchain.cmake"
            file.write_text('set(GPU_TARGETS gfx942 CACHE STRING "original" FORCE)')
            with self.assertRaises(ValueError):
                product.check_toolchain(file)

    def test_environment_export_is_literal_and_rejects_multiline_before_write(self):
        with tempfile.TemporaryDirectory() as raw:
            file = Path(raw) / "github-env"
            file.write_text("existing=value\n")
            with self.assertRaises(ValueError):
                product.export_environment(
                    {"runtime_environment": {"ONE": "safe", "TWO": "bad\nvalue"}}, file
                )
            self.assertEqual(file.read_text(), "existing=value\n")
            product.export_environment(
                {"runtime_environment": {"PATH_WITH_SPACE": "/build with space/cache"}},
                file,
            )
            self.assertIn("PATH_WITH_SPACE=/build with space/cache\n", file.read_text())

    def test_private_metadata_snapshot_is_complete_and_content_sensitive_to_size(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "cache-entry").write_bytes(b"original")
            first = product.cache_inventory(root)
            (root / "cache-entry").write_bytes(b"modified larger")
            second = product.cache_inventory(root)
            self.assertTrue(first["complete"])
            self.assertEqual(first["file_count"], 1)
            self.assertNotEqual(first["metadata_sha256"], second["metadata_sha256"])

    def test_private_cache_inventory_does_not_inherit_generic_100k_cap(self):
        from types import SimpleNamespace

        class Entry:
            def __init__(self, index):
                self.index = index

            def is_symlink(self):
                return False

            def is_file(self):
                return True

            def stat(self):
                return SimpleNamespace(st_size=1)

            def relative_to(self, root):
                return f"entry-{self.index:06d}"

        class Root:
            def rglob(self, pattern):
                return (Entry(i) for i in range(100005))

            def stat(self):
                return SimpleNamespace(st_dev=1)

        inventory = product.cache_inventory(Root())
        self.assertTrue(inventory["complete"])
        self.assertEqual(inventory["file_count"], 100005)
        self.assertEqual(inventory["bytes"], 100005)
        self.assertEqual(inventory["files"][-1]["path"], "entry-100004")

    def test_patch_has_no_generator_flag_or_dependency_edits_and_commit_precedes_copy(
        self,
    ):
        text = (ROOT / product.PATCH).read_text(encoding="utf-8")
        paths = set(
            product.re.findall(r"^diff --git a/(\S+) b/\1$", text, product.re.MULTILINE)
        )
        self.assertEqual(paths, product.HOST_FILES)
        self.assertNotIn("CMakeLists.txt", text)
        self.assertNotIn("rtc_compile.cpp", text)
        additions = "\n".join(
            l[1:]
            for l in text.splitlines()
            if l.startswith("+") and not l.startswith("+++")
        )
        self.assertNotIn("work_queue.h", additions)
        self.assertNotIn("LOG_RTC", additions)
        self.assertNotIn("ftime", additions)
        self.assertLess(
            text.index("normal_aot_transaction(false)"), text.index("->write_aot_cache")
        )
        self.assertLess(
            text.index("RTCCache::single.reset()"),
            text.index("normal_aot.publish_success"),
        )


class ExactSourceApplyTests(unittest.TestCase):
    def test_exact_f4_host_patch_preserves_history_and_generator_inputs(self):
        fixture = os.environ.get("ROCFFT_AOT_TEST_SOURCE_FIXTURE")
        if not fixture:
            self.skipTest("Declare the exact-f4 selected-source fixture repository")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "libraries"
            subprocess.run(
                [
                    "git",
                    "-c",
                    "core.autocrlf=false",
                    "clone",
                    "--quiet",
                    "--no-hardlinks",
                    fixture,
                    str(source),
                ],
                check=True,
            )
            identity = product.source_identity(source)
            before = product.generator_inputs(source / "projects/rocfft")
            self.assertGreater(len(before), 60)
            with self.assertRaisesRegex(ValueError, "pristine pinned"):
                product.apply_product(source, root / "wrong-head")
            receipt = product.apply_product(
                source, root / "receipts", expected_head=identity["head"]
            )
            self.assertEqual(product.source_identity(source), identity)
            self.assertEqual(receipt["generator_inputs"], before)
            self.assertEqual(
                receipt["header_sha256"], product.digest(ROOT / product.HEADER)
            )
            product.verify_source(source, receipt)
            changed = source / "projects/rocfft" / before[0]["path"]
            changed.write_bytes(changed.read_bytes() + b"\n// changed checksum input\n")
            with self.assertRaisesRegex(ValueError, "Generator checksum"):
                product.verify_source(source, receipt)


class RealSQLiteHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = os.environ.get("ROCFFT_PROFILE_TEST_CXX")
        include = os.environ.get("ROCFFT_AOT_TEST_SQLITE_DIR")
        library = os.environ.get("ROCFFT_AOT_TEST_SQLITE_LIB")
        if not all([compiler, include, library, shutil.which("ninja")]):
            raise unittest.SkipTest(
                "Declare host compiler, real SQLite3.51.3 include/library and Ninja"
            )
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.binary = cls.root / "sqlite-fixture.exe"
        subprocess.run(
            [
                *json.loads(compiler),
                "-std=c++17",
                "-pthread",
                "-I",
                include,
                str(ROOT / "build_tools/tests/normal_rocfft_aot_fixture.cpp"),
                library,
                "-o",
                str(cls.binary),
            ],
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def make_case(self, case):
        root = self.root / case
        build, output, state = fixture_prepare(root, case)
        nested = Path(state["nested"])
        directory = nested / "library/src"
        for name in ["rocfft_aot_helper", "rocfft_rtc_helper"]:
            (directory / name).write_bytes(b"host fixture, not executed by Ninja")
        ninja = ["rule cache", "  command = $command"]
        for i, arch in enumerate(product.ARCHES):
            target = f"library/src/rocfft_kernel_cache_{arch}.db"
            command = (
                f'cd "{directory}" && cmake -E env PATH=/fixture '
                f'"{directory / "rocfft_aot_helper"}" "" "{nested / target}" '
                f'"{directory / "rocfft_rtc_helper"}" {arch}'
            )
            ninja += [
                f"build {target}: cache",
                f"  command = {command}",
                f"build library/src/rocfft_kernel_cache_{arch}_target: phony {target}"
                + (
                    f" library/src/rocfft_kernel_cache_{product.ARCHES[i-1]}_target"
                    if i
                    else ""
                ),
            ]
        (nested / "build.ninja").write_text("\n".join(ninja) + "\n")
        result = subprocess.run(
            [str(self.binary), str(directory)],
            env={**os.environ, **state["runtime_environment"]},
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("COMMIT rollback", result.stdout)
        return root, output, state

    def test_real_sqlite_end_receipts_full_output_gate_and_payload_parity(self):
        inventories = []
        for case in ["baseline", "candidate"]:
            root, output, state = self.make_case(case)
            with patch.object(product, "verify_source"):
                result = product.validate_product(
                    output, root / "source", case, environment(), entries=1
                )
            self.assertTrue(result["success"])
            self.assertEqual(result["total_entries"], 7)
            self.assertTrue(result["comgr_metadata_complete"])
            records = json.loads((output / "cache-inventories.json").read_text())
            inventories.append(
                [
                    {
                        k: r[k]
                        for k in ["arches", "entries", "file_sha256", "semantic_sha256"]
                    }
                    for r in records
                ]
            )
            for arch in product.ARCHES:
                receipt = json.loads(
                    (Path(state["receipts"]) / f"{arch}.json").read_text()
                )
                self.assertEqual(receipt["commit_succeeded"], case == "candidate")
                self.assertEqual(receipt["policy_before"], receipt["policy_after"])
            # A success exit/output count alone must not hide a partial transaction,
            # fallback or missing completion receipt on later acceptance attempts.
            path = Path(state["receipts"]) / "gfx1036.json"
            original = path.read_text()
            for field, value in [
                ("commit_succeeded", case != "candidate"),
                ("subprocess_fallback_errors", 1),
            ]:
                row = json.loads(original)
                row[field] = value
                path.write_text(json.dumps(row))
                with patch.object(product, "verify_source"), self.assertRaises(
                    ValueError
                ):
                    product.validate_product(
                        output, root / "source", case, environment(), entries=1
                    )
                path.write_text(original)
            with patch.object(product, "verify_source"), self.assertRaisesRegex(
                ValueError, "another case/run"
            ):
                product.validate_product(
                    output,
                    root / "source",
                    case,
                    {**environment(), "GITHUB_RUN_ID": "another"},
                    entries=1,
                )
        self.assertEqual(inventories[0], inventories[1])


if __name__ == "__main__":
    unittest.main()
