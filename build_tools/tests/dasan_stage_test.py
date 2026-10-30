# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dasan_stage


class DasanStageTest(unittest.TestCase):
    def test_imported_compiler_version_and_paths(self):
        with patch(
            "dasan_stage.subprocess.check_output",
            return_value="AMD clang version 24.0.0\n",
        ) as query:
            options = dasan_stage.compiler_options("math-libs", Path("/build"))
        self.assertEqual(query.call_count, 1)
        self.assertTrue(any(o.endswith("clang++") for o in options))
        self.assertTrue(any("LLVM_VERSION_MAJOR=24" in o for o in options))
        self.assertIn("-DTHEROCK_TEST_AMDGPU_TARGETS=gfx950", options)
        self.assertIn("-DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=gfx950-dcgpu", options)

    def test_unknown_compiler_rejected(self):
        with patch(
            "dasan_stage.subprocess.check_output", return_value="unknown compiler"
        ):
            with self.assertRaisesRegex(RuntimeError, "Cannot determine"):
                dasan_stage.compiler_options("emulation", Path("/build"))

    def test_bootstrap_without_imported_compiler(self):
        with patch("dasan_stage.subprocess.check_output") as query:
            dasan_stage.compiler_options("compiler-runtime", Path("/build"))
        query.assert_not_called()

    def test_complete_compiler_fetch_and_patch_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = []
            with patch("dasan_stage.REPO", root), patch(
                "dasan_stage.run", side_effect=lambda cmd, log: events.append(cmd)
            ), patch(
                "dasan_stage.apply_dasan_patches",
                side_effect=lambda log: events.append(["DASAN_PATCHES"]),
            ), patch(
                "dasan_stage.capture", return_value="sha"
            ):
                dasan_stage.prepare("compiler-runtime", root / "build", root / "output")
            fetches = [c for c in events if "build_tools/fetch_sources.py" in c]
            self.assertEqual(len(fetches), 2)
            self.assertIn("--no-apply-patches", fetches[0])
            self.assertIn("--no-update-submodules", fetches[1])
            self.assertEqual(events[-1], ["DASAN_PATCHES"])
            self.assertTrue(all("--artifacts" not in c for c in fetches))
            manifest = json.loads((root / "output/sources.json").read_text())
            self.assertEqual(
                manifest["binary_dependency_policy"], "current workflow run only"
            )

    def test_full_math_stage_not_ml_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = []
            with patch("dasan_stage.REPO", root), patch(
                "dasan_stage.run", side_effect=lambda cmd, log: events.append(cmd)
            ), patch("dasan_stage.apply_dasan_patches") as patches, patch(
                "dasan_stage.capture", return_value="sha"
            ), patch(
                "dasan_stage.compiler_options", return_value=[]
            ):
                dasan_stage.prepare("math-libs", root / "build", root / "output")
            patches.assert_not_called()
            for command in events:
                if "build_tools/fetch_sources.py" in command:
                    self.assertEqual(command[command.index("--stage") + 1], "math-libs")
                    self.assertNotIn("--artifacts", command)

    def test_audit_command_and_argument_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = root / "build/comm-libs/rccl/build/compile_commands.json"
            database.parent.mkdir(parents=True)
            database.write_text(
                json.dumps(
                    [
                        {"command": "clang++ -fsanitize=daddress -c kernel.cpp"},
                        {
                            "arguments": [
                                "clang++",
                                "-fsanitize=daddress",
                                "-c",
                                "host.cpp",
                            ]
                        },
                        {"command": "cc -c bootstrap.c"},
                    ]
                )
            )
            result = dasan_stage.audit("comm-libs", root / "build", root / "logs")
            self.assertEqual(result["native_commands"], 3)
            self.assertEqual(result["daddress_commands"], 2)
            self.assertEqual(
                result["databases"][0]["without_daddress"], ["cc -c bootstrap.c"]
            )

    def test_uninstrumented_native_stage_fails_and_retains_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "build"
            build.mkdir()
            (build / "compile_commands.json").write_text(
                json.dumps([{"command": "clang++ -c file.cpp"}])
            )
            with self.assertRaisesRegex(RuntimeError, "none enable"):
                dasan_stage.audit("comm-libs", build, root / "logs")
            report = json.loads((root / "logs/instrumentation-audit.json").read_text())
            self.assertTrue(report["failures"])
            self.assertTrue(report["exclusions"])

    def test_bootstrap_native_stage_has_explicit_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "compile_commands.json").write_text(
                json.dumps([{"command": "c++ -c bootstrap.cpp"}])
            )
            result = dasan_stage.audit("compiler-runtime", root, root / "logs")
            self.assertEqual(result["failures"], [])

    def test_combined_sanitizer_flags_count_as_instrumented(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "compile_commands.json").write_text(
                json.dumps(
                    [{"command": "clang++ -fsanitize=undefined,daddress -c file.cpp"}]
                )
            )
            result = dasan_stage.audit("emulation", root, root / "logs")
            self.assertEqual(result["daddress_commands"], 1)


if __name__ == "__main__":
    unittest.main()
