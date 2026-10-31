# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import ast
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import profile_cache_probe as probe
from profile_cache_hash_inputs import rewrite_outputs


def arguments(root):
    return [
        str(root / "sdk/clang++"),
        "-O3",
        "-std=c++17",
        "-MD",
        "-MF",
        "object.d",
        *["--offload-arch=" + arch for arch in sorted(probe.preparation.ARCHES)],
        "-c",
        str(root / "benchmark_device_segmented_radix_sort_pairs.cpp"),
        "-o",
        "object.o",
    ]


class CacheProbeTest(unittest.TestCase):
    def test_actual_command_shape_and_output_containment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            good = arguments(root)
            self.assertEqual(
                probe.original_arguments({"arguments": ["ccache", *good]}), good
            )
            rewritten, changes = rewrite_outputs(good, root / "diagnostics")
            self.assertEqual(changes, {"-MF": "object.d", "-o": "object.o"})
            for flag in ("-MF", "-o"):
                self.assertTrue(
                    Path(rewritten[rewritten.index(flag) + 1]).is_relative_to(
                        root / "diagnostics"
                    )
                )
            for bad in (
                good + ["--offload-jobs=2"],
                good + ["@response"],
                [arg for arg in good if arg != "--offload-arch=gfx1036"],
                good + ["--offload-arch=gfx1036"],
                good + ["&&", "touch", "bad"],
                ["/usr/bin/env", *good],
            ):
                with self.subTest(bad=bad[-3:]), self.assertRaises(ValueError):
                    probe.original_arguments({"arguments": bad})
            for extra in (
                "-fplugin=bad.so",
                "-Xclang",
                "-fmodules-cache-path=inputs",
                "-fprofile-generate=inputs",
                "-ftime-trace",
                "--config=hidden.cfg",
                "-MJbad.json",
            ):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    rewrite_outputs([*good, extra], root / "diagnostics")

    def test_clone_is_real_hardlinked_tree_with_auditable_symlinks_and_content_checks(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "A", Path(temp) / "B"
            a.mkdir()
            (a / "header.h").write_text("unchanged input\n")
            try:
                (a / "absolute.h").symlink_to(a / "header.h")
            except OSError:
                self.skipTest("Symlink support required")
            remapping = probe.clone_prefix(a, b)
            self.assertFalse(b.is_symlink())
            self.assertTrue(os.path.samefile(a / "header.h", b / "header.h"))
            self.assertEqual(os.readlink(b / "absolute.h"), str(b / "header.h"))
            self.assertEqual(remapping[0]["relative_path"], "absolute.h")
            before = probe.content_manifest(a, {})
            self.assertEqual(before, probe.content_manifest(b, {}))
            (b / "header.h").write_text("mutated fixture\n")
            self.assertNotEqual(before, probe.content_manifest(a, {}))
            with self.assertRaises(ValueError):
                probe.clone_prefix(a, b)

    def test_prepare_targets_configure_only_with_pinned_sdk_and_unchanged_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            build, output = root / "build", root / "diagnostics"
            output.mkdir()
            nested = build / "math-libs/hipCCL_tests/build"
            nested.mkdir(parents=True)
            (build / "CMakeCache.txt").write_text(
                "THEROCK_PACKAGE_VERSION:STRING="
                + probe.preparation.PACKAGE_VERSION
                + "\n"
            )
            (nested / "CMakeCache.txt").write_text(
                "BUILD_TEST:BOOL=ON\nBUILD_BENCHMARK:BOOL=ON\nBENCHMARK_CONFIG_TUNING:BOOL=OFF\nBENCHMARK_AUTOTUNED_TYPES_ONLY:BOOL=OFF\n"
            )
            commands = []

            def capture(command, *unused):
                if command[:2] == ["git", "rev-parse"]:
                    return probe.preparation.LIBRARIES_PIN
                return "96" if command == ["nproc"] else ""

            module = SimpleNamespace(
                get_topology=lambda: SimpleNamespace(
                    get_python_requires_for_stage=lambda unused: []
                )
            )
            with patch.dict(sys.modules, {"configure_stage": module}), patch.object(
                probe.preparation, "ROOT", root
            ), patch.object(
                probe.preparation, "capture", side_effect=capture
            ), patch.object(
                probe.preparation,
                "run_logged",
                side_effect=lambda command, *unused: commands.append(command),
            ), patch.object(
                probe.preparation, "snapshot"
            ), patch.object(
                probe.preparation.subprocess, "check_output", return_value="-DTEST=1"
            ):
                self.assertEqual(probe.preparation.prepare(build, output, {}), nested)
            self.assertIn(probe.preparation.SDK_RUN, commands[0])
            self.assertIn("--bootstrap", commands[0])
            builds = [command for command in commands if "--build" in command]
            self.assertEqual(len(builds), 1)
            self.assertEqual(
                builds[0][builds[0].index("--target") + 1], "hipCCL_tests+configure"
            )
            self.assertNotIn(probe.preparation.OBJECT, json.dumps(commands))
            calls = [
                node.func.attr
                for node in ast.walk(ast.parse(Path(probe.__file__).read_text()))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ]
            for forbidden in (
                "run_trial",
                "force_object_rebuild",
                "capability_probe",
                "patch_flags",
            ):
                self.assertNotIn(forbidden, calls)

    def test_repository_execution_guard_rejects_rockrel(self):
        for env in (
            {},
            {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "ROCm/rockrel"},
        ):
            with self.assertRaises(RuntimeError):
                probe.preparation.require_ci(env)

    def test_actual_preprocessor_line_markers_capture_content_without_an_extra_compile(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            header = root / "header with space.h"
            header.write_text("int header;\n")
            raw = root / "raw"
            raw.mkdir()
            (raw / "cpp_stdout.tmp.i").write_text(
                '# 1 "<built-in>"\n# 1 ' + json.dumps(str(header)) + " 1\nint header;\n"
            )
            rows = probe.consumed_inputs(raw, root, root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["path"], "$CHECKOUT/header with space.h".replace("/", os.sep)
            )
            self.assertEqual(rows[0]["sha256"], probe.sha256(header))

    def test_workflow_is_default_off_the_rock_only_and_suppresses_full_builds(self):
        workflow = yaml.load(
            (
                Path(probe.__file__).parents[1] / ".github/workflows/multi_arch_ci.yml"
            ).read_text(),
            Loader=yaml.BaseLoader,
        )
        for event in ("workflow_dispatch", "workflow_call"):
            self.assertEqual(
                workflow["on"][event]["inputs"]["profile_cache_probe"]["default"],
                "false",
            )
        for name in (
            "setup",
            "linux_build_and_test",
            "windows_build_and_test",
            "manifest_diff",
            "ci_summary",
            "profile_offload_trial",
        ):
            self.assertIn("!inputs.profile_cache_probe", workflow["jobs"][name]["if"])
        job = workflow["jobs"]["profile_cache_probe"]
        self.assertIn("github.repository == 'ROCm/TheRock'", job["if"])
        self.assertEqual(job["runs-on"], "aws-linux-scale-rocm-prod")
        self.assertIn("/../cache-probe-diagnostics", job["env"]["CACHE_PROBE_OUTPUT"])
        self.assertEqual(job["permissions"], {"contents": "read", "actions": "read"})
        text = json.dumps(job)
        for forbidden in (
            "configure_aws",
            "artifact_manager.py push",
            "run_trial",
            "id-token",
        ):
            self.assertNotIn(forbidden, text)
        self.assertEqual(job["steps"][-1]["if"], "always()")


if __name__ == "__main__":
    unittest.main()
