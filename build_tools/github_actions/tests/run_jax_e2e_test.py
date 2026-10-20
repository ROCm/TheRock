# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for external-builds/jax/run_jax_e2e.py."""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
SELECTION_DIR = REPO_ROOT / "external-builds" / "jax" / "test_selection"

sys.path.insert(0, os.fspath(REPO_ROOT / "external-builds" / "jax"))

import run_jax_e2e as e2e

STACK = {
    "jax": "0.11.2",
    "jaxlib": "0.11.2",
    "jax-rocm10-plugin": "0.11.2+rocm10.2.0a20261016",
    "jax-rocm10-pjrt": "0.11.2+rocm10.2.0a20261016",
}


class WorkloadListTest(unittest.TestCase):
    def read(self, text: str) -> list[e2e.Workload]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.txt"
            path.write_text(text)
            return e2e.read_workload_list(path)

    def test_lines_comments_and_blanks(self):
        workloads = self.read(
            "# header\n\nmaxtext_release llama3_1_8b 8  # dense\nflax_release resnet 4\n"
        )

        self.assertEqual(
            workloads,
            [
                e2e.Workload("maxtext_release", "llama3_1_8b", 8),
                e2e.Workload("flax_release", "resnet", 4),
            ],
        )

    def test_a_malformed_line_names_its_place(self):
        for line in ["maxtext_release llama3_1_8b", "a b eight", "a b 0", "a b 1 c"]:
            with self.subTest(line=line), self.assertRaisesRegex(ValueError, ":1:"):
                self.read(line + "\n")

    def test_a_list_naming_nothing_is_an_error(self):
        with self.assertRaises(ValueError):
            self.read("# nothing here\n")


class CheckedInListsTest(unittest.TestCase):
    def test_medium_is_the_two_maxtext_workloads(self):
        workloads = e2e.read_workload_list(SELECTION_DIR / "e2e_medium.txt")

        self.assertEqual(
            [(w.target, w.name, w.gpus) for w in workloads],
            [
                ("maxtext_release", "llama3_1_8b", 8),
                ("maxtext_release", "deepseek2_16b", 8),
            ],
        )

    def test_large_is_every_jax_lab_workload_and_covers_medium(self):
        medium = set(e2e.read_workload_list(SELECTION_DIR / "e2e_medium.txt"))
        large = e2e.read_workload_list(SELECTION_DIR / "e2e_large.txt")

        self.assertEqual(len(large), 13)
        self.assertEqual(len(set(large)), 13)
        self.assertLessEqual(medium, set(large))
        self.assertEqual(
            {w.target for w in large},
            {
                "maxtext_release",
                "flax_release",
                "hf_diffusers_release",
                "hf_transformers_release",
            },
        )
        self.assertLessEqual({w.gpus for w in large}, {1, 4, 8})


class VisibleDevicesTest(unittest.TestCase):
    def test_without_a_restriction_takes_the_first_ones(self):
        self.assertEqual(e2e.visible_devices({}, 4), "0,1,2,3")

    def test_stays_inside_an_existing_restriction(self):
        env = {"HIP_VISIBLE_DEVICES": "4,5,6,7"}

        self.assertEqual(e2e.visible_devices(env, 2), "4,5")
        self.assertEqual(e2e.visible_devices(env, 4), "4,5,6,7")

    def test_too_few_visible_is_an_error(self):
        with self.assertRaises(ValueError):
            e2e.visible_devices({"HIP_VISIBLE_DEVICES": "0,1"}, 8)


class StackTest(unittest.TestCase):
    def test_constraints_pin_every_wheel_under_test(self):
        self.assertEqual(
            e2e.constraints(STACK).splitlines(),
            [
                "jax==0.11.2",
                "jax-rocm10-pjrt==0.11.2+rocm10.2.0a20261016",
                "jax-rocm10-plugin==0.11.2+rocm10.2.0a20261016",
                "jaxlib==0.11.2",
            ],
        )

    def test_a_missing_plugin_is_an_error(self):
        dists = [
            mock.Mock(metadata={"Name": name}, version=version)
            for name, version in [("jax", "0.11.2"), ("jaxlib", "0.11.2")]
        ]
        with mock.patch.object(e2e.importlib.metadata, "distributions", lambda: dists):
            with self.assertRaisesRegex(RuntimeError, "plugin"):
                e2e.installed_jax_stack()

    def test_names_are_normalized(self):
        dists = [
            mock.Mock(metadata={"Name": name}, version="1")
            for name in [
                "jax",
                "jaxlib",
                "jax_rocm10_plugin",
                "jax_rocm10_pjrt",
                "numpy",
            ]
        ]
        with mock.patch.object(e2e.importlib.metadata, "distributions", lambda: dists):
            stack = e2e.installed_jax_stack()

        self.assertEqual(
            set(stack), {"jax", "jaxlib", "jax-rocm10-plugin", "jax-rocm10-pjrt"}
        )


class WorkloadEnvTest(unittest.TestCase):
    def test_what_jax_lab_run_sh_reads(self):
        env = e2e.workload_env(
            {"PATH": "/usr/bin", "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
            e2e.Workload("flax_release", "resnet", 4),
            runner="linux-gfx942-8gpu-ossci-rocm",
            rocm_version="10.2.0a20261016",
            constraint_file=Path("/tmp/constraints.txt"),
            sdk_root=Path("/sdk"),
        )

        python = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.assertEqual(env["JAXLIB_VERSION"], "preinstalled")
        self.assertEqual(env["PIP_CONSTRAINT"], "/tmp/constraints.txt")
        self.assertEqual(env["COMBO"], f"4gpu-resnet-flax_release-py{python}")
        self.assertEqual(env["RUNNER"], "linux-gfx942-8gpu-ossci-rocm")
        self.assertEqual(env["ROCM_VERSION"], "10.2.0a20261016")
        self.assertEqual(env["ROCM_TAG"], "therock-10.2.0a20261016")
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertEqual(env["PATH"], "/sdk/bin:/usr/bin")
        self.assertEqual(env["LD_LIBRARY_PATH"], "/sdk/lib")
        self.assertEqual(env["PYTHON"], sys.executable)


class RunWorkloadTest(unittest.TestCase):
    def run_script(self, script: str, timeout_minutes: float) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            lab = Path(tmp)
            (lab / "targets" / "t").mkdir(parents=True)
            (lab / "targets" / "t" / "run.sh").write_text(script)
            return e2e.run_workload(
                lab, e2e.Workload("t", "w", 1), dict(os.environ), timeout_minutes
            )

    def test_exit_codes(self):
        self.assertEqual(self.run_script('[ "$1" = w ]\n', 1), "passed")
        self.assertEqual(self.run_script("exit 3\n", 1), "failed")

    def test_a_hang_is_killed_with_its_children(self):
        # 0.02 minutes is about a second.
        self.assertEqual(self.run_script("sleep 60 &\nwait\n", 0.02), "timed out")


class MainTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.lab = self.tmp / "jax-lab"
        (self.lab / "targets" / "maxtext_release").mkdir(parents=True)
        (self.lab / "targets" / "maxtext_release" / "run.sh").write_text("exit 0\n")
        self.list = self.tmp / "list.txt"
        self.list.write_text("maxtext_release llama3_1_8b 8\n")
        for name, value in [
            ("installed_jax_stack", lambda: STACK),
            ("rocm_sdk_root", lambda: Path("/sdk")),
        ]:
            patch = mock.patch.object(e2e, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def argv(self, *extra: str) -> list[str]:
        return [
            "--jax-lab-dir",
            os.fspath(self.lab),
            "--workload-list",
            os.fspath(self.list),
            "--results-dir",
            os.fspath(self.tmp / "results"),
            "--runner",
            "local",
            "--rocm-version",
            "10.2.0a20261016",
            *extra,
        ]

    def test_dry_run_prints_without_running(self):
        with mock.patch.object(e2e, "run_workload") as run:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(e2e.main(self.argv("--dry-run")), 0)

        run.assert_not_called()
        self.assertIn("targets/maxtext_release/run.sh llama3_1_8b", out.getvalue())
        self.assertEqual(
            (self.tmp / "results" / "constraints.txt").read_text(),
            e2e.constraints(STACK),
        )

    def test_a_target_jax_lab_lacks_fails_before_running_anything(self):
        self.list.write_text("maxtext_release llama3_1_8b 8\nno_such_target x 1\n")

        with mock.patch.object(e2e, "run_workload") as run:
            with self.assertRaises(FileNotFoundError):
                e2e.main(self.argv())
        run.assert_not_called()

    def test_every_workload_runs_and_one_failure_fails_the_run(self):
        self.list.write_text(
            "maxtext_release llama3_1_8b 8\nmaxtext_release gemma3_4b 8\n"
        )
        outcomes = iter(["failed", "passed"])

        def fake_run(jax_lab_dir, workload, env, timeout_minutes):
            result = jax_lab_dir / "targets" / workload.target / "run_artifacts"
            (result / workload.name).mkdir(parents=True)
            (result / workload.name / "result.json").write_text("{}")
            return next(outcomes)

        with mock.patch.object(e2e, "run_workload", fake_run):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(e2e.main(self.argv()), 1)

        self.assertIn("1 of 2 workloads failed", out.getvalue())
        for name in ["llama3_1_8b", "gemma3_4b"]:
            self.assertTrue(
                (
                    self.tmp / "results" / "maxtext_release" / name / "result.json"
                ).is_file()
            )

    def test_a_pass_without_a_result_is_a_failure(self):
        with mock.patch.object(e2e, "run_workload", lambda *a: "passed"):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(e2e.main(self.argv()), 1)

    def test_replacing_the_wheels_under_test_stops_the_run(self):
        stacks = iter([STACK, {**STACK, "jax": "0.12.0"}, {**STACK, "jax": "0.12.0"}])

        with mock.patch.object(e2e, "installed_jax_stack", lambda: next(stacks)):
            with mock.patch.object(e2e, "run_workload", lambda *a: "failed"):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "replaced"):
                        e2e.main(self.argv())


if __name__ == "__main__":
    unittest.main()
