# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from packaging.version import Version

sys.path.insert(
    0,
    os.fspath(Path(__file__).parent.parent / "test_executable_scripts"),
)

import test_tensilelite as runner


class ReleaseWheelDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.wheels_dir = Path(self._temp.name)
        self.version = Version("5.0.0+rocm10.1.0")

    def tearDown(self):
        self._temp.cleanup()

    def _wheel(self, name: str, version: str = "5.0.0+rocm10.1.0") -> Path:
        path = self.wheels_dir / f"{name}-{version}-py3-none-any.whl"
        path.touch()
        return path

    def test_selects_exact_canonical_and_compatibility_wheels(self):
        canonical = self._wheel("tensilelite")
        compatibility = self._wheel("tensilelite_tensile_compat")

        wheels = runner.discover_release_wheels(self.wheels_dir, self.version)

        self.assertEqual(wheels.canonical, canonical)
        self.assertEqual(wheels.compatibility, compatibility)

    def test_rejects_missing_wheel(self):
        self._wheel("tensilelite")
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "No .*compat"):
            runner.discover_release_wheels(self.wheels_dir, self.version)

    def test_rejects_ambiguous_wheel(self):
        self._wheel("tensilelite")
        self._wheel("tensilelite", "5.0.1+rocm10.1.0")
        self._wheel("tensilelite_tensile_compat")
        with self.assertRaisesRegex(
            runner.TensileLiteRunnerError, "exactly one tensilelite wheel"
        ):
            runner.discover_release_wheels(self.wheels_dir, self.version)

    def test_rejects_wrong_project(self):
        self._wheel("tensilelite")
        self._wheel("tensilelite_tensile_compat")
        self._wheel("not_tensilelite")
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "Unexpected"):
            runner.discover_release_wheels(self.wheels_dir, self.version)

    def test_rejects_version_mismatch(self):
        self._wheel("tensilelite", "5.0.1+rocm10.1.0")
        self._wheel("tensilelite_tensile_compat")
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "does not match"):
            runner.discover_release_wheels(self.wheels_dir, self.version)


class ClientVersionTest(unittest.TestCase):
    def _result(self, returncode=0, stdout="5.0.0+rocm10.1.0\n", stderr=""):
        return subprocess.CompletedProcess(
            ["client", "--version"], returncode, stdout, stderr
        )

    def _query(self, result=None, side_effect=None):
        env = {"ROCM_PATH": "/rocm"}
        with mock.patch.object(
            runner.subprocess, "run", return_value=result, side_effect=side_effect
        ) as run:
            version = runner.query_client_version(Path("/rocm/client"), env)
        self.assertEqual(run.call_args.args[0], ["/rocm/client", "--version"])
        self.assertEqual(run.call_args.kwargs["env"], env)
        self.assertEqual(run.call_args.kwargs["timeout"], 5)
        self.assertTrue(run.call_args.kwargs["capture_output"])
        self.assertTrue(run.call_args.kwargs["text"])
        self.assertFalse(run.call_args.kwargs["check"])
        return version

    def test_valid_version(self):
        self.assertEqual(self._query(self._result()), Version("5.0.0+rocm10.1.0"))

    def test_launch_failure(self):
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "file not found"):
            self._query(side_effect=FileNotFoundError())

    def test_loader_failure(self):
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "load or launch"):
            self._query(side_effect=OSError("missing shared library"))

    def test_dynamic_loader_diagnostic(self):
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "Failed to load"):
            self._query(
                self._result(
                    returncode=127,
                    stderr="error while loading shared libraries: libomp.so\n",
                )
            )

    def test_timeout(self):
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "timed out"):
            self._query(
                side_effect=subprocess.TimeoutExpired(["client", "--version"], 5)
            )

    def test_signal(self):
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "signal 9"):
            self._query(self._result(returncode=-9))

    def test_nonzero_exit(self):
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "status 7"):
            self._query(self._result(returncode=7))

    def test_stderr(self):
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "stderr"):
            self._query(self._result(stderr="warning\n"))

    def test_missing_output(self):
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "no version"):
            self._query(self._result(stdout=""))

    def test_extra_output(self):
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "exactly one"):
            self._query(self._result(stdout="5.0.0+rocm10.1.0\nextra\n"))

    def test_malformed_output(self):
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "malformed"):
            self._query(self._result(stdout="not-a-version\n"))


class PhaseOrchestrationTest(unittest.TestCase):
    def setUp(self):
        self.rocm_path = Path("/artifact/rocm")
        self.env = {"ROCM_PATH": str(self.rocm_path)}
        self.wheels = runner.ReleaseWheels(
            canonical=Path("/wheels/tensilelite.whl"),
            compatibility=Path("/wheels/compat.whl"),
        )

    def _run(self, install_statuses=(0, 0), phase_statuses=(0, 0)):
        with mock.patch.object(
            runner, "query_client_version", return_value=Version("5.0.0")
        ) as query, mock.patch.object(
            runner, "discover_release_wheels", return_value=self.wheels
        ), mock.patch.object(
            runner, "install_wheel", side_effect=install_statuses
        ) as install, mock.patch.object(
            runner.pytest_runner, "run_phase", side_effect=phase_statuses
        ) as phase:
            status = runner.run_test_phases(
                self.rocm_path, "quick", "gfx942", self.env
            )
        return status, query, install, phase

    def test_installs_and_runs_both_phases_in_order(self):
        status, query, install, phase = self._run()

        self.assertEqual(status, 0)
        self.assertIs(query.call_args.args[1], self.env)
        self.assertEqual(
            install.call_args_list,
            [
                mock.call(self.wheels.canonical, self.env),
                mock.call(self.wheels.compatibility, self.env),
            ],
        )
        self.assertEqual(phase.call_count, 2)
        self.assertEqual(phase.call_args_list[0].args[-1], self.env)
        self.assertEqual(phase.call_args_list[1].args[-1], self.env)
        self.assertIsNone(phase.call_args_list[1].args[1])
        self.assertEqual(
            phase.call_args_list[1].kwargs,
            {
                "test_paths_override": ["compat/tests"],
                "marker_expression_override": "",
                "pytest_args_override": ["--run-compat"],
            },
        )

    def test_phase_operations_are_strictly_ordered(self):
        events = []

        def install(wheel, env):
            self.assertIs(env, self.env)
            events.append(("install", wheel))
            return 0

        def phase(*args, **kwargs):
            self.assertIs(args[-1], self.env)
            events.append(("phase", kwargs.get("test_paths_override")))
            return 0

        with mock.patch.object(
            runner, "query_client_version", return_value=Version("5.0.0")
        ), mock.patch.object(
            runner, "discover_release_wheels", return_value=self.wheels
        ), mock.patch.object(
            runner, "install_wheel", side_effect=install
        ), mock.patch.object(
            runner.pytest_runner, "run_phase", side_effect=phase
        ):
            status = runner.run_test_phases(
                self.rocm_path, "quick", "gfx942", self.env
            )

        self.assertEqual(status, 0)
        self.assertEqual(
            events,
            [
                ("install", self.wheels.canonical),
                ("phase", None),
                ("install", self.wheels.compatibility),
                ("phase", ["compat/tests"]),
            ],
        )

    def test_canonical_install_failure_stops_before_pytest(self):
        status, _query, install, phase = self._run(
            install_statuses=(3,), phase_statuses=()
        )
        self.assertEqual(status, 3)
        self.assertEqual(install.call_count, 1)
        phase.assert_not_called()

    def test_canonical_failure_stops_before_compatibility_install(self):
        status, _query, install, phase = self._run(
            install_statuses=(0,), phase_statuses=(4,)
        )
        self.assertEqual(status, 4)
        self.assertEqual(install.call_count, 1)
        self.assertEqual(phase.call_count, 1)

    def test_compatibility_install_failure_propagates(self):
        status, _query, install, phase = self._run(
            install_statuses=(0, 5), phase_statuses=(0,)
        )
        self.assertEqual(status, 5)
        self.assertEqual(install.call_count, 2)
        self.assertEqual(phase.call_count, 1)

    def test_compatibility_failure_propagates(self):
        status, _query, install, phase = self._run(phase_statuses=(0, 6))
        self.assertEqual(status, 6)
        self.assertEqual(install.call_count, 2)
        self.assertEqual(phase.call_count, 2)


class PipInstallTest(unittest.TestCase):
    def test_uses_complete_active_interpreter_commands_and_environment(self):
        env = {"ROCM_PATH": "/artifact/rocm"}
        for wheel in (
            Path("/artifact/tensilelite.whl"),
            Path("/artifact/tensilelite_tensile_compat.whl"),
        ):
            with self.subTest(wheel=wheel), mock.patch.object(
                runner.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run:
                status = runner.install_wheel(wheel, env)

            self.assertEqual(status, 0)
            self.assertEqual(
                run.call_args.args[0],
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--force-reinstall",
                    "--no-deps",
                    str(wheel),
                ],
            )
            self.assertIs(run.call_args.kwargs["env"], env)
