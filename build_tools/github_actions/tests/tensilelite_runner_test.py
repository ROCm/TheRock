# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(
    0,
    os.fspath(Path(__file__).parent.parent / "test_executable_scripts"),
)

import test_tensilelite as runner


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        self.tmp = Path(temporary_dir.name)


class LayoutCheckTest(TempDirTestCase):
    def test_requires_wheels(self):
        with self.assertRaisesRegex(
            runner.TensileLiteRunnerError, "No TensileLite wheels"
        ):
            runner.check_artifact_layout(self.tmp)

    def test_rejects_unpacked_package(self):
        (self.tmp / "wheels").mkdir()
        (self.tmp / "tensilelite").mkdir()
        (self.tmp / "tensilelite/__init__.py").touch()
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "unpacked"):
            runner.check_artifact_layout(self.tmp)

    def test_accepts_installed_wheel_layout(self):
        (self.tmp / "wheels").mkdir()
        # Installed tests may live under tensilelite/Tests without a package.
        (self.tmp / "tensilelite/Tests").mkdir(parents=True)
        runner.check_artifact_layout(self.tmp)


class FindWheelTest(TempDirTestCase):
    def _wheel(self, name):
        path = self.tmp / name
        path.touch()
        return path

    def test_selects_by_project_name_for_every_label(self):
        # TheRock does not parse the label (rocm-libraries owns its format).
        for version in (
            "5.0.0+rocm7.12.0",
            "5.0.0+devrocm7.12.0.dev0.abc123",
            "5.0.0+g0123456789abcdef0123456789abcdef01234567",
            "5.0.0+g0123456789abcdef0123456789abcdef01234567.dirty",
        ):
            with self.subTest(version=version):
                for old in self.tmp.glob("*.whl"):
                    old.unlink()
                wheel = self._wheel(f"tensilelite-{version}-py3-none-any.whl")
                self.assertEqual(runner.find_tensilelite_wheel(self.tmp), wheel)

    def test_rejects_missing_wheel(self):
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "exactly one"):
            runner.find_tensilelite_wheel(self.tmp)

    def test_rejects_multiple_wheels(self):
        self._wheel("tensilelite-5.0.0+rocm7.12.0-py3-none-any.whl")
        self._wheel("tensilelite-5.0.1+rocm7.12.0-py3-none-any.whl")
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "exactly one"):
            runner.find_tensilelite_wheel(self.tmp)

    def test_rejects_other_projects(self):
        self._wheel("tensilelite-5.0.0+rocm7.12.0-py3-none-any.whl")
        self._wheel("tensilelite_tensile_compat-5.0.0+rocm7.12.0-py3-none-any.whl")
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "Unexpected"):
            runner.find_tensilelite_wheel(self.tmp)

    def test_rejects_malformed_filename(self):
        self._wheel("tensilelite.whl")
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "Malformed"):
            runner.find_tensilelite_wheel(self.tmp)


class ShimTest(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.root = self.tmp / "artifact"
        self.shim = self.tmp / "shim"
        self.shim.mkdir()
        for name in ("rocisa", "stinkytofu", "tensilelite", "other"):
            (self.root / name).mkdir(parents=True)

    def test_exposes_only_binding_packages_and_guard(self):
        runner.create_shim(self.root, self.shim)

        self.assertEqual(
            sorted(path.name for path in self.shim.iterdir()),
            sorted(["rocisa", "stinkytofu", f"{runner.GUARD_MODULE}.py"]),
        )
        self.assertEqual(
            (self.shim / "rocisa").resolve(), (self.root / "rocisa").resolve()
        )

    def test_requires_rocisa(self):
        (self.root / "rocisa").rmdir()
        with self.assertRaisesRegex(runner.TensileLiteRunnerError, "rocisa"):
            runner.create_shim(self.root, self.shim)

    def test_environment_resets_pythonpath_and_loads_guard(self):
        env = runner.wheel_test_environment(
            {
                "PYTHONPATH": "/artifact/root:/checkout",
                "PYTEST_ADDOPTS": "-x",
                "A": "1",
            },
            self.shim,
        )
        self.assertEqual(env["PYTHONPATH"], str(self.shim))
        self.assertEqual(env["PYTEST_ADDOPTS"], f"-p {runner.GUARD_MODULE} -x")
        self.assertEqual(env["A"], "1")


class ImportGuardTest(TempDirTestCase):
    def _guard(self):
        module = types.ModuleType(runner.GUARD_MODULE)
        exec(runner.GUARD_SOURCE, module.__dict__)
        return module

    def _fake_tensilelite(self, parent):
        package = parent / "tensilelite"
        package.mkdir(parents=True)
        (package / "__init__.py").touch()
        return package

    def test_accepts_package_under_an_installed_root(self):
        guard = self._guard()
        package = self._fake_tensilelite(self.tmp / "site-packages")
        with (
            mock.patch.dict(
                sys.modules,
                {"tensilelite": mock.Mock(__file__=str(package / "__init__.py"))},
            ),
            mock.patch.object(
                guard,
                "installed_roots",
                return_value=[(self.tmp / "site-packages").resolve()],
            ),
            mock.patch.object(
                guard.importlib.metadata, "version", return_value="5.0.0+g1"
            ),
        ):
            self.assertEqual(
                guard.check(), ((package / "__init__.py").resolve(), "5.0.0+g1")
            )

    def test_rejects_package_outside_installed_roots(self):
        guard = self._guard()
        package = self._fake_tensilelite(self.tmp / "artifact")
        with (
            mock.patch.dict(
                sys.modules,
                {"tensilelite": mock.Mock(__file__=str(package / "__init__.py"))},
            ),
            mock.patch.object(
                guard,
                "installed_roots",
                return_value=[(self.tmp / "site-packages").resolve()],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "not from the installed wheel"):
                guard.check()

    def test_pytest_fails_when_artifact_copy_shadows_wheel(self):
        # pytest's default import mode puts the artifact root on sys.path for
        # tests inside a package; the guard must fail those tests.
        artifact = self.tmp / "artifact"
        package = self._fake_tensilelite(artifact)
        (package / "Tests").mkdir()
        (package / "Tests/__init__.py").touch()
        (package / "Tests/test_x.py").write_text("def test_x():\n    pass\n")
        shim = self.tmp / "shim"
        shim.mkdir()
        (shim / f"{runner.GUARD_MODULE}.py").write_text(runner.GUARD_SOURCE)
        env = runner.wheel_test_environment(dict(os.environ), shim)
        env.pop("PYTEST_ADDOPTS")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                runner.GUARD_MODULE,
                "-p",
                "no:cacheprovider",
                "tensilelite/Tests",
            ],
            cwd=artifact,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("not from the installed wheel", result.stdout)


class MainTest(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.rocm = self.tmp / "rocm"
        (self.rocm / "bin").mkdir(parents=True)
        self.component_root = self.rocm / "share/hipblaslt/tensilelite"
        self.component_root.mkdir(parents=True)
        patcher = mock.patch.dict(
            os.environ,
            {
                "TEST_COMPONENT": "tensilelite",
                "THEROCK_BIN_DIR": str(self.rocm / "bin"),
                "TEST_TYPE": "standard",
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("TEST_CATEGORY", None)

    def test_rejects_unknown_component(self):
        os.environ["TEST_COMPONENT"] = "rocblas"
        self.assertEqual(runner.main(), 1)

    def test_missing_wheels_is_an_error(self):
        with mock.patch.object(runner.pytest_runner, "main") as existing:
            self.assertEqual(runner.main(), 1)
        existing.assert_not_called()

    def test_common_job_uses_same_runner(self):
        os.environ["TEST_COMPONENT"] = "tensilelite-common"
        os.environ["TEST_CATEGORY"] = "hw-common"
        (self.component_root / "wheels").mkdir()
        with mock.patch.object(runner, "run_installed_wheel", return_value=0) as run:
            self.assertEqual(runner.main(), 0)
        self.assertEqual(run.call_args.args[:2], ("tensilelite-common", "hw-common"))

    def test_installed_wheel_path(self):
        (self.component_root / "wheels").mkdir()
        wheel = self.component_root / "wheels/tensilelite-5.0.0+g1-py3-none-any.whl"
        wheel.touch()
        (self.component_root / "rocisa").mkdir()
        calls = []

        def fake_run_phase(component, test_type, families, targets, rocm_path, env):
            shim = Path(env["PYTHONPATH"])
            calls.append((component, test_type, sorted(p.name for p in shim.iterdir())))
            return 3

        with (
            mock.patch.object(runner, "install_wheel") as install,
            mock.patch.object(
                runner, "check_installed_import", return_value="5.0.0+g1"
            ) as imported,
            mock.patch.object(runner, "check_package_help") as package_help,
            mock.patch.object(
                runner.pytest_runner, "run_phase", side_effect=fake_run_phase
            ),
        ):
            self.assertEqual(runner.main(), 3)

        install.assert_called_once_with(wheel)
        imported.assert_called_once()
        package_help.assert_called_once()
        self.assertEqual(
            calls,
            [
                (
                    "tensilelite",
                    "standard",
                    sorted(["rocisa", f"{runner.GUARD_MODULE}.py"]),
                )
            ],
        )

    def test_setup_failure_returns_error(self):
        (self.component_root / "wheels").mkdir()
        self.assertEqual(runner.main(), 1)


class ProvenanceTest(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.wheel = Path("tensilelite-5.0.0+devrocm7.12.0.dev0.abc-py3-none-any.whl")

    def test_missing_values_are_recorded_as_absent(self):
        provenance = runner.collect_provenance(
            self.tmp, self.wheel, "5.0.0+devrocm7.12.0.dev0.abc", {}
        )

        self.assertEqual(provenance["tensilelite.wheel"], self.wheel.name)
        self.assertEqual(
            provenance["tensilelite.version"], "5.0.0+devrocm7.12.0.dev0.abc"
        )
        self.assertEqual(provenance["tensilelite_client.version"], "absent")
        self.assertEqual(provenance["tensilelite_client.source_revision"], "absent")
        self.assertEqual(provenance["therock.rocm_package_version"], "absent")
        self.assertEqual(provenance["therock.the_rock_commit"], "absent")
        self.assertTrue(provenance["python.version"])

    @unittest.skipIf(os.name == "nt", "uses a shell-script client")
    def test_records_client_and_manifest_verbatim(self):
        client = self.tmp / "libexec/hipblaslt/tensilelite/tensilelite-client"
        client.parent.mkdir(parents=True)
        client.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = --version ]; then echo "client 5.0.0"; exit 0; fi\n'
            "exit 2\n"
        )
        client.chmod(0o755)
        manifest = self.tmp / "share/therock/therock_manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            '{"rocm_package_version": "7.12.0.dev0+abc", "the_rock_commit": "f00"}'
        )

        provenance = runner.collect_provenance(
            self.tmp, self.wheel, "v", dict(os.environ)
        )

        self.assertEqual(provenance["tensilelite_client.version"], "client 5.0.0")
        self.assertEqual(
            provenance["tensilelite_client.source_revision"], "unavailable (exit 2)"
        )
        self.assertEqual(provenance["therock.rocm_package_version"], "7.12.0.dev0+abc")
        self.assertEqual(provenance["therock.the_rock_commit"], "f00")

    def test_writes_summary_and_junit_properties(self):
        junit = self.tmp / "tensilelite.xml"
        junit.write_text(
            '<?xml version="1.0" encoding="utf-8"?>'
            '<testsuites><testsuite name="pytest"><testcase name="t"/></testsuite></testsuites>'
        )
        summary = self.tmp / "summary.md"
        with mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary)}):
            runner.write_provenance({"tensilelite.version": "5.0.0+g1"}, str(junit))

        self.assertIn("| tensilelite.version | `5.0.0+g1` |", summary.read_text())
        root = runner.ElementTree.parse(junit).getroot()
        prop = root.find("testsuite/properties/property")
        self.assertEqual(
            prop.attrib, {"name": "tensilelite.version", "value": "5.0.0+g1"}
        )

    def test_missing_junit_is_ignored(self):
        runner.write_provenance({"a": "b"}, str(self.tmp / "missing.xml"))


if __name__ == "__main__":
    unittest.main()
