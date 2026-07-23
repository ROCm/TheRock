# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

# pytest_runner.py lives under github_actions/test_executable_scripts/.
sys.path.insert(
    0,
    os.fspath(Path(__file__).parent.parent / "test_executable_scripts"),
)

import pytest_runner


class ExtractGpuArchTest(unittest.TestCase):
    def test_none_and_empty(self):
        self.assertIsNone(pytest_runner.extract_gpu_arch(None))
        self.assertIsNone(pytest_runner.extract_gpu_arch(""))

    def test_simple_arch(self):
        self.assertEqual(pytest_runner.extract_gpu_arch("gfx942"), "gfx942")

    def test_family_suffix(self):
        # First gfx token is returned, suffixes/families included in the token.
        self.assertEqual(pytest_runner.extract_gpu_arch("gfx94X-dcgpu"), "gfx94X")

    def test_first_of_multiple(self):
        self.assertEqual(pytest_runner.extract_gpu_arch("gfx1250,gfx942"), "gfx1250")

    def test_unparseable(self):
        self.assertIsNone(pytest_runner.extract_gpu_arch("no-arch-here"))


class BuildMarkerExpressionTest(unittest.TestCase):
    def test_empty_config_no_arch(self):
        self.assertEqual(pytest_runner.build_marker_expression({}, None), "")

    def test_include_only(self):
        cfg = {"pytest_markers": ["gfx1250"]}
        self.assertEqual(pytest_runner.build_marker_expression(cfg, None), "(gfx1250)")

    def test_include_multiple_ored(self):
        cfg = {"pytest_markers": ["gfx1250", "gfx12"]}
        self.assertEqual(
            pytest_runner.build_marker_expression(cfg, None),
            "(gfx1250 or gfx12)",
        )

    def test_exclude_only(self):
        cfg = {"exclude_markers": ["gpu"]}
        self.assertEqual(pytest_runner.build_marker_expression(cfg, None), "not gpu")

    def test_skip_marker_appended_for_arch(self):
        cfg = {"exclude_markers": ["gpu"]}
        self.assertEqual(
            pytest_runner.build_marker_expression(cfg, "gfx942"),
            "not gpu and not skip-gfx942",
        )

    def test_include_exclude_and_arch_combined(self):
        cfg = {"pytest_markers": ["gfx1250"], "exclude_markers": ["slow"]}
        self.assertEqual(
            pytest_runner.build_marker_expression(cfg, "gfx1250"),
            "(gfx1250) and not slow and not skip-gfx1250",
        )

    def test_none_values_treated_as_empty(self):
        cfg = {"pytest_markers": None, "exclude_markers": None}
        self.assertEqual(pytest_runner.build_marker_expression(cfg, None), "")


class ResolveComponentPathTest(unittest.TestCase):
    def test_known_component(self):
        rocm = Path("/opt/rocm")
        result = pytest_runner.resolve_component_path("tensilelite", rocm)
        self.assertEqual(result, (rocm / "share/hipblaslt/tensilelite").resolve())

    def test_unknown_component_exits(self):
        with self.assertRaises(SystemExit):
            pytest_runner.resolve_component_path("bogus", Path("/opt/rocm"))


class GetEnvIntOverrideTest(unittest.TestCase):
    def test_unset_returns_zero(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(pytest_runner.get_env_int_override("PYTEST_X"), 0)

    def test_blank_returns_zero(self):
        with mock.patch.dict(os.environ, {"PYTEST_X": ""}, clear=True):
            self.assertEqual(pytest_runner.get_env_int_override("PYTEST_X"), 0)

    def test_positive_value(self):
        with mock.patch.dict(os.environ, {"PYTEST_X": "42"}, clear=True):
            self.assertEqual(pytest_runner.get_env_int_override("PYTEST_X"), 42)

    def test_non_integer_returns_zero(self):
        with mock.patch.dict(os.environ, {"PYTEST_X": "abc"}, clear=True):
            self.assertEqual(pytest_runner.get_env_int_override("PYTEST_X"), 0)

    def test_negative_returns_zero(self):
        with mock.patch.dict(os.environ, {"PYTEST_X": "-5"}, clear=True):
            self.assertEqual(pytest_runner.get_env_int_override("PYTEST_X"), 0)


class LoadTestCategoriesYamlTest(unittest.TestCase):
    def setUp(self):
        self._tmp = (
            Path(os.environ.get("PYTEST_RUNNER_TMP", "/tmp")) / f"pr_yaml_{os.getpid()}"
        )
        self._tmp.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        for p in self._tmp.glob("*"):
            p.unlink()
        self._tmp.rmdir()

    def test_valid_yaml(self):
        path = self._tmp / "ok.yaml"
        path.write_text("test_categories:\n  quick:\n    test_paths: ['a']\n")
        cfg = pytest_runner.load_test_categories_yaml(path)
        self.assertIn("test_categories", cfg)
        self.assertEqual(cfg["test_categories"]["quick"]["test_paths"], ["a"])

    def test_missing_file_exits(self):
        with self.assertRaises(SystemExit):
            pytest_runner.load_test_categories_yaml(self._tmp / "nope.yaml")

    def test_invalid_yaml_exits(self):
        path = self._tmp / "bad.yaml"
        path.write_text("test_categories: [unclosed\n")
        with self.assertRaises(SystemExit):
            pytest_runner.load_test_categories_yaml(path)


class BuildEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self._orig_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._orig_env)

    def test_paths_and_ld_library_path(self):
        # Clear inherited values so assertions are deterministic.
        for var in ("PYTHONPATH", "LD_LIBRARY_PATH", "PATH"):
            os.environ.pop(var, None)
        rocm = Path("/opt/rocm")

        env = pytest_runner.build_environment(rocm, "tensilelite")

        component_root = (rocm / "share/hipblaslt/tensilelite").resolve()
        self.assertEqual(env["PYTHONPATH"], str(component_root))
        self.assertEqual(env["ROCM_PATH"], str(rocm))

        # Both lib/ and lib/llvm/lib/ must be present (libomp for the client).
        ld = env["LD_LIBRARY_PATH"].split(os.pathsep)
        self.assertIn(str(rocm / "lib"), ld)
        self.assertIn(str(rocm / "lib" / "llvm" / "lib"), ld)

        # bin/ and llvm/bin/ prepended to PATH.
        path_entries = env["PATH"].split(os.pathsep)
        self.assertEqual(path_entries[0], str(rocm / "bin"))
        self.assertEqual(path_entries[1], str(rocm / "lib" / "llvm" / "bin"))

    def test_prepends_to_existing_values(self):
        os.environ["PYTHONPATH"] = "/pre/existing"
        os.environ["LD_LIBRARY_PATH"] = "/pre/ld"
        rocm = Path("/opt/rocm")

        env = pytest_runner.build_environment(rocm, "tensilelite")

        self.assertTrue(env["PYTHONPATH"].endswith("/pre/existing"))
        self.assertIn("/pre/ld", env["LD_LIBRARY_PATH"].split(os.pathsep))


class RunPytestTest(unittest.TestCase):
    def setUp(self):
        # A cwd with one existing and one missing test path.
        self._cwd = Path("/tmp") / f"pr_run_{os.getpid()}"
        (self._cwd / "exists").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        (self._cwd / "exists").rmdir()
        self._cwd.rmdir()

    def _run(self, **kwargs):
        """Invoke run_pytest with subprocess.run mocked; return (rc, cmd)."""
        captured = {}
        returncode = kwargs.pop("returncode", 0)

        def fake_run(cmd, cwd, env, check):
            captured["cmd"] = cmd
            captured["cwd"] = cwd
            return mock.Mock(returncode=returncode)

        defaults = dict(
            test_paths=["exists"],
            marker_expr="",
            extra_args=[],
            junit_xml=None,
            timeout=None,
            num_workers=1,
            cwd=self._cwd,
            env={},
        )
        defaults.update(kwargs)
        # No optional plugins -> no --timeout / --numprocesses noise.
        with mock.patch.object(
            pytest_runner.subprocess, "run", fake_run
        ), mock.patch.object(pytest_runner, "find_spec", return_value=None):
            rc = pytest_runner.run_pytest(**defaults)
        return rc, captured["cmd"]

    def test_basic_command(self):
        rc, cmd = self._run()
        self.assertEqual(rc, 0)
        self.assertEqual(cmd[:4], ["pytest", "exists", "-v", "--color=yes"])

    def test_returncode_propagates(self):
        rc, _ = self._run(returncode=7)
        self.assertEqual(rc, 7)

    def test_missing_paths_filtered(self):
        _, cmd = self._run(test_paths=["exists", "missing"])
        self.assertIn("exists", cmd)
        self.assertNotIn("missing", cmd)

    def test_all_paths_missing_exits(self):
        with self.assertRaises(SystemExit):
            self._run(test_paths=["missing"])

    def test_marker_junit_and_extra_args_order(self):
        _, cmd = self._run(
            marker_expr="(gfx1250) and not skip-gfx1250",
            junit_xml="/out/tensilelite.xml",
            extra_args=["--prebuilt-client=/x", "-k", "foo"],
        )
        # marker, then junit, then extra args.
        self.assertIn("-m", cmd)
        self.assertEqual(cmd[cmd.index("-m") + 1], "(gfx1250) and not skip-gfx1250")
        self.assertIn("--junit-xml=/out/tensilelite.xml", cmd)
        self.assertIn("--prebuilt-client=/x", cmd)
        self.assertLess(
            cmd.index("--junit-xml=/out/tensilelite.xml"),
            cmd.index("--prebuilt-client=/x"),
        )

    def test_timeout_flag_present_when_plugin_available(self):
        captured = {}

        def fake_run(cmd, cwd, env, check):
            captured["cmd"] = cmd
            return mock.Mock(returncode=0)

        # pytest_timeout importable -> --timeout appended.
        with mock.patch.object(
            pytest_runner.subprocess, "run", fake_run
        ), mock.patch.object(pytest_runner, "find_spec", return_value=object()):
            pytest_runner.run_pytest(
                test_paths=["exists"],
                marker_expr="",
                extra_args=[],
                junit_xml=None,
                timeout=300,
                num_workers=16,
                cwd=self._cwd,
                env={},
            )
        self.assertIn("--timeout=300", captured["cmd"])
        self.assertIn("--numprocesses=16", captured["cmd"])

    def test_timeout_flag_absent_when_plugin_missing(self):
        _, cmd = self._run(timeout=300, num_workers=16)
        self.assertNotIn("--timeout=300", cmd)
        self.assertNotIn("--numprocesses=16", cmd)


if __name__ == "__main__":
    unittest.main()
