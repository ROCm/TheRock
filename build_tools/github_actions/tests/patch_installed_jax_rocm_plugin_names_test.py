"""
Unit tests for external-builds/jax/patch_installed_jax_rocm_plugin_names.py
"""

import importlib
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
JAX_DIR = THIS_DIR.parents[2] / "external-builds" / "jax"

sys.path.insert(0, os.fspath(JAX_DIR))

import patch_installed_jax_rocm_plugin_names as patcher

# jaxlib/plugin_support.py as released in jaxlib 0.10.0 through 0.11.0, i.e.
# before jax-ml/jax#39634.
JAXLIB_RELEASED = textwrap.dedent(
    '''\
    from collections.abc import Sequence
    import importlib
    import re

    _PLUGIN_MODULE_NAMES = {
        "cuda": ["jax_cuda13_plugin", "jax_cuda12_plugin"],
        "rocm": ["jax_rocm7_plugin", "jax_rocm60_plugin"],
        "oneapi": ["jax_oneapi_plugin"],
    }


    def import_from_plugin(plugin_name: str, submodule_name: str):
      """Docstring mentioning "rocm": ["..."] should not confuse the patch."""
      return _PLUGIN_MODULE_NAMES[plugin_name]
    '''
)

# The same file once a jaxlib carrying jax-ml/jax#39634 ships.
JAXLIB_FIXED = textwrap.dedent(
    '''\
    _PLUGIN_MODULE_NAMES = {
        "cuda": ["jax_cuda13_plugin", "jax_cuda12_plugin"],
        "rocm": ["jax_rocm10_plugin", "jax_rocm7_plugin", "jax_rocm60_plugin"],
        "oneapi": ["jax_oneapi_plugin"],
    }


    @functools.cache
    def _discovered_rocm_plugin_module_names() -> tuple[str, ...]:
      """Finds installed ROCm plugin packages not listed above."""
      return ()
    '''
)

# jax_plugins/rocm/__init__.py before the ROCm/jax fix, installed as
# jax_plugins/xla_rocm<major>/__init__.py.
SHIM_RELEASED = textwrap.dedent(
    """\
    import importlib
    import jax._src.xla_bridge as xb

    # rocm_plugin_extension locates inside jaxlib. `jaxlib` is for testing without
    # preinstalled jax rocm plugin packages.
    for pkg_name in ['jax_rocm7_plugin', 'jax_rocm60_plugin', 'jaxlib.rocm']:
      try:
        rocm_plugin_extension = importlib.import_module(
            f'{pkg_name}.rocm_plugin_extension'
        )
      except ImportError:
        rocm_plugin_extension = None
      else:
        break
    """
)

# The same file with the ROCm/jax fix, which every rocm-jaxlib-v0.10.x and
# v0.11.0 branch now carries.
SHIM_FIXED = textwrap.dedent(
    """\
    import importlib

    _pkg_names = [
        'jax_rocm10_plugin', 'jax_rocm7_plugin', 'jax_rocm60_plugin', 'jaxlib.rocm'
    ]

    _rocm_major = (__package__ or '').rpartition('xla_rocm')[2]
    if _rocm_major.isdigit() and f'jax_rocm{_rocm_major}_plugin' not in _pkg_names:
      _pkg_names.insert(0, f'jax_rocm{_rocm_major}_plugin')

    for pkg_name in _pkg_names:
      try:
        rocm_plugin_extension = importlib.import_module(
            f'{pkg_name}.rocm_plugin_extension'
        )
      except ImportError:
        rocm_plugin_extension = None
      else:
        break
    """
)


class RewriteJaxlibTest(unittest.TestCase):
    def test_plugin_is_inserted_first(self) -> None:
        result = patcher.rewrite_jaxlib_plugin_names(
            JAXLIB_RELEASED, "jax_rocm10_plugin"
        )

        self.assertIn(
            '"rocm": ["jax_rocm10_plugin", "jax_rocm7_plugin", "jax_rocm60_plugin"]',
            result.text,
        )
        self.assertIn("jax_rocm10_plugin", result.detail)

    def test_other_platforms_are_untouched(self) -> None:
        result = patcher.rewrite_jaxlib_plugin_names(
            JAXLIB_RELEASED, "jax_rocm10_plugin"
        )

        self.assertIn('"cuda": ["jax_cuda13_plugin", "jax_cuda12_plugin"]', result.text)
        self.assertIn('"oneapi": ["jax_oneapi_plugin"]', result.text)

    def test_result_is_valid_python(self) -> None:
        result = patcher.rewrite_jaxlib_plugin_names(
            JAXLIB_RELEASED, "jax_rocm10_plugin"
        )

        compile(result.text, "plugin_support.py", "exec")

    def test_wrapped_list_is_rewritten(self) -> None:
        wrapped = textwrap.dedent(
            """\
            _PLUGIN_MODULE_NAMES = {
                "rocm": [
                    "jax_rocm7_plugin",
                    "jax_rocm60_plugin",
                ],
            }
            """
        )

        result = patcher.rewrite_jaxlib_plugin_names(wrapped, "jax_rocm10_plugin")

        self.assertIn('"jax_rocm10_plugin", "jax_rocm7_plugin"', result.text)
        compile(result.text, "plugin_support.py", "exec")

    def test_rewrite_is_idempotent(self) -> None:
        once = patcher.rewrite_jaxlib_plugin_names(JAXLIB_RELEASED, "jax_rocm10_plugin")
        twice = patcher.rewrite_jaxlib_plugin_names(once.text, "jax_rocm10_plugin")

        self.assertIsNone(twice.text)
        self.assertIn("already lists jax_rocm10_plugin", twice.detail)

    def test_fixed_jaxlib_is_left_alone(self) -> None:
        result = patcher.rewrite_jaxlib_plugin_names(JAXLIB_FIXED, "jax_rocm99_plugin")

        self.assertIsNone(result.text)
        self.assertIn("discovers", result.detail)

    def test_unexpected_layout_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "plugin list"):
            patcher.rewrite_jaxlib_plugin_names(
                "_PLUGIN_MODULE_NAMES = load_plugin_names()\n", "jax_rocm10_plugin"
            )


class RewritePjrtShimTest(unittest.TestCase):
    def test_plugin_is_inserted_first(self) -> None:
        result = patcher.rewrite_pjrt_shim_plugin_names(
            SHIM_RELEASED, "jax_rocm10_plugin"
        )

        self.assertIn(
            "for pkg_name in ['jax_rocm10_plugin', 'jax_rocm7_plugin',"
            " 'jax_rocm60_plugin', 'jaxlib.rocm']:",
            result.text,
        )
        compile(result.text, "__init__.py", "exec")

    def test_rewrite_is_idempotent(self) -> None:
        once = patcher.rewrite_pjrt_shim_plugin_names(
            SHIM_RELEASED, "jax_rocm10_plugin"
        )
        twice = patcher.rewrite_pjrt_shim_plugin_names(once.text, "jax_rocm10_plugin")

        self.assertIsNone(twice.text)
        self.assertIn("already lists jax_rocm10_plugin", twice.detail)

    def test_fixed_shim_is_left_alone(self) -> None:
        result = patcher.rewrite_pjrt_shim_plugin_names(SHIM_FIXED, "jax_rocm99_plugin")

        self.assertIsNone(result.text)
        self.assertIn("derives the plugin name", result.detail)

    def test_unexpected_layout_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "plugin list"):
            patcher.rewrite_pjrt_shim_plugin_names(
                "rocm_plugin_extension = None\n", "jax_rocm10_plugin"
            )


class MainTest(unittest.TestCase):
    """Drives main() against a fake install on sys.path."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.site_packages = Path(self._tmp.name)

        self._saved_path = list(sys.path)
        self._saved_modules = {
            name: module for name, module in sys.modules.items() if self._is_faked(name)
        }
        self.addCleanup(self._restore_import_state)

        # A real jaxlib or plugin in this environment must not shadow the fake
        # one, and neither must a module cached by an earlier test.
        for name in self._saved_modules:
            del sys.modules[name]
        sys.path.insert(0, os.fspath(self.site_packages))
        importlib.invalidate_caches()

        self.jaxlib_path = self._write_package(
            "jaxlib", "plugin_support.py", JAXLIB_RELEASED
        )
        self.shim_path = self._write_package(
            "jax_plugins/xla_rocm10", "__init__.py", SHIM_RELEASED
        )
        # The plugin wheel itself, which main() requires to be installed.
        self._write_package("jax_rocm10_plugin", "__init__.py", "")

    @staticmethod
    def _is_faked(module_name: str) -> bool:
        return module_name.split(".")[0] in (
            "jaxlib",
            "jax_plugins",
            "jax_rocm10_plugin",
        )

    def _restore_import_state(self) -> None:
        sys.path[:] = self._saved_path
        for name in [name for name in sys.modules if self._is_faked(name)]:
            del sys.modules[name]
        sys.modules.update(self._saved_modules)
        importlib.invalidate_caches()

    def _write_package(self, package: str, file_name: str, text: str) -> Path:
        directory = self.site_packages / package
        directory.mkdir(parents=True, exist_ok=True)
        init = directory / "__init__.py"
        if not init.exists():
            init.write_text("")
        path = directory / file_name
        path.write_text(text)
        return path

    def test_patches_both_files(self) -> None:
        patcher.main(["--plugin-package", "jax_rocm10_plugin"])

        self.assertIn("jax_rocm10_plugin", self.jaxlib_path.read_text())
        self.assertIn(
            "for pkg_name in ['jax_rocm10_plugin',", self.shim_path.read_text()
        )

    def test_second_run_changes_nothing(self) -> None:
        patcher.main(["--plugin-package", "jax_rocm10_plugin"])
        after_first = (self.jaxlib_path.read_text(), self.shim_path.read_text())

        patcher.main(["--plugin-package", "jax_rocm10_plugin"])

        self.assertEqual(
            after_first, (self.jaxlib_path.read_text(), self.shim_path.read_text())
        )

    def test_skip_pjrt_shim_leaves_shim_installed(self) -> None:
        patcher.main(["--plugin-package", "jax_rocm10_plugin", "--skip-pjrt-shim"])

        self.assertIn("jax_rocm10_plugin", self.jaxlib_path.read_text())
        self.assertEqual(SHIM_RELEASED, self.shim_path.read_text())

    def test_unreadable_shim_leaves_jaxlib_unpatched(self) -> None:
        self.shim_path.write_text("rocm_plugin_extension = None\n")

        with self.assertRaisesRegex(ValueError, "plugin list"):
            patcher.main(["--plugin-package", "jax_rocm10_plugin"])

        self.assertEqual(JAXLIB_RELEASED, self.jaxlib_path.read_text())

    def test_unreadable_jaxlib_leaves_shim_unpatched(self) -> None:
        self.jaxlib_path.write_text("_PLUGIN_MODULE_NAMES = load_plugin_names()\n")

        with self.assertRaisesRegex(ValueError, "plugin list"):
            patcher.main(["--plugin-package", "jax_rocm10_plugin"])

        self.assertEqual(SHIM_RELEASED, self.shim_path.read_text())

    def test_missing_plugin_package_is_an_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            patcher.main(["--plugin-package", "jax_rocm99_plugin"])

        self.assertEqual(JAXLIB_RELEASED, self.jaxlib_path.read_text())

    def test_invalid_plugin_package_names_are_rejected(self) -> None:
        for name in ["jax_rocm_plugin", "jax_rocm10", "rocm10_plugin", ""]:
            with self.subTest(name=name):
                with self.assertRaises(SystemExit):
                    patcher.main(["--plugin-package", name])

        self.assertEqual(JAXLIB_RELEASED, self.jaxlib_path.read_text())


if __name__ == "__main__":
    unittest.main()
