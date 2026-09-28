# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the rocm-sdk-devel initialization state machine."""

import importlib.util
import io
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest import mock


DEVEL_MODULE_PATH = (
    Path(__file__).parent.parent
    / "packaging"
    / "python"
    / "templates"
    / "rocm"
    / "src"
    / "rocm_sdk"
    / "_devel.py"
)
CORE_CLI_MODULE_PATH = (
    Path(__file__).parent.parent
    / "packaging"
    / "python"
    / "templates"
    / "rocm-sdk-core"
    / "src"
    / "rocm_sdk_core"
    / "_cli.py"
)


class DevelInitializationTest(unittest.TestCase):
    def setUp(self):
        self.temp_context = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_context.name)
        self.site_lib_path = self.temp_dir / "site-packages"
        self.site_lib_path.mkdir()
        self.pure_package_path = self.site_lib_path / "rocm_sdk_devel"
        self.pure_package_path.mkdir()
        pure_init_path = self.pure_package_path / "__init__.py"
        pure_init_path.touch()
        pure_package = types.ModuleType("rocm_sdk_devel")
        pure_package.__file__ = os.fspath(pure_init_path)
        sys.modules[pure_package.__name__] = pure_package
        self.addCleanup(sys.modules.pop, pure_package.__name__, None)
        self.record_path = (
            self.site_lib_path / "rocm_sdk_devel-0.0.1.dist-info" / "RECORD"
        )
        self.record_path.parent.mkdir()
        (self.record_path.parent / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: rocm-sdk-devel\nVersion: 0.0.1\n"
        )
        self.record_path.write_text("rocm_sdk_devel/__init__.py,,\n")
        self.devel = self._load_devel_module()

    def tearDown(self):
        self.temp_context.cleanup()

    def _load_devel_module(self):
        package_name = f"_test_rocm_sdk_{id(self)}"
        package_module = types.ModuleType(package_name)
        package_module.__path__ = []
        dist_info_module = types.ModuleType(f"{package_name}._dist_info")
        package_entry = types.SimpleNamespace(
            get_py_package_name=lambda: "_rocm_sdk_devel"
        )
        dist_info_module.ALL_PACKAGES = {"devel": package_entry}
        dist_info_module.__version__ = "0.0.1"
        sys.modules[package_name] = package_module
        sys.modules[dist_info_module.__name__] = dist_info_module
        self.addCleanup(sys.modules.pop, dist_info_module.__name__, None)
        self.addCleanup(sys.modules.pop, package_name, None)

        module_name = f"{package_name}._devel"
        spec = importlib.util.spec_from_file_location(module_name, DEVEL_MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)
        spec.loader.exec_module(module)
        return module

    def test_get_devel_root_uses_manifest_as_completion_sentinel(self):
        devel_root = self.site_lib_path / "_rocm_sdk_devel"
        devel_root.mkdir()
        (devel_root / "__init__.py").touch()
        tar_path = self.pure_package_path / "_devel.tar"
        tar_path.touch()

        with mock.patch.object(self.devel, "_expand_devel_contents") as expand:
            self.assertEqual(self.devel.get_devel_root(), devel_root)
            expand.assert_called_once_with(
                self.pure_package_path,
                self.site_lib_path,
            )

            tar_path.unlink()
            self.assertEqual(self.devel.get_devel_root(), devel_root)
            self.assertEqual(expand.call_count, 1)

    def _write_manifest(self, members: list[tarfile.TarInfo], data: bytes = b""):
        tar_path = self.pure_package_path / "_devel.tar"
        with tarfile.open(tar_path, mode="w") as tf:
            for member in members:
                fileobj = None
                if member.isfile():
                    member.size = len(data)
                    fileobj = io.BytesIO(data)
                tf.addfile(member, fileobj=fileobj)
        return tar_path

    @staticmethod
    def _directory(name: str) -> tarfile.TarInfo:
        member = tarfile.TarInfo(name)
        member.type = tarfile.DIRTYPE
        member.mode = 0o755
        return member

    @staticmethod
    def _symlink(name: str, target: str) -> tarfile.TarInfo:
        member = tarfile.TarInfo(name)
        member.type = tarfile.SYMTYPE
        member.linkname = target
        return member

    def test_link_manifest_preserves_wheel_files_and_is_idempotent(self):
        devel_root = self.site_lib_path / "_rocm_sdk_devel"
        direct_file = devel_root / "include" / "direct.h"
        direct_file.parent.mkdir(parents=True)
        direct_file.write_text("wheel-owned")
        direct_directory = devel_root / "lib" / "llvm"
        direct_directory.mkdir(parents=True)

        runtime_file = self.site_lib_path / "_rocm_sdk_core" / "lib" / "shared.txt"
        runtime_file.parent.mkdir(parents=True)
        runtime_file.write_text("runtime-owned")

        stale_alias = devel_root / "lib" / "shared.txt"
        stale_alias.write_text("stale")
        tar_path = self._write_manifest(
            [
                self._directory("_rocm_sdk_devel"),
                self._directory("_rocm_sdk_devel/include"),
                self._directory("_rocm_sdk_devel/lib"),
                self._symlink(
                    "_rocm_sdk_devel/lib/shared.txt",
                    "../../_rocm_sdk_core/lib/shared.txt",
                ),
                self._symlink("_rocm_sdk_devel/llvm", "lib/llvm"),
            ]
        )

        self.devel._lock_and_expand(
            self.site_lib_path,
            self.pure_package_path,
            self.record_path,
            set(),
        )

        self.assertEqual(direct_file.read_text(), "wheel-owned")
        self.assertTrue(os.path.samefile(stale_alias, runtime_file))
        self.assertTrue((devel_root / "llvm").is_symlink())
        self.assertEqual(os.readlink(devel_root / "llvm"), "lib/llvm")
        self.assertFalse(tar_path.exists())
        record_lines = self.record_path.read_text().splitlines()
        self.assertIn("_rocm_sdk_devel/lib/shared.txt,,", record_lines)
        self.assertIn("_rocm_sdk_devel/llvm,,", record_lines)

        # A second process can enter after the first consumed the manifest. The
        # locked recheck makes this a successful no-op.
        self.devel._lock_and_expand(
            self.site_lib_path,
            self.pure_package_path,
            self.record_path,
            set(),
        )
        self.assertEqual(direct_file.read_text(), "wheel-owned")

    def test_regular_file_manifest_member_is_rejected(self):
        devel_root = self.site_lib_path / "_rocm_sdk_devel"
        direct_file = devel_root / "include" / "direct.h"
        direct_file.parent.mkdir(parents=True)
        direct_file.write_text("wheel-owned")
        members = [
            self._directory("_rocm_sdk_devel"),
            self._directory("_rocm_sdk_devel/include"),
        ]
        regular_file = tarfile.TarInfo("_rocm_sdk_devel/include/direct.h")
        regular_file.type = tarfile.REGTYPE
        regular_file.mode = 0o644
        members.append(regular_file)
        tar_path = self._write_manifest(members, data=b"legacy payload")

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported devel manifest member: _rocm_sdk_devel/include/direct.h",
        ):
            self.devel._lock_and_expand(
                self.site_lib_path,
                self.pure_package_path,
                self.record_path,
                set(),
            )

        self.assertEqual(direct_file.read_text(), "wheel-owned")
        self.assertTrue(tar_path.exists())


class CoreCliInitializationStateTest(unittest.TestCase):
    def setUp(self):
        self.temp_context = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_context.name)
        self.cli = self._load_cli_module()

    def tearDown(self):
        self.temp_context.cleanup()

    def _load_cli_module(self):
        package_name = f"_test_rocm_sdk_core_{id(self)}"
        package_module = types.ModuleType(package_name)
        package_module.__path__ = []
        dist_info_module = types.ModuleType(f"{package_name}._dist_info")
        core_entry = types.SimpleNamespace(get_py_package_name=lambda: "_rocm_sdk_core")
        devel_entry = types.SimpleNamespace(
            pure_py_package_name="rocm_sdk_devel",
            get_py_package_name=lambda: "_rocm_sdk_devel",
        )
        dist_info_module.ALL_PACKAGES = {
            "core": core_entry,
            "devel": devel_entry,
        }
        sys.modules[package_name] = package_module
        sys.modules[dist_info_module.__name__] = dist_info_module
        self.addCleanup(sys.modules.pop, dist_info_module.__name__, None)
        self.addCleanup(sys.modules.pop, package_name, None)

        module_name = f"{package_name}._cli"
        spec = importlib.util.spec_from_file_location(module_name, CORE_CLI_MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)
        spec.loader.exec_module(module)
        return module

    def test_platform_package_is_not_initialized_while_manifest_exists(self):
        pure_package_path = self.temp_dir / "rocm_sdk_devel"
        pure_package_path.mkdir()
        pure_init_path = pure_package_path / "__init__.py"
        pure_init_path.touch()
        platform_spec = types.SimpleNamespace(origin="platform/__init__.py")
        pure_spec = types.SimpleNamespace(origin=os.fspath(pure_init_path))

        def find_spec(name: str):
            if name == "_rocm_sdk_devel":
                return platform_spec
            if name == "rocm_sdk_devel":
                return pure_spec
            return None

        with mock.patch.object(
            self.cli.importlib.util,
            "find_spec",
            side_effect=find_spec,
        ):
            manifest_path = pure_package_path / "_devel.tar"
            manifest_path.touch()
            self.assertFalse(self.cli._is_devel_module_expanded())

            manifest_path.unlink()
            self.assertTrue(self.cli._is_devel_module_expanded())

    def test_malformed_pure_devel_package_is_rejected(self):
        non_file_spec = types.SimpleNamespace(origin=None)
        cases = [
            (
                "missing pure package",
                None,
                AssertionError,
                "Package 'rocm_sdk_devel' disappeared after it was detected",
            ),
            (
                "non-file-backed pure package",
                non_file_spec,
                ImportError,
                "Required package 'rocm_sdk_devel' is not file-backed",
            ),
        ]

        for name, pure_spec, error_type, error in cases:
            with self.subTest(name=name):
                with mock.patch.object(
                    self.cli.importlib.util,
                    "find_spec",
                    return_value=pure_spec,
                ):
                    with self.assertRaisesRegex(error_type, error):
                        self.cli._is_devel_module_expanded()


if __name__ == "__main__":
    unittest.main()
