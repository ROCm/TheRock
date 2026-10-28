#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the Windows ``upload_package_repo.py``.

The uploader is a thin wrapper: it resolves the destination via
``WorkflowOutputRoot`` and uploads ``*.msi`` through a storage backend. These
tests mock both collaborators and assert the wiring and the failure modes
(missing directory, no MSIs).
"""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

THIS_DIR = Path(__file__).resolve().parent
WINDOWS_DIR = THIS_DIR.parent
BUILD_TOOLS_DIR = WINDOWS_DIR.parent.parent

if os.fspath(BUILD_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, os.fspath(BUILD_TOOLS_DIR))


def _load_module(name: str, path: Path) -> types.ModuleType:
    """Load the module under test by path under a unique registered name.

    The Linux packaging dir has a sibling ``upload_package_repo.py``; loading by
    an explicit unique name avoids the sys.modules clash that a bare
    ``import upload_package_repo`` would cause when both test files run together.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


upload_repo = _load_module(
    "windows_upload_package_repo", WINDOWS_DIR / "upload_package_repo.py"
)


class UploadPackageRepoTest(unittest.TestCase):
    def test_missing_package_dir_raises(self):
        # A missing directory is a caller error, surfaced as a specific
        # exception rather than sys.exit so callers/tests can handle it.
        with self.assertRaises(FileNotFoundError):
            upload_repo.upload_package_repo(
                run_id="123", package_dir=Path("/no/such/dir"), dry_run=False
            )

    @patch.object(upload_repo, "create_storage_backend")
    @patch.object(upload_repo, "WorkflowOutputRoot")
    def test_uploads_msi_to_resolved_destination(self, mock_root_cls, mock_backend_fn):
        dest = MagicMock()
        dest.s3_uri = "s3://bucket/123-windows/packages/msi"
        mock_root_cls.from_workflow_run.return_value.native_windows_packages.return_value = (
            dest
        )
        backend = MagicMock()
        backend.upload_directory.return_value = 1
        mock_backend_fn.return_value = backend

        with tempfile.TemporaryDirectory() as tmp:
            pkg_dir = Path(tmp)
            (pkg_dir / "amdrocm-runtime.msi").write_bytes(b"msi")
            upload_repo.upload_package_repo(
                run_id="123", package_dir=pkg_dir, dry_run=False
            )

        # Destination resolved for the windows msi packages of that run.
        mock_root_cls.from_workflow_run.assert_called_once_with(
            run_id="123", platform="windows"
        )
        mock_root_cls.from_workflow_run.return_value.native_windows_packages.assert_called_once_with(
            "msi"
        )
        # Only *.msi are uploaded, to the resolved destination.
        args, kwargs = backend.upload_directory.call_args
        self.assertEqual(args[0], pkg_dir)
        self.assertEqual(args[1], dest)
        self.assertEqual(kwargs.get("include"), ["*.msi"])

    @patch.object(upload_repo, "create_storage_backend")
    @patch.object(upload_repo, "WorkflowOutputRoot")
    def test_no_msi_files_raises(self, mock_root_cls, mock_backend_fn):
        mock_root_cls.from_workflow_run.return_value.native_windows_packages.return_value = (
            MagicMock()
        )
        backend = MagicMock()
        backend.upload_directory.return_value = 0  # nothing matched *.msi
        mock_backend_fn.return_value = backend

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                upload_repo.upload_package_repo(
                    run_id="123", package_dir=Path(tmp), dry_run=False
                )

    @patch.object(upload_repo, "create_storage_backend")
    @patch.object(upload_repo, "WorkflowOutputRoot")
    def test_dry_run_forwarded_to_backend(self, mock_root_cls, mock_backend_fn):
        mock_root_cls.from_workflow_run.return_value.native_windows_packages.return_value = (
            MagicMock()
        )
        backend = MagicMock()
        backend.upload_directory.return_value = 1
        mock_backend_fn.return_value = backend

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "x.msi").write_bytes(b"msi")
            upload_repo.upload_package_repo(
                run_id="123", package_dir=Path(tmp), dry_run=True
            )

        mock_backend_fn.assert_called_once_with(dry_run=True)


if __name__ == "__main__":
    unittest.main()
