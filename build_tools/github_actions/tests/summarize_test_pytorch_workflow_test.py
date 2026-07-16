#!/usr/bin/env python
"""Unit tests for summarize_test_pytorch_workflow.py.

Focus on the index-URL construction: per-family installs append a family
subdir, while multi-arch installs select the GPU via device extras on the
flat whl-multi-arch index and must NOT append a subdir.
"""

import argparse
import os
import sys
import unittest
import unittest.mock
from pathlib import Path

# Add github_actions to path so the module under test is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import summarize_test_pytorch_workflow


def _make_args(
    index_url,
    index_subdir="",
    device_extras="",
    torch_version="2.10.0+rocm7.12.0a20260501",
    pytorch_git_ref="release/2.10",
    python_version="3.12",
):
    return argparse.Namespace(
        index_url=index_url,
        index_subdir=index_subdir,
        device_extras=device_extras,
        torch_version=torch_version,
        pytorch_git_ref=pytorch_git_ref,
        python_version=python_version,
    )


def _run_and_capture(args) -> str:
    with unittest.mock.patch.object(
        summarize_test_pytorch_workflow, "gha_append_step_summary"
    ) as mock_summary:
        summarize_test_pytorch_workflow.run(args)
    return mock_summary.call_args[0][0]


class TestIndexUrl(unittest.TestCase):
    def test_per_family_appends_subdir(self):
        """Per-family mode (no device extras) appends the family subdir."""
        text = _run_and_capture(
            _make_args(
                index_url="https://rocm.nightlies.amd.com/v2-staging",
                index_subdir="gfx110X-dgpu",
            )
        )
        self.assertIn(
            "--index-url=https://rocm.nightlies.amd.com/v2-staging/gfx110X-dgpu/",
            text,
        )

    def test_multi_arch_does_not_append_subdir(self):
        """Multi-arch mode uses the flat index even if a family is passed."""
        text = _run_and_capture(
            _make_args(
                index_url="https://rocm.nightlies.amd.com/whl-multi-arch/",
                index_subdir="gfx94X-dcgpu",
                device_extras="device-gfx942",
            )
        )
        self.assertIn(
            "--index-url=https://rocm.nightlies.amd.com/whl-multi-arch/",
            text,
        )
        self.assertNotIn("gfx94X-dcgpu/", text)
        # Device extras select the GPU via the package spec instead.
        self.assertIn("torch[device-gfx942]", text)

    def test_no_subdir_no_extras(self):
        """With neither subdir nor extras, the index URL is used as-is."""
        text = _run_and_capture(
            _make_args(index_url="https://rocm.nightlies.amd.com/v2")
        )
        self.assertIn("--index-url=https://rocm.nightlies.amd.com/v2/", text)


if __name__ == "__main__":
    unittest.main()
