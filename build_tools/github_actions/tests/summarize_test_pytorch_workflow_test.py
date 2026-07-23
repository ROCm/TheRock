#!/usr/bin/env python
"""Unit tests for summarize_test_pytorch_workflow.py."""

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
    device_extras="",
    torch_version="2.10.0+rocm7.12.0a20260501",
    pytorch_git_ref="release/2.10",
    python_version="3.12",
):
    return argparse.Namespace(
        index_url=index_url,
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
    def test_with_device_extras(self):
        """Device extras are joined with the torch package name."""
        text = _run_and_capture(
            _make_args(
                index_url="https://rocm.nightlies.amd.com/whl-multi-arch/",
                device_extras="device-gfx942",
            )
        )
        self.assertIn(
            "--index-url=https://rocm.nightlies.amd.com/whl-multi-arch/",
            text,
        )
        # Device extras select the GPU via the package spec.
        self.assertIn("torch[device-gfx942]", text)
        # No GPU family subdirectory is added (per-family releases used this).
        self.assertNotIn("gfx94X-dcgpu/", text)

    def test_without_device_extras(self):
        """Torch can also be installed without any extras."""
        text = _run_and_capture(
            _make_args(index_url="https://rocm.nightlies.amd.com/whl-multi-arch/")
        )
        self.assertIn(
            "--index-url=https://rocm.nightlies.amd.com/whl-multi-arch/", text
        )
        # No device extra added anywhere.
        self.assertNotIn("device-]", text)


if __name__ == "__main__":
    unittest.main()
