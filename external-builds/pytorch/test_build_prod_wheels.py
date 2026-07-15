#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for helpers in build_prod_wheels.py.

These cover the pure-Python argument-defaulting logic (no build/network side
effects), in particular the `--root-checkout-dir` handling introduced alongside
the `python -m build` wheel-build change.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_prod_wheels  # noqa: E402


def _make_build_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        root_checkout_dir=None,
        pytorch_dir=None,
        pytorch_audio_dir=None,
        pytorch_vision_dir=None,
        triton_dir=None,
        apex_dir=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_directory_if_exists_returns_path_when_present(tmp_path):
    assert build_prod_wheels.directory_if_exists(tmp_path) == tmp_path


def test_directory_if_exists_returns_none_when_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert build_prod_wheels.directory_if_exists(missing) is None


def test_apply_root_checkout_dir_populates_existing_subdirs(tmp_path):
    for name in ("pytorch", "pytorch_audio", "pytorch_vision", "triton", "apex"):
        (tmp_path / name).mkdir()

    args = _make_build_args(root_checkout_dir=tmp_path)
    build_prod_wheels.apply_root_checkout_dir(args)

    assert args.pytorch_dir == tmp_path / "pytorch"
    assert args.pytorch_audio_dir == tmp_path / "pytorch_audio"
    assert args.pytorch_vision_dir == tmp_path / "pytorch_vision"
    assert args.triton_dir == tmp_path / "triton"
    assert args.apex_dir == tmp_path / "apex"


def test_apply_root_checkout_dir_leaves_missing_subdirs_none(tmp_path):
    # Only pytorch exists under the root; the rest should stay None.
    (tmp_path / "pytorch").mkdir()

    args = _make_build_args(root_checkout_dir=tmp_path)
    build_prod_wheels.apply_root_checkout_dir(args)

    assert args.pytorch_dir == tmp_path / "pytorch"
    assert args.pytorch_audio_dir is None
    assert args.pytorch_vision_dir is None
    assert args.triton_dir is None
    assert args.apex_dir is None


def test_apply_root_checkout_dir_does_not_override_explicit_dirs(tmp_path):
    # Explicit per-project dirs take precedence over the root default even when
    # a matching subdir exists under the root.
    (tmp_path / "pytorch").mkdir()
    explicit = tmp_path / "custom-pytorch"
    explicit.mkdir()

    args = _make_build_args(root_checkout_dir=tmp_path, pytorch_dir=explicit)
    build_prod_wheels.apply_root_checkout_dir(args)

    assert args.pytorch_dir == explicit


def test_apply_root_checkout_dir_noop_when_root_is_none(tmp_path):
    args = _make_build_args(root_checkout_dir=None)
    build_prod_wheels.apply_root_checkout_dir(args)

    assert args.pytorch_dir is None
    assert args.pytorch_audio_dir is None
    assert args.pytorch_vision_dir is None
    assert args.triton_dir is None
    assert args.apex_dir is None
