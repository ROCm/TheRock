#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for compute_stage_sparse_checkout.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from compute_stage_sparse_checkout import (
    compute_stage_sparse_checkout,
    extract_source_path_from_project,
    get_stage_source_paths,
)


class TestExtractSourcePathFromProject:
    """Tests for extract_source_path_from_project function."""

    def test_projects_path(self):
        """Test extraction from projects/ path."""
        assert extract_source_path_from_project("projects/rocprim") == "rocprim"

    def test_shared_path(self):
        """Test extraction from shared/ path."""
        assert extract_source_path_from_project("shared/rocroller") == "rocroller"

    def test_dnn_providers_path(self):
        """Test extraction from dnn-providers/ path."""
        result = extract_source_path_from_project("dnn-providers/miopen-provider")
        assert result == "miopen-provider"

    def test_simple_path(self):
        """Test extraction from simple path without prefix."""
        assert extract_source_path_from_project("rocprim") == "rocprim"

    def test_empty_path(self):
        """Test extraction from empty path."""
        assert extract_source_path_from_project("") is None
        assert extract_source_path_from_project("  ") is None


class TestGetStageSourcePaths:
    """Tests for get_stage_source_paths function."""

    def test_math_libs_includes_expected(self):
        """Test math-libs stage includes expected source paths."""
        source_paths = get_stage_source_paths("math-libs")
        assert "rocprim" in source_paths
        assert "rocblas" in source_paths
        assert "rocroller" in source_paths

    def test_cv_libs_includes_rpp(self):
        """Test cv-libs stage includes rpp."""
        source_paths = get_stage_source_paths("cv-libs")
        assert "rpp" in source_paths

    def test_unknown_stage_returns_empty(self):
        """Test unknown stage returns empty set."""
        source_paths = get_stage_source_paths("unknown-stage")
        assert source_paths == set()


class TestComputeStageSparseCheckout:
    """Tests for compute_stage_sparse_checkout function."""

    def test_affected_stage(self):
        """Test stage that is affected by changed projects."""
        paths = compute_stage_sparse_checkout(
            "math-libs", "projects/rocprim,shared/rocroller"
        )
        assert "projects/rocprim" in paths
        assert "shared/rocroller" in paths

    def test_unaffected_stage(self):
        """Test stage that is NOT affected by changed projects."""
        paths = compute_stage_sparse_checkout("cv-libs", "projects/rocprim")
        assert paths == []

    def test_empty_changed_projects(self):
        """Test with empty changed_projects."""
        paths = compute_stage_sparse_checkout("math-libs", "")
        assert paths == []

    def test_partial_affect(self):
        """Test when only some changed projects affect the stage."""
        paths = compute_stage_sparse_checkout(
            "math-libs", "projects/rocprim,projects/rpp"
        )
        # rocprim affects math-libs, rpp does not
        assert "projects/rocprim" in paths
        assert "projects/rpp" not in paths

    def test_paths_are_sorted(self):
        """Test that returned paths are sorted."""
        paths = compute_stage_sparse_checkout(
            "math-libs", "shared/rocroller,projects/rocprim"
        )
        assert paths == sorted(paths)
