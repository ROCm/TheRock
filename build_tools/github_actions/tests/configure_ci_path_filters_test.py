from pathlib import Path
import os
import sys
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from configure_ci_path_filters import is_ci_run_required


class ConfigureCIPathFiltersTest(unittest.TestCase):
    def test_run_ci_if_source_file_edited(self):
        paths = ["source_file.h"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

    def test_dont_run_ci_if_only_markdown_files_edited(self):
        paths = ["README.md", "build_tools/README.md"]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

    def test_dont_run_ci_if_only_external_builds_edited(self):
        paths = ["external-builds/pytorch/CMakeLists.txt"]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

    def test_dont_run_ci_if_only_external_builds_edited(self):
        paths = ["experimental/file.h"]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

    def test_run_ci_if_related_workflow_file_edited(self):
        paths = [".github/workflows/ci.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

        paths = [".github/workflows/build_portable_linux_artifacts.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

        paths = [".github/workflows/build_artifact.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

        # External repos use therock-*.yml naming
        paths = [".github/workflows/therock-ci.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

    def test_dont_run_ci_if_unrelated_workflow_file_edited(self):
        paths = [".github/workflows/pre-commit.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

        paths = [".github/workflows/test_jax_dockerfile.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

    def test_run_ci_if_source_file_and_unrelated_workflow_file_edited(self):
        paths = ["source_file.h", ".github/workflows/pre-commit.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

    def test_is_ci_run_required_with_none_paths_skips_build(self):
        """Test that None paths (git diff failure) skips build for internal repos."""
        run_ci = is_ci_run_required(paths=None)
        self.assertFalse(run_ci)

    def test_is_ci_run_required_with_external_repo_skip_patterns(self):
        """Test that external repo custom skip patterns are used correctly."""
        # External repo provides custom skip pattern
        custom_skip_patterns = ["custom-docs/*", "*.rst"]

        # Path matching custom pattern should skip
        paths = ["custom-docs/README.rst"]
        run_ci = is_ci_run_required(paths, skip_patterns=custom_skip_patterns)
        self.assertFalse(run_ci)

        # Path not matching custom pattern should trigger build
        paths = ["source_file.cpp"]
        run_ci = is_ci_run_required(paths, skip_patterns=custom_skip_patterns)
        self.assertTrue(run_ci)

        # Path matching default pattern but not custom pattern should trigger build
        # (custom patterns override defaults)
        paths = ["README.md"]  # Would be skipped by default, but not by custom patterns
        run_ci = is_ci_run_required(paths, skip_patterns=custom_skip_patterns)
        self.assertTrue(run_ci)

    def test_is_ci_run_required_with_empty_skip_patterns_does_not_skip(self):
        """Test that empty skip_patterns list means no patterns to skip (all paths trigger build)."""
        # Empty list means no skip patterns, so all paths trigger build
        paths = [
            "README.md"
        ]  # Would be skipped by default patterns, but empty list = no skipping
        run_ci = is_ci_run_required(paths, skip_patterns=[])
        self.assertTrue(run_ci)

        # None means use defaults
        paths = ["README.md"]  # Should be skipped by default patterns
        run_ci = is_ci_run_required(paths, skip_patterns=None)
        self.assertFalse(run_ci)


if __name__ == "__main__":
    unittest.main()
