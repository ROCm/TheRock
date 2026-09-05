# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import platform
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Add repo root to PYTHONPATH
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import overlay_coverage_artifacts


def is_windows() -> bool:
    return platform.system() == "Windows"


class TempDirTestBase(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.root = Path(self._temp_dir.name)

    def write(self, relative_path: str, contents: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        return path


class FetchInstrumentedArtifactsTest(TempDirTestBase):
    def test_narrows_the_fetch_to_the_named_artifacts(self):
        with mock.patch.object(
            overlay_coverage_artifacts, "artifact_manager_main"
        ) as fetch:
            artifacts_dir = overlay_coverage_artifacts.fetch_instrumented_artifacts(
                run_id="99999",
                artifact_names=["rand", "prim"],
                amdgpu_families="gfx94X-dcgpu",
                amdgpu_targets="",
                output_dir=self.root / "staging",
                run_github_repo=None,
            )

        (argv,) = fetch.call_args.args
        self.assertEqual(argv[0], "fetch")
        # These artifacts are stage outputs, not inbound to any stage.
        self.assertEqual(argv[argv.index("--stage") + 1], "all")
        self.assertEqual(argv[argv.index("--artifact-names") + 1], "rand,prim")
        self.assertEqual(artifacts_dir, self.root / "staging" / "artifacts")

    def test_skips_components_that_cannot_be_instrumented(self):
        with mock.patch.object(
            overlay_coverage_artifacts, "artifact_manager_main"
        ) as fetch:
            overlay_coverage_artifacts.fetch_instrumented_artifacts(
                run_id="99999",
                artifact_names=["rand"],
                amdgpu_families="gfx94X-dcgpu",
                amdgpu_targets="gfx942",
                output_dir=self.root / "staging",
                run_github_repo=None,
            )

        (argv,) = fetch.call_args.args
        excluded = argv[argv.index("--exclude-components") + 1].split(",")
        self.assertCountEqual(excluded, ["run", "dbg", "doc"])
        self.assertEqual(argv[argv.index("--amdgpu-targets") + 1], "gfx942")


class OverlayRelpathsTest(TempDirTestBase):
    def _staged_artifact(self, artifact: str, relpath: str, files: dict[str, str]):
        for name, contents in files.items():
            self.write(f"staging/{artifact}/{relpath}/{name}", contents)

    def test_copies_only_the_requested_relpath(self):
        self._staged_artifact(
            "rand_lib_gfx942",
            "math-libs/hipRAND/stage",
            {"lib/libhiprand.so": "instrumented"},
        )
        self._staged_artifact(
            "rand_lib_gfx942",
            "math-libs/rocRAND/stage",
            {"lib/librocrand.so": "instrumented"},
        )
        self.write("install/lib/libhiprand.so", "baseline")
        self.write("install/lib/librocrand.so", "baseline")

        copied = overlay_coverage_artifacts.overlay_relpaths(
            staging_dir=self.root / "staging",
            relpaths=["math-libs/hipRAND/stage"],
            install_dir=self.root / "install",
        )

        self.assertEqual(copied, 1)
        self.assertEqual(
            (self.root / "install/lib/libhiprand.so").read_text(), "instrumented"
        )
        # A sibling project in the same grouped artifact stays as it was.
        self.assertEqual(
            (self.root / "install/lib/librocrand.so").read_text(), "baseline"
        )

    def test_merges_into_the_existing_tree(self):
        self._staged_artifact(
            "rand_lib_gfx942",
            "math-libs/hipRAND/stage",
            {"lib/libhiprand.so": "instrumented"},
        )
        self.write("install/lib/libamdhip64.so", "baseline")

        overlay_coverage_artifacts.overlay_relpaths(
            staging_dir=self.root / "staging",
            relpaths=["math-libs/hipRAND/stage"],
            install_dir=self.root / "install",
        )

        self.assertTrue((self.root / "install/lib/libamdhip64.so").exists())
        self.assertTrue((self.root / "install/lib/libhiprand.so").exists())

    def test_counts_every_matching_artifact(self):
        for component in ("lib", "test"):
            self._staged_artifact(
                f"rand_{component}_gfx942",
                "math-libs/hipRAND/stage",
                {f"{component}/file": component},
            )
        (self.root / "install").mkdir()

        self.assertEqual(
            overlay_coverage_artifacts.overlay_relpaths(
                staging_dir=self.root / "staging",
                relpaths=["math-libs/hipRAND/stage"],
                install_dir=self.root / "install",
            ),
            2,
        )

    def test_reports_nothing_copied_when_the_relpath_is_absent(self):
        self._staged_artifact(
            "rand_lib_gfx942", "math-libs/rocRAND/stage", {"lib/librocrand.so": "x"}
        )
        (self.root / "install").mkdir()

        self.assertEqual(
            overlay_coverage_artifacts.overlay_relpaths(
                staging_dir=self.root / "staging",
                relpaths=["math-libs/hipRAND/stage"],
                install_dir=self.root / "install",
            ),
            0,
        )

    @unittest.skipIf(is_windows(), "symlinks require elevation on Windows")
    def test_preserves_symlinks(self):
        # The loader follows libfoo.so -> libfoo.so.1; copying the link as a
        # regular file would leave a stale non-instrumented copy behind it.
        stage = "staging/rand_lib_gfx942/math-libs/hipRAND/stage/lib"
        self.write(f"{stage}/libhiprand.so.1", "instrumented")
        (self.root / stage / "libhiprand.so").symlink_to("libhiprand.so.1")
        (self.root / "install").mkdir()

        overlay_coverage_artifacts.overlay_relpaths(
            staging_dir=self.root / "staging",
            relpaths=["math-libs/hipRAND/stage"],
            install_dir=self.root / "install",
        )

        self.assertTrue((self.root / "install/lib/libhiprand.so").is_symlink())


class MainTest(TempDirTestBase):
    def _run(self, **overrides) -> int:
        args = {
            "--run-id": "99999",
            "--amdgpu-families": "gfx94X-dcgpu",
            "--artifact-names": "rand",
            "--artifact-relpaths": "math-libs/hipRAND/stage",
            "--install-dir": os.fspath(self.root / "install"),
            "--staging-dir": os.fspath(self.root / "staging"),
        }
        args.update(overrides)
        argv = [item for pair in args.items() for item in pair]
        return overlay_coverage_artifacts.main(argv)

    def test_fails_when_the_install_tree_is_missing(self):
        with mock.patch.object(
            overlay_coverage_artifacts, "artifact_manager_main"
        ) as fetch:
            self.assertEqual(self._run(), 1)
        fetch.assert_not_called()

    def test_fails_when_nothing_was_overlaid(self):
        # A silent pass here would report coverage for binaries that were never
        # instrumented.
        (self.root / "install").mkdir()
        with mock.patch.object(overlay_coverage_artifacts, "artifact_manager_main"):
            self.assertEqual(self._run(), 1)

    def test_fetches_and_overlays(self):
        (self.root / "install").mkdir()

        def fake_fetch(argv):
            self.write(
                "staging/artifacts/rand_lib_gfx942/math-libs/hipRAND/stage"
                "/lib/libhiprand.so",
                "instrumented",
            )

        with mock.patch.object(
            overlay_coverage_artifacts,
            "artifact_manager_main",
            side_effect=fake_fetch,
        ):
            self.assertEqual(self._run(), 0)

        self.assertEqual(
            (self.root / "install/lib/libhiprand.so").read_text(), "instrumented"
        )


if __name__ == "__main__":
    unittest.main()
