#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for install_rocm_code_coverage_build.py.

Focuses on the generic-vs-instrumented repo split: the generic install is keyed
on --run-github-repo while the instrumented replacement fetch must be keyed on
--code-coverage-run-github-repo (defaulting to $GITHUB_REPOSITORY). Also checks
which of the replaced files get their program header table looked at.
"""

import io
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import install_rocm_code_coverage_build as mod


class _FakeBackend:
    """Minimal ArtifactBackend stand-in sufficient for the replacement fetch."""

    def __init__(self):
        self.base_uri = "s3://fake-bucket/fake"

    def list_artifacts(self, name_filter=None):
        # One instrumented blas tar for the requested family.
        return ["blas_lib_generic.tar.xz"]


class TestCodeCoverageRepoSplit(unittest.TestCase):
    def _run_main(self, extra_argv, env=None):
        """Run main() with all network boundaries mocked, capturing kwargs.

        Returns the github_repository kwarg passed to create_backend_from_env
        for the instrumented replacement fetch.
        """
        captured = {}

        def fake_create_backend(**kwargs):
            captured["github_repository"] = kwargs.get("github_repository")
            captured["run_id"] = kwargs.get("run_id")
            return _FakeBackend()

        def fake_find_available(artifact_names, target_families, available):
            # Pretend the single listed artifact matches.
            return list(available)

        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "--code-coverage-run-id",
                "222",
                "--replace-rocblas",
                "--artifact-group",
                "gfx94X-dcgpu",
                "--output-dir",
                tmp,
                "--run-id",
                "111",
                "--run-github-repo",
                "ROCm/rocm-libraries",
            ] + extra_argv

            patched_env = env or {}
            with (
                mock.patch.object(mod, "install_from_artifacts_main"),
                mock.patch.object(
                    mod, "create_backend_from_env", side_effect=fake_create_backend
                ),
                mock.patch.object(
                    mod, "find_available_artifacts", side_effect=fake_find_available
                ),
                mock.patch.object(
                    mod, "download_artifact", return_value=Path(tmp) / "x"
                ),
                # No archives to extract -> replace step is a no-op.
                mock.patch.object(mod, "replace_instrumented_libraries"),
                mock.patch.dict(os.environ, patched_env, clear=False),
            ):
                mod.main(argv)
        return captured

    def test_replacement_uses_code_coverage_repo_not_generic_repo(self):
        captured = self._run_main(["--code-coverage-run-github-repo", "ROCm/TheRock"])
        self.assertEqual(captured["run_id"], "222")
        self.assertEqual(captured["github_repository"], "ROCm/TheRock")
        self.assertNotEqual(captured["github_repository"], "ROCm/rocm-libraries")

    def test_replacement_repo_defaults_to_github_repository_env(self):
        # main() must be re-imported so the argparse default picks up the env,
        # since the default is evaluated at parse time inside main().
        captured = self._run_main([], env={"GITHUB_REPOSITORY": "ROCm/TheRock"})
        self.assertEqual(captured["github_repository"], "ROCm/TheRock")


def _write_archive(path, members):
    """Write an artifact archive of `members`: name -> content, None for a dir."""
    with tarfile.open(path, "w:xz") as tf:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            if content is None:
                info.type = tarfile.DIRTYPE
                tf.addfile(info)
            else:
                info.size = len(content)
                tf.addfile(info, io.BytesIO(content))


class TestReplaceInstrumentedLibraries(unittest.TestCase):
    def _replaced_files(self, archive_name, artifacts, members):
        """Run the replacement into an empty tree; return the files it wrote."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            downloads = tmp / "downloads"
            downloads.mkdir()
            _write_archive(downloads / archive_name, members)
            output_dir = tmp / "install"
            with mock.patch.object(mod, "pin_phdr_table", return_value=False):
                mod.replace_instrumented_libraries(artifacts, downloads, output_dir)
            return sorted(
                p.relative_to(output_dir).as_posix()
                for p in output_dir.rglob("*")
                if p.is_file()
            )

    def test_pins_program_headers_of_every_replaced_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            downloads = tmp / "downloads"
            downloads.mkdir()
            prefix = "math-libs/BLAS/rocBLAS/stage"
            other_prefix = "math-libs/BLAS/hipBLAS/stage"
            members = {
                "artifact_manifest.txt": f"{prefix}\n{other_prefix}\n".encode(),
                f"{prefix}/lib/librocblas.so.5.8": b"instrumented library",
                f"{prefix}/lib/rocblas/library": None,
                f"{prefix}/lib/rocblas/library/TensileLibrary.dat": b"data",
                f"{other_prefix}/lib/libhipblas.so.3.8": b"some other library",
            }
            _write_archive(downloads / "blas_lib_generic.tar.xz", members)
            output_dir = tmp / "install"

            with mock.patch.object(mod, "pin_phdr_table", return_value=False) as pin:
                mod.replace_instrumented_libraries(
                    {"blas": ["rocBLAS"]}, downloads, output_dir
                )

            library = output_dir / "lib" / "librocblas.so.5.8"
            data = output_dir / "lib" / "rocblas" / "library" / "TensileLibrary.dat"
            self.assertEqual(library.read_bytes(), b"instrumented library")
            self.assertFalse((output_dir / "lib" / "libhipblas.so.3.8").exists())
            self.assertEqual(
                pin.call_args_list,
                [
                    mock.call(library, require_section="__llvm_prf_cnts"),
                    mock.call(data, require_section="__llvm_prf_cnts"),
                ],
            )

    def test_replaces_every_file_in_the_component_stage_dirs(self):
        # Laid out like prim_test: tests named after what they test, and the
        # device code of all three projects in hipCUB's stage directory.
        stages = {
            "hipCUB": "math-libs/hipCUB/stage",
            "rocPRIM": "math-libs/rocPRIM_tests/stage",
            "rocThrust": "math-libs/rocThrust/stage",
        }
        files = {
            "hipCUB": ["bin/test_hipcub_block_scan"],
            "rocPRIM": ["bin/test_block_scan"],
            "rocThrust": ["bin/merge.hip", "bin/rocthrust/CTestTestfile.cmake"],
        }
        members = {
            "artifact_manifest.txt": "".join(
                f"{s}\n" for s in stages.values()
            ).encode(),
            f"{stages['hipCUB']}/.kpack/prim_test_gfx942.kpack": b"device code",
        }
        for folder, paths in files.items():
            for path in paths:
                members[f"{stages[folder]}/{path}"] = b"instrumented test"

        # The shared kpack follows whichever project's host binaries are
        # swapped in, or they look up kernels the baseline's kpack lacks.
        for folder, paths in files.items():
            with self.subTest(folder=folder):
                self.assertEqual(
                    self._replaced_files(
                        "prim_test_generic.tar.xz", {"prim": [folder]}, members
                    ),
                    sorted([".kpack/prim_test_gfx942.kpack", *paths]),
                )

    def test_keeps_device_code_of_artifacts_without_the_component_stage(self):
        prefix = "math-libs/BLAS/hipBLASLt/stage"
        members = {
            "artifact_manifest.txt": f"{prefix}\n".encode(),
            f"{prefix}/.kpack/blas_lib_gfx942.kpack": b"device code",
            f"{prefix}/lib/libhipblaslt.so": b"some other library",
        }
        self.assertEqual(
            self._replaced_files(
                "blas_lib_gfx942.tar.xz", {"blas": ["rocBLAS"]}, members
            ),
            [],
        )

    def test_replaces_device_code_named_after_the_component(self):
        prefix = "math-libs/rocWMMA/stage"
        members = {
            "artifact_manifest.txt": f"{prefix}\n".encode(),
            f"{prefix}/.kpack/rocwmma_test_gfx942.kpack": b"device code",
            f"{prefix}/bin/gemm_xdl-validate": b"instrumented test",
        }
        self.assertEqual(
            self._replaced_files(
                "rocwmma_test_gfx942.tar.xz", {"rocwmma": ["rocWMMA"]}, members
            ),
            [".kpack/rocwmma_test_gfx942.kpack", "bin/gemm_xdl-validate"],
        )

    def test_wrapper_brings_its_backend_from_the_same_build(self):
        # hipRAND's linkage test compares the librocrand version against the
        # rocRAND headers it was built with, so rocRAND must be swapped too.
        hiprand = "math-libs/hipRAND/stage"
        rocrand = "math-libs/rocRAND/stage"
        members = {
            "artifact_manifest.txt": f"{hiprand}\n{rocrand}\n".encode(),
            f"{hiprand}/lib/libhiprand.so.1.1": b"instrumented hiprand",
            f"{rocrand}/lib/librocrand.so.1.1": b"instrumented rocrand",
        }
        self.assertEqual(
            self._replaced_files(
                "rand_lib_generic.tar.xz",
                mod.replacement_folders(["hiprand"]),
                members,
            ),
            ["lib/libhiprand.so.1.1", "lib/librocrand.so.1.1"],
        )


class TestReplacementFolders(unittest.TestCase):
    def test_wrapper_adds_its_same_artifact_backend(self):
        self.assertEqual(
            mod.replacement_folders(["hiprand"]), {"rand": ["hipRAND", "rocRAND"]}
        )

    def test_backend_is_not_listed_twice(self):
        self.assertEqual(
            mod.replacement_folders(["hipfft", "rocfft"]),
            {"fft": ["hipFFT", "rocFFT"]},
        )

    def test_backend_alone_does_not_bring_its_wrapper(self):
        self.assertEqual(mod.replacement_folders(["rocrand"]), {"rand": ["rocRAND"]})

    def test_backends_ship_in_their_wrappers_artifact(self):
        folders = {folder for _, folder in mod.COMPONENT_MAP.values()}
        for comp, backends in mod.SAME_ARTIFACT_BACKENDS.items():
            artifact = mod.COMPONENT_MAP[comp][0]
            for backend in backends:
                with self.subTest(comp=comp, backend=backend):
                    self.assertIn(backend, folders)
                    owner = next(
                        name
                        for name, folder in mod.COMPONENT_MAP.values()
                        if folder == backend
                    )
                    self.assertEqual(owner, artifact)


class TestHostLibraries(unittest.TestCase):
    def test_hipsparselt_takes_tensilelite_host_and_origami_from_blas(self):
        self.assertEqual(
            mod.host_library_deps(["hipsparselt"]),
            {
                "blas": [
                    ("hipBLASLt", "libtensilelite-host"),
                    ("origami", "liborigami"),
                ]
            },
        )

    def test_component_without_host_libraries_needs_none(self):
        self.assertEqual(mod.host_library_deps(["rocblas"]), {})

    def test_downloads_only_the_generic_lib_archive_of_host_artifacts(self):
        class _Backend:
            base_uri = "s3://fake-bucket/fake"

            def list_artifacts(self, name_filter=None):
                return [
                    "sparse_lib_generic.tar.zst",
                    "sparse_test_gfx942.tar.zst",
                    "blas_lib_generic.tar.zst",
                    "blas_lib_gfx942.tar.zst",
                    "blas_dev_generic.tar.zst",
                    "blas_test_generic.tar.zst",
                ]

        with tempfile.TemporaryDirectory() as tmp:
            opts = mock.Mock(
                artifact_group="gfx94X-dcgpu", amdgpu_targets="", output_dir=Path(tmp)
            )
            with (
                mock.patch.object(
                    mod, "create_backend_from_env", return_value=_Backend()
                ),
                mock.patch.object(
                    mod, "download_artifact", return_value=Path(tmp) / "x"
                ) as download,
            ):
                mod.download_replacement_artifacts(
                    "222", {"sparse"}, opts, "ROCm/TheRock", "ci", {"blas"}
                )
            fetched = sorted(c.args[0].artifact_key for c in download.call_args_list)
        self.assertEqual(
            fetched,
            [
                "blas_lib_generic.tar.zst",
                "sparse_lib_generic.tar.zst",
                "sparse_test_gfx942.tar.zst",
            ],
        )

    def test_replaces_only_the_named_host_libraries(self):
        hipblaslt = "math-libs/BLAS/hipBLASLt/stage"
        origami = "math-libs/BLAS/origami/stage"
        rocblas = "math-libs/BLAS/rocBLAS/stage"
        members = {
            "artifact_manifest.txt": f"{hipblaslt}\n{origami}\n{rocblas}\n".encode(),
            f"{hipblaslt}/lib/libhipblaslt.so.1.0": b"hipblaslt",
            f"{hipblaslt}/lib/libtensilelite-host.so.1.0": b"tensilelite-host",
            f"{origami}/lib/liborigami.so.1.0": b"origami",
            f"{rocblas}/lib/librocblas.so.5": b"rocblas",
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            downloads = tmp / "downloads"
            downloads.mkdir()
            _write_archive(downloads / "blas_lib_generic.tar.xz", members)
            output_dir = tmp / "install"
            with mock.patch.object(mod, "pin_phdr_table", return_value=False):
                mod.replace_host_libraries(
                    mod.host_library_deps(["hipsparselt"]), downloads, output_dir
                )
            written = sorted(
                p.relative_to(output_dir).as_posix()
                for p in output_dir.rglob("*")
                if p.is_file()
            )
        self.assertEqual(
            written, ["lib/liborigami.so.1.0", "lib/libtensilelite-host.so.1.0"]
        )

    def test_missing_host_library_is_an_error(self):
        prefix = "math-libs/BLAS/hipBLASLt/stage"
        members = {
            "artifact_manifest.txt": f"{prefix}\n".encode(),
            f"{prefix}/lib/libhipblaslt.so.1.0": b"hipblaslt",
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            downloads = tmp / "downloads"
            downloads.mkdir()
            _write_archive(downloads / "blas_lib_generic.tar.xz", members)
            with self.assertRaises(IOError):
                mod.replace_host_libraries(
                    {"blas": [("hipBLASLt", "libtensilelite-host")]},
                    downloads,
                    tmp / "install",
                )


if __name__ == "__main__":
    unittest.main()
