# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import platform
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import msgpack
import pyzstd

# Add repo root to PYTHONPATH
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import merge_coverage_report

ROCRAND_KEY = "math-libs/rocRAND/stage/lib/librocrand.so.1.1"
HIPRAND_KEY = "math-libs/hipRAND/stage/lib/libhiprand.so.1.1"


def is_windows() -> bool:
    return platform.system() == "Windows"


def tiny_elf(*section_names: str) -> bytes:
    """A 64-bit little-endian AMDGPU ELF image with empty sections of these names."""
    names = ["", *section_names, ".shstrtab"]
    strtab = b"".join(name.encode() + b"\0" for name in names)
    name_offsets = [sum(len(n) + 1 for n in names[:i]) for i in range(len(names))]
    shoff = 64 + len(strtab)
    header = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    header += struct.pack(
        "<HHIQQQIHHHHHH",
        3,
        224,
        1,
        0,
        0,
        shoff,
        0,
        64,
        0,
        0,
        64,
        len(names),
        len(names) - 1,
    )
    sections = b"\0" * 64
    for index in range(1, len(names)):
        is_strtab = index == len(names) - 1
        sections += struct.pack(
            "<IIQQQQIIQQ",
            name_offsets[index],
            3 if is_strtab else 1,
            0,
            0,
            64 if is_strtab else 0,
            len(strtab) if is_strtab else 0,
            0,
            0,
            1,
            0,
        )
    return header + strtab + sections


def write_kpack(path: Path, kernels: dict) -> None:
    """Writes {(key, arch): code object} in rocm_kpack's zstd-per-kernel layout."""
    toc, frames = {}, []
    for ordinal, ((key, arch), data) in enumerate(kernels.items()):
        toc.setdefault(key, {})[arch] = {"type": "hsaco", "ordinal": ordinal}
        frames.append(pyzstd.compress(data))
    blob = struct.pack("<I", len(frames)) + b"".join(
        struct.pack("<I", len(f)) + f for f in frames
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sIQ", b"KPAK", 1, 0) + b"\0" * 48 + blob)
        toc_offset = f.tell()
        msgpack.pack(
            {
                "format_version": 1,
                "toc": toc,
                "compression_scheme": "zstd-per-kernel",
                "zstd_offset": 64,
                "zstd_size": len(blob),
            },
            f,
            use_bin_type=True,
        )
        f.seek(8)
        f.write(struct.pack("<Q", toc_offset))


class TempDirTestBase(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.root = Path(self._temp_dir.name)

    def touch(self, relative_path: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path


class FindLlvmToolTest(TempDirTestBase):
    def test_prefers_the_rocm_distribution_copy(self):
        bin_dir = self.root / "lib" / "llvm" / "bin"
        bin_dir.mkdir(parents=True)
        expected = bin_dir / f"llvm-cov{merge_coverage_report.EXECUTABLE_SUFFIX}"
        expected.touch()

        with mock.patch("shutil.which", return_value="/usr/bin/llvm-cov"):
            self.assertEqual(
                merge_coverage_report.find_llvm_tool(bin_dir, "llvm-cov"), expected
            )

    def test_falls_back_to_path(self):
        bin_dir = self.root / "empty"
        bin_dir.mkdir()

        with mock.patch("shutil.which", return_value="/usr/bin/llvm-cov"):
            self.assertEqual(
                merge_coverage_report.find_llvm_tool(bin_dir, "llvm-cov"),
                Path("/usr/bin/llvm-cov"),
            )

    def test_raises_when_the_tool_is_missing_everywhere(self):
        bin_dir = self.root / "empty"
        bin_dir.mkdir()

        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(FileNotFoundError):
                merge_coverage_report.find_llvm_tool(bin_dir, "llvm-cov")


class FindProfrawFilesTest(TempDirTestBase):
    def test_searches_recursively_and_ignores_other_files(self):
        self.touch("shard0/a-1234.profraw")
        self.touch("shard1/nested/b-5678.profraw")
        self.touch("shard1/test-output.xml")

        found = merge_coverage_report.find_profraw_files(self.root)

        self.assertEqual([f.name for f in found], ["a-1234.profraw", "b-5678.profraw"])

    def test_empty_directory_yields_nothing(self):
        self.assertEqual(merge_coverage_report.find_profraw_files(self.root), [])


class ResolveObjectsTest(TempDirTestBase):
    @unittest.skipIf(is_windows(), "symlinks require elevated privileges on Windows")
    def test_versioned_symlinks_collapse_to_one_object(self):
        real = self.touch("lib/libhiprand.so.1.0")
        (self.root / "lib" / "libhiprand.so").symlink_to(real)
        (self.root / "lib" / "libhiprand.so.1").symlink_to(real)

        objects = merge_coverage_report.resolve_objects(
            self.root, ["lib/libhiprand.so*"]
        )

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].resolve(), real.resolve())

    def test_multiple_globs_are_combined(self):
        self.touch("lib/libhiprand.so")
        self.touch("bin/hiprand_tool")

        objects = merge_coverage_report.resolve_objects(
            self.root, ["lib/libhiprand.so*", "bin/hiprand_*"]
        )

        self.assertEqual(
            sorted(o.name for o in objects), ["hiprand_tool", "libhiprand.so"]
        )

    def test_directories_are_not_treated_as_objects(self):
        (self.root / "lib" / "libhiprand.so.d").mkdir(parents=True)

        self.assertEqual(
            merge_coverage_report.resolve_objects(self.root, ["lib/libhiprand.so*"]), []
        )

    def test_unmatched_glob_yields_nothing(self):
        self.assertEqual(
            merge_coverage_report.resolve_objects(self.root, ["lib/nothing*"]), []
        )

    def test_bang_glob_takes_matches_back_out(self):
        self.touch("bin/test_basic")
        self.touch("bin/test_hipcub_basic")

        objects = merge_coverage_report.resolve_objects(
            self.root, ["bin/test_*", "!bin/test_hipcub_*"]
        )

        self.assertEqual([o.name for o in objects], ["test_basic"])


class CountProfiledFunctionsTest(unittest.TestCase):
    def _count(self, stdout: str):
        with mock.patch("subprocess.run", return_value=mock.Mock(stdout=stdout)) as run:
            count = merge_coverage_report.count_profiled_functions(
                Path("/llvm/llvm-profdata"), Path("/tmp/c.profdata")
            )
        self.assertEqual(run.call_args.args[0][1], "show")
        return count

    def test_reads_the_summary_line(self):
        self.assertEqual(
            self._count("Instrumentation level: Front-end\nTotal functions: 1305\n"),
            1305,
        )

    def test_profile_without_counters_reads_as_zero(self):
        self.assertEqual(self._count("Total functions: 0\n"), 0)

    def test_unrecognized_output_skips_the_check(self):
        self.assertIsNone(self._count("something else entirely\n"))


class CountLcovLinesTest(TempDirTestBase):
    def test_sums_over_every_file(self):
        lcov = self.root / "coverage.info"
        lcov.write_text(
            "SF:a.cpp\nDA:1,1\nLF:10\nLH:4\nend_of_record\n"
            "SF:b.cpp\nDA:1,0\nLF:5\nLH:0\nend_of_record\n"
        )

        self.assertEqual(merge_coverage_report.count_lcov_lines(lcov), (15, 4))


class CommandConstructionTest(TempDirTestBase):
    def test_merge_passes_every_profraw_file(self):
        profraw_files = [self.touch("a.profraw"), self.touch("b.profraw")]
        output = self.root / "out" / "coverage.profdata"
        llvm_profdata = Path("/llvm/llvm-profdata")

        with mock.patch("subprocess.run") as run:
            merge_coverage_report.merge_profraw(llvm_profdata, profraw_files, output)

        command = run.call_args.args[0]
        self.assertEqual(command[:3], [str(llvm_profdata), "merge", "-sparse"])
        self.assertEqual(command[-2:], [str(profraw_files[0]), str(profraw_files[1])])
        self.assertTrue(output.parent.is_dir())

    def test_export_passes_the_first_object_positionally(self):
        objects = [self.touch("lib/a.so"), self.touch("lib/b.so")]
        output = self.root / "out" / "coverage.info"
        llvm_cov = Path("/llvm/llvm-cov")
        profdata = Path("/tmp/coverage.profdata")

        with mock.patch("subprocess.run") as run:
            merge_coverage_report.export_lcov(llvm_cov, profdata, objects, output)

        command = run.call_args.args[0]
        self.assertEqual(command[:3], [str(llvm_cov), "export", str(objects[0])])
        self.assertIn("-object", command)
        self.assertEqual(command[-2:], [f"-instr-profile={profdata}", "--format=lcov"])

    def test_summary_reports_and_captures_the_table(self):
        objects = [self.touch("lib/a.so")]
        output = self.root / "out" / "coverage_summary.txt"
        table = "Filename  Cover\nTOTAL     77.78%\n"

        with mock.patch("subprocess.run", return_value=mock.Mock(stdout=table)) as run:
            merge_coverage_report.write_summary(
                Path("/llvm/llvm-cov"), Path("/tmp/c.profdata"), objects, output
            )

        self.assertEqual(run.call_args.args[0][1], "report")
        self.assertEqual(output.read_text(), table)

    def test_html_renders_into_a_directory(self):
        objects = [self.touch("lib/a.so")]
        output_dir = self.root / "out" / "html"

        with mock.patch("subprocess.run") as run:
            merge_coverage_report.write_html(
                Path("/llvm/llvm-cov"),
                Path("/tmp/c.profdata"),
                objects,
                output_dir,
                project_title="hiprand",
                demangler=Path("/llvm/llvm-cxxfilt"),
            )

        command = run.call_args.args[0]
        self.assertEqual(command[1], "show")
        self.assertIn("--format=html", command)
        self.assertIn(f"-output-dir={output_dir}", command)
        self.assertIn("--project-title=hiprand", command)
        self.assertIn(f"-Xdemangler={Path('/llvm/llvm-cxxfilt')}", command)

    def test_path_equivalence_reaches_every_rendering(self):
        objects = [self.touch("lib/a.so")]
        remap = "/__w/TheRock/TheRock,/home/runner/work"

        for build in (
            lambda: merge_coverage_report.export_lcov(
                Path("/llvm/llvm-cov"),
                Path("/tmp/c.profdata"),
                objects,
                self.root / "out" / "coverage.info",
                remap,
            ),
            lambda: merge_coverage_report.write_html(
                Path("/llvm/llvm-cov"),
                Path("/tmp/c.profdata"),
                objects,
                self.root / "out" / "html",
                remap,
            ),
        ):
            with mock.patch("subprocess.run") as run:
                build()
            self.assertIn(f"-path-equivalence={remap}", run.call_args.args[0])


class MainTest(TempDirTestBase):
    def _argv(self, *extra: str) -> list[str]:
        return [
            "--profraw-dir",
            os.fspath(self.root / "profraw"),
            "--rocm-dir",
            os.fspath(self.root / "rocm"),
            "--object-globs",
            "lib/libhiprand.so*",
            *extra,
        ]

    def test_missing_profiles_fail_by_default(self):
        (self.root / "profraw").mkdir()

        self.assertEqual(merge_coverage_report.main(self._argv()), 1)

    def test_missing_profiles_are_tolerated_with_allow_empty(self):
        (self.root / "profraw").mkdir()

        self.assertEqual(merge_coverage_report.main(self._argv("--allow-empty")), 0)

    def test_unmatched_object_globs_fail(self):
        self.touch("profraw/a.profraw")
        llvm_bin_dir = self.root / "rocm" / "lib" / "llvm" / "bin"
        llvm_bin_dir.mkdir(parents=True)
        for tool in ("llvm-profdata", "llvm-cov"):
            (llvm_bin_dir / f"{tool}{merge_coverage_report.EXECUTABLE_SUFFIX}").touch()

        self.assertEqual(merge_coverage_report.main(self._argv()), 1)

    def test_happy_path_invokes_both_llvm_tools(self):
        self.touch("profraw/shard0/a.profraw")
        self.touch("rocm/lib/libhiprand.so")
        llvm_bin_dir = self.root / "rocm" / "lib" / "llvm" / "bin"
        llvm_bin_dir.mkdir(parents=True)
        for tool in ("llvm-profdata", "llvm-cov"):
            (llvm_bin_dir / f"{tool}{merge_coverage_report.EXECUTABLE_SUFFIX}").touch()

        with mock.patch(
            "subprocess.run", return_value=mock.Mock(stdout="Total functions: 3\n")
        ) as run:
            exit_code = merge_coverage_report.main(
                self._argv(
                    "--profdata-output",
                    os.fspath(self.root / "out" / "coverage.profdata"),
                    "--lcov-output",
                    os.fspath(self.root / "out" / "coverage.info"),
                )
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(run.call_count, 3)
        invoked = [Path(call.args[0][0]).name for call in run.call_args_list]
        self.assertEqual(
            invoked,
            [
                f"llvm-profdata{merge_coverage_report.EXECUTABLE_SUFFIX}",
                f"llvm-profdata{merge_coverage_report.EXECUTABLE_SUFFIX}",
                f"llvm-cov{merge_coverage_report.EXECUTABLE_SUFFIX}",
            ],
        )

    def test_summary_and_html_are_opt_in(self):
        self.touch("profraw/shard0/a.profraw")
        self.touch("rocm/lib/libhiprand.so")
        llvm_bin_dir = self.root / "rocm" / "lib" / "llvm" / "bin"
        llvm_bin_dir.mkdir(parents=True)
        for tool in ("llvm-profdata", "llvm-cov"):
            (llvm_bin_dir / f"{tool}{merge_coverage_report.EXECUTABLE_SUFFIX}").touch()

        with mock.patch("subprocess.run", return_value=mock.Mock(stdout="")) as run:
            exit_code = merge_coverage_report.main(
                self._argv(
                    "--profdata-output",
                    os.fspath(self.root / "out" / "coverage.profdata"),
                    "--lcov-output",
                    os.fspath(self.root / "out" / "coverage.info"),
                    "--summary-output",
                    os.fspath(self.root / "out" / "coverage_summary.txt"),
                    "--html-output",
                    os.fspath(self.root / "out" / "html"),
                )
            )

        self.assertEqual(exit_code, 0)
        # merge and check the index, then export, report and show off it.
        subcommands = [call.args[0][1] for call in run.call_args_list]
        self.assertEqual(subcommands, ["merge", "show", "export", "report", "show"])


class MainRejectsEmptyReportsTest(TempDirTestBase):
    """Every way the pipeline breaks upstream still leaves profraw files behind."""

    def setUp(self):
        super().setUp()
        self.touch("profraw/shard0/a.profraw")
        self.touch("rocm/lib/libhiprand.so")
        llvm_bin_dir = self.root / "rocm" / "lib" / "llvm" / "bin"
        llvm_bin_dir.mkdir(parents=True)
        for tool in ("llvm-profdata", "llvm-cov"):
            (llvm_bin_dir / f"{tool}{merge_coverage_report.EXECUTABLE_SUFFIX}").touch()

    def _main(
        self, *extra: str, functions: int = 3, lcov: str = "", export_fails=False
    ):
        def fake_run(command, **kwargs):
            tool, subcommand = Path(command[0]).stem, command[1]
            if (tool, subcommand) == ("llvm-profdata", "show"):
                return mock.Mock(stdout=f"Total functions: {functions}\n")
            if subcommand == "export":
                if export_fails:
                    raise subprocess.CalledProcessError(1, command)
                kwargs["stdout"].write(lcov)
            return mock.Mock(stdout="")

        with mock.patch("subprocess.run", side_effect=fake_run) as run:
            exit_code = merge_coverage_report.main(
                [
                    "--profraw-dir",
                    os.fspath(self.root / "profraw"),
                    "--rocm-dir",
                    os.fspath(self.root / "rocm"),
                    "--object-globs",
                    "lib/libhiprand.so*",
                    "--profdata-output",
                    os.fspath(self.root / "out" / "coverage.profdata"),
                    "--lcov-output",
                    os.fspath(self.root / "out" / "coverage.info"),
                    *extra,
                ]
            )
        return exit_code, [call.args[0][1] for call in run.call_args_list]

    def test_profiles_without_counters_fail_before_export(self):
        # What an uninstrumented library writes: a header and nothing else.
        exit_code, subcommands = self._main(functions=0)

        self.assertEqual(exit_code, 1)
        self.assertEqual(subcommands, ["merge", "show"])

    def test_profiles_without_counters_are_tolerated_with_allow_empty(self):
        exit_code, _ = self._main("--allow-empty", functions=0)

        self.assertEqual(exit_code, 0)

    def test_objects_without_a_coverage_mapping_fail(self):
        exit_code, _ = self._main(export_fails=True)

        self.assertEqual(exit_code, 1)

    def test_report_that_never_reaches_the_objects_fails(self):
        exit_code, subcommands = self._main(
            "--summary-output",
            os.fspath(self.root / "out" / "coverage_summary.txt"),
            lcov="SF:a.cpp\nLF:10\nLH:0\nend_of_record\n",
        )

        self.assertEqual(exit_code, 1)
        # The summary is still written, so the job page shows the zero.
        self.assertEqual(subcommands, ["merge", "show", "export", "report"])

    def test_report_with_any_line_hit_passes(self):
        exit_code, _ = self._main(lcov="SF:a.cpp\nLF:10\nLH:1\nend_of_record\n")

        self.assertEqual(exit_code, 0)


class ElfSectionNamesTest(unittest.TestCase):
    def test_lists_every_section(self):
        names = merge_coverage_report.elf_section_names(
            tiny_elf("__llvm_covfun", "__llvm_covmap", ".text")
        )

        self.assertEqual(
            names, {"", "__llvm_covfun", "__llvm_covmap", ".text", ".shstrtab"}
        )

    def test_anything_but_an_elf_image_has_no_sections(self):
        for image in (b"", b"KPAK" + b"\0" * 60, tiny_elf(".text")[:70]):
            with self.subTest(size=len(image)):
                self.assertEqual(merge_coverage_report.elf_section_names(image), set())


class CountDeviceProfilesTest(unittest.TestCase):
    def test_counts_profiles_named_after_a_gpu_target(self):
        names = [
            "rocrand-shard1-123-456.profraw",
            "gfx942_sramecc+_xnack-.0.rocrand-shard1-123-456.profraw",
            "gfx90a.rocrand-shard1-123-456.profraw",
        ]

        self.assertEqual(
            merge_coverage_report.count_device_profiles([Path(n) for n in names]), 2
        )


class ExtractDeviceObjectsTest(TempDirTestBase):
    @unittest.skipIf(is_windows(), "symlinks require elevated privileges on Windows")
    def test_takes_the_objects_instrumented_code_objects_only(self):
        self.touch("rocm/lib/librocrand.so.1.1")
        (self.root / "rocm/lib/librocrand.so").symlink_to("librocrand.so.1.1")
        instrumented = tiny_elf("__llvm_covfun", "__llvm_covmap")
        write_kpack(
            self.root / "rocm/.kpack/rand_lib_gfx942.kpack",
            {
                (f"{ROCRAND_KEY}#0", "gfx942"): instrumented,
                # A translation unit with no instrumented device code.
                (f"{ROCRAND_KEY}#1", "gfx942"): tiny_elf(".text"),
                # Shares the archive, but is not what the report is about.
                (f"{HIPRAND_KEY}#0", "gfx942"): tiny_elf("__llvm_covfun"),
            },
        )
        objects = merge_coverage_report.resolve_objects(
            self.root / "rocm", ["lib/librocrand.so*"]
        )

        extracted = merge_coverage_report.extract_device_objects(
            self.root / "rocm", objects, self.root / "device-code"
        )

        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0].read_bytes(), instrumented)
        self.assertNotIn("#", extracted[0].name)

    def test_no_kpack_archives_yield_nothing(self):
        self.touch("rocm/lib/librocrand.so.1.1")

        self.assertEqual(
            merge_coverage_report.extract_device_objects(
                self.root / "rocm",
                [self.root / "rocm/lib/librocrand.so.1.1"],
                self.root / "device-code",
            ),
            [],
        )


class MainDeviceCodeTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        self.touch("profraw/shard1/rocrand-shard1-1-2.profraw")
        self.touch("rocm/lib/librocrand.so.1.1")
        llvm_bin_dir = self.root / "rocm" / "lib" / "llvm" / "bin"
        llvm_bin_dir.mkdir(parents=True)
        for tool in ("llvm-profdata", "llvm-cov"):
            (llvm_bin_dir / f"{tool}{merge_coverage_report.EXECUTABLE_SUFFIX}").touch()

    def _main(self, *extra: str):
        def fake_run(command, **kwargs):
            tool, subcommand = Path(command[0]).stem, command[1]
            if (tool, subcommand) == ("llvm-profdata", "show"):
                return mock.Mock(stdout="Total functions: 3\n")
            if subcommand == "export":
                kwargs["stdout"].write("SF:a.cpp\nLF:10\nLH:1\nend_of_record\n")
            return mock.Mock(stdout="")

        with mock.patch("subprocess.run", side_effect=fake_run) as run:
            exit_code = merge_coverage_report.main(
                [
                    "--profraw-dir",
                    os.fspath(self.root / "profraw"),
                    "--rocm-dir",
                    os.fspath(self.root / "rocm"),
                    "--object-globs",
                    "lib/librocrand.so*",
                    "--profdata-output",
                    os.fspath(self.root / "out" / "coverage.profdata"),
                    "--lcov-output",
                    os.fspath(self.root / "out" / "coverage.info"),
                    "--device-code",
                    "--device-code-dir",
                    os.fspath(self.root / "out" / "device-code"),
                    *extra,
                ]
            )
        exports = [c.args[0] for c in run.call_args_list if c.args[0][1] == "export"]
        return exit_code, exports

    def _add_device_code(self):
        write_kpack(
            self.root / "rocm/.kpack/rand_lib_gfx942.kpack",
            {(f"{ROCRAND_KEY}#0", "gfx942"): tiny_elf("__llvm_covfun")},
        )

    def _add_device_profile(self):
        self.touch("profraw/shard1/gfx942_sramecc+_xnack-.0.rocrand-shard1-1-2.profraw")

    def test_device_code_objects_reach_llvm_cov(self):
        self._add_device_code()
        self._add_device_profile()

        exit_code, (export,) = self._main()

        self.assertEqual(exit_code, 0)
        # The host library stays the positional object.
        self.assertEqual(Path(export[2]).name, "librocrand.so.1.1")
        device_objects = [export[i + 1] for i, a in enumerate(export) if a == "-object"]
        self.assertEqual(len(device_objects), 1)
        self.assertTrue(device_objects[0].endswith(".co"))

    def test_no_instrumented_device_code_fails(self):
        self._add_device_profile()

        exit_code, exports = self._main()

        self.assertEqual(exit_code, 1)
        self.assertEqual(exports, [])

    def test_no_device_side_profile_fails(self):
        self._add_device_code()

        exit_code, exports = self._main()

        self.assertEqual(exit_code, 1)
        self.assertEqual(exports, [])

    def test_allow_empty_falls_back_to_host_code(self):
        self._add_device_code()

        exit_code, (export,) = self._main("--allow-empty")

        self.assertEqual(exit_code, 0)
        self.assertNotIn("-object", export)


if __name__ == "__main__":
    unittest.main()
