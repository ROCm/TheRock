# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import fnmatch
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_rocrtst_csan import MAX_CASE, calibrate, run_suite, split_command


def selected(command, cases):
    expression = next(
        x.split("=", 1)[1] for x in command if x.startswith("--gtest_filter=")
    )
    positive, _, negative = expression.partition("-")
    return {
        case
        for case in cases
        if any(fnmatch.fnmatchcase(case, p) for p in positive.split(":"))
        and not any(fnmatch.fnmatchcase(case, p) for p in negative.split(":"))
    }


class RocrtstPartitionTest(unittest.TestCase):
    def test_calibration_compiles_with_csan_and_preserves_the_measurement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch(
                "run_rocrtst_csan.subprocess.run",
                side_effect=[
                    subprocess.CompletedProcess([], 0),
                    subprocess.CompletedProcess([], 0, "actual rate\n", ""),
                ],
            ) as run:
                calibrate(root, root)
            compile_command = run.call_args_list[0].args[0]
            self.assertIn("-fsanitize=concurrency", compile_command)
            self.assertIn("-lhsa-runtime64", compile_command)
            self.assertEqual((root / "calibration.log").read_text(), "actual rate\n")

    @unittest.skipUnless(shutil.which("c++"), "host C++ compiler required")
    def test_calibration_source_balances_real_api_calls_in_cpu_fixture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "hsa").mkdir()
            (root / "hsa" / "hsa.h").write_text(
                "using hsa_status_t = unsigned;\nconstexpr unsigned HSA_STATUS_SUCCESS = 0;\n"
                'extern "C" unsigned hsa_init();\nextern "C" unsigned hsa_shut_down();\n'
            )
            fake = root / "fake.cpp"
            fake.write_text(
                "#include <cstdlib>\nstatic unsigned refs = 0, calls = 0;\n"
                'extern "C" unsigned hsa_init() { ++refs; ++calls; return 0; }\n'
                'extern "C" unsigned hsa_shut_down() {\n'
                "if (!refs) std::abort();\n"
                "if (!--refs && calls != 100001) std::abort();\nreturn 0; }\n"
            )
            source = Path(__file__).parent.parent / "rocrtst_refcount_probe.cpp"
            binary = root / "probe"
            subprocess.run(
                [
                    shutil.which("c++"),
                    "-O2",
                    "-I" + str(root),
                    str(source),
                    str(fake),
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            result = subprocess.run(
                [str(binary)], check=True, capture_output=True, text=True
            )
            self.assertIn("100000 init/shutdown references", result.stdout)
            self.assertIn("INT32_MAX linear estimate=", result.stdout)

    def test_partitions_preserve_exact_coverage_and_existing_exclusions(self):
        cases = {
            MAX_CASE,
            "rocrtstFunc.Reference_Count",
            "rocrtstFunc.Memory_Max_Mem",
            "rocrtstStress.Concurrent",
            "rocrtstNeg.Invalid",
        }
        for expression in (
            MAX_CASE
            + ":rocrtstFunc.Reference_Count:rocrtstStress.*-rocrtstFunc.Memory_Max_Mem",
            "rocrtstFunc.*:rocrtstStress.*-rocrtstFunc.Memory_Max_Mem",
        ):
            original = ["rocrtst64", "--verbose", "--gtest_filter=" + expression]
            regular = split_command(original, "regular")
            maximum = split_command(original, "max")
            self.assertEqual(
                selected(regular, cases) | selected(maximum, cases),
                selected(original, cases),
            )
            self.assertFalse(selected(regular, cases) & selected(maximum, cases))
            self.assertEqual(maximum[:2], original[:2])
            self.assertEqual(selected(maximum, cases), {MAX_CASE})

    def test_category_without_max_fails_instead_of_inventing_coverage(self):
        with self.assertRaisesRegex(ValueError, "does not contain"):
            split_command(
                ["rocrtst64", "--gtest_filter=rocrtstFunc.Reference_Count"], "max"
            )

    def test_original_category_command_exit_code_and_environment_are_retained(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            test_dir = root / "bin" / "rocrtst"
            test_dir.mkdir(parents=True)
            runner = test_dir / "fake.py"
            runner.write_text(
                "import pathlib, sys\npathlib.Path('ran').write_text(str(sys.argv))\nsys.exit(9)\n"
            )
            spec = {
                "tests": [
                    {
                        "command": [
                            sys.executable,
                            str(runner),
                            "--gtest_filter="
                            + MAX_CASE
                            + ":rocrtstFunc.Reference_Count",
                        ],
                        "properties": [
                            {"name": "WORKING_DIRECTORY", "value": str(test_dir)}
                        ],
                    }
                ]
            }
            real_run = subprocess.run

            def execute(command, **kwargs):
                if command[0] == "ctest":
                    return subprocess.CompletedProcess(command, 0, json.dumps(spec), "")
                return real_run(command, **kwargs)

            with patch("run_rocrtst_csan.subprocess.run", side_effect=execute), patch(
                "run_rocrtst_csan.calibrate"
            ) as calibration:
                code = run_suite(root, root / "diagnostics", "comprehensive", "regular")
            self.assertEqual(code, 9)
            calibration.assert_not_called()
            self.assertIn("Reference_Count", (test_dir / "ran").read_text())
            self.assertTrue((root / "diagnostics" / "command.json").is_file())

    def test_max_calibrates_before_running_the_unmodified_case(self):
        calls = []
        spec = {
            "tests": [
                {
                    "command": [
                        "rocrtst64",
                        "--gtest_filter=" + MAX_CASE + ":rocrtstFunc.Reference_Count",
                    ]
                }
            ]
        }

        def execute(command, **kwargs):
            if command[0] == "ctest":
                return subprocess.CompletedProcess(command, 0, json.dumps(spec), "")
            calls.append(command)
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "run_rocrtst_csan.subprocess.run", side_effect=execute
        ), patch(
            "run_rocrtst_csan.calibrate",
            side_effect=lambda *args: calls.append("calibration"),
        ):
            code = run_suite(Path(temp_dir), Path(temp_dir) / "logs", "full", "max")
        self.assertEqual(code, 0)
        self.assertEqual(
            calls, ["calibration", ["rocrtst64", "--gtest_filter=" + MAX_CASE]]
        )


if __name__ == "__main__":
    unittest.main()
