#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import argparse
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from typing import List, Dict, Tuple
from re import Pattern


class TestNameValidator:

    KEYWORDS: Dict[str, List[str]] = {
        "test_types": ["Test", "Integration"],
        "gpu": ["Gpu"],
        "datatypes": ["Bfp16", "Fp16", "Fp32", "Fp64"],
        "shapes": ["Nhwc", "Nchw", "Ndhwc", "Ncdhw"],
    }

    FLATTENED_KEYWORDS: List[str] = [
        kw for sublist in KEYWORDS.values() for kw in sublist
    ]

    POSITIONAL_KEYWORDS: List[str] = (
        KEYWORDS["test_types"] + KEYWORDS["datatypes"] + KEYWORDS["gpu"]
    )

    FULL_NAME_RE: Pattern[str] = re.compile(
        r"^(?:(?P<prefix>[A-Z][A-Za-z0-9]*)/)?"
        r"(?P<suite>[A-Z][A-Za-z0-9]*)"
        r"\.(?P<case>(?:DISABLED_[A-Z][A-Za-z0-9]+|[A-Z][A-Za-z0-9]*))"
        r"(?:/.*)?$"
    )

    def _validate_test_case(self, case_name: str) -> List[str]:
        """
        Validate a test case name.
        Returns a list of issues found, or empty list if valid.
        """
        issues = []

        if case_name.startswith("DISABLED_"):
            case_name = case_name[9:]

        # Check for disallowed positional keywords
        found_keywords = [kw for kw in self.POSITIONAL_KEYWORDS if kw in case_name]
        if found_keywords:
            issues.append(
                f"Test case name should not contain keywords: {', '.join(found_keywords)}. These belong in the test suite name."
            )

        return issues

    def _validate_suite_structure(self, suite_name: str) -> List[str]:
        """
        Validate the structure of a test suite name.
        Returns a list of issues found, or empty list if valid.
        """
        issues = []

        prefix_part = f"({'|'.join(self.KEYWORDS['test_types'])})"
        gpu_part = f"({self.KEYWORDS['gpu'][0]})?"
        feature_part = r"(?P<feature>[A-Z][a-zA-Z0-9]*?)"
        datatypes_part = f"({'|'.join(self.KEYWORDS['datatypes'])})?"

        structure_regex = re.compile(
            f"^{prefix_part}{gpu_part}{feature_part}{datatypes_part}$"
        )

        match = structure_regex.match(suite_name)

        if not match:
            issues.append(
                "Suite name does not follow the structure: (Test|Integration)[Gpu?]FeatureName[Datatype?]"
            )
            return issues

        feature_name = match.group("feature")

        for keyword in self.POSITIONAL_KEYWORDS:
            if keyword in feature_name:
                issues.append(
                    f"Keyword '{keyword}' is misplaced and should not be in the middle of the suite name."
                )

        return issues

    def validate_test_name(self, test_name: str) -> List[str]:
        """
        Validate a single full test name.
        Returns a list of issues found, or empty list if valid
        """
        issues = []

        parsed_match = self.FULL_NAME_RE.match(test_name)
        if not parsed_match:
            issues.append(
                "Test name does not match expected PascalCase format 'TestSuite.TestCase' or 'TestSuite/Instance.TestCase' without special characters."
            )
            return issues

        prefix = parsed_match.group("prefix") or ""
        suite_name = parsed_match.group("suite")
        case_name = parsed_match.group("case")

        for keyword in self.FLATTENED_KEYWORDS:
            matches = re.findall(
                re.escape(keyword), f"{prefix}/{suite_name}.{case_name}", re.IGNORECASE
            )

            valid_matches = [m for m in matches if m == keyword or m == keyword.upper()]
            # Check capitalization
            issues.extend(
                [
                    f"Keyword '{match}' should be capitalized as '{keyword}'"
                    for match in valid_matches
                    if match != keyword
                ]
            )

            # Check duplicates
            if len(valid_matches) > 1:
                issues.append(
                    f"Keyword '{keyword}' appears more than once."
                )  # Potentially useful to make test names more concise

        issues.extend(self._validate_suite_structure(suite_name))
        issues.extend(self._validate_test_case(case_name))

        return issues

    @staticmethod
    def extract_test_names_from_ctest_json(json_path: Path) -> List[str]:
        """Extract test names from CTest JSON output."""
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            test_names = []
            if "tests" in data:
                for test in data["tests"]:
                    if "name" in test:
                        test_names.append(test["name"].split("#")[0].strip())

            return test_names
        except FileNotFoundError:
            print(f"Error: CTest JSON file not found: {json_path}", file=sys.stderr)
            return []
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in file {json_path}: {e}", file=sys.stderr)
            return []
        except Exception as e:
            print(f"Unexpected error reading CTest JSON file: {e}", file=sys.stderr)
            return []

    @staticmethod
    def extract_test_names_from_executables(
        executables_file: Path, build_dir: Path, verbose: bool = False
    ) -> List[str]:
        """Extract test names by running executables with --gtest_list_tests."""
        test_names = []

        try:
            with open(executables_file, "r") as f:
                executables = [line.strip() for line in f if line.strip()]
                if verbose:
                    print(f"Found {len(executables)} test executables to process.")
                    print(f"Executables: {executables}")
        except FileNotFoundError:
            print(
                f"Error: Executables file not found: {executables_file}",
                file=sys.stderr,
            )
            return []

        for executable in executables:
            exec_path = (build_dir / executable).resolve()
            if not exec_path.exists() or not exec_path.is_file():
                print(f"Warning: Executable not found: {exec_path}", file=sys.stderr)
                continue

            try:
                result = subprocess.run(
                    [str(exec_path), "--gtest_list_tests"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(exec_path.parent),
                )

                if result.returncode != 0:
                    print(
                        f"Warning: Failed to list tests from {executable}: {result.stderr}",
                        file=sys.stderr,
                    )
                    continue

                current_suite = None
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if not line:
                        continue

                    if line.endswith(".") and not line.endswith("..."):
                        current_suite = line[:-1]
                    elif line and current_suite:
                        test_case = line.strip()
                        test_names.append(
                            f"{current_suite}.{test_case.split("#")[0].strip()}"
                        )

            except subprocess.TimeoutExpired:
                print(f"Warning: Timeout running {executable}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Error running {executable}: {e}", file=sys.stderr)

        return test_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate test names against hipDNN rules",
        epilog="At least one of --ctest-json or --test-executables must be provided.",
    )

    input_group = parser.add_argument_group(
        "input methods", "Specify one or both methods to gather test names"
    )

    input_group.add_argument(
        "--ctest-json",
        type=Path,
        help="Path to CTest JSON output (from ctest --show-only=json-v1)",
    )

    input_group.add_argument(
        "--test-executables",
        type=Path,
        help="Path to file containing list of test executables",
    )

    input_group.add_argument(
        "--build-dir",
        type=Path,
        help="Build directory where test executables are located. Required if --test-executables is provided.",
    )

    options_group = parser.add_argument_group("options")

    options_group.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any test names are invalid",
        default=False,
    )

    options_group.add_argument(
        "--verbose",
        action="store_true",
        help="Show all test names, not just invalid ones",
        default=False,
    )

    options_group.add_argument(
        "--run-tests",
        action="store_true",
        help="Run unit tests for this validator",
        default=False,
    )

    args = parser.parse_args()

    if not args.run_tests:
        if args.test_executables and not args.build_dir:
            parser.error("--build-dir is required when using --test-executables")

        if not args.ctest_json and not args.test_executables:
            parser.error(
                "At least one of --ctest-json or --test-executables must be provided"
            )

    return args


def main() -> int:
    args = parse_args()

    if args.run_tests:
        unittest.main(argv=[sys.argv[0]], exit=True)

    validator = TestNameValidator()
    test_names = []

    if args.test_executables and args.build_dir:
        test_names.extend(
            validator.extract_test_names_from_executables(
                args.test_executables, args.build_dir, args.verbose
            )
        )

    if args.ctest_json:
        test_names.extend(validator.extract_test_names_from_ctest_json(args.ctest_json))

    if not test_names:
        print("Warning: No test names found to validate", file=sys.stderr)
        return 0

    invalid_count = 0
    results: Dict[str, List[str]] = {}

    for test_name in test_names:
        issues = validator.validate_test_name(test_name)
        results[test_name] = issues
        if issues:
            invalid_count += 1

    print(f"\nTest Name Validation Report")
    print(f"{'=' * 60}")
    print(f"Total tests found: {len(test_names)}")
    print(f"Valid test names: {len(test_names) - invalid_count}")
    print(f"Invalid test names: {invalid_count}\n")

    if invalid_count == 0:
        return 0

    print(f"{'Test Name':<50} {'Status':<10}")
    print(f"{'-' * 60}")

    for test_name, issues in sorted(results.items()):
        is_valid = len(issues) == 0
        if not is_valid or args.verbose:
            status = "[PASS]" if is_valid else "[FAIL]"
            print(f"{test_name:<50} {status:<10}")
            if issues:
                for issue in issues:
                    print(f"  → {issue}")

    print(f"\nWarning: {invalid_count} test(s) have non-conforming names")
    print(
        " - For detailed hipDNN test naming rules, see: docs/CodingStyleAndNamingGuidelines.md\n"
    )

    return 1 if args.strict else 0


class TestTestNameValidator(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = TestNameValidator()

    def test_valid_test_names_basic(self) -> None:
        """Test basic valid test names in PascalCase"""
        valid_names = [
            "TestMyClass.Something",
            "IntegrationConvolution.Forward",
            "TestBatchNorm.Backward",
            "TestExecutionPlanBuilder.Build",
        ]

        for name in valid_names:
            with self.subTest(name=name):
                issues = self.validator.validate_test_name(name)
                self.assertEqual(
                    issues, [], f"Expected {name} to be valid, but got issues: {issues}"
                )

    def test_valid_test_names_with_gpu(self) -> None:
        """Test valid test names with Gpu keyword"""
        valid_names = [
            "TestGpuConvolution.Forward",
            "IntegrationGpuBatchNorm.Backward",
            "TestGpuExecutionPlanBuilder.Build",
        ]

        for name in valid_names:
            with self.subTest(name=name):
                issues = self.validator.validate_test_name(name)
                self.assertEqual(
                    issues, [], f"Expected {name} to be valid, but got issues: {issues}"
                )

    def test_valid_test_names_with_datatypes(self) -> None:
        """Test valid test names with datatype suffixes"""
        valid_names = [
            "TestConvolutionFp32.Forward",
            "TestBatchNormFp16.Backward",
            "IntegrationConvolutionBfp16.Wrw",
            "TestGpuConvolutionFp64.Performance",
            "IntegrationGpuBatchNormFp32.Convergence",
        ]

        for name in valid_names:
            with self.subTest(name=name):
                issues = self.validator.validate_test_name(name)
                self.assertEqual(
                    issues, [], f"Expected {name} to be valid, but got issues: {issues}"
                )

    def test_valid_test_names_with_instance(self) -> None:
        """Test valid parameterized test names with instance"""
        valid_names = [
            "Temp/TestConvolution.Forward",
            "Group/IntegrationGpuBatchNormFp32.Accuracy",
            "Config/TestMyClass.Method",
        ]

        for name in valid_names:
            with self.subTest(name=name):
                issues = self.validator.validate_test_name(name)
                self.assertEqual(
                    issues, [], f"Expected {name} to be valid, but got issues: {issues}"
                )

    def test_valid_disabled_test_names(self) -> None:
        """Test valid disabled test names"""
        valid_names = [
            "TestMyClass.DISABLED_Something",
            "IntegrationGpuConvolutionFp32.DISABLED_Forward",
        ]

        for name in valid_names:
            with self.subTest(name=name):
                issues = self.validator.validate_test_name(name)
                self.assertEqual(
                    issues, [], f"Expected {name} to be valid, but got issues: {issues}"
                )

    def test_invalid_format(self) -> None:
        """Test names with invalid format"""
        invalid_names = [
            "test_my_class.test_something",  # lowercase
            "TestMyClass_Something",  # underscore instead of dot
            "TestMyClass.test_something",  # lowercase test case
            "testMyClass.Something",  # lowercase suite
            "Test-MyClass.Something",  # hyphens
            "Test My Class.Something",  # spaces
        ]

        for name in invalid_names:
            with self.subTest(name=name):
                issues = self.validator.validate_test_name(name)
                self.assertTrue(len(issues) > 0, f"Expected format error for {name}")

    def test_keywords_in_test_case(self) -> None:
        """Test that positional keywords are not allowed in test case names"""
        invalid_names = [
            "TestMyClass.TestGpuFunction",  # Gpu in test case
            "TestMyClass.TestFp32Precision",  # Fp32 in test case
            "TestMyClass.IntegrationTest",  # Integration in test case
            "TestMyClass.TestBfp16Type",  # Bfp16 in test case
        ]

        for name in invalid_names:
            with self.subTest(name=name):
                issues = self.validator.validate_test_name(name)
                self.assertTrue(
                    any("should not contain keywords" in issue for issue in issues),
                    f"Expected keyword error in test case for {name}",
                )

    def test_invalid_suite_structure(self) -> None:
        """Test suite names that don't follow the required structure"""
        invalid_names = [
            "MyClass.Something",  # Missing Test/Integration prefix
            "GpuTestConvolution.Forward",  # Gpu must come after Test/Integration
            "TestConvolutionGpu.Forward",  # Gpu must come before feature name
            "ConvolutionTest.Forward",  # Test must be at the beginning
            "TestFp32Convolution.Forward",  # Datatype must be at the end
        ]

        for name in invalid_names:
            with self.subTest(name=name):
                issues = self.validator.validate_test_name(name)
                self.assertTrue(len(issues) > 0, f"Expected structure error for {name}")

    def test_keyword_misplacement(self) -> None:
        """Test keywords misplaced in the feature name"""
        invalid_names = [
            "TestConvolutionGpuPlannerFp32.Forward",  # Gpu in middle
            "TestConvolutionTestPlannerFp32.Forward",  # Test in middle
            "IntegrationConvolutionFp32Planner.Forward",  # Fp32 in middle
        ]

        for name in invalid_names:
            with self.subTest(name=name):
                issues = self.validator.validate_test_name(name)
                self.assertTrue(
                    any("misplaced" in issue for issue in issues),
                    f"Expected misplacement error for {name}",
                )

    def test_keyword_capitalization(self) -> None:
        """Test incorrect keyword capitalization"""
        invalid_names = [
            "TestGPUConvolution.Forward",  # GPU instead of Gpu
            "TestConvolutionFP32.Forward",  # FP32 instead of Fp32
            "testConvolution.Forward",  # test instead of Test
            "INTEGRATIONConvolution.Forward",  # INTEGRATION instead of Integration
        ]

        for name in invalid_names:
            with self.subTest(name=name):
                issues = self.validator.validate_test_name(name)
                self.assertTrue(
                    len(issues) > 0, f"Expected capitalization issues for {name}"
                )

    def test_keyword_duplicates(self) -> None:
        """Test duplicate keywords"""
        invalid_names = [
            "TestTestConvolution.Forward",  # Duplicate Test
            "TestGpuConvolutionGpu.Forward",  # Duplicate Gpu
        ]

        for name in invalid_names:
            with self.subTest(name=name):
                issues = self.validator.validate_test_name(name)
                self.assertTrue(
                    len(issues) > 0, f"Expected issues for duplicate keywords in {name}"
                )

    def test_complex_valid_names(self) -> None:
        """Test more complex but valid test names"""
        valid_names = [
            "IntegrationGpuConvolutionPlannerNchwFp32.Forward",
            "TestActivationKernelNchwFp32.Relu",
            "TestExecutionPlanBuilderFp32.Optimization",
            "IntegrationGraphFusion.MultipleOps",
            "TestConvolutionHeuristicsFp32.Performance",
            "TestConvolutionHeuristics.Accuracy",
        ]

        for name in valid_names:
            with self.subTest(name=name):
                issues = self.validator.validate_test_name(name)
                self.assertEqual(
                    issues, [], f"Expected {name} to be valid, but got issues: {issues}"
                )


if __name__ == "__main__":
    sys.exit(main())
