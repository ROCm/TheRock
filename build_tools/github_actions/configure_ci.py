#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


"""Configures metadata for a CI workflow run.

----------
| Inputs |
----------

  Environment variables (for all triggers):
  * GITHUB_EVENT_NAME    : GitHub event name, e.g. pull_request.
  * GITHUB_OUTPUT        : path to write workflow output variables.
  * GITHUB_STEP_SUMMARY  : path to write workflow summary output.
  * INPUT_LINUX_AMDGPU_FAMILIES (optional): Comma-separated string of Linux AMD GPU families
  * LINUX_TEST_LABELS (optional): Comma-separated list of test labels to test
  * LINUX_USE_PREBUILT_ARTIFACTS (optional): If enabled, CI will only run Linux tests
  * INPUT_WINDOWS_AMDGPU_FAMILIES (optional): Comma-separated string of Windows AMD GPU families
  * WINDOWS_TEST_LABELS (optional): Comma-separated list of test labels to test
  * WINDOWS_USE_PREBUILT_ARTIFACTS (optional): If enabled, CI will only run Windows tests
  * BRANCH_NAME (optional): The branch name
  * BUILD_VARIANT (optional): The build variant to run (ex: release, asan, tsan)

  Environment variables (for pull requests):
  * PR_LABELS (optional) : JSON list of PR label names.
  * BASE_REF  (required) : base commit SHA of the PR.

  Local git history with at least fetch-depth of 2 for file diffing.

-----------
| Outputs |
-----------

  Written to GITHUB_OUTPUT:
  * linux_amdgpu_families : List of valid Linux AMD GPU families to execute build and test jobs
  * linux_test_labels : List of test names to run on Linux, optionally filtered by PR labels.
  * windows_amdgpu_families : List of valid Windows AMD GPU families to execute build and test jobs
  * windows_test_labels : List of test names to run on Windows, optionally filtered by PR labels.
  * enable_build_jobs: If true, builds will be enabled
  * test_type: The type of test that component tests will run (i.e. quick, full)
  * run_functional_tests: If true, functional tests will be enabled (nightly/scheduled builds)

  Written to GITHUB_STEP_SUMMARY:
  * Human-readable summary for most contributors

  Written to stdout/stderr:
  * Detailed information for CI maintainers
"""

import json
import os
from pathlib import Path
import random
import sys
from typing import Iterable, List, Optional
import string
from amdgpu_family_matrix import (
    all_build_variants,
    get_all_families_for_trigger_types,
)
from fetch_test_configurations import test_matrix, functional_matrix

from configure_ci_path_filters import (
    get_git_modified_paths,
    get_git_submodule_paths,
    is_ci_run_required,
)
from github_actions_api import *

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent.parent

# --------------------------------------------------------------------------- #
# Matrix creation logic based on PR, push, or workflow_dispatch
# --------------------------------------------------------------------------- #


def get_pr_labels(args) -> List[str]:
    """Gets a list of labels applied to a pull request."""
    data = json.loads(args.get("pr_labels", "{}"))
    labels = []
    for label in data.get("labels", []):
        labels.append(label["name"])
    return labels


def get_workflow_dispatch_additional_label_options(args) -> List[str]:
    """Gets a list of additional label options from workflow_dispatch."""
    additional_label_options = args.get(
        "workflow_dispatch_additional_label_options", ""
    )
    if additional_label_options:
        return [
            label.strip()
            for label in additional_label_options.split(",")
            if label.strip()
        ]
    return []


def filter_known_names(
    requested_names: List[str], name_type: str, target_matrix=None
) -> List[str]:
    """Filters a requested names list down to known names.

    Args:
        requested_names: List of names to filter
        name_type: Type of name ('target' or 'test')
        target_matrix: For 'target' type, the specific family matrix to use. Required for 'target' type.
    """
    if name_type == "target":
        assert (
            target_matrix is not None
        ), "target_matrix must be provided for 'target' name_type"
        known_references = {"target": target_matrix}
    else:
        # Merge test_matrix and functional_matrix so that functional test
        # names/labels will be recognised.
        combined_test_matrix = {**test_matrix, **functional_matrix}
        known_references = {"test": combined_test_matrix}

    filtered_names = []
    if name_type not in known_references:
        print(f"WARNING: unknown name_type '{name_type}'")
        return filtered_names
    for name in requested_names:
        # Standardize on lowercase names.
        # This helps prevent potential user-input errors.
        name = name.lower()

        if name in known_references[name_type]:
            filtered_names.append(name)
        else:
            print(
                f"WARNING: unknown {name_type} name '{name}' not found in matrix:\n{known_references[name_type]}"
            )

    return filtered_names


def determine_long_lived_branch(branch_name: str) -> bool:
    # For long-lived branches (main, releases) we want to run both presubmit and postsubmit jobs on push,
    # instead of just presubmit jobs (as for other branches)
    is_long_lived_branch = False
    # Let's differentiate between full/complete matches and prefix matches for long-lived branches
    long_lived_full_match = ["main"]
    long_lived_prefix_match = ["release/therock-"]
    if branch_name in long_lived_full_match or any(
        branch_name.startswith(prefix) for prefix in long_lived_prefix_match
    ):
        is_long_lived_branch = True

    return is_long_lived_branch


def matrix_generator(
    is_pull_request=False,
    is_workflow_dispatch=False,
    is_push=False,
    is_schedule=False,
    base_args={},
    families={},
    platform="linux",
):
    """
    Generates a matrix of "family" and "test-runs-on" parameters based on the workflow inputs.
    Second return value is a list of test names to run, if any.
    """

    # Select target names based on inputs. Targets will be filtered by platform afterwards.
    selected_target_names = []
    # Select only test names based on label inputs, if applied. If no test labels apply, use default logic.
    selected_test_names = []

    branch_name = base_args.get("branch_name", "")
    # For long-lived branches (main, releases) we want to run both presubmit and postsubmit jobs on push,
    # instead of just presubmit jobs (as for other branches)
    is_long_lived_branch = determine_long_lived_branch(branch_name)

    print(f"* {branch_name} is considered a long-lived branch: {is_long_lived_branch}")

    # Determine which trigger types are active for proper matrix lookup
    active_trigger_types = []
    if is_pull_request:
        active_trigger_types.append("presubmit")
    if is_push:
        if is_long_lived_branch:
            active_trigger_types.extend(["presubmit", "postsubmit"])
        else:
            # Non-long-lived branch pushes use presubmit defaults
            active_trigger_types.append("presubmit")
    if is_schedule:
        active_trigger_types.extend(["presubmit", "postsubmit", "nightly"])

    # Get the appropriate family matrix based on active triggers
    # For workflow_dispatch and PR labels, we need to check all matrices
    if is_workflow_dispatch or is_pull_request:
        # For workflow_dispatch, check all possible matrices
        lookup_trigger_types = ["presubmit", "postsubmit", "nightly"]
        lookup_matrix = get_all_families_for_trigger_types(lookup_trigger_types)
        print(f"Using family matrix for trigger types: {lookup_trigger_types}")
    elif active_trigger_types:
        lookup_matrix = get_all_families_for_trigger_types(active_trigger_types)
        print(f"Using family matrix for trigger types: {active_trigger_types}")
    else:
        # This code path should never be reached in production workflows
        # as they only trigger on main branch pushes, PRs, workflow_dispatch, or schedule.
        # If this error is raised, it indicates an unexpected trigger combination.
        raise AssertionError(
            f"Unreachable code: no trigger types determined. "
            f"is_pull_request={is_pull_request}, is_workflow_dispatch={is_workflow_dispatch}, "
            f"is_push={is_push}, is_schedule={is_schedule}, "
            f"branch_name={branch_name}"
        )

    if is_workflow_dispatch:
        print(f"[WORKFLOW_DISPATCH] Generating build matrix with {str(base_args)}")

        # Parse inputs into a targets list.
        input_gpu_targets = families.get("amdgpu_families")
        # Sanitizing the string to remove any punctuation from the input
        # After replacing punctuation with spaces, turning string input to an array
        # (ex: ",gfx94X ,|.gfx1201" -> "gfx94X   gfx1201" -> ["gfx94X", "gfx1201"])
        translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
        requested_target_names = input_gpu_targets.translate(translator).split()

        selected_target_names.extend(
            filter_known_names(requested_target_names, "target", lookup_matrix)
        )

        # If any workflow dispatch test labels are specified, we run full tests for those specific tests
        workflow_dispatch_test_labels_str = (
            base_args.get("workflow_dispatch_linux_test_labels", "")
            if platform == "linux"
            else base_args.get("workflow_dispatch_windows_test_labels", "")
        )
        # (ex: "test:rocprim, test:hipcub" -> ["test:rocprim", "test:hipcub"])
        workflow_dispatch_test_labels = [
            test_label.strip()
            for test_label in workflow_dispatch_test_labels_str.split(",")
        ]

        requested_test_names = []
        for label in workflow_dispatch_test_labels:
            if "test:" in label:
                _, test_name = label.split(":")
                requested_test_names.append(test_name)
                print(
                    f"    Workflow dispatch test label '{label}' -> test: {test_name}"
                )

        if requested_test_names:
            print(f"  Requested tests from workflow_dispatch: {requested_test_names}")

        selected_test_names.extend(filter_known_names(requested_test_names, "test"))

    if is_pull_request:
        print(f"[PULL_REQUEST] Generating build matrix with {str(base_args)}")

        # Add presubmit targets.
        for target in get_all_families_for_trigger_types(["presubmit"]):
            selected_target_names.append(target)

        # Extend with any additional targets that PR labels opt-in to running.
        # TODO(#1097): This (or the code below) should handle opting in for
        #     a GPU family for only one platform (e.g. Windows but not Linux)
        requested_target_names = []
        requested_test_names = []
        pr_labels = get_pr_labels(base_args)
        print(f"  Processing {len(pr_labels)} PR label(s): {pr_labels}")

        for label in pr_labels:
            # if a GPU target label was added, we add the GPU target to the build and test matrix
            if "gfx" in label:
                target = label.split("-")[0]
                requested_target_names.append(target)
                print(f"    Label '{label}' matched 'gfx*' pattern -> target: {target}")
            # If a test label was added, we run the full test for the specified test
            if "test:" in label:
                _, test_name = label.split(":")
                requested_test_names.append(test_name)
                print(
                    f"    Label '{label}' matched 'test:*' pattern -> test: {test_name}"
                )
            # If the "ci:skip" label was added, we skip all builds and tests
            # We don't want to check for anymore labels
            if "ci:skip" == label:
                print(f"    Label 'ci:skip' detected -> skipping all builds and tests")
                selected_target_names = []
                selected_test_names = []
                requested_target_names = []
                requested_test_names = []
                break
            if "ci:run-all-archs" == label:
                print(
                    f"    Label 'ci:run-all-archs' detected -> enabling all architectures"
                )
                selected_target_names = [
                    target
                    for target in get_all_families_for_trigger_types(
                        ["presubmit", "postsubmit", "nightly"]
                    )
                ]

        if requested_target_names:
            print(f"  Requested targets from labels: {requested_target_names}")
        if requested_test_names:
            print(f"  Requested tests from labels: {requested_test_names}")

        selected_target_names.extend(
            filter_known_names(requested_target_names, "target", lookup_matrix)
        )
        selected_test_names.extend(filter_known_names(requested_test_names, "test"))

    if is_push:
        if is_long_lived_branch:
            print(
                f"[PUSH - {branch_name.upper()}] Generating build matrix with {str(base_args)}"
            )

            # Add presubmit and postsubmit targets.
            for target in get_all_families_for_trigger_types(
                ["presubmit", "postsubmit"]
            ):
                selected_target_names.append(target)
        else:
            print(
                f"[PUSH - {branch_name}] Generating build matrix with {str(base_args)}"
            )

            # Non-long-lived branch pushes use presubmit targets
            for target in get_all_families_for_trigger_types(["presubmit"]):
                selected_target_names.append(target)

    if is_schedule:
        print(f"[SCHEDULE] Generating build matrix with {str(base_args)}")

        # For nightly runs, we run all builds and full tests
        amdgpu_family_info_matrix_all = get_all_families_for_trigger_types(
            ["presubmit", "postsubmit", "nightly"]
        )
        for key in amdgpu_family_info_matrix_all:
            selected_target_names.append(key)

    # Ensure the lists are unique
    unique_target_names = list(set(selected_target_names))
    unique_test_names = list(set(selected_test_names))

    platform_build_variants = all_build_variants.get(platform)
    assert isinstance(
        platform_build_variants, dict
    ), f"Expected build variant {platform} in {all_build_variants}"

    # Expand selected target names back to a matrix (cross-product of families × variants).
    matrix_output = []
    for target_name in unique_target_names:
        # Filter targets to only those matching the requested platform.
        # Use the trigger-appropriate lookup matrix
        platform_set = lookup_matrix.get(target_name)
        if platform in platform_set:
            platform_info = platform_set.get(platform)
            assert isinstance(platform_info, dict)

            # Further expand it based on build_variant.
            build_variant_names = platform_info.get("build_variants")
            assert isinstance(
                build_variant_names, list
            ), f"Expected 'build_variant' in platform: {platform_info}"
            for build_variant_name in build_variant_names:
                # We have custom build variants for specific CI flows.
                # For CI, we use the release build variant (for PRs, pushes to main, nightlies)
                # For CI ASAN/TSAN, we use the ASAN/TSAN build variant (for pushes to main)
                # In the case that the build variant is not requested, we skip it
                if build_variant_name != base_args.get("build_variant"):
                    continue

                # Merge platform_info and build_variant_info into a matrix_row.
                matrix_row = dict(platform_info)

                build_variant_info = platform_build_variants.get(build_variant_name)
                assert isinstance(
                    build_variant_info, dict
                ), f"Expected {build_variant_name} in {platform_build_variants} for {platform_info}"

                # If the build variant level notes expect_failure, set it on the overall row.
                # But if not, honor what is already there.
                if build_variant_info.get("expect_failure", False):
                    matrix_row["expect_failure"] = True

                # Enable pytorch builds for families without known build failures.
                # TODO(#3291): Add finer-grained controls over when pytorch is built
                expect_failure = matrix_row.get("expect_failure", False)
                expect_pytorch_failure = matrix_row.get("expect_pytorch_failure", False)
                matrix_row["build_pytorch"] = (
                    not expect_failure and not expect_pytorch_failure
                )

                del matrix_row["build_variants"]
                matrix_row.update(build_variant_info)

                # Assign a computed "artifact_group" combining the family and variant.
                artifact_group = platform_info["family"]
                build_variant_suffix = build_variant_info["build_variant_suffix"]
                if build_variant_suffix:
                    artifact_group += f"-{build_variant_suffix}"
                matrix_row["artifact_group"] = artifact_group

                # Handle dual-label configuration with weighted random selection.
                # Some families (e.g. gfx94x) have multiple runner labels available.
                if "test-runs-on-alternate" in platform_info:
                    alternate_label = platform_info["test-runs-on-alternate"]
                    alternate_weight = platform_info.get(
                        "test-runs-on-alternate-weight", 0.5
                    )
                    if random.random() < alternate_weight:
                        matrix_row["test-runs-on"] = alternate_label
                        print(
                            f"  {target_name}: selected alternate runner (weight={alternate_weight}): "
                            f"{alternate_label}"
                        )
                    else:
                        print(
                            f"  {target_name}: selected primary runner (weight={1-alternate_weight}): "
                            f"{matrix_row['test-runs-on']}"
                        )

                # We retrieve labels from both PR and workflow_dispatch to customize the build and test jobs
                label_options = []
                label_options.extend(get_pr_labels(base_args))
                label_options.extend(
                    get_workflow_dispatch_additional_label_options(base_args)
                )
                for label in label_options:
                    # If a specific test kernel type was specified, we use that kernel-enabled test runners
                    # We disable the other machines that do not have the specified kernel type
                    # If a kernel test label was added, we set the test-runs-on accordingly to kernel-specific test machines
                    if "test_runner" in label:
                        _, kernel_type = label.split(":")
                        # If the architecture has a valid kernel machine, we set it here
                        if (
                            "test-runs-on-kernel" in platform_info
                            and kernel_type in platform_info["test-runs-on-kernel"]
                        ):
                            matrix_row["test-runs-on"] = platform_info[
                                "test-runs-on-kernel"
                            ][kernel_type]
                        # Otherwise, we disable the test runner for this architecture
                        else:
                            matrix_row["test-runs-on"] = ""
                            if "test-runs-on-multi-gpu" in platform_info:
                                matrix_row["test-runs-on-multi-gpu"] = ""
                        break

                # TODO(#3433): Remove sandbox logic once ASAN tests are passing and environment is no longer required
                # To avoid impact on the production environment, we use the custom sandbox runners if this is an ASAN test run
                if (
                    "asan" in base_args.get("build_variant")
                    and "test-runs-on-sandbox" in matrix_row
                ):
                    matrix_row["test-runs-on"] = matrix_row["test-runs-on-sandbox"]

                matrix_output.append(matrix_row)

    print(f"Generated build matrix: {str(matrix_output)}")
    print(f"Generated test list: {str(unique_test_names)}")
    return matrix_output, unique_test_names


# --------------------------------------------------------------------------- #
# Core script logic
# --------------------------------------------------------------------------- #


def main(base_args, linux_families, windows_families):
    github_event_name = base_args.get("github_event_name")
    is_push = github_event_name == "push"
    is_workflow_dispatch = github_event_name == "workflow_dispatch"
    is_pull_request = github_event_name == "pull_request"
    is_schedule = github_event_name == "schedule"

    branch_name = base_args.get("branch_name", "")
    base_ref = base_args.get("base_ref")
    build_variant = base_args.get("build_variant", "")

    linux_use_prebuilt_artifacts = base_args.get("linux_use_prebuilt_artifacts")
    windows_use_prebuilt_artifacts = base_args.get("windows_use_prebuilt_artifacts")

    print("Found metadata:")
    print(f"  github_event_name: {github_event_name}")
    print(f"    is_push: {is_push}")
    print(f"    is_workflow_dispatch: {is_workflow_dispatch}")
    print(f"    is_pull_request: {is_pull_request}")
    print(f"    is_schedule: {is_schedule}")
    print(f"  branch_name: {branch_name}")
    print(f"  base_ref: {base_ref}")
    print(f"  build_variant: {build_variant}")
    print(f"  linux_use_prebuilt_artifacts: {linux_use_prebuilt_artifacts}")
    print(f"  windows_use_prebuilt_artifacts: {windows_use_prebuilt_artifacts}")
    pr_labels = None
    if is_pull_request:
        pr_labels = get_pr_labels(base_args)
        print(f"  pr_labels: {pr_labels}")
    if is_workflow_dispatch:
        print(
            f"  workflow_dispatch_linux_test_labels: {base_args.get('workflow_dispatch_linux_test_labels', '')}"
        )
        print(
            f"  workflow_dispatch_windows_test_labels: {base_args.get('workflow_dispatch_windows_test_labels', '')}"
        )
    print("")

    print(f"Generating build matrix for Linux: {str(linux_families)}")
    linux_variants_output, linux_test_output = matrix_generator(
        is_pull_request,
        is_workflow_dispatch,
        is_push,
        is_schedule,
        base_args,
        linux_families,
        platform="linux",
    )
    print("")

    print(f"Generating build matrix for Windows: {str(windows_families)}")
    windows_variants_output, windows_test_output = matrix_generator(
        is_pull_request,
        is_workflow_dispatch,
        is_push,
        is_schedule,
        base_args,
        windows_families,
        platform="windows",
    )
    print("")

    test_type = "quick"
    test_type_reason = "default (quick tests)"
    run_functional_tests = False

    if is_schedule:
        # Always build and run full tests on scheduled runs.
        enable_build_jobs = True
        test_type = "comprehensive"
        test_type_reason = "scheduled run triggers comprehensive tests"
        # Functional tests run on nightly/scheduled builds
        run_functional_tests = True
    elif is_workflow_dispatch:
        # Always build and conditionally run full tests for workflow dispatch.
        enable_build_jobs = True
        if linux_test_output or windows_test_output:
            combined_test_labels = list(set(linux_test_output + windows_test_output))
            test_type = "full"
            test_type_reason = f"test label(s) specified: {combined_test_labels}"
            # Functional tests run on nightly/scheduled builds
            run_functional_tests = True
    else:
        # Conditionally build and conditionally run full tests for other
        # triggers (pull_request), based on modified paths and other inputs.
        modified_paths = get_git_modified_paths(base_ref)
        print("modified_paths (max 200):", modified_paths[:200])
        print(f"Checking modified files since this had a {github_event_name} trigger")
        # TODO(#199): other behavior changes
        #     * workflow_dispatch or workflow_call with inputs controlling enabled jobs?
        enable_build_jobs = is_ci_run_required(modified_paths)

        # multi_arch_ci.yml is now the default, so the "non-multi-arch" ci.yml
        # now requires an opt-in to run on pull requests.
        # This avoids doubling CI load during the transition from ci.yml
        # to multi_arch_ci.yml. See https://github.com/ROCm/TheRock/issues/3337
        # TODO(#3399): move multi-arch CI configuration to its own script
        if is_pull_request and "ci:run-non-multi-arch" not in (pr_labels or []):
            print(
                "Skipping non-multi-arch CI: 'ci:run-non-multi-arch' label not found. "
                "Add the label to opt in."
            )
            enable_build_jobs = False

        # If the modified path contains any git submodules, we want to run a full test suite.
        # Otherwise, we just run quick tests
        submodule_paths = get_git_submodule_paths(repo_root=THEROCK_DIR)
        matching_submodule_paths = list(set(submodule_paths) & set(modified_paths))
        if matching_submodule_paths:
            test_type = "full"
            test_type_reason = f"submodule(s) changed: {matching_submodule_paths}"

        # If any test label is included, run full test suite for specified tests
        if linux_test_output or windows_test_output:
            combined_test_labels = list(set(linux_test_output + windows_test_output))
            test_type = "full"
            test_type_reason = f"test label(s) specified: {combined_test_labels}"

        for matrix_row in linux_variants_output + windows_variants_output:
            # If the "run-full-tests-only" flag is set for this family, we do not run tests if it is a quick test type
            if matrix_row.get("run-full-tests-only", False) and test_type == "quick":
                matrix_row["test-runs-on"] = ""

        # If a test filter label is included, we set the "test_type" to the designated filter
        if pr_labels and any("test_filter:" in label for label in pr_labels):
            for label in pr_labels:
                if "test_filter:" in label:
                    filter_type = label.split(":")[1]
                    # If the filter type is not recognized, we ignore the label and keep the default test type
                    if filter_type not in [
                        "quick",
                        "standard",
                        "comprehensive",
                        "full",
                    ]:
                        continue
                    test_type = filter_type
                    test_type_reason = f"test filter label specified: {label}"
                    break

    print(f"test_type decision: '{test_type}' (reason: {test_type_reason})")

    # Format variants for summary
    def format_variants(variants):
        result = []
        for item in variants:
            if "family" in item:
                label = item["family"]
                # Also show flags for the family, if any.
                flags = []
                if item.get("expect_failure"):
                    flags.append("expect_failure")
                if item.get("build_pytorch"):
                    flags.append("build_pytorch")
                if flags:
                    label += f" ({', '.join(flags)})"
                result.append(label)
        return result

    gha_append_step_summary(
        f"""## Workflow configure results

* `linux_variants`: {str(format_variants(linux_variants_output))}
* `linux_test_labels`: {str([test for test in linux_test_output])}
* `linux_use_prebuilt_artifacts`: {json.dumps(linux_use_prebuilt_artifacts)}
* `windows_variants`: {str(format_variants(windows_variants_output))}
* `windows_test_labels`: {str([test for test in windows_test_output])}
* `windows_use_prebuilt_artifacts`: {json.dumps(windows_use_prebuilt_artifacts)}
* `enable_build_jobs`: {json.dumps(enable_build_jobs)}
* `test_type`: {test_type}
* `run_functional_tests`: {json.dumps(run_functional_tests)}
    """
    )

    output = {
        "linux_variants": json.dumps(linux_variants_output),
        "linux_test_labels": json.dumps(linux_test_output),
        "windows_variants": json.dumps(windows_variants_output),
        "windows_test_labels": json.dumps(windows_test_output),
        "enable_build_jobs": json.dumps(enable_build_jobs),
        "test_type": test_type,
        "run_functional_tests": json.dumps(run_functional_tests),
    }
    gha_set_output(output)


if __name__ == "__main__":
    base_args = {}
    linux_families = {}
    windows_families = {}

    linux_families["amdgpu_families"] = os.environ.get(
        "INPUT_LINUX_AMDGPU_FAMILIES", ""
    )

    windows_families["amdgpu_families"] = os.environ.get(
        "INPUT_WINDOWS_AMDGPU_FAMILIES", ""
    )

    base_args["pr_labels"] = os.environ.get("PR_LABELS", '{"labels": []}')
    base_args["branch_name"] = os.environ.get("GITHUB_REF_NAME", "")
    if base_args["branch_name"] == "":
        print(
            "[ERROR] GITHUB_REF_NAME is not set! No branch name detected. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)
    base_args["github_event_name"] = os.environ.get("GITHUB_EVENT_NAME", "")
    base_args["base_ref"] = os.environ.get("BASE_REF", "HEAD^1")
    base_args["linux_use_prebuilt_artifacts"] = (
        os.environ.get("LINUX_USE_PREBUILT_ARTIFACTS") == "true"
    )
    base_args["windows_use_prebuilt_artifacts"] = (
        os.environ.get("WINDOWS_USE_PREBUILT_ARTIFACTS") == "true"
    )
    base_args["workflow_dispatch_linux_test_labels"] = os.getenv(
        "LINUX_TEST_LABELS", ""
    )
    base_args["workflow_dispatch_windows_test_labels"] = os.getenv(
        "WINDOWS_TEST_LABELS", ""
    )
    base_args["workflow_dispatch_additional_label_options"] = os.getenv(
        "ADDITIONAL_LABEL_OPTIONS", ""
    )
    base_args["build_variant"] = os.getenv("BUILD_VARIANT", "release")

    main(base_args, linux_families, windows_families)
