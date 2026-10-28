# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check that jobs which upload artifacts are not pinned to GitHub-hosted runners.

Fork PRs skip OIDC and rely on baseline storage credentials on our self-hosted
runners, including AWS and Azure runners. See the Authentication section in
docs/development/s3_buckets.md.

This code would *fail* the tests as the job cannot use runner base credentials
on fork PRs:

    jobs:
      upload:
        runs-on: windows-2022  # Rejected: always GitHub-hosted.
        steps:
          - uses: ./.github/actions/configure_aws_artifacts_credentials
          - run: aws s3 cp artifact.zip s3://therock-ci-artifacts-external/example/
            # This would fail to upload on fork PRs: no baseline credentials.

This code would *pass* the tests as runners are always self-hosted
(note that our runners are unavailable to fork repositories so this is harder
for contributors to run themselves):

    jobs:
      upload:
        runs-on: aws-linux-scale-rocm-prod
        steps:
          - uses: ./.github/actions/configure_aws_artifacts_credentials
          - run: aws s3 cp artifact.zip s3://therock-ci-artifacts-external/example/

This code would *pass* the tests as runners are conditionally self-hosted
(fork repositories can run, but skip uploads):

    jobs:
      upload:
        runs-on: ${{ github.repository_owner == 'ROCm' && inputs.build_runs_on || 'ubuntu-26.04' }}
        steps:
          - uses: ./.github/actions/configure_aws_artifacts_credentials
            if: ${{ github.repository_owner == 'ROCm' }}
          - run: aws s3 cp artifact.zip s3://therock-ci-artifacts-external/example/
            if: ${{ github.repository_owner == 'ROCm' }}

This is a static guard against fixed GitHub-hosted runner labels. Expressions
are allowed so reusable workflows can select self-hosted runners via inputs,
with GitHub-hosted fallbacks for forks. It does not evaluate expressions or
verify that callers supply runners with credentials.
"""

import re
import unittest

from workflow_utils import WORKFLOWS_DIR, load_workflow


_ARTIFACT_CREDENTIALS_ACTION = "./.github/actions/configure_aws_artifacts_credentials"
_GITHUB_HOSTED_LABEL = re.compile(r"(?:ubuntu|windows|macos)-(?:latest|\d).*", re.I)


def uses_fixed_github_hosted_runner(runs_on: object) -> bool:
    """Recognize literal hosted labels in string, list, and mapping syntax.

    Lists are label constraints on a single runner, not alternative runners.
    An explicit self-hosted label or runner group selects a custom runner pool.
    """
    if isinstance(runs_on, dict):
        if runs_on.get("group"):
            return False
        return uses_fixed_github_hosted_runner(runs_on.get("labels"))
    if isinstance(runs_on, list):
        if "self-hosted" in runs_on:
            return False
        return any(uses_fixed_github_hosted_runner(label) for label in runs_on)
    return (
        isinstance(runs_on, str)
        and "${{" not in runs_on
        and _GITHUB_HOSTED_LABEL.fullmatch(runs_on.strip()) is not None
    )


class WorkflowSelfHostedRunnersTest(unittest.TestCase):
    def test_artifact_credentials_jobs_allow_self_hosted_runners(self):
        workflow_paths = sorted(
            [*WORKFLOWS_DIR.glob("*.yml"), *WORKFLOWS_DIR.glob("*.yaml")]
        )
        self.assertTrue(workflow_paths, f"No workflows found in {WORKFLOWS_DIR}")
        errors = []
        for workflow_path in workflow_paths:
            workflow = load_workflow(workflow_path)
            for job_name, job in workflow.get("jobs", {}).items():
                if not any(
                    step.get("uses") == _ARTIFACT_CREDENTIALS_ACTION
                    for step in job.get("steps", [])
                ):
                    continue
                runs_on = job.get("runs-on")
                if uses_fixed_github_hosted_runner(runs_on):
                    errors.append(
                        f"{workflow_path.name}: jobs.{job_name}: runs-on: {runs_on!r}"
                    )

        if errors:
            self.fail(
                "Jobs using configure_aws_artifacts_credentials must allow "
                "self-hosted runners with baseline storage credentials for fork PRs.\n"
                "Use a self-hosted runner label or a conditional runner input, "
                "as in multi_arch_build_native_linux_packages.yml.\n"
                "See docs/development/s3_buckets.md#authentication.\n"
                + "\n".join(f"  - {error}" for error in errors)
            )


class RunnerClassificationTest(unittest.TestCase):
    def test_fixed_github_hosted_runners(self):
        for runs_on in (
            "windows-2022",
            "ubuntu-26.04",
            "ubuntu-24.04-arm",
            "macos-latest",
            "macos-15-large",
            ["ubuntu-latest"],
            {"labels": "windows-latest"},
        ):
            with self.subTest(runs_on=runs_on):
                self.assertTrue(uses_fixed_github_hosted_runner(runs_on))

    def test_self_hosted_and_configurable_runners(self):
        for runs_on in (
            "self-hosted",
            "aws-windows-scale-rocm-prod-mix",
            "azure-linux-scale-rocm",
            ["self-hosted", "linux", "x64"],
            ["self-hosted", "ubuntu-26.04"],
            {"group": "rocm-builders", "labels": "linux"},
            "${{ inputs.build_runs_on }}",
            "${{ github.repository_owner == 'ROCm' && inputs.build_runs_on || 'ubuntu-26.04' }}",
            "${{ github.repository_owner == 'ROCm' && inputs.build_runs_on || 'windows-2022' }}",
            "${{ fromJSON(inputs.build_config).build_runs_on_small }}",
        ):
            with self.subTest(runs_on=runs_on):
                self.assertFalse(uses_fixed_github_hosted_runner(runs_on))


if __name__ == "__main__":
    unittest.main()
