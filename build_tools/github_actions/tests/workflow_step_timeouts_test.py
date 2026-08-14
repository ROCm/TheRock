# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests that checkout and fetch_sources steps have timeout-minutes set.

Long-running git operations (actions/checkout, fetch_sources.py) can hang
indefinitely if GitHub throttles or a network issue occurs. Every such step
must declare a timeout-minutes so hung jobs fail fast instead of consuming
runner capacity until the job-level timeout fires (which may be hours away).

See: ROCm/TheRock#7343
"""

import unittest
import warnings

from workflow_utils import WORKFLOWS_DIR, load_workflow

# Step uses: prefixes that identify git checkout actions.
_CHECKOUT_ACTION_PREFIX = "actions/checkout@"

# Substrings in a step's run: command that identify a fetch_sources invocation.
_FETCH_SOURCES_PATTERNS = ("fetch_sources.py",)

# Known violations that predate this enforcement. Each entry is the string
# "workflow_file / job_name / 'step_name'" exactly as produced by
# _find_violations(). New violations are NOT added here; they must be fixed.
# Tracked for follow-up in: ROCm/TheRock#7388
_KNOWN_VIOLATIONS = frozenset(
    [
        "build_portable_linux_python_packages.yml / build_rocm_wheels / 'Checkout'",
        "build_windows_python_packages.yml / build_rocm_wheels / 'Checkout'",
        "bump_submodules.yml / bump-submodules / 'Checkout ROCm/TheRock'",
        "codeql.yml / analyze / 'Checkout repository'",
        "copy_release.yml / copy_python_packages / 'Checkout Repository'",
        "gitleaks.yml / scan / 'Bootstrap checkout'",
        "gitleaks.yml / scan / 'actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1'",
        "hip_tagging_automation.yml / tag-rocm-systems / 'Checkout ROCm/TheRock'",
        "manifest-diff.yml / generate-report / 'Checkout repository'",
        "multi_arch_build_linux_jax_wheels.yml / generate_target_to_run / 'Checking out repository'",
        "multi_arch_build_linux_jax_wheels.yml / generate_target_to_run / 'Checkout CI config'",
        "multi_arch_build_native_linux_packages.yml / build_native_packages / 'Checking out repository'",
        "multi_arch_build_portable_linux_pytorch_wheels.yml / build_pytorch_wheels / 'Checkout'",
        "multi_arch_build_portable_linux_pytorch_wheels.yml / configure_pytorch_tests / 'Checkout'",
        "multi_arch_build_portable_linux_pytorch_wheels.yml / configure_pytorch_tests / 'Checkout CI config'",
        "multi_arch_build_portable_linux_pytorch_wheels_ci.yml / build_pytorch_wheels / 'Checkout'",
        "multi_arch_build_tarballs.yml / build_tarballs / 'Checkout'",
        "multi_arch_build_windows_pytorch_wheels.yml / build_pytorch_wheels / 'Checkout'",
        "multi_arch_build_windows_pytorch_wheels.yml / configure_pytorch_tests / 'Checkout'",
        "multi_arch_build_windows_pytorch_wheels.yml / configure_pytorch_tests / 'Checkout CI config'",
        "multi_arch_build_windows_pytorch_wheels_ci.yml / build_pytorch_wheels / 'Checkout'",
        "multi_arch_ci.yml / ci_summary / 'actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1'",
        "multi_arch_ci_asan.yml / ci_summary / 'actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1'",
        "multi_arch_ci_linux.yml / copy_prebuilt_stages / 'Checking out repository'",
        "multi_arch_ci_windows.yml / copy_prebuilt_stages / 'Checking out repository'",
        "multi_arch_release_linux.yml / publish_to_release_buckets / 'Checkout'",
        "multi_arch_release_windows.yml / publish_to_release_buckets / 'Checkout'",
        "multi_arch_release_windows_pytorch_wheels.yml / setup_matrix / 'Checkout'",
        "pre-commit.yml / pre-commit / 'actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1'",
        "publish_dockerfile.yml / build-and-push-image / 'Checkout repository'",
        "setup_multi_arch.yml / setup / 'Checking out repository'",
        "setup_multi_arch.yml / setup / 'Checkout CI config'",
        "test_artifacts.yml / configure_test_matrix / 'Checking out repository'",
        "test_artifacts.yml / configure_test_matrix / 'Checkout CI config'",
        "test_artifacts_structure.yml / test_artifact_structure / 'Checkout repository'",
        "test_component.yml / test_component / 'Fetch 'build_tools' from repository'",
        "test_component.yml / test_component / 'Checking repository'",
        "test_jax_dockerfile.yml / test_wheels / 'Checkout'",
        "test_linux_jax_wheels.yml / test_jax_wheels / 'Checkout'",
        "test_linux_jax_wheels.yml / test_jax_wheels / 'Checkout rocm-jax (plugin + build scripts)'",
        "test_linux_jax_wheels.yml / test_jax_wheels / 'Checkout JAX extended tests repo'",
        "test_multi_arch_linux_jax_wheels.yml / test_jax_wheels / 'Checkout'",
        "test_multi_arch_linux_jax_wheels.yml / test_jax_wheels / 'Checkout rocm-jax'",
        "test_multi_arch_linux_jax_wheels.yml / test_jax_wheels / 'Checkout JAX'",
        "test_native_linux_packages_install.yml / prepare_install_context / 'Checkout'",
        "test_native_linux_packages_install.yml / test_native_linux_packages_install / 'Checkout'",
        "test_pytorch_wheels.yml / test_wheels / 'Checkout'",
        "test_pytorch_wheels_full.yml / prepare_matrix / 'Checkout'",
        "test_pytorch_wheels_full.yml / test_wheels / 'Checkout'",
        "test_pytorch_wheels_full.yml / test_summary / 'Checkout'",
        "test_rocm_wheels.yml / test_wheels / 'Checkout'",
        "therock-pr-bot.yml / policy / 'actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1'",
        "unit_tests.yml / unit_tests / 'actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1'",
        "unit_tests.yml / unit_tests_summary / 'actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1'",
    ]
)


def _step_is_checkout(step: dict) -> bool:
    """Returns True if the step uses actions/checkout."""
    uses = step.get("uses", "")
    return isinstance(uses, str) and uses.startswith(_CHECKOUT_ACTION_PREFIX)


def _step_is_fetch_sources(step: dict) -> bool:
    """Returns True if the step runs fetch_sources.py."""
    run = step.get("run", "")
    if not isinstance(run, str):
        return False
    return any(pattern in run for pattern in _FETCH_SOURCES_PATTERNS)


def _find_violations(workflow: dict, workflow_name: str) -> list[str]:
    """Returns violation strings for every checkout/fetch_sources step missing a timeout."""
    violations = []
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        return violations

    for job_name, job_def in jobs.items():
        if not isinstance(job_def, dict):
            continue
        steps = job_def.get("steps")
        if not isinstance(steps, list):
            continue

        for step in steps:
            if not isinstance(step, dict):
                continue
            if not (_step_is_checkout(step) or _step_is_fetch_sources(step)):
                continue
            if step.get("timeout-minutes") is None:
                step_name = (
                    step.get("name") or step.get("uses") or step.get("run", "")[:60]
                )
                violations.append(f"{workflow_name} / {job_name} / '{step_name}'")

    return violations


class WorkflowStepTimeoutsTest(unittest.TestCase):
    """Verifies that checkout and fetch_sources steps declare timeout-minutes."""

    def test_checkout_and_fetch_sources_have_timeouts(self):
        """Every actions/checkout and fetch_sources.py step must have timeout-minutes.

        New violations fail the test immediately. Violations in _KNOWN_VIOLATIONS
        are pre-existing debt; they emit a warning and must be fixed in follow-up
        PRs tracked by ROCm/TheRock#7388.
        """
        new_violations = []
        known_seen = []

        for workflow_path in sorted(WORKFLOWS_DIR.glob("*.yml")):
            workflow = load_workflow(workflow_path)
            for v in _find_violations(workflow, workflow_path.name):
                if v in _KNOWN_VIOLATIONS:
                    known_seen.append(v)
                else:
                    new_violations.append(v)

        if known_seen:
            warnings.warn(
                f"{len(known_seen)} pre-existing checkout/fetch_sources steps are"
                " missing timeout-minutes (fix in follow-up PRs;"
                " tracked in ROCm/TheRock#7388):\n"
                + "\n".join(f"  - {v}" for v in known_seen),
                stacklevel=2,
            )

        if new_violations:
            self.fail(
                "New checkout/fetch_sources steps are missing timeout-minutes.\n"
                "Add 'timeout-minutes: 15' to each step listed below, or add it\n"
                "to _KNOWN_VIOLATIONS only if it is genuinely pre-existing debt:\n"
                + "\n".join(f"  - {v}" for v in new_violations)
            )


if __name__ == "__main__":
    unittest.main()
