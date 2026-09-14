# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate repository-wide uniqueness of summary job display names.

Required status checks identify jobs by display name, so a summary job must
not share its name with another job in the repository. Step names do not
affect required checks. Scan only this repository's workflows, not submodules.

This checks statically declared names; it does not expand matrix expressions
or compose caller/callee names for reusable workflows.
"""

from collections import defaultdict
import unittest

from workflow_utils import WORKFLOWS_DIR, load_workflow


def is_summary_job(job_id: str, job: dict) -> bool:
    """Recognize summary job IDs and jobs invoking the summary script."""
    return (
        job_id == "summary"
        or job_id.endswith("_summary")
        or any(
            "workflow_summary.py" in step.get("run", "")
            for step in job.get("steps", [])
        )
    )


class WorkflowSummaryNamesTest(unittest.TestCase):
    def test_summary_names_are_unique(self):
        workflow_paths = sorted(
            [*WORKFLOWS_DIR.glob("*.yml"), *WORKFLOWS_DIR.glob("*.yaml")]
        )
        self.assertTrue(workflow_paths, f"No workflows found in {WORKFLOWS_DIR}")

        jobs_by_name: dict[str, list[str]] = defaultdict(list)
        summary_names: set[str] = set()
        for workflow_path in workflow_paths:
            workflow = load_workflow(workflow_path)
            for job_id, job in workflow.get("jobs", {}).items():
                name = job.get("name", job_id)
                jobs_by_name[name].append(f"{workflow_path.name}: jobs.{job_id}")
                if is_summary_job(job_id, job):
                    summary_names.add(name)

        self.assertTrue(summary_names, "No summary jobs found in workflows")
        errors = []
        for name in sorted(summary_names):
            locations = jobs_by_name[name]
            if len(locations) > 1:
                errors.append(f"{name!r} is used by:\n  " + "\n  ".join(locations))
        if errors:
            self.fail("Summary job names must be unique:\n" + "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
