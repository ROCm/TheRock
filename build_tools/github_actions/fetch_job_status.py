"""
This script determines the job status for different job runs
as part of GitHub workflow based on RUN_ID and ATTEMPT

Required environment variables:
  - RUN_ID
  - ATTEMPT
"""

import json
import os
from urllib.request import urlopen, Request
import logging

logging.basicConfig(level=logging.INFO)

RUN_ID = os.getenv("RUN_ID")
ATTEMPT = os.getenv("ATTEMPT")

# Check for missing values
if not RUN_ID or not ATTEMPT:
    raise ValueError(
        f"Missing required environment variable RUN_ID or ATTEMPT. "
        f"Ensure these are exported or set in the CI environment."
    )


def run():
    github_workflow_jobs_url = f"https://api.github.com/repos/ROCm/TheRock/actions/runs/{RUN_ID}/attempts/{ATTEMPT}/jobs"

    logging.info(f"Constructed GitHub workflow jobs URL: {github_workflow_jobs_url}")

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    # If GITHUB_TOKEN environment variable is available, include it in the API request to avoid a lower rate limit
    gh_token = os.getenv("GITHUB_TOKEN", "")
    if gh_token:
        headers["Authentication"] = f"Bearer {gh_token}"

    request = Request(github_workflow_jobs_url, headers=headers)
    with urlopen(request) as response:
        if response.status == 403:
            raise Exception(
                f"Access denied (403 Forbidden) while retrieving workflow jobs from GitHub. "
                f"Check if your token has the necessary permissions (e.g., `repo`, `workflow`)."
            )
        elif response.status != 200:
            raise Exception(
                f"Failed to retrieve GitHub workflow job data for run ID '{RUN_ID}' and attempt '{ATTEMPT}'. "
                f"Received unexpected status code: {response.status}. Please verify the URL or check GitHub API status {response.status}."
            )

        job_data = json.loads(response.read().decode("utf-8"))
        # Check if API output shows number of jobs run in the workflow to be atleast 1
        if not job_data.get("jobs"):
            raise Exception("No jobs found in the GitHub workflow run.")
        # Output the job summary JSON string directly to stdout
        print(json.dumps(job_data))


if __name__ == "__main__":
    run()
