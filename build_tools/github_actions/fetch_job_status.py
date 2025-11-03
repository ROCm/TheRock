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

from github_actions_utils import *

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

    job_data = gha_send_request(github_workflow_jobs_url)

    # Check if API output shows number of jobs run in the workflow to be atleast 1
    if not job_data.get("jobs"):
        raise Exception("No jobs found in the GitHub workflow run.")

    # Output the job summary JSON string directly to stdout
    print(json.dumps(job_data))


if __name__ == "__main__":
    run()
