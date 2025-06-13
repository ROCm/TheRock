"""
This script determines the job status for different jobs run
as part of GitHub workflow based on RUN_ID and ATTEMPT

Required environment variables:
  - RUN_ID
  - ATTEMPT
"""

from configure_ci import set_github_output
import json
import os
from urllib.request import urlopen, Request

RUN_ID = os.getenv("RUN_ID")
ATTEMPT = os.getenv("ATTEMPT")

def run():
    github_workflow_jobs_url = (
        f"https://api.github.com/repos/RoCm/TheRock/actions/runs/{RUN_ID}/attempts/{ATTEMPT}/jobs"
    )
    print(github_workflow_jobs_url)
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
                f"Error when retrieving GitHub response. This is most likely a rate limiting issue, so please try again"
            )
        elif response.status != 200:
            raise Exception(
                f"Error when retrieving GitHub response assets for {RUN_ID} tag with status code {response.status}. Exiting..."
            )

        job_data = json.loads(response.read().decode("utf-8"))
        # Check if API output shows number of jobs run in the workflow to be atleast 1
        if len(job_data["jobs"]) > 0:
            set_github_output({"job_summary": json.dumps(job_data)})


if __name__ == "__main__":
    run()
