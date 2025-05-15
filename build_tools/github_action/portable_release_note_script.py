from configure_ci import set_github_output
import json
import os
from urllib.request import urlopen, Request

TAG_NAME = os.getenv("TAG_NAME")
VERSION = os.getenv("VERSION")

def run():
    github_release_url = (
        f"https://api.github.com/repos/ROCm/TheRock/releases/tags/{TAG_NAME}"
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    # If GITHUB_TOKEN environment variable is available, include it in the API request to avoid a lower rate limit
    gh_token = os.getenv("GITHUB_TOKEN", "")
    if gh_token:
        headers["Authentication"] = f"Bearer {gh_token}"

    request = Request(github_release_url, headers=headers)
    with urlopen(request) as response:
        if response.status == 403:
            print(
                f"Error when retrieving GitHub release assets for release tag '{TAG_NAME}'. This is most likely a rate limiting issue, so please try again"
            )
            return
        elif response.status != 200:
            print(
                f"Error when retrieving GitHub release assets for release tag '{TAG_NAME}' with status code {response.status}. Exiting..."
            )
            return

        release_data = json.loads(response.read().decode("utf-8"))
        append_release_note = VERSION in release_data["body"]
        set_github_output({"append": json.dumps(append_release_note)})

if __name__ == "__main__":
    run()
