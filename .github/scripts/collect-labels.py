import json
import os
import sys
import requests
import yaml


def get_labels(repo, token):
    headers = {"Authorization": f"token {token}"}
    labels = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/labels?page={page}&per_page=100"
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            raise Exception(f"Failed to fetch labels from {repo}: {resp.text}")
        data = resp.json()
        if not data:
            break
        labels.extend(data)
        page += 1
    return labels


def main(file_path):
    with open(file_path, "r") as f:
        repos_data = json.load(f)["repositories"]

    token = os.environ["GH_TOKEN"]
    all_labels = {}

    for repo_entry in repos_data:
        repo_url = repo_entry["url"]
        print(f"Collecting labels from {repo_url}")
        for label in get_labels(repo_url, token):
            name = label["name"]
            if name not in all_labels:
                all_labels[name] = {
                    "name": name,
                    "color": label["color"],
                    "description": label.get("description", ""),
                }

    sorted_labels = sorted(all_labels.values(), key=lambda l: l["name"].lower())
    os.makedirs(".github", exist_ok=True)  # Ensure the .github directory exists
    with open(".github/labels.yml", "w") as out:
        yaml.dump(sorted_labels, out, sort_keys=False)


if __name__ == "__main__":
    main(sys.argv[1])
