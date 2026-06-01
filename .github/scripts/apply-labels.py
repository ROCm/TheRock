import os
import sys
import yaml
import requests


def get_existing_labels(repo, token):
    headers = {"Authorization": f"token {token}"}
    labels = {}
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/labels?page={page}&per_page=100"
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            raise Exception(f"Failed to fetch existing labels: {resp.text}")
        data = resp.json()
        if not data:
            break
        for label in data:
            labels[label["name"]] = {
                "color": label["color"],
                "description": label.get("description", ""),
            }
        page += 1
    return labels


def create_or_update_label(repo, token, label, existing):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }

    if label["name"] not in existing:
        # Create label
        print(f"Creating label: {label['name']}")
        url = f"https://api.github.com/repos/{repo}/labels"
        resp = requests.post(url, json=label, headers=headers)
    else:
        # Update if different
        current = existing[label["name"]]
        if label["color"].lower() != current["color"].lower() or label.get(
            "description", ""
        ) != current.get("description", ""):
            print(f"Updating label: {label['name']}")
            url = f"https://api.github.com/repos/{repo}/labels/{label['name']}"
            resp = requests.patch(url, json=label, headers=headers)
        else:
            print(f"Label '{label['name']}' already up to date. Skipping.")
            return

    if not resp.ok:
        print(f"Failed to apply label {label['name']}: {resp.status_code} {resp.text}")


def main(label_file):
    token = os.environ["GH_TOKEN"]
    repo = os.environ["GITHUB_REPO"]
    existing = get_existing_labels(repo, token)

    with open(label_file, "r") as f:
        labels = yaml.safe_load(f)

    for label in labels:
        create_or_update_label(repo, token, label, existing)


if __name__ == "__main__":
    main(sys.argv[1])
