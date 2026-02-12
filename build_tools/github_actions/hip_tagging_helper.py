import os
import argparse
import subprocess
import json
from datetime import datetime
from github import Github


ROCM_SYSTEMS_REPO = "ROCm/rocm-systems"


def run_git_command(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def submodule_changed(before, after, path="rocm-systems"):
    """
    Returns True if the submodule SHA changed between commits.
    """
    diff = run_git_command(["git", "diff", before, after, "--", path])
    return bool(diff.strip())


def get_submodule_sha(commit, path="rocm-systems"):
    """
    Returns the submodule SHA recorded in a commit.
    """
    output = run_git_command(["git", "ls-tree", commit, path])
    return output.split()[2]


def get_rocm_version():
    """
    Reads version.json from repo root.
    Returns major.minor
    """
    with open("version.json", "r") as f:
        data = json.load(f)

    full_version = data["rocm-version"]
    return ".".join(full_version.split(".")[:2])


def calculate_patch_tag(dt, version_prefix):
    last_digit_year = str(dt.year)[-1]
    day_of_year = dt.timetuple().tm_yday
    tag_number = f"{last_digit_year}{day_of_year:03}0"
    return f"hip-version_{version_prefix}.{tag_number}"


def tag_exists_for_commit(commit_hash, pat):
    g = Github(pat)
    repo = g.get_repo(ROCM_SYSTEMS_REPO)

    for tag in repo.get_tags():
        if tag.commit.sha == commit_hash:
            print(
                f"[INFO] Tag '{tag.name}' already exists for commit {commit_hash}. Skipping."
            )
            return True
    return False


def create_tag(commit_hash, tag_name, pat):
    from github import Github, Auth

    auth = Auth.Token(pat)
    g = Github(auth=auth)
    repo = g.get_repo("ROCm/rocm-systems")

    print(f"[INFO] Creating tag '{tag_name}' on commit {commit_hash}")

    tag = repo.create_git_tag(
        tag=tag_name,
        message=f"Tag for build {tag_name}",
        object=commit_hash,
        type="commit",
    )

    repo.create_git_ref(ref=f"refs/tags/{tag_name}", sha=tag.sha)

    print("[INFO] Tag created successfully.")

    # Verify
    ref = repo.get_git_ref(f"tags/{tag_name}")
    print(f"[INFO] Tag confirmed at {ref.object.sha}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Repository (ROCm/therock)")
    parser.add_argument("--before", required=True, help="Previous commit SHA")
    parser.add_argument("--after", required=True, help="New commit SHA")
    parser.add_argument("--pat", required=True, help="GitHub PAT")
    args = parser.parse_args()

    print(f"[INFO] Checking submodule changes between {args.before} → {args.after}")

    if not submodule_changed(args.before, args.after):
        print("[INFO] rocm-systems submodule not changed. Exiting.")
        return

    old_sha = get_submodule_sha(args.before)
    new_sha = get_submodule_sha(args.after)

    print("[INFO] rocm-systems submodule updated:")
    print(f"       Old SHA: {old_sha}")
    print(f"       New SHA: {new_sha}")

    if old_sha == new_sha:
        print("[INFO] Submodule SHA identical. Exiting.")
        return

    if tag_exists_for_commit(new_sha, args.pat):
        return

    version_prefix = get_rocm_version()
    current_dt = datetime.now()
    tag_name = calculate_patch_tag(current_dt, version_prefix)

    print(f"[INFO] Generated tag: {tag_name}")

    create_tag(new_sha, tag_name, args.pat)


if __name__ == "__main__":
    main()
