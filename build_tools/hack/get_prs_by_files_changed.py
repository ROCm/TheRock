#!/usr/bin/env python3

"""Returns list of PRs that change the same files matching a given pattern.
Pattern is applied as partial matching for the full path name (= dir name + file name).

For more info look below in def get_args(argv)

"""
import subprocess
import argparse
from functools import partial
import sys

from github import Github

# Authentication is defined via github.Auth
from github import Auth

from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor


def check_pr_for_matches(pr, partial_matches_files, owner, repo_name):
    matches = []
    skipped = []

    # skip too large PRs
    # 0 can also be the result if too many lines where changed
    adds = pr.additions
    dels = pr.deletions
    if adds + dels > 100000 or adds + dels == 0:
        skipped.append(
            (
                f"SKIPPED https://github.com/{owner}/{repo_name}/pull/{pr.number}    {pr.title}"
            )
        )
        return [matches, skipped]

    for file in pr.get_files():
        for match in partial_matches_files:
            if match in file.filename:
                matches.append(
                    (
                        f"https://github.com/{owner}/{repo_name}/pull/{pr.number}    {pr.title}"
                    )
                )
                return [matches, skipped]


def get_prs(g, owner: str, repo_name: str, partial_matches_files):
    repo = g.get_repo(f"{owner}/{repo_name}")
    pulls = repo.get_pulls(state="open")
    print("Turn page-wise pulls into a list")
    pull_list = list(pulls)
    print("...done")

    check_fn = partial(
        check_pr_for_matches,
        partial_matches_files=partial_matches_files,
        owner=owner,
        repo_name=repo_name,
    )

    prs_found = []
    prs_skipped = []

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(check_fn, pull_list), total=len(pull_list)))

    # Flatten results
    for r in results:
        if r == None:
            continue
        if not r[0] == None and len(r[0]) > 0:
            prs_found.extend(r[0])
        if not r[1] == None and len(r[1]) > 0:
            prs_skipped.extend(r[1])

    print()
    print("PRs SKIPPED (due to having either zero or >100000 lines changed)")
    for pr in prs_skipped:
        print(pr)
    print()
    print("PRs FOUND")
    for pr in prs_found:
        print(pr)


def get_args(argv):
    p = argparse.ArgumentParser(
        prog="get_prs_by_files_changed.py",
        description="""Returns list of PRs that change the same files matching a given pattern.
Pattern is applied as partial matching for the full path name (= dir name + file name).

GitHub token:
- Creation: Settings --> Developer Settings --> Personal access tokens --> Fine-grained access token
GitHub token needs the following access: None, just the default selection of Public repositories (Read-only access to public repositories.)

Extra pip requirements:
pip install PyGithub tqdm

If you see:
Request GET /repos/ROCm/TheRock/pulls/1879/files failed with 403: Forbidden
Setting next backoff to 1626.936613s

Then you run into the GitHub rate limit.
See https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api
""",
        usage='python3 get_prs_by_files_changed.py --token-file "./git_token.txt" --repo rocm-systems --match "README,.py"',
    )
    p.add_argument(
        "--owner",
        default="ROCm",
        help="GitHub Repo Owner (default: ROCm)",
    )
    p.add_argument(
        "--repo",
        default="TheRock",
        help="GitHub Repository Name (default: TheRock)",
    )
    p.add_argument(
        "--match",
        required=True,
        help="Comma separated list of partial matches of directoy and file names.",
    )
    tokengroup = p.add_mutually_exclusive_group(required=True)
    tokengroup.add_argument(
        "--token-file",
        type=str,
        help="Path to (fine-grained) GitHub Token that can access the owner/repo read-only",
    )
    tokengroup.add_argument(
        "--token",
        type=str,
        help="Clear-text (fine-grained) GitHub Token <THINK TWICE WHO CAN HAVE ACCESS TO YOUR CMD HISTORY>",
    )
    args = p.parse_args(argv)
    return args


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    if args.token == None:
        with open(args.token_file, "r", encoding="utf-8") as file:
            content = file.read().strip()
            auth = Auth.Token(content)
    else:
        auth = Auth.Token(args.token)
    g = Github(auth=auth)

    matches = args.match.split(",")
    print(f"GitHub Repo: {args.owner}/{args.repo}")
    print(f"Looking for PRs that changed files matching: {matches}")
    get_prs(g, args.owner, args.repo, matches)
    g.close()
