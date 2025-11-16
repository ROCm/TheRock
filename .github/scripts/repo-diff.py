# This script generates a report for TheRock highlighting the difference in commits for each component between 2 builds.

# Imports
import requests
import re
import os
from urllib.parse import urlparse
import argparse
import base64
import subprocess

from collections import defaultdict

# Global variables for GitHub API access
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
GITHUB_HEADERS = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

def get_rocm_components(repo):
    """Get components from ROCm repositories (shared and projects directories)"""
    components = []

    # If the repo is rocm-libraries fetch from shared and projects subfolders
    if repo == "rocm-libraries":
        url = f"https://api.github.com/repos/ROCm/{repo}/contents/shared"
        print(f"Requesting: {url}")
        resp = requests.get(url, headers=GITHUB_HEADERS)
        print(f"GitHub API status (shared): {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            for item in data:
                print(f"Item: {item.get('name')} type: {item.get('type')}")
                if item['type'] == 'dir':
                    components.append("shared/" + item['name'])
        else:
            print(f"Failed to fetch shared folder from GitHub: {resp.status_code} {resp.text}")

    # Fetch the components in the projects directory
    url = f"https://api.github.com/repos/ROCm/{repo}/contents/projects"
    print(f"Requesting: {url}")
    resp = requests.get(url, headers=GITHUB_HEADERS)
    print(f"GitHub API status (projects): {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        for item in data:
            print(f"Item: {item.get('name')} type: {item.get('type')}")
            if item['type'] == 'dir':
                components.append("projects/" + item['name'])
    else:
        print(f"Failed to fetch projects folder from GitHub: {resp.status_code} {resp.text}")

    return components

def fetch_commits_range(repo, start_rev, end_rev):
    """Fetch commits between start_rev and end_rev from ROCm repository"""
    url = f"https://api.github.com/repos/ROCm/{repo}/commits"

    params = {
        "sha": end_rev,
        "per_page": 100
    }
    commits = []
    found_start = False
    while True:
        resp = requests.get(url, headers=GITHUB_HEADERS, params=params)
        if resp.status_code != 200:
            break
        data = resp.json()
        for commit in data:
            sha = commit['sha']
            print(f"Processing commit: {sha}")
            # Enrich commit with files
            commit_url = f"https://api.github.com/repos/ROCm/{repo}/commits/{sha}"
            commit_resp = requests.get(commit_url, headers=GITHUB_HEADERS)
            if commit_resp.status_code == 200:
                commit_full = commit_resp.json()
                commits.append(commit_full)
            else:
                commits.append(commit)  # fallback to original if API fails
            if sha == start_rev:
                found_start = True
                break
        if found_start:
            break
        # Pagination
        if 'next' in resp.links:
            url = resp.links['next']['url']
            params = None
        else:
            break
    return commits

def allocate_commits_to_projects(commits, projects):
    """Allocate commits to projects based on file paths touched."""
    allocation = defaultdict(list)

    for commit in commits:
        sha = commit.get('sha', '-')
        print(f"Handling commit: {sha}")
        files = commit.get('files', [])
        assigned_projects = set()

        # Check which projects this commit touches
        for f in files:
            filename = f.get('filename', '')
            print(f"  Analyzing file: {filename}")

            # Check each project to see if this file belongs to it
            for project in projects:
                if filename.startswith(f"{project}/"):
                    assigned_projects.add(project)
                    print(f"    Matches project: {project}")
                    break  # Found a match, no need to check other projects for this file

        # Assign commit to all matching projects
        if assigned_projects:
            for project in assigned_projects:
                allocation[project].append(commit)
                print(f"    Assigned to project: {project}")
        else:
            allocation['Unassigned'].append(commit)
            print(f"  Assigned to: Unassigned")

    return dict(allocation)

def create_commit_badge_html(sha, repo_name):
    """Create a styled HTML badge for a commit SHA with link"""
    short_sha = sha[:7] if sha != '-' and sha != 'N/A' else sha
    commit_url = f"https://github.com/ROCm/{repo_name}/commit/{sha}"
    return (
        f"<a href='{commit_url}' target='_blank' style='text-decoration:none;'>"
        f"<span style='display:inline-block; background-color:#2196F3; color:#fff; padding:2px 8px; font-family:Verdana,sans-serif; font-size:12px; border-radius:2px;'>{short_sha}</span>"
        f"</a>"
    )

def create_commit_item_html(commit, repo_name):
    """Create HTML for a single commit item with badge and message"""
    sha = commit.get('sha', '-')
    msg = commit.get('commit', {}).get('message', '-').split('\n')[0]
    badge_html = create_commit_badge_html(sha, repo_name)
    return f"<div style='margin-bottom:4px;'>{badge_html} {msg}</div>"

def create_commit_list_container(commit_items):
    """Create a scrollable container for commit items"""
    return (
        f"<div class='commit-list' style='max-height:120px; overflow-y:auto; padding:4px; border:1px solid #ccc; font-family:Verdana,sans-serif; font-size:13px; line-height:1.4; background-color:#f9f9f9;'>"
        f"{''.join(commit_items)}</div>"
    )

def create_table_wrapper(headers, rows):
    """Create a styled HTML table with headers and rows"""
    header_html = "".join([f"<th style='padding:8px; border:1px solid #ccc; text-align:left;'>{header}</th>" for header in headers])
    return (
        "<table style='width:100%; border-collapse:collapse; border:1px solid #ccc; font-family:Verdana,sans-serif; font-size:14px;'>"
        f"<tr style='background-color:#f1f1f1;'>{header_html}</tr>"
        + "".join(rows) +
        "</table>"
    )

def generate_commit_diff_html_table(allocation, original_commits=None, repo_name="rocm-libraries"):
    """Create a styled HTML table for commit differences"""
    rows = []
    commit_seen = set()
    commit_to_projects = {}
    unassigned_proj = None
    unassigned_list = None

    for component, commits in allocation.items():
        commit_items = []
        for commit in commits:
            sha = commit.get('sha', '-')
            if sha not in commit_to_projects:
                commit_to_projects[sha] = set()
            # Only add 'Unassigned' if this is the only association
            if component != 'Unassigned':
                commit_to_projects[sha].add(component)
            elif not commit_to_projects[sha]:
                commit_to_projects[sha].add('Unassigned')

            commit_items.append(create_commit_item_html(commit, repo_name))
            commit_seen.add(sha)

        # Create scrollable list for commits
        commit_list_html = create_commit_list_container(commit_items)

        if component == "Unassigned":
            unassigned_proj = component
            unassigned_list = commit_list_html
        else:
            rows.append(
                f"<tr>"
                f"<td style='padding:8px; border:1px solid #ccc; vertical-align:top;'>{component}</td>"
                f"<td style='padding:8px; border:1px solid #ccc; vertical-align:top;'>{commit_list_html}</td>"
                f"</tr>"
            )

    if unassigned_list:
        rows.append(
            f"<tr>"
            f"<td style='padding:8px; border:1px solid #ccc; vertical-align:top;'>{unassigned_proj}</td>"
            f"<td style='padding:8px; border:1px solid #ccc; vertical-align:top;'>{unassigned_list}</td>"
            f"</tr>"
        )

    table = create_table_wrapper(["Component", "Commits"], rows)
    # List commit-project associations below the table
    commit_project_list = []
    if original_commits is not None:
        for commit in original_commits:
            sha = commit.get('sha', '-')
            msg = commit.get('commit', {}).get('message', '-').split('\n')[0]
            projects = ', '.join(sorted(commit_to_projects.get(sha, []))) if sha in commit_to_projects else 'Unassigned'
            badge_html = create_commit_badge_html(sha, repo_name)

            commit_project_list.append(
                f"<div style='margin-bottom:4px;'>"
                f"<b>{projects}</b> {badge_html} {msg}</div>"
            )

    commit_projects_html = (
        "<div style='margin-top:16px; font-weight:bold;'>Commit Project Associations (in commit history order):</div>"
        + "".join(commit_project_list)
    )

    return table + commit_projects_html

def find_repo_head(workflowId):
    """Get commit SHA for a TheRock workflow run using GitHub API"""
    print(f"Looking for commit SHA for workflow {workflowId}")

    url = f"https://api.github.com/repos/ROCm/TheRock/actions/runs/{workflowId}"

    try:
        response = requests.get(url, headers=GITHUB_HEADERS)
        response.raise_for_status()
        data = response.json()

        # The head_sha field contains the commit SHA that triggered the workflow
        head_commit = data.get('head_sha')
        print(f"Found commit SHA via API: {head_commit}")
        return head_commit

    except requests.exceptions.RequestException as e:
        print(f"Error fetching workflow info via API: {e}")
        return None

def get_submodule_paths_from_gitmodules(commit_sha):
    """Get submodule paths from the .gitmodules file for a specific commit"""
    gitmodules_url = f"https://api.github.com/repos/ROCm/TheRock/contents/.gitmodules?ref={commit_sha}"

    try:
        response = requests.get(gitmodules_url, headers=GITHUB_HEADERS)
        response.raise_for_status()

        gitmodules_data = response.json()
        if gitmodules_data.get('encoding') == 'base64':
            # Decode the base64 content
            content = base64.b64decode(gitmodules_data['content']).decode('utf-8')

            # Parse the .gitmodules content to extract paths
            submodule_paths = set()
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('path ='):
                    path = line.split('path =')[1].strip()
                    submodule_paths.add(path)
                    print(f"Found submodule path in .gitmodules: {path}")

            return submodule_paths
        else:
            print("Error: .gitmodules file encoding not supported")
            return set()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching .gitmodules file: {e}")
        return set()

def get_submodule_commit_at_path(commit_sha, submodule_path):
    """Get commit SHA for a submodule at a specific path using GitHub API"""
    try:
        # Use Contents API to get the submodule at the specific path
        contents_url = f"https://api.github.com/repos/ROCm/TheRock/contents/{submodule_path}?ref={commit_sha}"
        response = requests.get(contents_url, headers=GITHUB_HEADERS)
        response.raise_for_status()

        content_data = response.json()
        # For submodules, the 'sha' field contains the pinned commit SHA
        if content_data.get('type') == 'submodule':
            return content_data.get('sha')
        else:
            print(f"Warning: {submodule_path} is not a submodule")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error getting submodule commit for {submodule_path}: {e}")
        return None

def find_submodules(commit_sha):
    """Find submodules and their commit SHAs for a TheRock commit"""

    # Get submodule paths from .gitmodules file
    submodule_paths = get_submodule_paths_from_gitmodules(commit_sha)
    if not submodule_paths:
        print("No submodules found in .gitmodules")
        return {}

    # Get the actual commit SHAs from the git tree
    submodules = {}
    for path in submodule_paths:
        submodule_sha = get_submodule_commit_at_path(commit_sha, path)
        if submodule_sha:
            # Extract just the submodule name from the path (last part after /)
            submodule_name = path.split('/')[-1]

            # Hard-coded fixes for special submodule names
            if path == "profiler/rocprof-trace-decoder/binaries":
                submodule_name = "rocprof-trace-decoder"
            elif submodule_name == "amd-llvm":
                submodule_name = "llvm-project"

            submodules[submodule_name] = submodule_sha
            print(f"Found submodule: {submodule_name} (path: {path}) -> {submodule_sha}")
        else:
            print(f"Warning: Could not find commit SHA for submodule at {path}")

    return submodules

def get_commits_in_range(start, end, repo_name):
    """Fetch commits between start and end commit SHAs from ROCm repository"""

    # Fetch the commits between the start and end range
    commits_url = f"https://api.github.com/repos/ROCm/{repo_name}/commits?sha={end}&since={start}&until={end}"
    response = requests.get(commits_url, headers=GITHUB_HEADERS)
    response.raise_for_status()

    return response.json()

def generate_non_monorepo_html_table(submodule_commits):
    """Generate an HTML table for non-monorepo components"""
    rows = []

    for submodule, commits in submodule_commits.items():
        commit_items = []
        if commits:
            for commit in commits:
                commit_items.append(create_commit_item_html(commit, submodule))
        else:
            commit_items.append("<div>No commits found</div>")

        # Create scrollable list for commits
        commit_list_html = create_commit_list_container(commit_items)

        rows.append(
            f"<tr>"
            f"<td style='padding:8px; border:1px solid #ccc; vertical-align:top;'>{submodule}</td>"
            f"<td style='padding:8px; border:1px solid #ccc; vertical-align:top;'>{commit_list_html}</td>"
            f"</tr>"
        )

    return create_table_wrapper(["Submodule", "Commits"], rows)

def generate_therock_html_report(html_reports):
    """Generate a comprehensive HTML report for TheRock repository diff"""
    print(f"\n=== Generating Comprehensive HTML Report ===")

    # Read template
    with open("report_template.html", "r") as f:
        template = f.read()

    # Check what sections have content and populate accordingly
    rocm_lib_data = html_reports.get('rocm-libraries')
    rocm_sys_data = html_reports.get('rocm-systems')
    non_monorepo_data = html_reports.get('non-monorepo')

    # Populate ROCm-Libraries Monorepo container
    if rocm_lib_data:
        template = template.replace('<span id="commit-diff-start-rocm-libraries-monorepo"></span>', rocm_lib_data['start_commit'])
        template = template.replace('<span id="commit-diff-end-rocm-libraries-monorepo"></span>', rocm_lib_data['end_commit'])
        template = template.replace(
            '<div id="commit-diff-job-content-rocm-libraries-monorepo" style="margin-top:8px;">\n      </div>',
            f'<div id="commit-diff-job-content-rocm-libraries-monorepo" style="margin-top:8px;">{rocm_lib_data["content_html"]}</div>'
        )
        print("Populated ROCm-Libraries monorepo section")

    # Populate ROCm-Systems Monorepo container
    if rocm_sys_data:
        template = template.replace('<span id="commit-diff-start-rocm-systems-monorepo"></span>', rocm_sys_data['start_commit'])
        template = template.replace('<span id="commit-diff-end-rocm-systems-monorepo"></span>', rocm_sys_data['end_commit'])
        template = template.replace(
            '<div id="commit-diff-job-content-rocm-systems-monorepo" style="margin-top:8px;">\n      </div>',
            f'<div id="commit-diff-job-content-rocm-systems-monorepo" style="margin-top:8px;">{rocm_sys_data["content_html"]}</div>'
        )
        print("Populated ROCm-Systems monorepo section")

    # Populate Non-Monorepo container
    if non_monorepo_data and non_monorepo_data.get('content_html') and non_monorepo_data['content_html'].strip():
        template = template.replace('<span id="commit-diff-start-non-monorepo"></span>', non_monorepo_data['start_commit'])
        template = template.replace('<span id="commit-diff-end-non-monorepo"></span>', non_monorepo_data['end_commit'])
        template = template.replace(
            '<div id="commit-diff-job-content-non-monorepo" style="margin-top:8px;">\n      </div>',
            f'<div id="commit-diff-job-content-non-monorepo" style="margin-top:8px;">{non_monorepo_data["content_html"]}</div>'
        )
        print("Populated Non-Monorepo section")

    # Write the final TheRock HTML report
    with open("TheRockReport.html", "w") as f:
        f.write(template)

    print("Generated TheRockReport.html successfully!")

# Main Function
def main():
    # Arguments parsed
    parser = argparse.ArgumentParser(description="Generate HTML report for repo diffs")
    parser.add_argument("--start", required=True, help="Start workflow ID or commit SHA")
    parser.add_argument("--end", required=True, help="End workflow ID or commit SHA")
    parser.add_argument("--workflow-mode", action="store_true", help="Use workflow mode (extract commits from workflow logs). If not set, uses commit mode (direct commit comparison)")

    args = parser.parse_args()

    # Determine mode
    if args.workflow_mode:
        print("Running in WORKFLOW mode - extracting commits from workflow logs")
        mode = "workflow"
    else:
        print("Running in COMMIT mode - direct commit comparison")
        mode = "commit"

    print(f"Start: {args.start}")
    print(f"End: {args.end}")
    print(f"Mode: {mode}")

    if mode == "workflow":
        # Extract commits from workflow logs
        start = find_repo_head(args.start)
        end = find_repo_head(args.end)
    else:
        # Direct commit comparison
        start = args.start
        end = args.end

    print(f"Start commit: {start}")
    print(f"End commit: {end}")

    if not start or not end:
        print("Error: Could not determine start or end commits")
        return

    # Store the submodules and their commits for the start and end
    print("\n=== Getting submodules for START commit ===")
    old_submodules = find_submodules(start)

    print("\n=== Getting submodules for END commit ===")
    new_submodules = find_submodules(end)

    print(f"\n=== COMPARISON RESULTS ===")
    print(f"Found {len(old_submodules)} submodules in start commit")
    print(f"Found {len(new_submodules)} submodules in end commit")

    # Compare submodules and get commit history for changed ones
    submodule_commits = {}
    notseen_submodules = []
    html_reports = {}

    for submodule in old_submodules.keys():
        if submodule in new_submodules:
            if submodule == "rocm-systems" or submodule == "rocm-libraries":
                print(f"\n=== Processing {submodule.upper()} monorepo ===")

                # Get the components for this monorepo
                components = get_rocm_components(submodule)

                # Fetch commits between the old and new SHA
                commits = fetch_commits_range(submodule, old_submodules[submodule], new_submodules[submodule])

                # Allocate commits to components
                allocation = allocate_commits_to_projects(commits, components)

                # Generate HTML table
                html_table = generate_commit_diff_html_table(allocation, commits, submodule)

                # Store the HTML report
                html_reports[submodule] = {
                    'start_commit': old_submodules[submodule],
                    'end_commit': new_submodules[submodule],
                    'content_html': html_table
                }

                print(f"Generated HTML report for {submodule}")

            else:
                # For other submodules, we can just print the commit SHAs we want to store them in a dictonary
                submodule_commits[submodule] = get_commits_in_range(old_submodules[submodule], new_submodules[submodule], submodule)
        else:
            # Append a list of submodules not in new submodules
            notseen_submodules.append(submodule)

    # Print all the submodules and their commits
    print(f"\n=== SUBMODULE COMMIT DETAILS ===")
    for submodule, commits in submodule_commits.items():
        print(f"\n--- {submodule.upper()} ---")
        if commits:
            for commit in commits:
                sha = commit.get('sha', 'N/A')
                short_sha = sha[:7] if sha != 'N/A' else 'N/A'
                message = commit.get('commit', {}).get('message', 'No message').split('\n')[0]
                author = commit.get('commit', {}).get('author', {}).get('name', 'Unknown')
                date = commit.get('commit', {}).get('author', {}).get('date', 'Unknown')
                print(f"  {short_sha} - {author} ({date}): {message}")
        else:
            print(f"  No commits found for {submodule}")

    print(f"\nTotal submodules with commits: {len(submodule_commits)}")

    # Generate HTML report for non-monorepo submodules
    print(f"\n=== Generating Non-Monorepo HTML Report ===")
    non_monorepo_html = generate_non_monorepo_html_table(submodule_commits)

    # Store non-monorepo HTML report with TheRock start/end commits
    html_reports['non-monorepo'] = {
        'start_commit': start,  # TheRock start commit
        'end_commit': end,      # TheRock end commit
        'content_html': non_monorepo_html
    }

    # Generate the comprehensive HTML report
    generate_therock_html_report(html_reports)


if __name__ == "__main__":
    main()