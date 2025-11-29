# This script generates a report for TheRock highlighting the difference in commits for each component between 2 builds.

# Imports
import re
import os
from urllib.parse import urlparse
import argparse
import base64
import subprocess
import urllib.parse
from collections import defaultdict

# Import GitHub Actions utilities
from github_actions_utils import gha_append_step_summary, gha_query_workflow_run_information, gha_send_request

# HTML Helper Functions
def format_commit_date(date_string):
    """Format ISO date string to readable format"""
    if date_string == 'Unknown' or not date_string:
        return 'Unknown'
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        return dt.strftime('%b %d, %Y')
    except:
        return date_string

def create_commit_badge_html(sha, repo_name):
    """Create a styled HTML badge for a commit SHA with link"""
    short_sha = sha[:7] if sha != '-' and sha != 'N/A' else sha
    commit_url = f"https://github.com/ROCm/{repo_name}/commit/{sha}"
    return (
        f"<a href='{commit_url}' target='_blank' style='text-decoration:none;'>"
        f"<span style='display:inline-block; background-color:#1976D2; color:#fff; padding:2px 8px; font-family:Verdana,sans-serif; font-size:12px; border-radius:2px;'>{short_sha}</span>"
        f"</a>"
    )

def extract_commit_data(commit):
    """Extract common commit data to avoid redundancy"""
    return {
        'sha': commit.get('sha', '-'),
        'message': commit.get('commit', {}).get('message', '-').split('\n')[0],
        'author': commit.get('commit', {}).get('author', {}).get('name', 'Unknown'),
        'date': commit.get('commit', {}).get('author', {}).get('date', 'Unknown'),
        'formatted_date': format_commit_date(commit.get('commit', {}).get('author', {}).get('date', 'Unknown'))
    }

def create_commit_item_html(commit, repo_name):
    """Create HTML for a single commit item with badge, message, author and date"""
    commit_data = extract_commit_data(commit)
    badge_html = create_commit_badge_html(commit_data['sha'], repo_name)

    return (
        f"<div style='margin-bottom:8px; border-left:3px solid #1976D2; padding-left:8px;'>"
        f"<div>{badge_html} {commit_data['message']}</div>"
        f"<div style='color:#666; font-size:11px; margin-top:2px;'>{commit_data['formatted_date']} • {commit_data['author']}</div>"
        f"</div>"
    )

def create_commit_list_container(commit_items):
    """Create a scrollable container for commit items"""
    content = ''.join(commit_items) if commit_items else '<div style="color:#888; font-style:italic; text-align:center; padding:8px;">Component not changed</div>'
    return (
        f"<div class='commit-list' style='max-height:160px; overflow-y:auto; padding:4px; border:1px solid #ccc; font-family:Verdana,sans-serif; font-size:13px; line-height:1.4; background-color:#f9f9f9;'>"
        f"{content}</div>"
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

# HTML Table Functions
def generate_superrepo_html_table(allocation, all_commits, repo_name):
    """Create a styled HTML table for superrepo commit differences with project allocation"""
    rows = []
    commit_to_projects = {}

    # Build commit-to-projects mapping efficiently from allocation
    for component, commits in allocation.items():
        for commit in commits:
            commit_data = extract_commit_data(commit)
            sha = commit_data['sha']
            if sha not in commit_to_projects:
                commit_to_projects[sha] = set()
            commit_to_projects[sha].add(component)

    # Generate component rows directly from allocation
    for component, commits in allocation.items():
        # Convert commits to HTML items
        commit_items = [create_commit_item_html(commit, repo_name) for commit in commits]
        commit_list_html = create_commit_list_container(commit_items)

        rows.append(
            f"<tr>"
            f"<td style='padding:8px; border:1px solid #ccc; vertical-align:top;'>{component}</td>"
            f"<td style='padding:8px; border:1px solid #ccc; vertical-align:top;'>{commit_list_html}</td>"
            f"</tr>"
        )

    table = create_table_wrapper(["Component", "Commits"], rows)

    # Create commit-project associations table using all_commits
    commit_projects_html = ""
    if all_commits:
        project_table_rows = []
        for commit in all_commits:
            commit_data = extract_commit_data(commit)
            projects = ', '.join(sorted(commit_to_projects.get(commit_data['sha'], []))) if commit_data['sha'] in commit_to_projects else 'Unassigned'
            badge_html = create_commit_badge_html(commit_data['sha'], repo_name)

            project_table_rows.append(
                f"<tr>"
                f"<td style='padding:8px; border:1px solid #ccc; font-size:12px; color:#666;'>{commit_data['formatted_date']}</td>"
                f"<td style='padding:8px; border:1px solid #ccc;'>{badge_html}</td>"
                f"<td style='padding:8px; border:1px solid #ccc; font-size:12px; color:#555;'>{commit_data['author']}</td>"
                f"<td style='padding:8px; border:1px solid #ccc; font-weight:bold;'>{projects}</td>"
                f"<td style='padding:8px; border:1px solid #ccc; font-size:13px;'>{commit_data['message']}</td>"
                f"</tr>"
            )

        if project_table_rows:
            commit_projects_html = (
                "<div style='margin-top:16px; font-weight:bold;'>Commit Project Associations (in newest commit to oldest commit order):</div>"
                "<table style='width:100%; border-collapse:collapse; border:1px solid #ccc; margin-top:8px;'>"
                "<tr style='background-color:#f8f9fa;'>"
                "<th style='padding:8px; border:1px solid #ccc; text-align:left; width:80px;'>Date</th>"
                "<th style='padding:8px; border:1px solid #ccc; text-align:left; width:60px;'>SHA</th>"
                "<th style='padding:8px; border:1px solid #ccc; text-align:left; width:120px;'>Author</th>"
                "<th style='padding:8px; border:1px solid #ccc; text-align:left; width:150px;'>Project(s)</th>"
                "<th style='padding:8px; border:1px solid #ccc; text-align:left;'>Message</th>"
                "</tr>"
                + "".join(project_table_rows) +
                "</table>"
            )

    return table + commit_projects_html

def generate_non_superrepo_html_table(submodule_commits):
    """Generate an HTML table for other components"""
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

def generate_therock_html_report(html_reports, removed_submodules=None, newly_added_submodules=None, unchanged_submodules=None, changed_submodules=None):
    """Generate a comprehensive HTML report for TheRock repository diff"""
    print(f"\n=== Generating Comprehensive HTML Report ===")

    # Read template
    with open("report_template.html", "r") as f:
        template = f.read()

    # Generate and populate submodule changes summary
    summary_html = ""
    if removed_submodules or newly_added_submodules or unchanged_submodules or changed_submodules:
        # Calculate total number of submodules
        total_submodules = len(newly_added_submodules or []) + len(removed_submodules or []) + len(unchanged_submodules or []) + len(changed_submodules or [])
        summary_html += '<div style="background-color:#ffffff; padding:16px; margin-bottom:3em; box-shadow:0 2px 5px rgba(0,0,0,0.16), 0 2px 10px rgba(0,0,0,0.12);">'
        summary_html += '<div style="text-align:center; color:#1976D2; font-size:2.2em; font-weight:bold; margin-bottom:16px;">Submodule Changes Summary</div>'
        summary_html += '<div style="text-align:center; color:#666; font-size:0.9em; margin-bottom:20px; padding:8px; background-color:#f8f9fa; border-radius:4px; border-left:4px solid #17a2b8;"><strong>Note:</strong> This summary shows only the direct submodules found in TheRock repository (start and end commits). Components within the superrepos (ROCm-Libraries and ROCm-Systems) are detailed separately in their respective sections below and are not counted in these submodule totals.</div>'

        if newly_added_submodules:
            summary_html += '<div style="margin-bottom:16px;">'
            summary_html += f'<h3 style="color:#28a745; margin-bottom:8px;">Newly Added Submodules ({len(newly_added_submodules)}/{total_submodules}):</h3>'
            summary_html += '<ul style="margin:0; padding-left:20px;">'
            for sub in sorted(newly_added_submodules):
                summary_html += f'<li style="color:#6c757d; margin-bottom:4px;"><code>{sub}</code></li>'
            summary_html += '</ul></div>'

        if removed_submodules:
            summary_html += '<div style="margin-bottom:16px;">'
            summary_html += f'<h3 style="color:#dc3545; margin-bottom:8px;">Removed Submodules ({len(removed_submodules)}/{total_submodules}):</h3>'
            summary_html += '<ul style="margin:0; padding-left:20px;">'
            for sub in sorted(removed_submodules):
                summary_html += f'<li style="color:#6c757d; margin-bottom:4px;"><code>{sub}</code></li>'
            summary_html += '</ul></div>'

        if changed_submodules:
            summary_html += '<div style="margin-bottom:16px;">'
            summary_html += f'<h3 style="color:#ffc107; margin-bottom:8px;">Changed Submodules ({len(changed_submodules)}/{total_submodules}):</h3>'
            summary_html += '<ul style="margin:0; padding-left:20px;">'
            for sub in sorted(changed_submodules):
                summary_html += f'<li style="color:#6c757d; margin-bottom:4px;"><code>{sub}</code></li>'
            summary_html += '</ul></div>'

        if unchanged_submodules:
            summary_html += '<div>'
            summary_html += f'<h3 style="color:#6c757d; margin-bottom:8px;">Unchanged Submodules ({len(unchanged_submodules)}/{total_submodules}):</h3>'
            summary_html += '<ul style="margin:0; padding-left:20px;">'
            for sub in sorted(unchanged_submodules):
                summary_html += f'<li style="color:#6c757d; margin-bottom:4px;"><code>{sub}</code></li>'
            summary_html += '</ul></div>'

        summary_html += '</div>'

    # Insert summary at the top
    template = template.replace(
        '<div id="submodule-summary"></div>',
        f'<div id="submodule-summary">{summary_html}</div>'
    )

    # Check what sections have content and populate accordingly
    rocm_lib_data = html_reports.get('rocm-libraries')
    rocm_sys_data = html_reports.get('rocm-systems')
    non_superrepo_data = html_reports.get('non-superrepo')

    # Populate ROCm-Libraries Superrepo container
    if rocm_lib_data:
        template = template.replace('<span id="commit-diff-start-rocm-libraries-superrepo"></span>', rocm_lib_data['start_commit'])
        template = template.replace('<span id="commit-diff-end-rocm-libraries-superrepo"></span>', rocm_lib_data['end_commit'])
        template = template.replace(
            '<div id="commit-diff-job-content-rocm-libraries-superrepo" style="margin-top:8px;">\n      </div>',
            f'<div id="commit-diff-job-content-rocm-libraries-superrepo" style="margin-top:8px;">{rocm_lib_data["content_html"]}</div>'
        )
        print("Populated ROCm-Libraries superrepo section")

    # Populate ROCm-Systems Superrepo container
    if rocm_sys_data:
        template = template.replace('<span id="commit-diff-start-rocm-systems-superrepo"></span>', rocm_sys_data['start_commit'])
        template = template.replace('<span id="commit-diff-end-rocm-systems-superrepo"></span>', rocm_sys_data['end_commit'])
        template = template.replace(
            '<div id="commit-diff-job-content-rocm-systems-superrepo" style="margin-top:8px;">\n      </div>',
            f'<div id="commit-diff-job-content-rocm-systems-superrepo" style="margin-top:8px;">{rocm_sys_data["content_html"]}</div>'
        )
        print("Populated ROCm-Systems superrepo section")

    # Populate Other Components container
    if non_superrepo_data and non_superrepo_data.get('content_html') and non_superrepo_data['content_html'].strip():
        template = template.replace('<span id="commit-diff-start-non-superrepo"></span>', non_superrepo_data['start_commit'])
        template = template.replace('<span id="commit-diff-end-non-superrepo"></span>', non_superrepo_data['end_commit'])
        template = template.replace(
            '<div id="commit-diff-job-content-non-superrepo" style="margin-top:8px;">\n      </div>',
            f'<div id="commit-diff-job-content-non-superrepo" style="margin-top:8px;">{non_superrepo_data["content_html"]}</div>'
        )
        print("Populated Other Components section")

    # Write the final TheRock HTML report
    with open("TheRockReport.html", "w") as f:
        f.write(template)

    print("Generated TheRockReport.html successfully!")

# TheRock Helper Functions
def get_rocm_components(repo):
    """Get components from ROCm superrepo repositories (shared and projects directories)"""
    components = []

    # If the repo is rocm-libraries fetch from shared and projects subfolders
    if repo == "rocm-libraries":
        url = f"https://api.github.com/repos/ROCm/{repo}/contents/shared"
        print(f"Requesting: {url}")
        try:
            data = gha_send_request(url)
            for item in data:
                print(f"Item: {item.get('name')} type: {item.get('type')}")
                if item['type'] == 'dir':
                    components.append("shared/" + item['name'])
        except Exception as e:
            print(f"Failed to fetch shared folder from GitHub: {e}")

    # Fetch the components in the projects directory
    url = f"https://api.github.com/repos/ROCm/{repo}/contents/projects"
    print(f"Requesting: {url}")
    try:
        data = gha_send_request(url)
        for item in data:
            print(f"Item: {item.get('name')} type: {item.get('type')}")
            if item['type'] == 'dir':
                components.append("projects/" + item['name'])
    except Exception as e:
        print(f"Failed to fetch projects folder from GitHub: {e}")

    return components

def get_commits_by_directories(repo_name, start_sha, end_sha, project_directories):
    """
    Get commits by project directories using GitHub API path parameter.

    Args:
        repo_name: Repository name (e.g., 'rocm-libraries')
        start_sha: Starting commit SHA (exclusive - commits after this)
        end_sha: Ending commit SHA (inclusive - commits up to this)
        project_directories: List of directory paths with trailing slashes (e.g., ['projects/hip/', 'shared/rocm-cmake/'])

    Returns:
        Tuple of (allocation, all_commits):
        - allocation: Dictionary with project names as keys, commit lists as values (includes 'Unassigned' if needed)
        - all_commits: List of all commits in chronological order (newest to oldest)
    """
    print(f"  Getting commits by directories for {repo_name} from {start_sha} to {end_sha}")

    # Step 1: Get all commits in range (GitHub API returns newest first by default)
    all_commits = []
    found_start = False
    page = 1
    max_pages = 20

    print(f"  Getting all commits in range...")
    while not found_start and page <= max_pages:
        params = {"sha": end_sha, "per_page": 100, "page": page}
        url = f"https://api.github.com/repos/ROCm/{repo_name}/commits?{urllib.parse.urlencode(params)}"

        try:
            data = gha_send_request(url)
            if not data:
                break

            for commit in data:
                all_commits.append(commit)
                if commit['sha'] == start_sha:
                    found_start = True
                    break

            if len(data) < params['per_page']:
                break
            page += 1

        except Exception as e:
            print(f"  Error fetching commits: {e}")
            break

    # Keep commits in newest-to-oldest order (as received from GitHub API)
    print(f"  Found {len(all_commits)} commits in range")

    # Create SHA set for fast range checking when filtering directory commits
    commit_shas_in_range = {commit['sha'] for commit in all_commits}

    # Step 2: Query each directory and build allocation
    allocation = {}
    all_seen_commits = set()

    for directory in project_directories:
        project_name = directory.rstrip('/')
        print(f"  Querying directory: {directory}")
        allocation[project_name] = []

        # Query commits that touched this specific directory path
        page = 1
        directory_commits = []

        while page <= max_pages:
            params = {
                "sha": end_sha,
                "path": directory,
                "per_page": 100,
                "page": page
            }
            url = f"https://api.github.com/repos/ROCm/{repo_name}/commits?{urllib.parse.urlencode(params)}"

            try:
                data = gha_send_request(url)
                if not data:
                    break

                commits_found_this_page = 0
                for commit in data:
                    sha = commit['sha']

                    # Only include commits that are in our target range
                    if sha in commit_shas_in_range:
                        directory_commits.append(commit)
                        all_seen_commits.add(sha)
                        commits_found_this_page += 1

                        # Stop if we've reached the start commit
                        if sha == start_sha:
                            break

                print(f"    Page {page}: Found {commits_found_this_page} commits in range")

                # If no commits found in range on this page, we've probably gone beyond our range
                if commits_found_this_page == 0 and page > 1:
                    break

                if len(data) < params['per_page']:
                    break

                page += 1

            except Exception as e:
                print(f"    Error querying directory {directory}: {e}")
                break

        # Sort directory commits to match all_commits chronological order (newest to oldest)
        # This is needed because directory API queries return commits in different order than the full commit list
        sha_to_position = {commit['sha']: i for i, commit in enumerate(all_commits)}
        directory_commits.sort(key=lambda c: sha_to_position.get(c['sha'], float('inf')))

        allocation[project_name] = directory_commits
        print(f"  Directory {project_name}: Found {len(directory_commits)} commits")

        # Show empty directories
        if not directory_commits:
            print(f"  Directory {project_name} has no commits - showing as empty")

    # Step 3: Find unassigned commits
    unassigned_commits = [commit for commit in all_commits if commit['sha'] not in all_seen_commits]

    if unassigned_commits:
        allocation['Unassigned'] = unassigned_commits
        print(f"  Found {len(unassigned_commits)} unassigned commits")

    return allocation, all_commits

def get_commits_between_shas(repo_name, start_sha, end_sha, enrich_with_files=False):
    """
    Get commits between two SHAs for any repository.

    Args:
        repo_name: Repository name (e.g., 'rocm-libraries', 'hip')
        start_sha: Starting commit SHA to stop at
        end_sha: Ending commit SHA to start from
        enrich_with_files: If True, fetch detailed file info for each commit (needed for superrepos)

    Returns:
        List of commit objects in chronological order (newest first)
    """
    commits = []
    found_start = False
    page = 1
    max_pages = 20

    repo_type = "superrepo" if enrich_with_files else "submodule"
    print(f"  Getting commits for {repo_type} {repo_name} from {start_sha} to {end_sha}")

    while not found_start and page <= max_pages:
        params = {"sha": end_sha, "per_page": 100, "page": page}
        url = f"https://api.github.com/repos/ROCm/{repo_name}/commits?{urllib.parse.urlencode(params)}"

        try:
            print(f"  Fetching page {page} for {repo_name}")
            data = gha_send_request(url)

            if not data:
                print(f"  No more commits found on page {page}")
                break

            print(f"  Page {page}: Received {len(data)} commits from API")

            for commit in data:
                sha = commit['sha']

                # Enrich with file details if requested (for superrepos)
                if enrich_with_files:
                    print(f"  Processing commit: {sha}")
                    try:
                        commit_url = f"https://api.github.com/repos/ROCm/{repo_name}/commits/{sha}"
                        commit = gha_send_request(commit_url)
                    except Exception:
                        pass  # Use original commit if enrichment fails

                commits.append(commit)

                if sha == start_sha:
                    found_start = True
                    print(f"  Found start commit {start_sha} for {repo_name}")
                    break

            # Stop if we got fewer commits than requested (end of history)
            if len(data) < params['per_page']:
                print(f"  Reached end of commits (got {len(data)} < {params['per_page']})")
                break

            page += 1

        except Exception as e:
            print(f"  Error fetching commits for {repo_name}: {e}")
            break

    # Report results
    if not found_start and page > max_pages:
        print(f"  Warning: Reached page limit ({max_pages}) without finding start commit {start_sha}")
    elif not found_start:
        print(f"  Warning: Did not find start commit {start_sha} for {repo_name}")

    print(f"  Found {len(commits)} commits for {repo_name}")
    return commits

def find_submodules(commit_sha):
    """Find submodules and their commit SHAs for a TheRock commit"""

    # Get .gitmodules file content
    gitmodules_url = f"https://api.github.com/repos/ROCm/TheRock/contents/.gitmodules?ref={commit_sha}"
    try:
        gitmodules_data = gha_send_request(gitmodules_url)
        if gitmodules_data.get('encoding') != 'base64':
            print("Error: .gitmodules file encoding not supported")
            return {}

        # Parse submodule paths from .gitmodules content
        content = base64.b64decode(gitmodules_data['content']).decode('utf-8')
        submodule_paths = {
            line.split('path =')[1].strip()
            for line in content.split('\n')
            if line.strip().startswith('path =')
        }

        if not submodule_paths:
            print("No submodules found in .gitmodules")
            return {}

        print(f"Found {len(submodule_paths)} submodule paths in .gitmodules")

    except Exception as e:
        print(f"Error fetching .gitmodules file: {e}")
        return {}

    # Special name mappings for submodules
    name_mappings = {
        "profiler/rocprof-trace-decoder/binaries": "rocprof-trace-decoder",
        "compiler/amd-llvm": "llvm-project"
    }

    # Get commit SHAs for all submodules
    submodules = {}
    for path in submodule_paths:
        try:
            # Get submodule commit SHA
            contents_url = f"https://api.github.com/repos/ROCm/TheRock/contents/{path}?ref={commit_sha}"
            content_data = gha_send_request(contents_url)
            if content_data.get('type') == 'submodule' and content_data.get('sha'):
                # Determine submodule name (with special mappings)
                submodule_name = name_mappings.get(path, path.split('/')[-1])
                submodules[submodule_name] = content_data['sha']
                print(f"Found submodule: {submodule_name} (path: {path}) -> {content_data['sha']}")
                if path == "compiler/amd-llvm":
                    print(f"  DEBUG: compiler/amd-llvm path mapped to {submodule_name}")
            else:
                print(f"Warning: {path} is not a valid submodule")
        except Exception as e:
            print(f"Warning: Could not get commit SHA for submodule at {path}: {e}")

    return submodules

# Workflow Summary Function
def generate_step_summary(start_commit, end_commit, html_reports, submodule_commits):
    """Generate simple GitHub Actions step summary"""
    superrepo_count = len([k for k in html_reports.keys() if k != 'non-superrepo'])
    non_superrepo_count = len(submodule_commits)
    total_submodules = superrepo_count + non_superrepo_count

    summary = f"""## TheRock Repository Diff Report

**TheRock Commit Range:** `{start_commit[:7]}` → `{end_commit[:7]}`

**Analysis:** Compared submodule changes between these two TheRock commits

**Status:** {' Report generated successfully' if os.path.exists('TheRockReport.html') else ' Report generation failed'}

**Submodules with Updates:** {total_submodules} submodules ({superrepo_count} superrepos + {non_superrepo_count} regular submodules)"""

    gha_append_step_summary(summary)

# Main Function
def main():
    # Arguments parsed
    parser = argparse.ArgumentParser(description="Generate HTML report for repo diffs")
    parser.add_argument("--start", required=True, help="Start workflow ID or commit SHA")
    parser.add_argument("--end", required=True, help="End workflow ID or commit SHA")

    args = parser.parse_args()

    # Determine mode from environment variable
    if os.getenv('WORKFLOW_MODE') == 'true':
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
        print(f"Looking for commit SHA for workflow {args.start}")
        try:
            start = gha_query_workflow_run_information("ROCm/TheRock", args.start).get('head_sha')
            print(f"Found start commit SHA via API: {start}")
        except Exception as e:
            print(f"Error fetching start workflow info via API: {e}")
            start = None
        print(f"Looking for commit SHA for workflow {args.end}")
        try:
            end = gha_query_workflow_run_information("ROCm/TheRock", args.end).get('head_sha')
            print(f"Found end commit SHA via API: {end}")
        except Exception as e:
            print(f"Error fetching end workflow info via API: {e}")
            end = None
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
    removed_submodules = []
    newly_added_submodules = []
    unchanged_submodules = []
    changed_submodules = []
    html_reports = {}

    # Get all unique submodules from both commits
    all_submodules = set(old_submodules.keys()) | set(new_submodules.keys())

    # Categorize submodules
    for submodule in all_submodules:
        old_sha = old_submodules.get(submodule)
        new_sha = new_submodules.get(submodule)

        if old_sha and not new_sha:
            # Submodule was removed
            removed_submodules.append(submodule)
            print(f"REMOVED: {submodule} (was at {old_sha[:7]})")

        elif new_sha and not old_sha:
            # Submodule was newly added
            newly_added_submodules.append(submodule)
            print(f"NEWLY ADDED: {submodule} -> {new_sha[:7]}")

            # Get commit message for the tip SHA
            commit_message = "N/A"
            try:
                commit_url = f"https://api.github.com/repos/ROCm/{submodule}/commits/{new_sha}"
                commit_api_response = gha_send_request(commit_url)
                commit_data = extract_commit_data(commit_api_response)
                commit_message = commit_data['message']
                print(f"  Retrieved commit message for newly added {submodule}")
            except Exception as e:
                print(f"  Warning: Could not get commit message for newly added {submodule}: {e}")

            # Process newly added superrepo (show as single commit entry)
            if submodule == "rocm-systems" or submodule == "rocm-libraries":
                # Create clickable commit badge for the current SHA
                commit_badge = create_commit_badge_html(new_sha, submodule)

                # Create informative content for newly added superrepo
                content_html = f"""
                <div style="padding: 20px; background-color: #f8f9fa; border-left: 4px solid #28a745; margin-bottom: 16px;">
                    <h3 style="margin-top: 0; color: #28a745; font-size: 1.4em;">Newly Added Superrepo</h3>
                    <p style="margin-bottom: 12px; font-size: 1.1em;">
                        This <strong>{submodule}</strong> superrepo has been newly added to TheRock repository.
                    </p>
                    <div style="background-color: #ffffff; padding: 12px; border-radius: 4px; border: 1px solid #dee2e6;">
                        <strong>Current(Tip) Commit:</strong> {commit_badge} {commit_message}
                    </div>
                    <p style="margin-top: 12px; margin-bottom: 0; color: #6c757d; font-style: italic;">
                        No previous version exists for comparison. Future reports will show detailed component-level changes.
                    </p>
                </div>
                """

                html_reports[submodule] = {
                    'start_commit': 'N/A (newly added)',
                    'end_commit': new_sha,
                    'content_html': content_html
                }
            else:
                submodule_commits[submodule] = [{
                    'sha': new_sha,
                    'commit': {
                        'message': f'Newly added submodule: {submodule} Tip -> {commit_message}',
                        'author': {'name': 'System', 'date': 'N/A'}
                    }
                }]

        elif old_sha and new_sha:
            # Submodule exists in both
            # Log and track changed and unchanged submodules
            if old_sha and new_sha and old_sha == new_sha:
                print(f" UNCHANGED: {submodule} -> {new_sha[:7]}")
                unchanged_submodules.append(submodule)
            else:
                print(f"CHANGED: {submodule} {old_sha[:7]} -> {new_sha[:7]}")
                changed_submodules.append(submodule)

            if submodule == "rocm-systems" or submodule == "rocm-libraries":
                print(f"\n=== Processing {submodule.upper()} superrepo ===")

                # Get the components for this superrepo
                components = get_rocm_components(submodule)

                # Add trailing slashes to ensure full path matching (projects/hip/ not projects/hipsparse)
                project_directories = [comp + "/" if not comp.endswith("/") else comp for comp in components]

                # Use new directory-based approach
                allocation, all_commits_for_display = get_commits_by_directories(submodule, old_sha, new_sha, project_directories)

                # Generate HTML table - allocation is ready to use directly
                html_table = generate_superrepo_html_table(allocation, all_commits_for_display, submodule)

                # Store the HTML report
                html_reports[submodule] = {
                    'start_commit': old_sha,
                    'end_commit': new_sha,
                    'content_html': html_table
                }

                print(f"Generated HTML report for {submodule}")

            else:
                # For other submodules, get commit history
                submodule_commits[submodule] = get_commits_between_shas(submodule, old_sha, new_sha, enrich_with_files=False)

    # Print summary
    print(f"\n=== SUBMODULE CHANGES SUMMARY ===")
    print(f" Total submodules: {len(all_submodules)}")
    print(f" Newly added: {len(newly_added_submodules)}")
    print(f" Removed: {len(removed_submodules)}")
    print(f" Unchanged: {len(unchanged_submodules)}")
    print(f" Changed: {len(changed_submodules)}")
    # Show detailed lists
    if newly_added_submodules:
        print(f"\n NEWLY ADDED SUBMODULES:")
        for sub in sorted(newly_added_submodules):
            print(f"  + {sub} -> {new_submodules[sub][:7]}")

    if removed_submodules:
        print(f"\n  REMOVED SUBMODULES:")
        for sub in sorted(removed_submodules):
            print(f"  - {sub} (was at {old_submodules[sub][:7]})")

    if unchanged_submodules:
        print(f"\n UNCHANGED SUBMODULES:")
        for sub in sorted(unchanged_submodules):
            print(f"  = {sub} -> {new_submodules[sub][:7]}")

    if changed_submodules:
        print(f"\n CHANGED SUBMODULES:")
        for sub in sorted(changed_submodules):
            print(f"  * {sub} {old_submodules[sub][:7]} -> {new_submodules[sub][:7]}")

    print(f" Changed: {len(changed_submodules)}")
    print(f" Unchanged: {len(unchanged_submodules)}")

    # Print all the submodules and their commits
    print(f"\n=== SUBMODULE COMMIT DETAILS ===")
    for submodule, commits in submodule_commits.items():
        print(f"\n--- {submodule.upper()} ---")
        if commits:
            for commit in commits:
                commit_data = extract_commit_data(commit)
                short_sha = commit_data['sha'][:7] if commit_data['sha'] != '-' else commit_data['sha']
                print(f"  {short_sha} - {commit_data['author']} ({commit_data['date']}): {commit_data['message']}")
        else:
            print(f"  No commits found for {submodule}")

    print(f"\nTotal other component submodules with commits: {len(submodule_commits)}")

    # Generate HTML report for other component submodules
    print(f"\n=== Generating Other Components HTML Report ===")
    non_superrepo_html = generate_non_superrepo_html_table(submodule_commits)

    # Store non-superrepo HTML report with TheRock start/end commits
    html_reports['non-superrepo'] = {
        'start_commit': start,  # TheRock start commit
        'end_commit': end,      # TheRock end commit
        'content_html': non_superrepo_html
    }

    # Generate the comprehensive HTML report
    generate_therock_html_report(html_reports, removed_submodules, newly_added_submodules, unchanged_submodules, changed_submodules)

    # Generate GitHub Actions step summary
    generate_step_summary(start, end, html_reports, submodule_commits)

if __name__ == "__main__":
    main()