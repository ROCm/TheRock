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

# HTML Table Functions
def generate_monorepo_html_table(allocation, original_commits=None, repo_name="rocm-libraries"):
    """Create a styled HTML table for monorepo commit differences with project allocation"""
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

def generate_therock_html_report(html_reports, removed_submodules=None, newly_added_submodules=None, unchanged_submodules=None):
    """Generate a comprehensive HTML report for TheRock repository diff"""
    print(f"\n=== Generating Comprehensive HTML Report ===")

    # Read template
    with open("report_template.html", "r") as f:
        template = f.read()

    # Generate and populate submodule changes summary
    summary_html = ""
    if removed_submodules or newly_added_submodules or unchanged_submodules:
        summary_html += '<div style="background-color:#ffffff; padding:16px; margin-bottom:3em; box-shadow:0 2px 5px rgba(0,0,0,0.16), 0 2px 10px rgba(0,0,0,0.12);">'
        summary_html += '<div style="text-align:center; color:#2196F3; font-size:2.2em; font-weight:bold; margin-bottom:16px;">Submodule Changes Summary</div>'

        if newly_added_submodules:
            summary_html += '<div style="margin-bottom:16px;">'
            summary_html += '<h3 style="color:#28a745; margin-bottom:8px;">Newly Added Submodules:</h3>'
            summary_html += '<ul style="margin:0; padding-left:20px;">'
            for sub in sorted(newly_added_submodules):
                summary_html += f'<li style="color:#6c757d; margin-bottom:4px;"><code>{sub}</code></li>'
            summary_html += '</ul></div>'

        if removed_submodules:
            summary_html += '<div style="margin-bottom:16px;">'
            summary_html += '<h3 style="color:#dc3545; margin-bottom:8px;">Removed Submodules:</h3>'
            summary_html += '<ul style="margin:0; padding-left:20px;">'
            for sub in sorted(removed_submodules):
                summary_html += f'<li style="color:#6c757d; margin-bottom:4px;"><code>{sub}</code></li>'
            summary_html += '</ul></div>'

        if unchanged_submodules:
            summary_html += '<div>'
            summary_html += '<h3 style="color:#6c757d; margin-bottom:8px;">Unchanged Submodules:</h3>'
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

# TheRock Helper Functions
def get_rocm_components(repo):
    """Get components from ROCm monorepo repositories (shared and projects directories)"""
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

def get_commits_between_shas(repo_name, start_sha, end_sha, enrich_with_files=False):
    """
    Get commits between two SHAs for any repository.

    Args:
        repo_name: Repository name (e.g., 'rocm-libraries', 'hip')
        start_sha: Starting commit SHA to stop at
        end_sha: Ending commit SHA to start from
        enrich_with_files: If True, fetch detailed file info for each commit (needed for monorepos)

    Returns:
        List of commit objects in chronological order (newest first)
    """
    commits = []
    found_start = False
    page = 1
    max_pages = 20

    repo_type = "monorepo" if enrich_with_files else "submodule"
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

                # Enrich with file details if requested (for monorepos)
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
    monorepo_count = len([k for k in html_reports.keys() if k != 'non-monorepo'])
    non_monorepo_count = len(submodule_commits)
    total_submodules = monorepo_count + non_monorepo_count

    summary = f"""## TheRock Repository Diff Report

**TheRock Commit Range:** `{start_commit[:7]}` → `{end_commit[:7]}`

**Analysis:** Compared submodule changes between these two TheRock commits

**Status:** {' Report generated successfully' if os.path.exists('TheRockReport.html') else ' Report generation failed'}

**Submodules with Updates:** {total_submodules} submodules ({monorepo_count} monorepos + {non_monorepo_count} regular submodules)"""

    gha_append_step_summary(summary)

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

            # Process newly added monorepo (show as single commit entry)
            if submodule == "rocm-systems" or submodule == "rocm-libraries":
                # Create clickable commit badge for the current SHA
                commit_badge = create_commit_badge_html(new_sha, submodule)

                # Get commit message for the current SHA
                commit_message = "N/A"
                try:
                    commit_url = f"https://api.github.com/repos/ROCm/{submodule}/commits/{new_sha}"
                    commit_data = gha_send_request(commit_url)
                    commit_message = commit_data.get('commit', {}).get('message', 'N/A').split('\n')[0]
                    print(f"  Retrieved commit message for newly added {submodule}")
                except Exception as e:
                    print(f"  Warning: Could not get commit message for newly added {submodule}: {e}")

                # Create informative content for newly added monorepo
                content_html = f"""
                <div style="padding: 20px; background-color: #f8f9fa; border-left: 4px solid #28a745; margin-bottom: 16px;">
                    <h3 style="margin-top: 0; color: #28a745; font-size: 1.4em;">Newly Added Monorepo</h3>
                    <p style="margin-bottom: 12px; font-size: 1.1em;">
                        This <strong>{submodule}</strong> monorepo has been newly added to TheRock repository.
                    </p>
                    <div style="background-color: #ffffff; padding: 12px; border-radius: 4px; border: 1px solid #dee2e6;">
                        <strong>Current Commit:</strong> {commit_badge} {commit_message}
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
                        'message': f'Newly added submodule: {submodule}',
                        'author': {'name': 'System', 'date': 'N/A'}
                    }
                }]

        elif old_sha and new_sha:
            # Submodule exists in both - process changed submodules
            print(f"🔄 CHANGED: {submodule} {old_sha[:7]} -> {new_sha[:7]}")

            if submodule == "rocm-systems" or submodule == "rocm-libraries":
                print(f"\n=== Processing {submodule.upper()} monorepo ===")

                # Get the components for this monorepo
                components = get_rocm_components(submodule)

                # Fetch commits between the old and new SHA
                commits = get_commits_between_shas(submodule, old_sha, new_sha, enrich_with_files=True)

                # Allocate commits to components
                allocation = allocate_commits_to_projects(commits, components)

                # Generate HTML table
                html_table = generate_monorepo_html_table(allocation, commits, submodule)

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

        # Handle unchanged submodules separately
        if old_sha and new_sha and old_sha == new_sha:
            print(f" UNCHANGED: {submodule} -> {new_sha[:7]}")
            unchanged_submodules.append(submodule)

    # Print summary
    print(f"\n=== SUBMODULE CHANGES SUMMARY ===")
    print(f" Total submodules: {len(all_submodules)}")
    print(f" Newly added: {len(newly_added_submodules)}")
    print(f"  Removed: {len(removed_submodules)}")
    print(f" Unchanged: {len(unchanged_submodules)}")
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

    changed_count = len([s for s in all_submodules if s in old_submodules and s in new_submodules and old_submodules[s] != new_submodules[s]])
    unchanged_count = len([s for s in all_submodules if s in old_submodules and s in new_submodules and old_submodules[s] == new_submodules[s]])
    print(f" Changed: {changed_count}")
    print(f" Unchanged: {unchanged_count}")

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

    print(f"\nTotal non-monorepo submodules with commits: {len(submodule_commits)}")

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
    generate_therock_html_report(html_reports, removed_submodules, newly_added_submodules, unchanged_submodules)

    # Generate GitHub Actions step summary
    generate_step_summary(start, end, html_reports, submodule_commits)

if __name__ == "__main__":
    main()