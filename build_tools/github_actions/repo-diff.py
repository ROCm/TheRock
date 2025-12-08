# This script generates a report for TheRock highlighting the difference in commits for each component between 2 builds.

# Imports
import os
import sys
from pathlib import Path
from urllib.parse import urlparse
import argparse
import base64
import urllib.parse
from collections import defaultdict
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime

# Establish script's location as reference point
THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent.parent

# Import GitHub Actions utilities
from github_actions_utils import (
    gha_append_step_summary,
    gha_query_workflow_run_information,
    gha_query_last_successful_workflow_run,
    gha_send_request,
)

# HTML Helper Functions
def format_commit_date(date_string: str) -> str:
    """Format ISO date string to readable format"""
    if date_string == "Unknown" or not date_string:
        return "Unknown"
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(date_string.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y")
    except:
        return date_string


def create_commit_badge_html(sha: str, repo_name: str) -> str:
    """Create a styled HTML badge for a commit SHA with link"""
    short_sha = sha[:7] if sha != "-" and sha != "N/A" else sha
    commit_url = f"https://github.com/ROCm/{repo_name}/commit/{sha}"
    return (
        f"<a href='{commit_url}' target='_blank' class='commit-badge-link'>"
        f"<span class='commit-badge'>{short_sha}</span>"
        f"</a>"
    )


def extract_commit_data(commit: Dict[str, Any]) -> Dict[str, str]:
    """Extract common commit data to avoid redundancy"""
    return {
        "sha": commit.get("sha", "-"),
        "message": commit.get("commit", {}).get("message", "-").split("\n")[0],
        "author": commit.get("commit", {}).get("author", {}).get("name", "Unknown"),
        "date": format_commit_date(
            commit.get("commit", {}).get("author", {}).get("date", "Unknown")
        ),
    }


def create_commit_item_html(commit: Dict[str, Any], repo_name: str) -> str:
    """Create HTML for a single commit item with badge, message, author and date"""
    commit_data = extract_commit_data(commit)
    badge_html = create_commit_badge_html(commit_data["sha"], repo_name)

    return (
        f"<div class='commit-item'>"
        f"<div>{badge_html} {commit_data['message']}</div>"
        f"<div class='commit-meta'>{commit_data['date']} • {commit_data['author']}</div>"
        f"</div>"
    )


def create_commit_list_container(commit_items: List[str], component_status: Optional[str] = None) -> str:
    """Create a scrollable container for commit items with support for component status"""
    if component_status == "newly_added":
        content = '<div class="newly-added"><strong>NEWLY ADDED:</strong> This component was newly added. Showing current tip commit.</div>' + "".join(commit_items)
    elif component_status == "removed":
        content = '<div class="removed">Component removed in this version</div>'
    elif component_status == "reverted":
        content = '<div class="reverted"><strong>REVERTED SUBMODULE:</strong> This submodule was reverted to an earlier commit. Displaying reverted commits</div>' + "".join(commit_items)
    elif not commit_items:
        content = '<div class="no-commits">Component has no commits in this range (Superrepo Component Unchanged)</div>'
    else:
        content = "".join(commit_items)

    container_classes = ['commit-list']
    if component_status == "reverted":
        container_classes.append('reverted-bg')
    elif component_status == "newly_added":
        container_classes.append('newly-added-bg')

    return f"<div class='{' '.join(container_classes)}'>" f"{content}</div>"


def create_table_wrapper(headers: List[str], rows: List[str]) -> str:
    """Create a styled HTML table with headers and rows"""
    header_html = "".join([f"<th>{header}</th>" for header in headers])
    return (
        "<table class='report-table'>"
        f"<tr>{header_html}</tr>" + "".join(rows) + "</table>"
    )

# HTML Table Functions
def generate_superrepo_html_table(
    allocation: Dict[str, List[str]],
    all_commits: Dict[str, List[Dict[str, Any]]],
    repo_name: str,
    component_status: Optional[Dict[str, str]] = None
) -> str:
    """Create a styled HTML table for superrepo commit differences with project allocation"""
    rows = []
    commit_to_projects = {}

    # Build commit-to-projects mapping efficiently from allocation
    for component, commits in allocation.items():
        for commit in commits:
            commit_data = extract_commit_data(commit)
            sha = commit_data["sha"]
            if sha not in commit_to_projects:
                commit_to_projects[sha] = set()
            commit_to_projects[sha].add(component)

    # Generate component rows directly from allocation
    for component, commits in allocation.items():
        # Convert commits to HTML items
        commit_items = [
            create_commit_item_html(commit, repo_name) for commit in commits
        ]

        # Determine component status from parameter
        status = component_status.get(component) if component_status else None
        commit_list_html = create_commit_list_container(
            commit_items=commit_items,
            component_status=status
        )

        rows.append(
            f"<tr>" f"<td>{component}</td>" f"<td>{commit_list_html}</td>" f"</tr>"
        )

    table = create_table_wrapper(headers=["Component", "Commits"], rows=rows)

    # Create commit-project associations table using all_commits
    commit_projects_html = ""
    if all_commits:
        project_table_rows = []
        for commit in all_commits:
            commit_data = extract_commit_data(commit)
            projects = (
                ", ".join(sorted(commit_to_projects.get(commit_data["sha"], [])))
                if commit_data["sha"] in commit_to_projects
                else "Unassigned"
            )
            badge_html = create_commit_badge_html(commit_data["sha"], repo_name)

            project_table_rows.append(
                f"<tr>"
                f"<td class='date-col'>{commit_data['date']}</td>"
                f"<td>{badge_html}</td>"
                f"<td class='author-col'>{commit_data['author']}</td>"
                f"<td class='project-col'>{projects}</td>"
                f"<td class='message-col'>{commit_data['message']}</td>"
                f"</tr>"
            )

        if project_table_rows:
            commit_projects_html = (
                "<div class='section-title'>Commit History (in newest commit to oldest commit order):</div>"
                "<table class='commit-history-table'>"
                "<tr>"
                "<th class='col-date'>Date</th>"
                "<th class='col-sha'>SHA</th>"
                "<th class='col-author'>Author</th>"
                "<th class='col-projects'>Project(s)</th>"
                "<th>Message</th>"
                "</tr>" + "".join(project_table_rows) + "</table>"
            )

    return table + commit_projects_html


def generate_non_superrepo_html_table(submodule_commits: Dict[str, List[Dict[str, Any]]], status_groups: Optional[Dict[str, List[str]]] = None) -> str:
    """Generate an HTML table for other components"""
    rows = []

    for submodule, commits in submodule_commits.items():
        commit_items = []
        if commits:
            for commit in commits:
                commit_items.append(create_commit_item_html(commit, submodule))
        else:
            commit_items.append("<div>No commits found</div>")

        # Check if this submodule has special status
        component_status = None
        row_classes = []
        if status_groups:
            if submodule in status_groups.get("reverted", []):
                component_status = "reverted"
                row_classes.append('reverted-bg')
            elif submodule in status_groups.get("newly_added", []):
                component_status = "newly_added"
                row_classes.append('newly-added-bg')

        # Create scrollable list for commits
        commit_list_html = create_commit_list_container(commit_items, component_status)

        row_class_attr = f'class="{" ".join(row_classes)}"' if row_classes else ''
        rows.append(
            f"<tr {row_class_attr}>" f"<td>{submodule}</td>" f"<td>{commit_list_html}</td>" f"</tr>"
        )

    return create_table_wrapper(headers=["Submodule", "Commits"], rows=rows)


def generate_summary_content(items_data: Dict[str, List[Any]], summary_type: str = "submodules") -> str:
    """Generate HTML content for summary categories (without container wrapper)"""
    if not any(items_data.values()):
        return ""

    total_items = sum(len(items) if items else 0 for items in items_data.values())
    if total_items == 0:
        return ""

    html = ""

    # Added items
    if items_data.get("added"):
        html += '<div class="summary-category added">'
        html += f'<h3>Newly Added {summary_type.title()} ({len(items_data["added"])}/{total_items}):</h3>'
        html += '<ul class="summary-list">'
        for item in sorted(items_data["added"]):
            html += f"<li><code>{item}</code></li>"
        html += "</ul></div>"

    # Removed items
    if items_data.get("removed"):
        html += '<div class="summary-category removed">'
        html += f'<h3>Removed {summary_type.title()} ({len(items_data["removed"])}/{total_items}):</h3>'
        html += '<ul class="summary-list">'
        for item in sorted(items_data["removed"]):
            html += f"<li><code>{item}</code></li>"
        html += "</ul></div>"

    # Changed items
    if items_data.get("changed"):
        html += '<div class="summary-category changed">'
        html += f'<h3>Changed {summary_type.title()} ({len(items_data["changed"])}/{total_items}):</h3>'
        html += '<ul class="summary-list">'
        for item in sorted(items_data["changed"]):
            html += f"<li><code>{item}</code></li>"
        html += "</ul></div>"

    # Unchanged items
    if items_data.get("unchanged"):
        html += '<div class="summary-category unchanged">'
        html += f'<h3>Unchanged {summary_type.title()} ({len(items_data["unchanged"])}/{total_items}):</h3>'
        html += '<ul class="summary-list">'
        for item in sorted(items_data["unchanged"]):
            html += f"<li><code>{item}</code></li>"
        html += "</ul></div>"

    # Reverted items
    if items_data.get("reverted"):
        html += '<div class="summary-category reverted">'
        html += f'<h3>Reverted {summary_type.title()} ({len(items_data["reverted"])}/{total_items}):</h3>'
        html += '<ul class="summary-list">'
        for item in sorted(items_data["reverted"]):
            html += f"<li><code>{item}</code></li>"
        html += "</ul></div>"

    return html


def generate_therock_html_report(
    html_reports: Dict[str, Dict[str, str]],
    therock_start_commit: str,
    therock_end_commit: str,
    status_groups: Dict[str, List[str]],
    superrepo_component_changes: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> None:
    """Generate a comprehensive HTML report for TheRock repository diff"""
    print(f"\n=== Generating Comprehensive HTML Report ===")

    # Read template
    template_path = THIS_SCRIPT_DIR / "report_template.html"
    with template_path.open("r") as f:
        template = f.read()

    # Generate submodule changes content
    submodule_data = {
        "added": status_groups.get("newly_added", []),
        "removed": status_groups.get("removed", []),
        "changed": status_groups.get("changed", []),
        "unchanged": status_groups.get("unchanged", []),
        "reverted": status_groups.get("reverted", []),
    }

    submodule_content = generate_summary_content(submodule_data, "submodules")

    # Generate superrepo components content
    superrepo_content = ""
    if superrepo_component_changes:
        for repo_name, component_data in superrepo_component_changes.items():
            if any(component_data.values()):  # Only show if there are changes
                repo_content = generate_summary_content(component_data, "components")
                if repo_content:
                    repo_title = f"{repo_name.title()} Components"
                    superrepo_content += f'<div class="summary-section"><h2>{repo_title}</h2>{repo_content}</div>'

    # Insert TheRock commit range into template
    template = template.replace(
        '<span id="therock-start-commit">START_COMMIT</span>',
        f'<span id="therock-start-commit">{therock_start_commit}</span>',
    )
    template = template.replace(
        '<span id="therock-end-commit">END_COMMIT</span>',
        f'<span id="therock-end-commit">{therock_end_commit}</span>',
    )

    # Insert content into template containers
    template = template.replace(
        '<div id="submodule-content"></div>',
        f'<div id="submodule-content">{submodule_content}</div>',
    )
    template = template.replace(
        '<div id="superrepo-content"></div>',
        f'<div id="superrepo-content">{superrepo_content}</div>',
    )  # Check what sections have content and populate accordingly
    rocm_lib_data = html_reports.get("rocm-libraries")
    rocm_sys_data = html_reports.get("rocm-systems")
    non_superrepo_data = html_reports.get("non-superrepo")

    # Populate ROCm-Libraries Superrepo container
    if rocm_lib_data:
        template = template.replace(
            '<span id="commit-diff-start-rocm-libraries-superrepo"></span>',
            rocm_lib_data["start_commit"],
        )
        template = template.replace(
            '<span id="commit-diff-end-rocm-libraries-superrepo"></span>',
            rocm_lib_data["end_commit"],
        )
        template = template.replace(
            '<div id="commit-diff-job-content-rocm-libraries-superrepo" style="margin-top:8px;">\n      </div>',
            f'<div id="commit-diff-job-content-rocm-libraries-superrepo" style="margin-top:8px;">{rocm_lib_data["content_html"]}</div>',
        )
        print("Populated ROCm-Libraries superrepo section")

    # Populate ROCm-Systems Superrepo container
    if rocm_sys_data:
        template = template.replace(
            '<span id="commit-diff-start-rocm-systems-superrepo"></span>',
            rocm_sys_data["start_commit"],
        )
        template = template.replace(
            '<span id="commit-diff-end-rocm-systems-superrepo"></span>',
            rocm_sys_data["end_commit"],
        )
        template = template.replace(
            '<div id="commit-diff-job-content-rocm-systems-superrepo" style="margin-top:8px;">\n      </div>',
            f'<div id="commit-diff-job-content-rocm-systems-superrepo" style="margin-top:8px;">{rocm_sys_data["content_html"]}</div>',
        )
        print("Populated ROCm-Systems superrepo section")

    # Populate Other Components container
    if (
        non_superrepo_data
        and non_superrepo_data.get("content_html")
        and non_superrepo_data["content_html"].strip()
    ):
        template = template.replace(
            '<span id="commit-diff-start-non-superrepo"></span>',
            non_superrepo_data["start_commit"],
        )
        template = template.replace(
            '<span id="commit-diff-end-non-superrepo"></span>',
            non_superrepo_data["end_commit"],
        )
        template = template.replace(
            '<div id="commit-diff-job-content-non-superrepo" style="margin-top:8px;">\n      </div>',
            f'<div id="commit-diff-job-content-non-superrepo" style="margin-top:8px;">{non_superrepo_data["content_html"]}</div>',
        )
        print("Populated Other Components section")

    # Write the final TheRock HTML report
    report_path = Path.cwd() / "TheRockReport.html"
    with report_path.open("w") as f:
        f.write(template)

    print("Generated TheRockReport.html successfully!")


# TheRock Helper Functions
def get_rocm_components(repo: str, commit_sha: Optional[str] = None) -> List[str]:
    """Get components from ROCm superrepo repositories (shared and projects directories) at specific commit"""
    components = []
    ref_param = f"?ref={commit_sha}" if commit_sha else ""

    # If the repo is rocm-libraries fetch from shared and projects subfolders
    if repo == "rocm-libraries":
        url = f"https://api.github.com/repos/ROCm/{repo}/contents/shared{ref_param}"
        print(f"Requesting: {url}")
        try:
            data = gha_send_request(url)
            for item in data:
                print(f"Item: {item.get('name')} type: {item.get('type')}")
                if item["type"] == "dir":
                    components.append("shared/" + item["name"])
        except Exception as e:
            print(f"Failed to fetch shared folder from GitHub: {e}")

    # Fetch the components in the projects directory
    url = f"https://api.github.com/repos/ROCm/{repo}/contents/projects{ref_param}"
    print(f"Requesting: {url}")
    try:
        data = gha_send_request(url)
        for item in data:
            print(f"Item: {item.get('name')} type: {item.get('type')}")
            if item["type"] == "dir":
                components.append("projects/" + item["name"])
    except Exception as e:
        print(f"Failed to fetch projects folder from GitHub: {e}")

    return components


def find_submodules(commit_sha: str) -> Dict[str, str]:
    """Find submodules and their commit SHAs for a TheRock commit"""

    # Get .gitmodules file content
    gitmodules_url = f"https://api.github.com/repos/ROCm/TheRock/contents/.gitmodules?ref={commit_sha}"
    try:
        gitmodules_data = gha_send_request(gitmodules_url)
        if gitmodules_data.get("encoding") != "base64":
            print("Error: .gitmodules file encoding not supported")
            return {}

        # Parse submodule paths from .gitmodules content
        content = base64.b64decode(gitmodules_data["content"]).decode("utf-8")
        submodule_paths = {
            line.split("path =")[1].strip()
            for line in content.split("\n")
            if line.strip().startswith("path =")
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
        "compiler/amd-llvm": "llvm-project",
    }

    # Get commit SHAs for all submodules
    submodules = {}
    for path in submodule_paths:
        try:
            # Get submodule commit SHA
            contents_url = f"https://api.github.com/repos/ROCm/TheRock/contents/{path}?ref={commit_sha}"
            content_data = gha_send_request(contents_url)
            if content_data.get("type") == "submodule" and content_data.get("sha"):
                # Determine submodule name (with special mappings)
                submodule_name = name_mappings.get(path, path.split("/")[-1])
                submodules[submodule_name] = content_data["sha"]
                print(
                    f"Found submodule: {submodule_name} (path: {path}) -> {content_data['sha']}"
                )
                if path == "compiler/amd-llvm":
                    print(f"  DEBUG: compiler/amd-llvm path mapped to {submodule_name}")
            else:
                print(f"Warning: {path} is not a valid submodule")
        except Exception as e:
            print(f"Warning: Could not get commit SHA for submodule at {path}: {e}")

    return submodules

def is_commit_newer_than(repo_name: str, sha1: str, sha2: str) -> bool:
    """Check if sha1 is newer than sha2 by comparing commit timestamps."""
    try:
        # Get commit info for both SHAs
        commit1_url = f"https://api.github.com/repos/ROCm/{repo_name}/commits/{sha1}"
        commit2_url = f"https://api.github.com/repos/ROCm/{repo_name}/commits/{sha2}"

        commit1_data = gha_send_request(commit1_url)
        commit2_data = gha_send_request(commit2_url)

        if not commit1_data or not commit2_data:
            print(f"  Warning: Could not fetch commit data for comparison in {repo_name}")
            return False

        # Extract commit dates
        date1_str = commit1_data.get("commit", {}).get("author", {}).get("date")
        date2_str = commit2_data.get("commit", {}).get("author", {}).get("date")

        if not date1_str or not date2_str:
            print(f"  Warning: Could not extract commit dates for comparison in {repo_name}")
            return False

        # Parse dates and compare
        date1 = datetime.fromisoformat(date1_str.replace("Z", "+00:00"))
        date2 = datetime.fromisoformat(date2_str.replace("Z", "+00:00"))

        return date1 > date2

    except Exception as e:
        print(f"  Error comparing commit timestamps in {repo_name}: {e}")
        return False

def fetch_commits_in_range(repo_name: str, start_sha: str, end_sha: str) -> List[Dict[str, Any]]:
    """Core function to fetch commits between two SHAs."""
    commits = []
    found_start = False
    page = 1
    max_pages = 20

    print(f"  Getting commits for {repo_name} from {start_sha[:7]} to {end_sha[:7]}")

    while not found_start and page <= max_pages:
        params = {"sha": end_sha, "per_page": 100, "page": page}
        url = f"https://api.github.com/repos/ROCm/{repo_name}/commits?{urllib.parse.urlencode(params)}"

        try:
            data = gha_send_request(url)
            if not data:
                break

            for commit in data:
                commits.append(commit)

                if commit["sha"] == start_sha:
                    found_start = True
                    break

            if len(data) < params["per_page"]:
                break
            page += 1

        except Exception as e:
            print(f"  Error fetching commits: {e}")
            break

    print(f"  Found {len(commits)} commits in range")
    return commits


def detect_component_changes(start_components: List[str], end_components: List[str], repo_name: Optional[str] = None) -> Dict[str, set]:
    """Compare two component lists to find added/removed components"""
    start_set = set(start_components)
    end_set = set(end_components)

    added = end_set - start_set
    removed = start_set - end_set

    if repo_name:
        if added:
            print(
                f"  Found {len(added)} newly added components in {repo_name}: {sorted(added)}"
            )
        if removed:
            print(
                f"  Found {len(removed)} removed components in {repo_name}: {sorted(removed)}"
            )

    return {"added": added, "removed": removed}


def get_commits_by_directories(repo_name: str, start_sha: str, end_sha: str, project_directories: List[str]) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Get commits by project directories using GitHub API path parameter."""
    print(
        f"  Getting commits by directories for {repo_name} from {start_sha} to {end_sha}"
    )

    # Step 1: Get all commits in range
    all_commits = fetch_commits_in_range(
        repo_name=repo_name,
        start_sha=start_sha,
        end_sha=end_sha
    )

    # Create SHA set for fast range checking when filtering directory commits
    commit_shas_in_range = {commit["sha"] for commit in all_commits}

    # Step 2: Query each directory and build allocation
    allocation = {}
    all_seen_commits = set()

    for directory in project_directories:
        project_name = directory.rstrip("/")
        print(f"  Querying directory: {directory}")
        allocation[project_name] = []

        # Query commits that touched this specific directory path
        page = 1
        max_pages = 20
        directory_commits = []

        while page <= max_pages:
            params = {"sha": end_sha, "path": directory, "per_page": 100, "page": page}
            url = f"https://api.github.com/repos/ROCm/{repo_name}/commits?{urllib.parse.urlencode(params)}"

            try:
                data = gha_send_request(url)
                if not data:
                    break

                commits_found_this_page = 0
                for commit in data:
                    sha = commit["sha"]

                    # Only include commits that are in our target range
                    if sha in commit_shas_in_range:
                        directory_commits.append(commit)
                        all_seen_commits.add(sha)
                        commits_found_this_page += 1

                        # Stop if we've reached the start commit
                        if sha == start_sha:
                            break

                print(
                    f"    Page {page}: Found {commits_found_this_page} commits in range"
                )

                # If no commits found in range on this page, we've probably gone beyond our range
                if commits_found_this_page == 0 and page > 1:
                    break

                if len(data) < params["per_page"]:
                    break

                page += 1

            except Exception as e:
                print(f"    Error querying directory {directory}: {e}")
                break

        # Sort directory commits to match all_commits chronological order (newest to oldest)
        # This is needed because directory API queries return commits in different order than the full commit list
        sha_to_position = {commit["sha"]: i for i, commit in enumerate(all_commits)}
        directory_commits.sort(
            key=lambda c: sha_to_position.get(c["sha"], float("inf"))
        )

        allocation[project_name] = directory_commits
        print(f"  Directory {project_name}: Found {len(directory_commits)} commits")

        # Show empty directories
        if not directory_commits:
            print(f"  Directory {project_name} has no commits - showing as empty")

    # Step 3: Find unassigned commits
    unassigned_commits = [
        commit for commit in all_commits if commit["sha"] not in all_seen_commits
    ]

    if unassigned_commits:
        allocation["Unassigned"] = unassigned_commits
        print(f"  Found {len(unassigned_commits)} unassigned commits")

    return allocation, all_commits


# Workflow Summary Function
def process_superrepo_changes(submodule: str, old_sha: str, new_sha: str) -> Dict[str, Any]:
    """Process component changes for superrepos and generate HTML content"""
    print(f"\n=== Processing {submodule.upper()} superrepo ===")

    # Get components from both commits
    start_components = get_rocm_components(repo=submodule, commit_sha=old_sha)
    end_components = get_rocm_components(repo=submodule, commit_sha=new_sha)

    # Detect component changes
    component_changes = detect_component_changes(
        start_components=start_components,
        end_components=end_components,
        repo_name=submodule
    )

    # Get all components and create directory list
    all_components = set(start_components) | set(end_components)
    project_directories = [
        comp + "/" if not comp.endswith("/") else comp for comp in all_components
    ]

    # Get commits by directory
    allocation, all_commits_for_display = get_commits_by_directories(
        repo_name=submodule,
        start_sha=old_sha,
        end_sha=new_sha,
        project_directories=project_directories
    )

    # Categorize components based on commit activity
    changed_components = set()
    unchanged_components = set()

    for comp in start_components:
        if comp in end_components:
            comp_key = comp.rstrip("/")
            if comp_key in allocation and allocation[comp_key]:
                changed_components.add(comp)
            else:
                unchanged_components.add(comp)

    # Create component categorization for summary
    component_summary = {
        "added": component_changes["added"],
        "removed": component_changes["removed"],
        "changed": changed_components,
        "unchanged": unchanged_components,
    }

    # Create component status mapping and fetch tip commits for newly added components
    component_status = {}
    for comp in component_changes["added"]:
        component_status[comp] = "newly_added"
        # Fetch tip commit for newly added component
        try:
            comp_path = comp + "/" if not comp.endswith("/") else comp
            params = {"sha": new_sha, "path": comp_path, "per_page": 1}
            url = f"https://api.github.com/repos/ROCm/{submodule}/commits?{urllib.parse.urlencode(params)}"
            tip_commits = gha_send_request(url)
            if tip_commits and len(tip_commits) > 0:
                allocation[comp] = [tip_commits[0]]  # Show just the tip commit
                print(f"  Found tip commit for newly added component {comp}: {tip_commits[0]['sha'][:7]}")
            else:
                allocation[comp] = []  # No commits found
                print(f"  No tip commit found for newly added component {comp}")
        except Exception as e:
            print(f"  Error fetching tip commit for {comp}: {e}")
            allocation[comp] = []  # Fallback to empty

    for comp in component_changes["removed"]:
        allocation[comp] = []  # Add removed components with empty commits
        component_status[comp] = "removed"

    # Generate HTML content
    content_html = generate_superrepo_html_table(
        allocation=allocation,
        all_commits=all_commits_for_display,
        repo_name=submodule,
        component_status=component_status
    )

    return {
        "html_content": content_html,
        "component_summary": component_summary,
        "start_commit": old_sha,
        "end_commit": new_sha,
    }


def create_newly_added_superrepo_html(submodule: str, new_sha: str, commit_message: str) -> str:
    """Create HTML for newly added superrepos"""
    commit_badge = create_commit_badge_html(new_sha, submodule)
    return f"""
    <div class="newly-added-superrepo">
        <h3>Newly Added Superrepo</h3>
        <p>
            This <strong>{submodule}</strong> superrepo has been newly added to TheRock repository.
        </p>
        <div class="commit-info">
            <strong>Current(Tip) Commit:</strong> {commit_badge} {commit_message}
        </div>
        <p class="note">
            No previous version exists for comparison. Future reports will show detailed component-level changes.
        </p>
    </div>
    """


def generate_step_summary(
    start_commit: str,
    end_commit: str,
    html_reports: Dict[str, Dict[str, str]],
    submodule_commits: Dict[str, List[Dict[str, Any]]],
    status_groups: Dict[str, List[str]],
    superrepo_component_changes: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> None:
    """Generate comprehensive GitHub Actions step summary with submodule and component change details"""
    superrepo_count = len([k for k in html_reports.keys() if k != "non-superrepo"])
    non_superrepo_count = len(submodule_commits)
    total_submodules = superrepo_count + non_superrepo_count

    # Calculate submodule change totals
    newly_added_count = len(status_groups.get("newly_added", []))
    removed_count = len(status_groups.get("removed", []))
    changed_count = len(status_groups.get("changed", []))
    unchanged_count = len(status_groups.get("unchanged", []))
    reverted_count = len(status_groups.get("reverted", []))

    summary = f"""## TheRock Repository Diff Report

**TheRock Commit Range:** `{start_commit[:7]}` → `{end_commit[:7]}`

**Analysis:** Compared submodule changes between these two TheRock commits

**Status:** {'Report generated successfully' if (Path.cwd() / 'TheRockReport.html').exists() else 'Report generation failed'}

### Submodule Changes Summary
- **Total Submodules:** {total_submodules} ({superrepo_count} superrepos + {non_superrepo_count} regular submodules)
- **Newly Added:** {newly_added_count}
- **Removed:** {removed_count}
- **Changed:** {changed_count}
- **Unchanged:** {unchanged_count}
- **Reverted:** {reverted_count}"""

    # Add component changes for each superrepo
    if superrepo_component_changes:
        summary += "\n\n### Superrepo Component Changes"
        for repo_name, component_data in superrepo_component_changes.items():
            if any(component_data.values()):
                added_components = len(component_data.get("added", []))
                removed_components = len(component_data.get("removed", []))
                changed_components = len(component_data.get("changed", []))
                unchanged_components = len(component_data.get("unchanged", []))
                total_components = (
                    added_components
                    + removed_components
                    + changed_components
                    + unchanged_components
                )

                summary += f"""

**{repo_name.title()} Components:** {total_components} total
- **Added:** {added_components}
- **Removed:** {removed_components}
- **Changed:** {changed_components}
- **Unchanged:** {unchanged_components}"""

    gha_append_step_summary(summary)


# Main Function
def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the script that can be called from tests or other scripts. """
    # Arguments parsed
    parser = argparse.ArgumentParser(description="Generate HTML report for repo diffs")
    parser.add_argument("--start", required=False, help="Start workflow ID or commit SHA")
    parser.add_argument("--end", required=True, help="End workflow ID or commit SHA")
    parser.add_argument("--find-last-successful", help="Workflow name to find last successful run (e.g., 'ci_nightly.yml')")

    args = parser.parse_args(argv)

    # Determine mode from environment variable first
    if os.getenv("WORKFLOW_MODE") == "true":
        print("Running in WORKFLOW mode - extracting commits from workflow logs")
        mode = "workflow"
    else:
        print("Running in COMMIT mode - direct commit comparison")
        mode = "commit"

    # Handle find last successful workflow run (after determining mode)
    if args.find_last_successful:
        print(f"Finding last successful run of workflow: {args.find_last_successful}")
        try:
            last_run = gha_query_last_successful_workflow_run("ROCm/TheRock", args.find_last_successful, branch="main")
            if last_run:
                if mode == "workflow":
                    # In workflow mode, use the workflow run ID
                    args.start = str(last_run['id'])
                    print(f"Found last successful run: {last_run['id']} (workflow mode)")
                else:
                    # In commit mode, use the head SHA
                    args.start = last_run['head_sha']
                    print(f"Found last successful run: {last_run['id']} with commit {args.start} (commit mode)")
            else:
                print(f"No previous successful run found for {args.find_last_successful} on main branch")
                return 1
        except Exception as e:
            print(f"Error finding last successful workflow run: {e}")
            return 1

    if not args.start:
        print("Error: --start is required unless --find-last-successful is provided")
        return 1

    print(f"Start: {args.start}")
    print(f"End: {args.end}")
    print(f"Mode: {mode}")

    if mode == "workflow":
        # Extract commits from workflow logs
        print(f"Looking for commit SHA for workflow {args.start}")
        try:
            start = gha_query_workflow_run_information("ROCm/TheRock", args.start).get(
                "head_sha"
            )
            print(f"Found start commit SHA via API: {start}")
        except Exception as e:
            print(f"Error fetching start workflow info via API: {e}")
            start = None
        print(f"Looking for commit SHA for workflow {args.end}")
        try:
            end = gha_query_workflow_run_information("ROCm/TheRock", args.end).get(
                "head_sha"
            )
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
        return 1

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
    status_groups = {
        "removed": [],
        "newly_added": [],
        "unchanged": [],
        "changed": [],
        "reverted": []
    }
    html_reports = {}
    superrepo_component_changes = {}

    # Get all unique submodules from both commits
    all_submodules = set(old_submodules.keys()) | set(new_submodules.keys())

    # Categorize submodules
    for submodule in all_submodules:
        old_sha = old_submodules.get(submodule)
        new_sha = new_submodules.get(submodule)

        if old_sha and not new_sha:
            # Submodule was removed
            status_groups["removed"].append(submodule)
            print(f"REMOVED: {submodule} (was at {old_sha[:7]})")

        elif new_sha and not old_sha:
            # Submodule was newly added
            status_groups["newly_added"].append(submodule)
            print(f"NEWLY ADDED: {submodule} -> {new_sha[:7]}")

            # Get commit info for the tip SHA
            commit_message = "N/A"
            commit_data = {"author": "System", "date": "N/A"}
            commit_url = (
                f"https://api.github.com/repos/ROCm/{submodule}/commits/{new_sha}"
            )
            commit_response = gha_send_request(commit_url)
            if commit_response:
                commit_data = extract_commit_data(commit_response)
                commit_message = commit_data["message"]
                print(f"  Retrieved commit info for newly added {submodule}")

            # Process newly added submodule
            if submodule == "rocm-systems" or submodule == "rocm-libraries":
                html_reports[submodule] = {
                    "start_commit": "N/A (newly added)",
                    "end_commit": new_sha,
                    "content_html": create_newly_added_superrepo_html(
                        submodule, new_sha, commit_message
                    ),
                }
            else:
                # Store the actual commit response for newly added submodules
                if commit_response:
                    submodule_commits[submodule] = [commit_response]

        elif old_sha and new_sha:
            # Submodule exists in both
            # Log and track changed and unchanged submodules
            if old_sha == new_sha:
                print(f" UNCHANGED: {submodule} -> {new_sha[:7]}")
                status_groups["unchanged"].append(submodule)
            else:
                # Check to see if this submodule was reverted
                if is_commit_newer_than(repo_name=submodule, sha1=old_sha, sha2=new_sha):
                    status_groups["reverted"].append(submodule)
                    print(f"REVERTED: {submodule} {old_sha[:7]} -> {new_sha[:7]} (reverted)")
                    # If reverted we still want to get the list of commits so switch the start and end commits
                    old_sha, new_sha = new_sha, old_sha
                else:
                    print(f"CHANGED: {submodule} {old_sha[:7]} -> {new_sha[:7]}")
                    status_groups["changed"].append(submodule)

            if submodule == "rocm-systems" or submodule == "rocm-libraries":
                # Process superrepo using consolidated function
                result = process_superrepo_changes(
                    submodule=submodule,
                    old_sha=old_sha,
                    new_sha=new_sha
                )
                html_reports[submodule] = {
                    "start_commit": result["start_commit"],
                    "end_commit": result["end_commit"],
                    "content_html": result["html_content"],
                    "component_changes": {
                        "added": result["component_summary"]["added"],
                        "removed": result["component_summary"]["removed"],
                    },
                }
                superrepo_component_changes[submodule] = result["component_summary"]
                print(f"Generated HTML report for {submodule}")
            else:
                # For other submodules, get commit history
                submodule_commits[submodule] = fetch_commits_in_range(
                    repo_name=submodule,
                    start_sha=old_sha,
                    end_sha=new_sha
                )

    # Print summary
    print(f"\n=== SUBMODULE CHANGES SUMMARY ===")
    print(f" Total submodules: {len(all_submodules)}")
    print(f" Newly added: {len(status_groups['newly_added'])}")
    print(f" Removed: {len(status_groups['removed'])}")
    print(f" Unchanged: {len(status_groups['unchanged'])}")
    print(f" Changed: {len(status_groups['changed'])}")
    print(f" Reverted: {len(status_groups['reverted'])}")
    # Show detailed lists
    if status_groups['newly_added']:
        print(f"\n NEWLY ADDED SUBMODULES:")
        for sub in sorted(status_groups['newly_added']):
            print(f"  + {sub} -> {new_submodules[sub][:7]}")

    if status_groups['removed']:
        print(f"\n  REMOVED SUBMODULES:")
        for sub in sorted(status_groups['removed']):
            print(f"  - {sub} (was at {old_submodules[sub][:7]})")

    if status_groups['unchanged']:
        print(f"\n UNCHANGED SUBMODULES:")
        for sub in sorted(status_groups['unchanged']):
            print(f"  = {sub} -> {new_submodules[sub][:7]}")

    if status_groups['changed']:
        print(f"\n CHANGED SUBMODULES:")
        for sub in sorted(status_groups['changed']):
            print(f"  * {sub} {old_submodules[sub][:7]} -> {new_submodules[sub][:7]}")

    if status_groups['reverted']:
        print(f"\n REVERTED SUBMODULES:")
        for sub in sorted(status_groups['reverted']):
            print(f"  ↩ {sub} {old_submodules[sub][:7]} -> {new_submodules[sub][:7]} (reverted)")

    # Print all the submodules and their commits
    print(f"\n=== SUBMODULE COMMIT DETAILS ===")
    for submodule, commits in submodule_commits.items():
        print(f"\n--- {submodule.upper()} ---")
        if commits:
            for commit in commits:
                commit_data = extract_commit_data(commit)
                short_sha = (
                    commit_data["sha"][:7]
                    if commit_data["sha"] != "-"
                    else commit_data["sha"]
                )
                print(
                    f"  {short_sha} - {commit_data['author']} ({commit_data['date']}): {commit_data['message']}"
                )
        else:
            print(f"  No commits found for {submodule}")

    print(f"\nTotal other component submodules with commits: {len(submodule_commits)}")

    # Generate HTML report for other component submodules
    print(f"\n=== Generating Other Components HTML Report ===")
    non_superrepo_html = generate_non_superrepo_html_table(submodule_commits, status_groups)

    # Store non-superrepo HTML report with TheRock start/end commits
    html_reports["non-superrepo"] = {
        "start_commit": start,  # TheRock start commit
        "end_commit": end,  # TheRock end commit
        "content_html": non_superrepo_html,
    }

    # Generate the comprehensive HTML report
    generate_therock_html_report(
        html_reports=html_reports,
        therock_start_commit=start,
        therock_end_commit=end,
        status_groups=status_groups,
        superrepo_component_changes=superrepo_component_changes,
    )

    # Generate GitHub Actions step summary
    generate_step_summary(
        start_commit=start,
        end_commit=end,
        html_reports=html_reports,
        submodule_commits=submodule_commits,
        status_groups=status_groups,
        superrepo_component_changes=superrepo_component_changes,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
