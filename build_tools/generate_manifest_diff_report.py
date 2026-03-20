"""Generate manifest diff reports between two TheRock commits.

Compares submodule versions and generates HTML reports showing commit changes
for each component between builds.

Arguments:
  --start                  Start commit SHA or workflow run ID (required unless using --find-last-successful)
  --end                    End commit SHA or workflow run ID (required)
  --find-last-successful   Workflow file to find last successful run (e.g., 'ci_nightly.yml')
  --workflow-mode          Treat --start and --end as workflow run IDs instead of commit SHAs

Example usage:
  python build_tools/generate_manifest_diff_report.py --start abc123 --end def456
  python build_tools/generate_manifest_diff_report.py --end def456 --find-last-successful ci_nightly.yml
  python build_tools/generate_manifest_diff_report.py --start 12345 --end 67890 --workflow-mode
"""

# Standard library imports
import argparse
import html
import sys
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError

# Establish script's location first (needed for local imports)
THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent
sys.path.insert(0, str(THIS_SCRIPT_DIR))

# Local imports
from generate_therock_manifest import build_manifest_schema
from github_actions.github_actions_api import (
    gha_append_step_summary,
    gha_query_last_successful_workflow_run,
    gha_query_workflow_run_by_id,
    gha_send_request,
)

# GitHub API constants
GITHUB_API_BASE = "https://api.github.com/repos"
ROCM_ORG = "ROCm"
THEROCK_REPO = "TheRock"

# Superrepos requiring component-level processing
SUPERREPO_NAMES = {"rocm-systems", "rocm-libraries"}

# Directories to scan within superrepos for components (In case new directories are added, we need to add them here)
SUPERREPO_COMPONENT_DIRS = ["shared", "projects"]

# Pagination constants for GitHub API commit fetching.
# MAX_PAGES * PER_PAGE = 2000 commits maximum.
# This limit prevents excessive API calls while covering most reasonable commit ranges.
# GitHub API returns commits in reverse chronological order, so we paginate until
# we find the start commit or hit the limit.
MAX_PAGES = 20  # Maximum number of API pages to fetch
PER_PAGE = 100  # Commits per page (GitHub API maximum is 100)

# Report identifiers
UNASSIGNED_KEY = "Unassigned"

# File paths
HTML_TEMPLATE_PATH = THIS_SCRIPT_DIR / "manifest_diff_report_template.html"
HTML_REPORT_PATH = THEROCK_DIR / "TheRockReport.html"


# =============================================================================
# Domain Classes
# =============================================================================


@dataclass
class Component:
    """A component within a superrepo (projects/rocblas, shared/hipblaslt)."""

    path: str
    name: str
    status: str = "unchanged"  # added, removed, unchanged, changed
    commits: list[dict] = field(default_factory=list)


@dataclass
class Submodule:
    """A submodule in TheRock manifest."""

    name: str
    sha: str
    api_base: str
    branch: str
    status: str = "unchanged"  # removed, added, unchanged, changed, reverted
    commits: list[dict] = field(default_factory=list)
    start_sha: str = ""
    end_sha: str = ""


@dataclass
class Superrepo(Submodule):
    """A superrepo submodule containing components (rocm-libraries, rocm-systems)."""

    components: dict[str, Component] = field(default_factory=dict)
    all_commits: list[dict] = field(default_factory=list)
    commit_allocation: dict[str, list[dict]] = field(default_factory=dict)

    @property
    def added_components(self) -> list[str]:
        return [c.name for c in self.components.values() if c.status == "added"]

    @property
    def removed_components(self) -> list[str]:
        return [c.name for c in self.components.values() if c.status == "removed"]

    @property
    def changed_components(self) -> list[str]:
        return [c.name for c in self.components.values() if c.status == "changed"]

    @property
    def unchanged_components(self) -> list[str]:
        return [c.name for c in self.components.values() if c.status == "unchanged"]


@dataclass
class ManifestDiff:
    """Result of comparing two TheRock manifests."""

    start_commit: str
    end_commit: str
    submodules: dict[str, Submodule] = field(default_factory=dict)
    superrepos: dict[str, Superrepo] = field(default_factory=dict)

    @property
    def added(self) -> list[str]:
        return [s.name for s in self.submodules.values() if s.status == "added"]

    @property
    def removed(self) -> list[str]:
        return [s.name for s in self.submodules.values() if s.status == "removed"]

    @property
    def changed(self) -> list[str]:
        return [s.name for s in self.submodules.values() if s.status == "changed"]

    @property
    def unchanged(self) -> list[str]:
        return [s.name for s in self.submodules.values() if s.status == "unchanged"]

    @property
    def reverted(self) -> list[str]:
        return [s.name for s in self.submodules.values() if s.status == "reverted"]

    @property
    def all_items(self) -> dict[str, Submodule | Superrepo]:
        """All submodules and superrepos combined."""
        return {**self.submodules, **self.superrepos}

    def get_status_groups(self) -> dict[str, list[str]]:
        """Return submodules grouped by status."""
        groups: dict[str, list[str]] = {
            "removed": [],
            "added": [],
            "unchanged": [],
            "changed": [],
            "reverted": [],
        }
        for item in self.all_items.values():
            groups[item.status].append(item.name)
        return groups


# =============================================================================
# CLI and Commit Resolution
# =============================================================================


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate manifest diff report")
    parser.add_argument(
        "--start", required=False, help="Start workflow ID or commit SHA"
    )
    parser.add_argument("--end", required=True, help="End workflow ID or commit SHA")
    parser.add_argument(
        "--find-last-successful",
        help="Workflow name to find last successful run (e.g., 'ci_nightly.yml')",
    )
    parser.add_argument(
        "--workflow-mode",
        action="store_true",
        help="Treat --start and --end as workflow run IDs instead of commit SHAs",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to search for last successful workflow run (default: main)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for the report (default: TheRock root directory)",
    )
    return parser.parse_args(argv)


def _optional_str(val: str | None) -> str | None:
    """Normalize optional string: None or blank -> None, otherwise stripped.

    Workflow callers (e.g. GitHub Actions) may pass empty string for unused
    optional inputs; treat those as not provided.
    """
    if val is None:
        return None
    s = val.strip()
    return s if s else None


def resolve_commits(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve start and end commit SHAs from arguments."""
    start = _optional_str(args.start)
    find_last = _optional_str(args.find_last_successful)
    end = _optional_str(args.end)

    if start is None and find_last is None:
        raise ValueError(
            "--start is required unless --find-last-successful is provided"
        )
    if end is None:
        raise ValueError("--end is required")

    therock_repo_full = f"{ROCM_ORG}/{THEROCK_REPO}"

    # Resolve start commit
    if find_last is not None:
        last_run = gha_query_last_successful_workflow_run(
            therock_repo_full, find_last, branch=args.branch
        )
        if not last_run:
            raise ValueError(f"No previous successful run found for {find_last}")
        start_sha = last_run["head_sha"]
    elif args.workflow_mode:
        workflow_info = gha_query_workflow_run_by_id(therock_repo_full, start)
        start_sha = workflow_info.get("head_sha")
    else:
        start_sha = start

    # Resolve end commit
    if args.workflow_mode:
        workflow_info = gha_query_workflow_run_by_id(therock_repo_full, end)
        end_sha = workflow_info.get("head_sha")
    else:
        end_sha = end

    return start_sha, end_sha


# =============================================================================
# GitHub API Utilities
# =============================================================================


def get_api_base_from_url(url: str, fallback_name: str) -> str:
    """Convert git remote URL to GitHub API base URL."""
    # To handle cases where urls are in ssh format, we need to convert them to https format
    normalized = url.replace("git@github.com:", "https://github.com/")
    parsed = urllib.parse.urlparse(normalized)
    if parsed.netloc == "github.com":
        path = parsed.path.lstrip("/")
        if path.endswith(".git"):
            path = path[:-4]
        return f"https://api.github.com/repos/{path}"
    print(f"  Warning: Could not parse URL '{url}', using ROCm/{fallback_name}")
    return f"{GITHUB_API_BASE}/{ROCM_ORG}/{fallback_name}"


def is_revert(old_sha: str, new_sha: str, api_base: str) -> bool:
    """Check if updating from old_sha to new_sha is a revert (going backwards).

    Uses GitHub compare API: compare/{new_sha}...{old_sha}
    - If old_sha is "ahead" of new_sha → old_sha came later → revert → return True
    - If old_sha is "behind" new_sha → normal forward progress → return False
    - If "diverged" → different branches, not a revert → return False
    - If 404 → commits deleted from repo → return False (can't determine)
    """
    try:
        # If old_sha is "ahead" of new_sha, we're moving backwards (revert)
        compare = gha_send_request(f"{api_base}/compare/{new_sha}...{old_sha}")
        status = compare.get("status", "")
        return status == "ahead"
    except HTTPError as e:
        # 404 = commits deleted from repo (hard orphaned, rare)
        # Other errors = network issues, rate limits, etc.
        if e.code == 404:
            print("    (Commits deleted from repo - cannot determine revert status)")
        return False
    except (KeyError, ValueError, TypeError) as e:
        print(f"  Warning: Could not compare commits for revert detection: {e}")
        return False


def fetch_commits_in_range(
    repo_name: str,
    start_sha: str,
    end_sha: str,
    api_base: str,
) -> list[dict]:
    """Fetch commits between two SHAs."""
    commits: list[dict] = []
    found_start = False
    page = 1

    print(f"  Fetching commits for {repo_name}: {start_sha[:7]} -> {end_sha[:7]}")

    while not found_start and page <= MAX_PAGES:
        params = {"sha": end_sha, "per_page": PER_PAGE, "page": page}
        url = f"{api_base}/commits?{urllib.parse.urlencode(params)}"
        try:
            data = gha_send_request(url)
            if not data:
                break
            for commit in data:
                if commit["sha"] == start_sha:
                    found_start = True
                    break
                commits.append(commit)
            if len(data) < PER_PAGE:
                break
            page += 1
        except (KeyError, ValueError, TypeError) as e:
            print(f"  Error fetching commits: {e}")
            break

    # Fallback for diverged/off-branch commits (soft orphaned)
    # This happens when commits exist in the repo but aren't in linear history,
    # e.g., after a rebase where the branch moved but old commits weren't deleted.
    # The compare API can still find commits unique to end_sha in these cases.
    if not found_start:
        try:
            compare_url = f"{api_base}/compare/{start_sha}...{end_sha}"
            compare_data = gha_send_request(compare_url)
            if compare_data.get("status") == "diverged":
                print("    (Commits are off-branch/diverged - using compare API)")
                diverged_commits = compare_data.get("commits", [])
                print(f"  Found {len(diverged_commits)} commits unique to end")
                return diverged_commits
        except (KeyError, ValueError, TypeError, RuntimeError) as e:
            print(f"  Warning: Could not use compare API fallback: {e}")

    print(f"  Found {len(commits)} commits")
    return commits


def fetch_superrepo_components(
    repo_name: str, commit_sha: str, api_base: str
) -> list[str]:
    """Get component paths from superrepo at given commit.

    Note: Some directories (e.g., shared/) may not exist at older commits.
    404 errors are handled gracefully - directory is skipped.
    """
    components: list[str] = []
    for directory in SUPERREPO_COMPONENT_DIRS:
        url = f"{api_base}/contents/{directory}?ref={commit_sha}"
        try:
            data = gha_send_request(url)
            for item in data:
                if item["type"] == "dir":
                    components.append(f"{directory}/{item['name']}")
        except HTTPError as e:
            if e.code == 404:
                print(f"    {directory}/ not found at this commit (skipping)")
            else:
                print(f"    HTTP {e.code} fetching {directory}/")
        except (KeyError, ValueError, TypeError) as e:
            print(f"    Failed to fetch {directory} folder: {e}")

    return components


def fetch_commits_by_directory(
    repo_name: str,
    start_sha: str,
    end_sha: str,
    api_base: str,
    directories: list[str],
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Get commits allocated by directory."""
    print(f"    Getting commits by directories for {repo_name}")

    all_commits = fetch_commits_in_range(
        repo_name=repo_name,
        start_sha=start_sha,
        end_sha=end_sha,
        api_base=api_base,
    )
    commit_shas_in_range = {c["sha"] for c in all_commits}

    allocation: dict[str, list[dict]] = {}
    seen_shas: set[str] = set()

    for directory in directories:
        dir_key = directory.rstrip("/")
        allocation[dir_key] = []

        page = 1
        while page <= MAX_PAGES:
            params = {
                "sha": end_sha,
                "path": directory,
                "per_page": PER_PAGE,
                "page": page,
            }
            url = f"{api_base}/commits?{urllib.parse.urlencode(params)}"
            try:
                data = gha_send_request(url)
                if not data:
                    break
                commits_found = 0
                for commit in data:
                    sha = commit["sha"]
                    if sha in commit_shas_in_range:
                        allocation[dir_key].append(commit)
                        seen_shas.add(sha)
                        commits_found += 1
                        if sha == start_sha:
                            break
                if commits_found == 0 and page > 1:
                    break
                if len(data) < PER_PAGE:
                    break
                page += 1
            except (KeyError, ValueError, TypeError) as e:
                print(f"    Error querying directory {directory}: {e}")
                break

        if allocation[dir_key]:
            print(f"    {dir_key}: {len(allocation[dir_key])} commits")

    # Unassigned commits
    unassigned = [c for c in all_commits if c["sha"] not in seen_shas]
    if unassigned:
        allocation[UNASSIGNED_KEY] = unassigned
        print(f"    {UNASSIGNED_KEY}: {len(unassigned)} commits")

    return allocation, all_commits


# =============================================================================
# Manifest Loading
# =============================================================================


def load_submodules_at_commit(commit_sha: str) -> dict[str, dict[str, str]]:
    """Load all submodules from TheRock at a specific commit.

    Uses build_manifest_schema from generate_therock_manifest to get submodule
    information and converts it to the expected format.

    Note: Requires the repository to be checked out with full history
    (fetch-depth: 0) for historical commits to be accessible via local git.
    """
    try:
        manifest = build_manifest_schema(THEROCK_DIR, commit_sha)

        submodules: dict[str, dict[str, str]] = {}
        for entry in manifest.get("submodules", []):
            name = entry["submodule_name"]
            url = entry["submodule_url"]
            pin_sha = entry.get("pin_sha")

            if pin_sha:
                submodules[name] = {
                    "sha": pin_sha,
                    "api_base": get_api_base_from_url(url, name),
                    "branch": None,  # Branch info not in manifest schema
                }

        print(f"Found {len(submodules)} submodules in manifest for {commit_sha[:8]}")
        return submodules

    except (RuntimeError, KeyError) as e:
        print(f"Error loading submodules for commit {commit_sha}: {e}")
        return {}


# =============================================================================
# Status Determination
# =============================================================================


def determine_status(
    old_sha: str | None,
    new_sha: str | None,
    api_base: str,
) -> tuple[str, str, str]:
    """Determine submodule status. Returns (status, fetch_start, fetch_end)."""
    if old_sha and not new_sha:
        return "removed", "", ""
    if new_sha and not old_sha:
        return "added", "", new_sha
    if old_sha == new_sha:
        return "unchanged", "", ""

    # Check for revert (is_revert handles orphaned/404 gracefully by returning False)
    if is_revert(old_sha, new_sha, api_base):
        return "reverted", new_sha, old_sha

    return "changed", old_sha, new_sha


# =============================================================================
# Submodule Processing
# =============================================================================


def process_regular_submodule(
    name: str,
    start_data: dict[str, str] | None,
    end_data: dict[str, str] | None,
) -> Submodule:
    """Process a regular (non-superrepo) submodule."""
    old_sha = start_data["sha"] if start_data else None
    new_sha = end_data["sha"] if end_data else None
    data = end_data or start_data or {}
    api_base = data.get("api_base", "")
    branch = data.get("branch", "main")

    status, fetch_start, fetch_end = determine_status(old_sha, new_sha, api_base)

    submodule = Submodule(
        name=name,
        sha=new_sha or old_sha or "",
        api_base=api_base,
        branch=branch,
        status=status,
        start_sha=old_sha or "",
        end_sha=new_sha or "",
    )

    if status == "removed":
        print(f"  {name}: REMOVED")

    elif status == "added":
        print(f"  {name}: ADDED -> {fetch_end[:7]}")
        # For a newly added submodule, we need to fetch only the tip commit
        try:
            tip_commit = gha_send_request(f"{api_base}/commits/{fetch_end}")
            submodule.commits = [tip_commit] if tip_commit else []
        except (KeyError, ValueError, TypeError) as e:
            print(f"  Warning: Could not fetch tip commit: {e}")

    elif status == "unchanged":
        print(f"  {name}: UNCHANGED")

    elif status in ("changed", "reverted"):
        label = "REVERTED" if status == "reverted" else "CHANGED"
        print(f"  {name}: {label} {fetch_start[:7]} -> {fetch_end[:7]}")
        submodule.commits = fetch_commits_in_range(
            repo_name=name,
            start_sha=fetch_start,
            end_sha=fetch_end,
            api_base=api_base,
        )

    return submodule


class SuperrepoProcessor:
    """Processes a superrepo with component-level analysis."""

    def __init__(
        self,
        name: str,
        start_data: dict[str, str] | None,
        end_data: dict[str, str] | None,
    ):
        self.name = name
        self.start_data = start_data
        self.end_data = end_data
        self.old_sha = start_data["sha"] if start_data else None
        self.new_sha = end_data["sha"] if end_data else None
        data = end_data or start_data or {}
        self.api_base = data.get("api_base", "")
        self.branch = data.get("branch", "main")
        self.superrepo: Superrepo | None = None
        self.fetch_start: str | None = None
        self.fetch_end: str | None = None

    def process(self) -> Superrepo:
        """Process the superrepo and return the result."""
        self.init_superrepo()
        print(f"  {self.name}: SUPERREPO ({self.superrepo.status})")

        if self.superrepo.status == "removed":
            return self.handle_removed()
        elif self.superrepo.status == "added":
            return self.handle_added()
        elif self.superrepo.status == "unchanged":
            return self.handle_unchanged()
        else:
            return self.handle_changed_or_reverted()

    def init_superrepo(self) -> None:
        """Initialize the Superrepo object with basic info."""
        status, self.fetch_start, self.fetch_end = determine_status(
            self.old_sha, self.new_sha, self.api_base
        )
        self.superrepo = Superrepo(
            name=self.name,
            sha=self.new_sha or self.old_sha or "",
            api_base=self.api_base,
            branch=self.branch,
            status=status,
            start_sha=self.old_sha or "",
            end_sha=self.new_sha or "",
        )

    def handle_removed(self) -> Superrepo:
        """Handle a removed superrepo."""
        start_components = fetch_superrepo_components(
            self.name, self.old_sha, self.api_base
        )
        print(f"    Removed superrepo had {len(start_components)} components")
        for comp_path in start_components:
            self.add_component(comp_path, status="removed")
        return self.superrepo

    def handle_added(self) -> Superrepo:
        """Handle a newly added superrepo."""
        end_components = fetch_superrepo_components(
            self.name, self.new_sha, self.api_base
        )
        print(f"    Added superrepo with {len(end_components)} components")
        print(f"    Fetching tip commits for each component...")
        for comp_path in end_components:
            tip_commits = self.fetch_tip_commit(comp_path)
            self.add_component(comp_path, status="added", commits=tip_commits)
        return self.superrepo

    def handle_unchanged(self) -> Superrepo:
        """Handle an unchanged superrepo."""
        end_components = fetch_superrepo_components(
            self.name, self.new_sha, self.api_base
        )
        print(f"    Unchanged superrepo with {len(end_components)} components")
        for comp_path in end_components:
            self.add_component(comp_path, status="unchanged")
        return self.superrepo

    def handle_changed_or_reverted(self) -> Superrepo:
        """Handle a changed or reverted superrepo with full commit analysis."""
        start_components = (
            fetch_superrepo_components(self.name, self.old_sha, self.api_base)
            if self.old_sha
            else []
        )
        end_components = (
            fetch_superrepo_components(self.name, self.new_sha, self.api_base)
            if self.new_sha
            else []
        )

        start_set = set(start_components)
        end_set = set(end_components)
        added_paths = end_set - start_set
        removed_paths = start_set - end_set

        print(
            f"    Components: {len(start_components)} -> {len(end_components)} "
            f"(+{len(added_paths)} -{len(removed_paths)})"
        )

        all_components = start_set | end_set
        directories = [c + "/" if not c.endswith("/") else c for c in all_components]

        allocation, all_commits = fetch_commits_by_directory(
            repo_name=self.name,
            start_sha=self.fetch_start,
            end_sha=self.fetch_end,
            api_base=self.api_base,
            directories=directories,
        )

        self.superrepo.all_commits = all_commits
        self.superrepo.commit_allocation = allocation

        for comp_path in all_components:
            comp_status, comp_commits = self.determine_component_status(
                comp_path, added_paths, removed_paths, allocation
            )
            self.add_component(comp_path, status=comp_status, commits=comp_commits)

        return self.superrepo

    def determine_component_status(
        self,
        comp_path: str,
        added_paths: set[str],
        removed_paths: set[str],
        allocation: dict[str, list[dict]],
    ) -> tuple[str, list[dict]]:
        """Determine the status and commits for a component."""
        comp_key = comp_path.rstrip("/")

        if comp_path in added_paths:
            comp_commits = allocation.get(comp_key, [])
            if comp_commits:
                comp_commits = [comp_commits[0]]
            return "added", comp_commits
        elif comp_path in removed_paths:
            return "removed", []
        elif comp_key in allocation and allocation[comp_key]:
            return "changed", allocation.get(comp_key, [])
        else:
            return "unchanged", []

    def fetch_tip_commit(self, comp_path: str) -> list[dict]:
        """Fetch the tip commit for a component."""
        comp_name = comp_path.split("/")[-1]
        try:
            params = {"sha": self.new_sha, "path": comp_path, "per_page": 1}
            url = f"{self.api_base}/commits?{urllib.parse.urlencode(params)}"
            tip_data = gha_send_request(url)
            if tip_data:
                return [tip_data[0]]
        except (KeyError, ValueError, TypeError, HTTPError) as e:
            print(f"      Warning: Could not fetch tip for {comp_name}: {e}")
        return []

    def add_component(
        self, comp_path: str, status: str, commits: list[dict] | None = None
    ) -> None:
        """Add a component to the superrepo."""
        comp_name = comp_path.split("/")[-1]
        self.superrepo.components[comp_path] = Component(
            path=comp_path, name=comp_name, status=status, commits=commits or []
        )


def process_superrepo(
    name: str,
    start_data: dict[str, str] | None,
    end_data: dict[str, str] | None,
) -> Superrepo:
    """Process a superrepo with component-level analysis."""
    return SuperrepoProcessor(name, start_data, end_data).process()


# =============================================================================
# ManifestDiff Construction
# =============================================================================


def compare_manifests(start_commit: str, end_commit: str) -> ManifestDiff:
    """Compare two TheRock commits and return a ManifestDiff."""
    print(f"\nComparing commits: {start_commit[:7]} -> {end_commit[:7]}")

    print("\n=== Getting submodules for START commit ===")
    start_subs = load_submodules_at_commit(start_commit)
    print(f"\nFound {len(start_subs)} submodules for START commit")

    print("\n=== Getting submodules for END commit ===")
    end_subs = load_submodules_at_commit(end_commit)
    print(f"\nFound {len(end_subs)} submodules for END commit")

    diff = ManifestDiff(start_commit=start_commit, end_commit=end_commit)

    all_names = set(start_subs.keys()) | set(end_subs.keys())

    print("\n=== Processing Submodules ===")
    for name in sorted(all_names):
        start_data = start_subs.get(name)
        end_data = end_subs.get(name)

        if name in SUPERREPO_NAMES:
            diff.superrepos[name] = process_superrepo(name, start_data, end_data)
        else:
            diff.submodules[name] = process_regular_submodule(
                name, start_data, end_data
            )

    return diff


# =============================================================================
# HTML Helpers
# =============================================================================


def format_commit_date(date_string: str) -> str:
    """Format ISO date string to readable format."""
    if date_string == "Unknown" or not date_string:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(date_string.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y")
    except (ValueError, TypeError):
        return date_string


def create_commit_badge(sha: str, repo_name: str) -> str:
    """Create HTML badge for a commit SHA."""
    short_sha = sha[:7] if sha not in ("-", "N/A", "") else sha or "N/A"
    url = f"https://github.com/ROCm/{repo_name}/commit/{sha}"
    return (
        f"<a href='{url}' target='_blank' class='commit-badge-link'>"
        f"<span class='commit-badge'>{short_sha}</span></a>"
    )


def extract_commit_data(commit: dict) -> dict[str, str]:
    """Extract commit data from GitHub API response."""
    return {
        "sha": commit.get("sha", "-"),
        "message": commit.get("commit", {}).get("message", "-").split("\n")[0],
        "author": commit.get("commit", {}).get("author", {}).get("name", "Unknown"),
        "date": format_commit_date(
            commit.get("commit", {}).get("author", {}).get("date", "Unknown")
        ),
    }


def create_commit_item_html(commit: dict, repo_name: str) -> str:
    """Create HTML for a single commit item."""
    data = extract_commit_data(commit)
    badge = create_commit_badge(data["sha"], repo_name)
    return (
        f"<div class='commit-item'>"
        f"<div>{badge} {data['message']}</div>"
        f"<div class='commit-meta'>{data['date']} | {data['author']}</div>"
        f"</div>"
    )


def create_commit_list_html(
    commits: list[dict],
    repo_name: str,
    status: str | None = None,
    start_sha: str | None = None,
    end_sha: str | None = None,
) -> str:
    """Create scrollable commit container with status-aware styling.

    Args:
        commits: List of commit dictionaries.
        repo_name: Repository name for creating commit links.
        status: Optional status ('newly_added', 'removed', 'reverted').
        start_sha: Optional start SHA for reverted message context.
        end_sha: Optional end SHA for reverted message context.
    """
    commit_items = [create_commit_item_html(c, repo_name) for c in commits]

    if status == "newly_added":
        if commit_items:
            content = (
                '<div class="newly-added-header"><strong>NEWLY ADDED:</strong> '
                "This component was newly added. Showing current tip commit.</div>"
                + "".join(commit_items)
            )
        else:
            content = (
                '<div class="newly-added-header"><strong>NEWLY ADDED:</strong> '
                "This component was newly added.</div>"
            )
    elif status == "removed":
        content = '<div class="removed-message">Component removed in this version</div>'
    elif status == "reverted":
        # Build informative revert message with commit links
        if start_sha and end_sha:
            start_badge = create_commit_badge(start_sha[:7], repo_name)
            end_badge = create_commit_badge(end_sha[:7], repo_name)
            revert_msg = f"Reverted from {start_badge} back to {end_badge}."
        else:
            # For components within a reverted superrepo (no individual SHAs)
            revert_msg = "Commits being undone by this revert."
        content = (
            f'<div class="reverted-header"><strong>REVERTED:</strong> '
            f"{revert_msg}</div>" + "".join(commit_items)
        )
    elif not commit_items:
        content = '<div class="no-commits">Component has no commits in this range (Superrepo Component Unchanged)</div>'
    else:
        content = "".join(commit_items)

    # Add background classes for status styling
    classes = ["commit-list"]
    if status == "newly_added":
        classes.append("newly-added-bg")
    elif status == "removed":
        classes.append("removed-bg")
    elif status == "reverted":
        classes.append("reverted-bg")

    return f"<div class='{' '.join(classes)}'>{content}</div>"


def create_table(headers: list[str], rows: list[str]) -> str:
    """Create HTML table."""
    header_html = "".join(f"<th>{h}</th>" for h in headers)
    return (
        "<table class='report-table'>"
        f"<tr class='report-table-header-row'>{header_html}</tr>"
        f"{''.join(rows)}</table>"
    )


# =============================================================================
# Summary HTML Generation
# =============================================================================


def generate_summary_html(items: dict[str, list[str]], summary_type: str) -> str:
    """Generate HTML summary lists."""
    total = sum(len(v) for v in items.values())
    if total == 0:
        return ""

    categories = [
        ("added", "Newly Added"),
        ("removed", "Removed"),
        ("changed", "Changed"),
        ("unchanged", "Unchanged"),
        ("reverted", "Reverted"),
    ]

    parts = []
    for cat, label in categories:
        cat_items = items.get(cat, [])
        if not cat_items:
            continue
        list_html = "".join(f"<li><code>{i}</code></li>" for i in sorted(cat_items))
        parts.append(
            f'<div class="summary-category {cat}">'
            f"<h3>{label} {summary_type.title()} ({len(cat_items)}/{total}):</h3>"
            f'<ul class="summary-list">{list_html}</ul></div>'
        )

    return "".join(parts)


# =============================================================================
# Superrepo HTML Generation
# =============================================================================


class SuperrepoHtmlBuilder:
    """Builds HTML for a superrepo section."""

    def __init__(
        self, superrepo: Superrepo, removed_submodules: list[str] | None = None
    ):
        self.superrepo = superrepo
        self.removed_submodules = removed_submodules
        self.commit_to_projects: dict[str, set[str]] = {}

    def build(self) -> str:
        """Build and return the complete HTML for the superrepo."""
        print(f"  Generating HTML for {self.superrepo.name} ({self.superrepo.status})")

        if not self.superrepo.components:
            return self.build_empty_html()

        self.build_commit_mapping()
        status_banner = self.build_status_banner()
        component_table = self.build_component_table()
        history_html = self.build_history_table()

        return status_banner + component_table + history_html

    def build_empty_html(self) -> str:
        """Build HTML for a superrepo with no components."""
        if self.superrepo.status == "removed":
            return f"<div class='removed'><strong>{self.superrepo.name}:</strong> REMOVED</div>"
        return f"<div class='unchanged'><strong>{self.superrepo.name}:</strong> No components</div>"

    def build_status_banner(self) -> str:
        """Build the status banner for added/removed/reverted superrepos."""
        if self.superrepo.status == "added":
            replaced_note = ""
            if self.removed_submodules:
                replaced_list = ", ".join(sorted(self.removed_submodules))
                replaced_note = (
                    f"<div class='replaced-submodules'>Replaces direct submodules: "
                    f"<code>{replaced_list}</code></div>"
                )
            return (
                "<div class='superrepo-status-banner added'>"
                f"<strong>NEWLY ADDED SUPERREPO</strong> - "
                f"This superrepo was newly added with {len(self.superrepo.components)} components. "
                f"Showing tip commit for each component.{replaced_note}</div>"
            )
        elif self.superrepo.status == "removed":
            return (
                "<div class='superrepo-status-banner removed'>"
                f"<strong>REMOVED SUPERREPO</strong> - "
                f"This superrepo was removed. It previously contained "
                f"{len(self.superrepo.components)} components.</div>"
            )
        elif self.superrepo.status == "reverted":
            start_badge = create_commit_badge(
                self.superrepo.start_sha[:7], self.superrepo.name
            )
            end_badge = create_commit_badge(
                self.superrepo.end_sha[:7], self.superrepo.name
            )
            return (
                "<div class='superrepo-status-banner reverted'>"
                f"<strong>REVERTED SUPERREPO</strong> - "
                f"This superrepo was reverted from {start_badge} back to {end_badge}. "
                f"Commits shown below are being undone by this revert.</div>"
            )
        return ""

    def build_commit_mapping(self) -> None:
        """Build mapping from commit SHA to component names."""
        for comp in self.superrepo.components.values():
            for commit in comp.commits:
                sha = commit.get("sha", "-")
                if sha not in self.commit_to_projects:
                    self.commit_to_projects[sha] = set()
                self.commit_to_projects[sha].add(comp.name)

    def build_component_table(self) -> str:
        """Build the component table HTML."""
        rows = []
        for comp in sorted(self.superrepo.components.values(), key=lambda c: c.path):
            row_classes = []
            status_class = None

            if comp.status == "added":
                status_class = "newly_added"
            elif comp.status == "removed":
                status_class = "removed"
            elif comp.status == "unchanged":
                row_classes.append("unchanged-row")
            elif self.superrepo.status == "reverted" and comp.commits:
                status_class = "reverted"

            row_classes.append("component-row")
            commit_html = create_commit_list_html(
                comp.commits, self.superrepo.name, status_class
            )
            row_class_attr = f" class='{' '.join(row_classes)}'" if row_classes else ""
            data_component = html.escape(comp.name, quote=True)
            rows.append(
                f"<tr{row_class_attr} data-component='{data_component}'><td>{comp.name}</td><td>{commit_html}</td></tr>"
            )

        return create_table(["Component", "Commits"], rows)

    def build_history_table(self) -> str:
        """Build the commit history table HTML."""
        if not self.superrepo.all_commits:
            return ""

        history_rows = []
        for commit in self.superrepo.all_commits:
            data = extract_commit_data(commit)
            projects = (
                ", ".join(sorted(self.commit_to_projects.get(data["sha"], [])))
                or "Unassigned"
            )
            badge = create_commit_badge(data["sha"], self.superrepo.name)
            history_rows.append(
                f"<tr>"
                f"<td class='date-col'>{data['date']}</td>"
                f"<td>{badge}</td>"
                f"<td class='author-col'>{data['author']}</td>"
                f"<td class='project-col'>{projects}</td>"
                f"<td class='message-col'>{data['message']}</td>"
                f"</tr>"
            )

        if not history_rows:
            return ""

        return (
            "<div class='section-title' style='margin-top:16px;font-weight:bold;color:#1976D2;'>"
            "Commit History (newest to oldest):</div>"
            "<table class='commit-history-table'>"
            "<tr><th class='col-date'>Date</th><th class='col-sha'>SHA</th>"
            "<th class='col-author'>Author</th><th class='col-projects'>Project(s)</th>"
            "<th>Message</th></tr>" + "".join(history_rows) + "</table>"
        )


def generate_superrepo_html(
    superrepo: Superrepo, removed_submodules: list[str] | None = None
) -> str:
    """Generate HTML for a superrepo section."""
    return SuperrepoHtmlBuilder(superrepo, removed_submodules).build()


# =============================================================================
# Non-Superrepo HTML Generation
# =============================================================================


def generate_non_superrepo_html(diff: ManifestDiff) -> str:
    """Generate HTML for non-superrepo submodules table."""
    submodules = list(diff.submodules.values())
    print(f"  Generating HTML for {len(submodules)} non-superrepo submodules")

    if not submodules:
        return "<div class='no-commits'>No non-superrepo submodules found</div>"

    # Banner for removed submodules (moved to superrepo)
    removed_banner = ""
    removed_subs = [s for s in submodules if s.status == "removed"]
    if removed_subs:
        # Check if any superrepo was added (likely the destination)
        added_superrepos = [
            s.name for s in diff.superrepos.values() if s.status == "added"
        ]
        destination = (
            f" (moved to {', '.join(added_superrepos)})" if added_superrepos else ""
        )
        removed_names = ", ".join(sorted(s.name for s in removed_subs))
        removed_banner = (
            "<div class='removed-submodules-banner'>"
            f"<strong>REMOVED SUBMODULES ({len(removed_subs)})</strong>{destination}<br>"
            f"<code>{removed_names}</code></div>"
        )

    rows = []
    for sub in sorted(submodules, key=lambda s: s.name):
        # Show range of commits actually displayed (oldest → newest)
        if sub.commits:
            oldest_sha = sub.commits[-1].get("sha", "-")
            newest_sha = sub.commits[0].get("sha", "-")
            start_badge = create_commit_badge(oldest_sha, sub.name)
            end_badge = create_commit_badge(newest_sha, sub.name)
        else:
            start_badge = create_commit_badge(sub.start_sha or "-", sub.name)
            end_badge = create_commit_badge(sub.end_sha or "-", sub.name)
        range_html = f"<div style='margin-bottom:8px;'><strong>Range:</strong> {start_badge} -&gt; {end_badge}</div>"

        row_classes = []
        status_class = None

        if sub.status == "added":
            status_class = "newly_added"
        elif sub.status == "removed":
            status_class = "removed"
        elif sub.status == "reverted":
            status_class = "reverted"
        elif sub.status == "unchanged":
            row_classes.append("unchanged-row")

        # Pass start/end SHAs for reverted status message
        commit_html = create_commit_list_html(
            sub.commits,
            sub.name,
            status_class,
            start_sha=sub.start_sha if sub.status == "reverted" else None,
            end_sha=sub.end_sha if sub.status == "reverted" else None,
        )
        full_content = range_html + commit_html
        row_classes.append("component-row")
        row_class_attr = f" class='{' '.join(row_classes)}'" if row_classes else ""
        data_component = html.escape(sub.name, quote=True)
        rows.append(
            f"<tr{row_class_attr} data-component='{data_component}'><td>{sub.name}</td><td>{full_content}</td></tr>"
        )

    return removed_banner + create_table(["Submodule", "Commits"], rows)


# =============================================================================
# Report Generation
# =============================================================================


class HtmlReportGenerator:
    """Generates HTML reports from ManifestDiff data."""

    def __init__(self, diff: ManifestDiff, output_dir: Path | None = None):
        """Initialize the generator.

        Args:
            diff: The manifest diff to generate the report from.
            output_dir: Optional output directory. If provided, creates the directory
                        and writes the report there. Otherwise writes to TheRock root.
        """
        self.diff = diff
        self.output_dir = output_dir
        self.html = ""
        self.report_path: Path | None = None

    def generate(self) -> Path:
        """Generate the HTML report.

        Returns:
            Path to the generated report file.
        """
        print("\n=== Writing HTML Report ===")
        self.load_template()
        self.inject_commit_range()
        self.inject_summaries()
        self.inject_section_badges()
        self.inject_section_content()
        return self.write_report()

    def load_template(self) -> None:
        """Load the HTML template and determine output path."""
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.report_path = self.output_dir / "TheRockReport.html"
        else:
            self.report_path = HTML_REPORT_PATH

        if not HTML_TEMPLATE_PATH.exists():
            print(f"  ERROR: Template not found at {HTML_TEMPLATE_PATH}")
            raise FileNotFoundError(f"Template not found at {HTML_TEMPLATE_PATH}")

        self.html = HTML_TEMPLATE_PATH.read_text(encoding="utf-8")

    def inject_commit_range(self) -> None:
        """Inject TheRock start and end commit badges into the template."""
        self.html = self.html.replace(
            '<span id="therock-start-commit">START_COMMIT</span>',
            f'<span id="therock-start-commit">{self.diff.start_commit[:8]}</span>',
        )
        self.html = self.html.replace(
            '<span id="therock-end-commit">END_COMMIT</span>',
            f'<span id="therock-end-commit">{self.diff.end_commit[:8]}</span>',
        )

    def inject_summaries(self) -> None:
        """Inject submodule and superrepo summary sections into the template."""
        # Submodule summary
        status_groups = self.diff.get_status_groups()
        submodule_summary = generate_summary_html(status_groups, "submodules")
        self.html = self.html.replace(
            '<div id="submodule-content"></div>',
            f'<div id="submodule-content">{submodule_summary}</div>',
        )

        # Superrepo summaries
        superrepo_parts = []
        for superrepo in self.diff.superrepos.values():
            comp_groups = {
                "added": superrepo.added_components,
                "removed": superrepo.removed_components,
                "changed": superrepo.changed_components,
                "unchanged": superrepo.unchanged_components,
            }
            comp_summary = generate_summary_html(comp_groups, "components")
            if comp_summary:
                superrepo_parts.append(
                    f"<div class='summary-section'><h2>{superrepo.name.title()} Components</h2>"
                    f"{comp_summary}</div>"
                )
        self.html = self.html.replace(
            '<div id="superrepo-content"></div>',
            f'<div id="superrepo-content">{"".join(superrepo_parts)}</div>',
        )

    def inject_section_badges(self) -> None:
        """Inject commit badges for each section header in the template."""
        # Superrepo section badges
        for repo_name in ["rocm-libraries", "rocm-systems"]:
            superrepo = self.diff.superrepos.get(repo_name)
            start_sha = superrepo.start_sha if superrepo else ""
            end_sha = superrepo.end_sha if superrepo else ""
            start_badge = (
                create_commit_badge(start_sha, repo_name) if start_sha else "N/A"
            )
            end_badge = create_commit_badge(end_sha, repo_name) if end_sha else "N/A"
            self.html = self.html.replace(
                f'<span id="commit-diff-start-{repo_name}-superrepo"></span>',
                f'<span id="commit-diff-start-{repo_name}-superrepo">{start_badge}</span>',
            )
            self.html = self.html.replace(
                f'<span id="commit-diff-end-{repo_name}-superrepo"></span>',
                f'<span id="commit-diff-end-{repo_name}-superrepo">{end_badge}</span>',
            )

        # Non-superrepo section badges
        start_badge = create_commit_badge(self.diff.start_commit, THEROCK_REPO)
        end_badge = create_commit_badge(self.diff.end_commit, THEROCK_REPO)
        self.html = self.html.replace(
            '<span id="commit-diff-start-non-superrepo"></span>',
            f'<span id="commit-diff-start-non-superrepo">{start_badge}</span>',
        )
        self.html = self.html.replace(
            '<span id="commit-diff-end-non-superrepo"></span>',
            f'<span id="commit-diff-end-non-superrepo">{end_badge}</span>',
        )

    def inject_section_content(self) -> None:
        """Inject detailed commit content for each section in the template."""
        # Superrepo sections
        for repo_name in ["rocm-libraries", "rocm-systems"]:
            superrepo = self.diff.superrepos.get(repo_name)
            if superrepo:
                # Pass removed submodules if this superrepo was newly added
                removed_subs = (
                    self.diff.removed if superrepo.status == "added" else None
                )
                content = generate_superrepo_html(
                    superrepo, removed_submodules=removed_subs
                )
            else:
                content = ""
            self.html = self.html.replace(
                f'<div id="commit-diff-job-content-{repo_name}-superrepo" style="margin-top:8px;">\n      </div>',
                f'<div id="commit-diff-job-content-{repo_name}-superrepo" style="margin-top:8px;">\n        {content}\n      </div>',
            )

        # Non-superrepo section
        non_superrepo_html = generate_non_superrepo_html(self.diff)
        self.html = self.html.replace(
            '<div id="commit-diff-job-content-non-superrepo" style="margin-top:8px;">\n      </div>',
            f'<div id="commit-diff-job-content-non-superrepo" style="margin-top:8px;">\n        {non_superrepo_html}\n      </div>',
        )

    def write_report(self) -> Path:
        """Write the HTML report to disk and return the path."""
        self.report_path.write_text(self.html, encoding="utf-8")
        if not self.report_path.exists() or self.report_path.stat().st_size == 0:
            raise RuntimeError(f"Failed to write HTML report to {self.report_path}")
        print(f"  Report written to: {self.report_path}")
        return self.report_path


def generate_html_report(diff: ManifestDiff, output_dir: Path | None = None) -> Path:
    """Generate TheRockReport.html from template.

    Args:
        diff: The manifest diff to generate the report from.
        output_dir: Optional output directory. If provided, creates the directory
                    and writes the report there. Otherwise writes to TheRock root.

    Returns:
        Path to the generated report file.
    """
    return HtmlReportGenerator(diff, output_dir).generate()


def generate_step_summary(diff: ManifestDiff) -> None:
    """Generate GitHub Actions step summary using the shared utility."""
    summary = "# TheRock Manifest Diff Report\n\n"
    summary += (
        f"**Commit Range:** `{diff.start_commit[:8]}` -> `{diff.end_commit[:8]}`\n\n"
    )
    summary += "## Submodule Changes\n\n"

    # Submodule changes
    status_configs = [
        ("added", "Added"),
        ("removed", "Removed"),
        ("changed", "Changed"),
        ("reverted", "Reverted"),
    ]

    status_groups = diff.get_status_groups()
    for status, title in status_configs:
        items = status_groups.get(status, [])
        if items:
            summary += f"### {title} ({len(items)})\n"
            for name in sorted(items):
                sub = diff.all_items.get(name)
                if status == "changed" and sub:
                    summary += f"- `{name}` ({len(sub.commits)} commits)\n"
                else:
                    summary += f"- `{name}`\n"
            summary += "\n"

    unchanged = status_groups.get("unchanged", [])
    if unchanged:
        summary += f"<details><summary>Unchanged ({len(unchanged)})</summary>\n\n"
        for n in sorted(unchanged):
            summary += f"- `{n}`\n"
        summary += "\n</details>\n\n"

    # Superrepo changes
    has_superrepo_changes = any(
        s.added_components or s.removed_components or s.changed_components
        for s in diff.superrepos.values()
    )
    if has_superrepo_changes:
        summary += "## Superrepo Component Changes\n\n"
        for superrepo in diff.superrepos.values():
            if not (
                superrepo.added_components
                or superrepo.removed_components
                or superrepo.changed_components
            ):
                continue
            summary += f"### {superrepo.name}\n\n"
            if superrepo.added_components:
                components = ", ".join(
                    f"`{c}`" for c in sorted(superrepo.added_components)
                )
                summary += f"**Added:** {components}\n"
            if superrepo.removed_components:
                components = ", ".join(
                    f"`{c}`" for c in sorted(superrepo.removed_components)
                )
                summary += f"**Removed:** {components}\n"
            if superrepo.changed_components:
                components = ", ".join(
                    f"`{c}`" for c in sorted(superrepo.changed_components)
                )
                summary += f"**Changed:** {components}\n"
            if superrepo.unchanged_components:
                summary += (
                    f"**Unchanged:** {len(superrepo.unchanged_components)} components\n"
                )
            summary += "\n"

    summary += "---\n*Generated by generate_manifest_diff_report.py*"

    gha_append_step_summary(summary)


# =============================================================================
# Entry Point
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    start_commit, end_commit = resolve_commits(args)

    diff = compare_manifests(start_commit, end_commit)

    output_dir = args.output_dir
    generate_html_report(diff, output_dir)

    print("\n=== Generating Step Summary ===")
    generate_step_summary(diff)

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
