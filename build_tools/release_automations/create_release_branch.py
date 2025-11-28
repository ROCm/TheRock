#!/usr/bin/env python3
"""
This script automates the creation of release branches for theRock projects and their submodules.
It supports branching from a specified source branch, including or excluding specific projects,
and optionally notifying stakeholders via email.

Main Features:
- Clones or uses an existing source tree for theRock and its submodules.
- Checks out the specified source branch for all relevant projects.
- Determines projects to branch based on .gitmodules and user input.
- Creates a plan for branching, including checks for branch existence and push permissions.
- Displays a summary of planned actions
- Executes branch creation and pushes to remote repositories.
- Sends notifications to specified email addresses.

Usage:
    python create_release_branch.py --release_branch <branch> [options]

Options:
    --release_branch         Name of the branch to create (required).
    --apitoken              API token for authentication with upstream repositories.
    --dry-run                Executed all part of the code except git push

Note:
    - This script assumes a specific repository structure and the presence of .gitmodules.
"""
import argparse
import configparser
import logging
import shutil
import re
import sys
import subprocess
import tempfile
from pathlib import Path
import shlex
from typing import Dict, Tuple, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class RockBranchingAutomation:
    """Class for rock branching automation"""

    def __init__(self, cli_args):
        self.release_branch = cli_args.release_branch
        self.apitoken = cli_args.apitoken
        self.dry_run = getattr(cli_args, "dry_run", False)

    def log(self, *args, **kwargs):
        """
        log function to print and flush output
        """
        print(*args, **kwargs)
        sys.stdout.flush()

    def exec(self, args: list[str | Path], cwd: Path):
        """
        Executes a command in a subprocess.
        """
        cmd = [str(arg) for arg in args]
        self.log(f"++ Exec [{cwd}]$ {shlex.join(cmd)}")
        sys.stdout.flush()
        subprocess.check_call(args, cwd=str(cwd), stdin=subprocess.DEVNULL)

    def checkout_source(self) -> Tuple[Path, Dict[str, str]]:
        """
        Clone theRock repository and return its path along with submodules dict.
        """
        the_rock_url = "https://github.com/ROCm/TheRock.git"
        the_rock_dir_path = Path.cwd() / "rock-branching"
        if the_rock_dir_path.is_dir():
            logging.info("Removing existing directory for fresh clone")
            shutil.rmtree(the_rock_dir_path)
        self.exec(
            ["git", "clone", the_rock_url, str(the_rock_dir_path)], cwd=Path.cwd()
        )
        the_rock_dir = the_rock_dir_path
        if not the_rock_dir.is_dir():
            logging.error("TheRock directory not found after clone")
            raise RuntimeError("TheRock directory not found after clone")
        submodules = self.get_submodules(the_rock_dir)
        return the_rock_dir, submodules

    def get_submodules(self, repo_path: Path) -> Dict[str, str]:
        """Run 'git submodule status' and parse each line into a dict.
        Format of each line typically:
        <prefix><commit> <path> (description)
        Prefix: ' ' initialized, '-' not initialized, '+' updated.
        Store as: key=path, value={ 'commit': commit, 'prefix': prefix, 'description': description }
        """
        # exec does not capture output; keep legacy behavior by falling back to subprocess
        try:
            output = subprocess.check_output(
                ["git", "submodule", "status"], cwd=repo_path, text=True
            )
        except subprocess.CalledProcessError:
            return {}
        submodule_dict: Dict[str, str] = {}
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line[0] in ["-", "+"]:
                line = line[1:]
            parts = line.split()
            if len(parts) < 2:
                continue
            commit = parts[0]
            path = parts[1]
            submodule_dict[path] = commit
        return submodule_dict

    # Determine projects to branch based on .gitmodules and user input.
    # This is a helper function for checkout_source.
    # Note that this returns a dict of project names to their key/value pairs from gitmodules.
    def determine_projects(self, working_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Parse .gitmodules to determine submodule projects.
        """
        logging.debug("Determining projects from gitmodules in %s", working_dir)
        gitmodules_path = working_dir / ".gitmodules"
        config = configparser.ConfigParser()
        config.read(gitmodules_path)
        projects: Dict[str, Dict[str, str]] = {}
        for section in config.sections():
            if section.startswith("submodule "):
                raw_name = section[len("submodule ") :]
                name = raw_name.strip('"')
            else:
                name = section
            projects[name] = {}
            for key, val in config.items(section):
                projects[name][key] = val
        return projects

    def get_upstream_branch(self, repo_path: Path) -> str:
        """Return default upstream branch (main/master) with minimal checks."""
        # Try symbolic-ref first
        try:
            out = subprocess.check_output(
                "git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'",
                shell=True,
                cwd=repo_path,
                text=True,
            ).strip()
            if out:
                return out
        except subprocess.CalledProcessError:
            pass
        # Simple fallbacks
        for branch in ("main", "master"):
            try:
                subprocess.check_call(
                    ["git", "rev-parse", "--verify", f"origin/{branch}"],
                    cwd=repo_path,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return branch
            except subprocess.CalledProcessError:
                continue
        raise RuntimeError("Cannot determine upstream branch")

    def tokenize_url(self, url: str) -> str:
        """Embed token into https URL if token available."""
        if self.apitoken and url.startswith("https://"):
            # Avoid double insertion
            if f"https://{self.apitoken}@" not in url:
                return url.replace("https://", f"https://{self.apitoken}@")
        return url

    def execute_plan(
        self, plan: Dict[str, Dict[str, str]], release_branch: str
    ) -> None:
        """
        Execute the branching plan.
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="rock-branching-"))
        logging.info("Temporary working directory: %s", temp_dir)
        successfull_components: Dict[str, Dict[str, str]] = {}
        failed_components: Dict[str, Dict[str, Any]] = {}
        validation_components: Dict[str, Dict[str, str]] = {}
        for path_key, meta in plan.items():
            url = meta.get("url")
            commit = meta.get("commit")
            if not url or not commit:
                msg = f"Skipping {path_key}: missing url or commit"
                logging.warning(msg)
                failed_components[path_key] = {"error": msg}
                validation_components[path_key] = {
                    "status": "Failure",
                    "plan_commit": commit or "N/A",
                    "branch_commit": "N/A",
                }
                continue
            token_url = self.tokenize_url(url)
            repo_name = Path(url.rstrip("/")).name
            if repo_name.endswith(".git"):
                repo_name = repo_name[:-4]
            clone_dir = temp_dir / repo_name
            logging.info("Cloning %s -> %s", url, clone_dir)
            try:
                self.exec(["git", "clone", token_url, str(clone_dir)], cwd=Path.cwd())
            except subprocess.CalledProcessError as e:
                err = f"Clone failed for {url}: {e}"
                logging.error(err)
                failed_components[path_key] = {"error": err}
                validation_components[path_key] = {
                    "status": "Failure",
                    "plan_commit": commit,
                    "branch_commit": "N/A",
                }
                continue
            try:
                subprocess.check_call(
                    ["git", "remote", "add", "rocm-github", token_url],
                    cwd=clone_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                pass
            try:
                self.exec(
                    ["git", "checkout", "-b", release_branch, commit], cwd=clone_dir
                )
            except subprocess.CalledProcessError as e:
                err = f"Branch creation failed for {repo_name} at {commit}: {e}"
                logging.error(err)
                failed_components[path_key] = {"error": err}
                validation_components[path_key] = {
                    "status": "Failure",
                    "plan_commit": commit,
                    "branch_commit": "N/A",
                }
                continue
            try:
                if self.dry_run:
                    self.log(
                        f"[DRY RUN] Skipping git push rocm-github {release_branch}"
                    )
                else:
                    self.exec(
                        ["git", "push", "rocm-github", release_branch], cwd=clone_dir
                    )
                logging.info("Pushed %s for %s", release_branch, repo_name)
                successfull_components[path_key] = {
                    "url": url,
                    "commit": commit,
                    "branch": release_branch,
                }
                try:
                    branch_head = subprocess.check_output(
                        ["git", "rev-parse", release_branch], cwd=clone_dir, text=True
                    ).strip()
                    status = "Success" if branch_head.strip() == commit else "Failure"
                    validation_components[path_key] = {
                        "status": status,
                        "plan_commit": commit,
                        "branch_commit": branch_head.strip(),
                    }
                except subprocess.CalledProcessError as e:
                    validation_components[path_key] = {
                        "status": "Failure",
                        "plan_commit": commit,
                        "branch_commit": f"Error: {e}",
                    }
            except subprocess.CalledProcessError as e:
                err = f"Push failed for {repo_name}: {e}"
                logging.error(err)
                failed_components[path_key] = {"error": err}
                validation_components[path_key] = {
                    "status": "Failure",
                    "plan_commit": commit,
                    "branch_commit": "N/A",
                }
        logging.info("Successfull components: %s", successfull_components)
        logging.info("Failed components: %s", failed_components)
        logging.info("Validation components: %s", validation_components)

    def prepare_plan(
        self, projects: Dict[str, str], gitmodule_projects: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, str]]:
        """Compare two dictionaries:
        - projects: submodule dict mapping submodule path -> commit hash
        - gitmodule_projects: dict from .gitmodules with keys (project name) and
        values containing 'path' and 'url'
         Build a new dictionary where keys are matching submodule paths and
        values are dicts with URL and commit.
        """
        # Build quick lookup from path to url based on gitmodule_projects
        path_to_meta = {}
        for _unused_name, meta in gitmodule_projects.items():
            path_val = meta.get("path")
            if path_val:  # ensure path key exists
                path_to_meta[path_val] = meta.get("url")

        matched = {}
        for submodule_path, commit in projects.items():
            if submodule_path in path_to_meta:
                matched[submodule_path] = {
                    "url": path_to_meta[submodule_path],
                    "commit": commit,
                }
        return matched

    def main(self) -> None:
        """
        Main entry point for the script.
        """
        working_dir, projects = self.checkout_source()
        gitmodule_projects = self.determine_projects(working_dir)
        plan = self.prepare_plan(projects, gitmodule_projects)
        try:
            default_branch = self.get_upstream_branch(working_dir)
            try:
                subprocess.check_call(["git", "fetch", "origin"], cwd=working_dir)
            except subprocess.CalledProcessError:
                pass
            root_commit = subprocess.check_output(
                ["git", "rev-parse", f"origin/{default_branch}"],
                cwd=working_dir,
                text=True,
            ).strip()
        except (subprocess.CalledProcessError, RuntimeError):
            root_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=working_dir,
                text=True,
            ).strip()
        root_url = "https://github.com/ROCm/TheRock.git"
        plan["TheRock"] = {"url": root_url, "commit": root_commit}
        self.execute_plan(plan, self.release_branch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scripts for branching automation")
    parser.add_argument(
        "-A",
        "--apitoken",
        help="enter the api key to be used with the upstream repositories",
    )
    parser.add_argument(
        "-B", "--release_branch", required=True, help="Name of branch to create"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run all steps but skip any git push operations.",
    )
    args = parser.parse_args()
    automation = RockBranchingAutomation(args)
    automation.main()
