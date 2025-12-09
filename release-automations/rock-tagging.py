#!/usr/bin/env python3
"""
Release branch automation for ROCm TheRock and its submodules.

End-to-end flow:
1. Fresh clone of TheRock (removes existing ./TheRock).
2. Parse 'git submodule status' -> mapping path -> commit (commit hash only).
3. Parse .gitmodules -> gather path -> url metadata; build plan joining path, url, commit.
4. Augment plan with TheRock itself using tip of its default upstream branch (origin/HEAD fallback to main/master).
5. For every entry in plan (components + TheRock):
     a. Clone into temp dir (token injected only for staging).
     b. Create branch <release_branch> at recorded commit.
     c. Push branch (classifies push failures: existing remote branch vs generic).
6. In original TheRock working clone:
     a. Rewrite workflow YAML triggers replacing 'main' dash/bracket entries with <release_branch>.
     b. Ensure each submodule in .gitmodules has branch=<release_branch>.
     c. checkout -B <release_branch>, stage workflow/.gitmodules, commit if changed.
     d. Push branch (simple failure message; no branch-exists classification).
7. Optional email notification (incomplete; relies on undefined self.environment/self.create_prs).

Arguments:
    -B / --release_branch  REQUIRED name of release branch (e.g. release/therock-7.9).
    -A / --apitoken        Personal access token (used only for staging remote URLs).
    -S / --staging         Use staging host/org (github.amd.com / ROCm-Staging).
    -P / --prod            Use production host/org (github.com / ROCm).
    -M / --mailing_list    Comma-separated emails (notification stub only).

Behavior / Notes:
    - Component branching always runs before workflow modification of TheRock.
    - TheRock is effectively processed twice: once in execute_plan (bare branch at upstream commit) then with workflow/.gitmodules changes.
    - Remote branch existence detected only implicitly during execute_plan push; commit_and_push_release may show generic 'Push failed'.
    - Workflow replacement targets '- main' and bracket list '[main'; assumes simple patterns.
    - Submodule branch keys added/replaced without validating commit compatibility.
    - Email notification not functional until missing attributes defined.
    - No dry-run or retry logic.

Example:
    python rock-tagging-code.py -S -A <token> -B release/therock-7.9 -V 7.9.0 
"""
import argparse
import configparser
import os
import shutil
import requests
import logging
import tarfile
import subprocess
import tempfile
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import re

class rockTagging():

    def __init__(self, args):
        
        self.release_branch = args.branch_name
        self.apitoken = args.apitoken
        self.release_version = args.release_version
        self.create_release = args.create_release
        self.is_staging = bool(args.staging)
        self.is_prod = bool(args.prod)
        if not (args.prod or args.staging):
            print("you should atleast pass --staging(-S) or --prod(-P) ")
            exit(1)

        if args.staging:
            self.gh_hostname = "github.amd.com"
            self.environment = "Staging"
            self.org = "ROCm-Staging"

        if args.prod:
            self.gh_hostname = "github.com"
            self.environment = "Prod"
            self.org = "ROCm"

        if args.mailing_list:
            self.mailing_list = args.mailing_list
        else:
            self.mailing_list = None

        if args.prod and args.staging:
            print("you cannot pass both --staging(-S) and --prod(-P) choose any one")
            exit(1)


    # Helper function to run shell commands via subprocess.run.
    # Return exit_code and output (if capture_output is True).
    def run(self, cmd, cwd=None, capture_output=False, check=True):
        # Support both single command (string) and a sequence of commands (list/tuple).
        # For a list/tuple, execute sequentially, aggregating outputs; return (True, outputs) if all succeed.
        if isinstance(cmd, (list, tuple)):
            agg_outputs = []
            for single_cmd in cmd:
                result = subprocess.run(
                    single_cmd,
                    shell=True,
                    cwd=cwd,
                    capture_output=True,  # always capture for aggregated mode
                    text=True,
                )
                agg_outputs.append(
                    {
                        'cmd': single_cmd,
                        'returncode': result.returncode,
                        'stdout': result.stdout.strip(),
                        'stderr': result.stderr.strip(),
                    }
                )
                if result.returncode != 0:
                    if check:
                        # Fail fast on first error
                        return (False, agg_outputs)
            # All commands succeeded (or check==False and we ignore failures)
            success = all(o['returncode'] == 0 for o in agg_outputs) if check else True
            return (success, agg_outputs)
        # Original single-command behavior
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
        )
        if capture_output:
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)
            return (result.returncode, result.stdout.strip())
        else:
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)
            return (result.returncode, None)
        


    # Check out source code and returns a path to where theRock is checked out, projects dict, and
    def checkout_source(self):
        if self.is_staging and self.apitoken:
            the_rock_url = f"https://{self.apitoken}@{self.gh_hostname}/{self.org}/TheRock.git"
        else:
            the_rock_url = f"https://{self.gh_hostname}/{self.org}/TheRock.git"
        the_rock_dir_path = os.path.join(os.getcwd(), "TheRock")
        # if os.path.isdir(the_rock_dir_path):
        #     shutil.rmtree(the_rock_dir_path)
        # self.run(f"git clone {the_rock_url}")
        cwd= os.getcwd()
        # Move into TheRock directory
        the_rock_dir = os.path.join(cwd, "TheRock")
        if not os.path.isdir(the_rock_dir):
            raise RuntimeError("TheRock directory not found after clone")

        # Gather submodule info
        submodules = self.get_submodules(the_rock_dir)
        return the_rock_dir, submodules

    def get_submodules(self, repo_path):
        """Run 'git submodule status' and parse each line into a dict.
        Format of each line typically:
        <prefix><commit> <path> (description)
        Prefix: ' ' initialized, '-' not initialized, '+' updated.
        Store as: key=path, value={ 'commit': commit, 'prefix': prefix, 'description': description }
        """
        code, output = self.run("git submodule status", cwd=repo_path, capture_output=True, check=False)
        submodule_dict = {}
        if code != 0:
            return submodule_dict
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            # Example line: "d7f3c1e2c3c4abcd1234567890abcdef12345678 libs/rocBLAS (heads/main)"
            # With prefix: "-d7f3c1e ..."
            prefix = ''
            if line[0] in ['-', '+']:
                prefix = line[0]
                line = line[1:]
            parts = line.split()
            if len(parts) < 2:
                continue
            commit = parts[0]
            path = parts[1]
            description = ' '.join(parts[2:]) if len(parts) > 2 else ''
            # Store only the commit hash per user request (omit prefix and description)
            submodule_dict[path] = commit
        return submodule_dict



    # Determine projects to branch based on .gitmodules and user input.
    # This is a helper function for checkout_source.
    # Note that this returns a dict of project names to their key/value pairs from gitmodules.
    def determine_projects(self,  working_dir):
        print(working_dir)
        gitmodules_path = os.path.join(working_dir, ".gitmodules")
        config = configparser.ConfigParser()
        config.read(gitmodules_path)
        projects = {}
        for section in config.sections():
            # Expect section like: submodule "path/to/project"
            if section.startswith('submodule '):
                raw_name = section[len('submodule '):]
                # Remove surrounding quotes if present
                name = raw_name.strip('"')
            else:
                name = section  # fallback
            projects[name] = {}
            for key, val in config.items(section):
                projects[name][key] = val
        print(projects)
        return projects

    def get_upstream_branch(self, repo_path):
        """Return default upstream branch (main/master) with minimal checks."""
        # Try symbolic-ref first
        code, out = self.run(
            "git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'",
            cwd=repo_path,
            capture_output=True,
            check=False,
        )
        if code == 0 and out.strip():
            return out.strip()
        # Simple fallbacks
        for branch in ("main", "master"):
            code, _ = self.run(
                f"git rev-parse --verify origin/{branch}",
                cwd=repo_path,
                capture_output=False,
                check=False,
            )
            if code == 0:
                return branch
        raise RuntimeError("Cannot determine upstream branch")



    def tokenize_url(self, url):
        """Embed token into https URL if token available."""
        if self.is_staging and self.apitoken and url.startswith('https://'):
            # Avoid double insertion
            if f'https://{self.apitoken}@' not in url:
                return url.replace('https://', f'https://{self.apitoken}@')
        return url

    # Execute the plan: clone each repo, create branch at commit, push to remote 'rocm-github'
    def execute_plan(self, plan, release_branch):
        # Normalize input: allow
        # 1) full plan dict {component: {url, commit, path?}}
        # 2) single component dict {url:..., commit:..., path:...}
        # 3) list/tuple of component meta dicts
        if isinstance(plan, dict):
            # Case 1: single component
            if 'url' in plan and 'commit' in plan:
                repo_name = os.path.splitext(os.path.basename(plan['url']))[0]
                work_items = {repo_name: plan}
            # Case 2: multiple components (normal plan)
            else:
                work_items = plan
        elif isinstance(plan, (list, tuple)):
            # Case 3: list of components
            work_items = {os.path.splitext(os.path.basename(meta['url']))[0]: meta for meta in plan}
        else:
            raise TypeError(f"Unsupported plan type: {type(plan)}")

        temp_dir = tempfile.mkdtemp(prefix="rock-tagging-")
        print(f"Temporary working directory: {temp_dir}")
        successful_components = dict()
        failed_components = dict()
        tag_Exists_components = dict()
        for path_key, meta in work_items.items():
            url = meta.get('url')
            commit = meta.get('commit')
            if not url or not commit:
                msg = f"Skipping {path_key}: missing url or commit"
                print(msg)
                failed_components[path_key] = {"error": msg}
                continue
            token_url = self.tokenize_url(url)
            repo_name = os.path.splitext(os.path.basename(url))[0]
            clone_dir = os.path.join(temp_dir, repo_name)
            print(f"Cloning {url} -> {clone_dir}")
            try:
                self.run(f"git clone {token_url} {clone_dir}")
            except subprocess.CalledProcessError as e:
                err = f"Clone failed for {url}: {e}"
                print(err)
                failed_components[path_key] = {"error": err}
                continue
            
            if self.environment == "Staging":
                print("printing inside staging")
                upstream_git_url = f"https://{self.gh_hostname}/{self.org}/{path_key}"
                url = f"https://github.amd.com/api/v3/repos/rocm-staging/{path_key}/tags"
            if self.environment == "Prod":
                upstream_git_url = f"https://{self.gh_hostname}/{self.org}/{path_key}"
                url = f"https://api.github.com/repos/ROCm/{path_key}/tags"
            headers = {
                'Authorization': f'token {self.apitoken}'
            }
            print(url)
            response = requests.get(url, headers=headers)
            tags=response.json()
            # print(tags)
            filtered_tags = [tag['name'] for tag in tags if tag['name'].startswith('therock-') or tag['name'].startswith('test')]
            # print(filtered_tags)
            boolean =False
            upstream_git_url = re.sub('(http[s]?://)', '\\1{}@'.format(self.apitoken), upstream_git_url)
            prepare_steps = [
                "git rev-parse HEAD",
                "git remote remove $(git remote)",
                f"git remote add rocm-github {upstream_git_url}",
                "git remote -v",
                f"git fetch rocm-github {commit}",
            ]
            success, outputs = self.run(prepare_steps, cwd=clone_dir, check=True)
            if not success:
                failed_components[path_key] = ["Failure", "Prepare steps failed"]
                continue
            # Optional: create annotated tag when release requested
            tag_name = f"therock-{self.release_version}" if self.release_version else None
            if tag_name and tag_name not in filtered_tags:
                tag_cmds = [f"git tag -a {tag_name} {commit} -m 'therock release v{tag_name}'"]
                tag_success, tag_outputs = self.run(tag_cmds, cwd=clone_dir, check=False)
                if not tag_success:
                    failed_components[path_key] = ["Failure", "Tag creation failed"]
                else:
                    push_tag_cmds = [f"git push rocm-github {tag_name}:refs/tags/{tag_name}"]
                    push_tag_success, _ = self.run(push_tag_cmds, cwd=clone_dir, check=False)
                    if not push_tag_success:
                        failed_components[path_key] = ["Failure", "Tag push failed"]
                    else:
                        successful_components[path_key] = ["Success"]
            else:
                boolean = True
                tag_Exists_components[path_key] = ["Failure"]
                failed_components[path_key] = ["Failure","Tag Already exists"]
            branching_step = [ f"git push rocm-github {commit}:refs/heads/{self.release_branch}" ]
            returnStatus, _ =  self.run(branching_step, cwd=clone_dir, check=False)
            print("Branching step is executed and the exit status is: ", returnStatus)
                
            if not returnStatus:
                failed_components[path_key] = ["Failure", "Release Branch update failed"]
                # continue
            
            # Tarball creation: each first-level directory under projects/ becomes <name>.tar.gz
            tarball_paths = []
            if path_key in ("rocm-libraries", "rocm-systems"):
                projects_dir = os.path.join(clone_dir, "projects")
                if os.path.isdir(projects_dir):
                    print(f"Creating tarballs for projects in '{path_key}' under: {projects_dir}")
                    for project_name in sorted(os.listdir(projects_dir)):
                        if project_name.startswith('.'):
                            continue  # skip hidden/system dirs
                        proj_path = os.path.join(projects_dir, project_name)
                        if not os.path.isdir(proj_path):
                            continue
                        tarball_name = f"{project_name}.tar.gz"
                        tarball_path = os.path.join(clone_dir, tarball_name)
                        try:
                            with tarfile.open(tarball_path, "w:gz") as tf:
                                tf.add(proj_path, arcname=project_name)
                            tarball_paths.append(tarball_path)
                            print(f"✅ Tarball created: {tarball_path}")
                        except Exception as e:
                            print(f"⚠️ Failed to create tarball for {project_name}: {e}")
                else:
                    print(f"⚠️ 'projects' directory not found for {path_key}: {projects_dir}")
            
            #Authenticate and create a release tag
            if self.create_release and boolean == False:
                # Build the tarball assets argument for gh release create
                tarball_str = " ".join(tarball_paths) if tarball_paths else ""
                rel_tagging_step  = [ "gh --version",
                                    "echo '{}' | gh auth login --hostname {} --with-token".format(self.apitoken, self.gh_hostname),
                                    f"gh release create therock-{self.release_version} --notes 'therock release v{self.release_version}' {tarball_str}" ]
                returnStatus, ouptut =  self.run(rel_tagging_step,  cwd=clone_dir, check=False)
                print("Release tagging step is executed and the exit status is: ", returnStatus)
                if not returnStatus:
                    failed_components[path_key] = ["Failure", "Release tagging failed"]

        print("Successful components:", successful_components)
        print("Failed components:", failed_components)
        return successful_components, failed_components

    def highlightCells(self, column):
        color = 'red'
        color1 = 'green'

        return [f'background-color: {color}' if cellvalue == 'Failure' or cellvalue == 'repo does not exist' else (f'background-color: {color1}' if cellvalue == 'Success' or cellvalue == 'admin' else '') for cellvalue in column ]

    def send_notifications(self, **kwargs):
        MAILING_LIST = kwargs['mailing_list']
        if kwargs["successfull_components"] is not None or kwargs["failed_components"] is not None:
                successfull_components = kwargs["successfull_components"]
                failed_components = kwargs["failed_components"]
                branch_validation = kwargs.get("branch_validation")
                tag_validation = kwargs.get("tag_validation")
                # pr_dict = kwargs["pr_dict"]
                print(successfull_components, failed_components)
                successfull_table = None
                failed_table = None
                html_tables = ""
                html_body = ""
                styles =   [
                    {'selector': 'th',
                    'props': [('background', 'lightblue'),

                                ('color', 'black'),
                                ('text-align', 'left') ]},
                    {'selector': 'th, td',
                    'props': [('padding', '5px'),

                                ('color', 'black'),
                                ('text-align', 'left') ]},
                    {'selector': 'td',
                        'props': [('color', 'black'),
                                ('text-align', 'left') ]},
                    {'selector': 'tr td:last-child',
                    'props': [('white-space', 'nowrap')]},

                    {'selector': 'tr:hover',
                    'props': [('background-color', '#bfeaf9')]}
                    ]

                if bool(successfull_components):
                    successfull_table = pd.DataFrame.from_dict(successfull_components, orient = 'index').reset_index()
                    successfull_table.columns = ["Component Name", "Status"]
                    successfull_table = successfull_table.style.apply(self.highlightCells, subset=['Status'], axis=0).set_table_attributes('border="1" class="dataframe table table-hover table-bordered"').set_table_styles(styles).hide()
                    print(successfull_table)
                    successfull_table = successfull_table.to_html()
                    print("printing successful table")
                    print(successfull_table)
                    table_title = "Success Component Table: "
                    html_tables = html_tables+"<br><p><b>{}</b></p><p>{}</p>".format(table_title, successfull_table)

                if bool(failed_components):
                    failed_table = pd.DataFrame.from_dict(failed_components, orient = 'index').reset_index()
                    failed_table.columns = ["Component Name", "Status", "Reason for failure"]
                    failed_table = failed_table.style.apply(self.highlightCells, subset=['Status'], axis=0).set_table_attributes('border="1" class="dataframe table table-hover table-bordered"').set_table_styles(styles).hide()
                    print(failed_table)
                    failed_table = failed_table.to_html()
                    print("printing failed_table")
                    print(failed_table)
                    table_title = "Failed Component Table: "
                    html_tables = html_tables+"<br><p><b>{}</b></p><p>{}</p>".format(table_title, failed_table)

                if branch_validation:
                    branch_table = pd.DataFrame.from_dict(branch_validation, orient='index').reset_index()
                    branch_table.columns = ["Component Name", "Status", "Branch Commit", "Plan Commit", "Info"]
                    branch_table = branch_table.style.apply(self.highlightCells, subset=['Status'], axis=0).set_table_attributes('border="1" class="dataframe table table-hover table-bordered"').set_table_styles(styles).hide()
                    branch_table_html = branch_table.to_html()
                    table_title = "Branch Validation Table:"
                    html_tables += f"<br><p><b>{table_title}</b></p><p>{branch_table_html}</p>"

                if tag_validation:
                    tag_table = pd.DataFrame.from_dict(tag_validation, orient='index').reset_index()
                    tag_table.columns = ["Component Name", "Status", "Tag Commit", "Plan Commit", "Tag URL"]
                    tag_table = tag_table.style.apply(self.highlightCells, subset=['Status'], axis=0).set_table_attributes('border="1" class="dataframe table table-hover table-bordered"').set_table_styles(styles).hide()
                    tag_table_html = tag_table.to_html()
                    table_title = "Tag Validation Table:"
                    html_tables += f"<br><p><b>{table_title}</b></p><p>{tag_table_html}</p>"



                html_body = html_body+html_tables

            #Setting the header message

                header_msg = "Below is the tagging/branching status of various rock components: "

                html="""<!DOCTYPE html><html><head></head><h4>{}</h4>{}</html>""".format(header_msg, html_body)

        sender_email = "jenkins-compute@amd.com"
        recipients = [email.strip() for email in MAILING_LIST.split(",") if email.strip()]
        msg =  MIMEMultipart('alternative')
        msg['Subject'] = f"Tagging/Branching Status Mail - {self.environment}"
        msg['From'] = sender_email
        msg['To'] = ", ".join(recipients)
        part1 = MIMEText(html, "html")
        msg.attach(part1)
        s = smtplib.SMTP('torsmtp10.amd.com')
        s.send_message(msg)
        s.quit()

    def prepare_plan(self, projects, gitmodule_projects):
        """Compare two dictionaries:
        - projects: submodule dict mapping submodule path -> commit hash
        - gitmodule_projects: dict from .gitmodules with keys (project name) and values containing 'path' and 'url'
    Build a new dictionary where keys are matching submodule paths and values are dicts with URL and commit.
        """
        # Build lookup from path -> (project_key, url)
        path_lookup = {}
        for project_key, meta in gitmodule_projects.items():
            p = meta.get('path')
            u = meta.get('url')
            if p and u:
                path_lookup[p] = (project_key, u)

        plan = {}
        for submodule_path, commit in projects.items():
            if submodule_path in path_lookup:
                project_key, url = path_lookup[submodule_path]
                # Use the project key (gitmodule section-derived name) as requested
                plan[project_key] = {
                    'url': url,
                    'commit': commit,
                }
        print("Plan mapping (project_key -> data):")
        print(plan)
        return plan

    def validate_branch_and_tag(self, plan, release_branch):
        """Return two separate dictionaries:
        validated_branches[component] = [Status, branch_commit, plan_commit, '']
        validated_tags[component]     = [Status, tag_commit,   plan_commit, tag_ref_url]
        Status is Success if commit matches plan commit, else Failure.
        Tag ref URL is constructed from repo URL for context when tag exists.
        """
        if isinstance(plan, dict):
            if 'url' in plan and 'commit' in plan:  # single component
                plan_items = { os.path.splitext(os.path.basename(plan['url']))[0]: plan }
            else:
                plan_items = plan
        elif isinstance(plan, (list, tuple)):
            plan_items = { os.path.splitext(os.path.basename(meta['url']))[0]: meta for meta in plan }
        else:
            raise TypeError(f"Unsupported plan type for validation: {type(plan)}")
        validated_branches = {}
        validated_tags = {}
        tag_name = f"therock-{self.release_version}" if (self.create_release and self.release_version) else None
        for comp, meta in plan_items.items():
            url = meta.get('url'); plan_commit = meta.get('commit')
            if not url or not plan_commit:
                validated_branches[comp] = ["Failure", 'MISSING', plan_commit or 'MISSING', '']
                validated_tags[comp] = ["Skipped", 'SKIPPED', plan_commit or 'MISSING', '']
                continue
            plain_url = url  # preserve original without token for reporting
            token_url = self.tokenize_url(url)  # used only for ls-remote queries
            branch_ref = f"refs/heads/{release_branch}"
            code_b, out_b = self.run(f"git ls-remote {token_url} {branch_ref}", capture_output=True, check=False)
            branch_commit = out_b.split()[0] if (code_b == 0 and out_b) else 'MISSING'
            branch_status = 'Success' if branch_commit == plan_commit else 'Failure'
            validated_branches[comp] = [branch_status, branch_commit, plan_commit, '']
            # Tag validation
            if tag_name:
                tag_ref = f"refs/tags/{tag_name}"+"^{}"
                code_t, out_t = self.run(f"git ls-remote {token_url} {tag_ref}", capture_output=True, check=False)
                tag_commit = out_t.split()[0] if (code_t == 0 and out_t) else 'MISSING'
                if tag_commit == 'MISSING':
                    validated_tags[comp] = ["Skipped", tag_commit, plan_commit, f"{plain_url}"]
                else:
                    tag_status = 'Success' if tag_commit == plan_commit else 'Failure'
                # Provide a browsable tag URL (tree view) without token leakage
                tag_url = f'https://{self.gh_hostname}/{self.org}/{comp}/releases/tag/{tag_name}'
                validated_tags[comp] = [tag_status, tag_commit, plan_commit, tag_url]
            else:
                validated_tags[comp] = ["Skipped", 'SKIPPED', plan_commit, '']
        return validated_branches, validated_tags
    
    def main(self):
        working_dir, projects = self.checkout_source()
        gitmodule_projects = self.determine_projects(working_dir)
        plan = self.prepare_plan(projects,gitmodule_projects)
        # Add TheRock itself using tip of its default/upstream branch
        try:
            default_branch = self.get_upstream_branch(working_dir)
            # Ensure we have latest
            self.run("git fetch origin", cwd=working_dir, check=False)
            _, root_commit = self.run(f"git rev-parse origin/{default_branch}", cwd=working_dir, capture_output=True, check=True)
        except Exception:
            # Fallback to current HEAD if upstream parsing fails
            _, root_commit = self.run("git rev-parse HEAD", cwd=working_dir, capture_output=True, check=True)
        if self.is_staging and self.apitoken:
            root_url = f"https://{self.apitoken}@{self.gh_hostname}/{self.org}/TheRock.git"
        else:
            root_url = f"https://{self.gh_hostname}/{self.org}/TheRock.git"
        plan['TheRock'] = { 'url': root_url, 'commit': root_commit }
        print(plan)
        successfull_components, failed_components = self.execute_plan(plan['rocm-systems'], self.release_branch)
        branch_validation, tag_validation = self.validate_branch_and_tag(plan['rocm-systems'], self.release_branch)
        if self.mailing_list:
            logging.info(f"Tagging completed. Notifications would be sent to: {self.mailing_list}")
            self.send_notifications(successfull_components = successfull_components, failed_components = failed_components, branch_validation=branch_validation, tag_validation=tag_validation, mailing_list = self.mailing_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for branching automation')
    parser.add_argument('-B', '--branch_name',
                        help='enter the release branch name')
    parser.add_argument('-V', '--release_version',
                        help='enter the rocm release version')                        
    parser.add_argument('-A', '--apitoken',
                        help='enter the api key to be used with the upstream repositories',
                        required='True')
    parser.add_argument('-S', '--staging', action='store_true',
                        help='Pass this option to select staging environment',
                        )
    parser.add_argument('-P', '--prod', action='store_true',
                        help='Pass this option to select Prod environment',
                        )
    parser.add_argument('-M', '--mailing_list',
                        type=str,
                        help='Pass this option to provide comma separated list of mail-ids for notification',
                        )
    parser.add_argument(
        '-CR',"--create_release",action='store_true', help="create only release"
    )
    args = parser.parse_args()
    
    TagObject = rockTagging(args)
    TagObject.main()
