import argparse
import shlex
import shutil
import subprocess
import sys
import os
import glob
import platform
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse, urlunparse, quote

import subprocess

TAG_UPSTREAM_DIFFBASE = "THEROCK_UPSTREAM_DIFFBASE"
TAG_HIPIFY_DIFFBASE = "THEROCK_HIPIFY_DIFFBASE"
HIPIFY_COMMIT_MESSAGE = "DO NOT SUBMIT: HIPIFY"

class RockProjectRepo():
    def __init__(self,
                project_name,
                project_root_dir,
                project_src_dir,
                project_build_dir,
                project_repo_url,
                project_version_hashtag):
        self.project_name = project_name
        self.project_src_dir = Path(project_src_dir)
        self.project_build_dir = Path(project_build_dir)
        self.project_repo_url = project_repo_url
        self.project_version_hashtag = project_version_hashtag
        self.project_patch_dir_base = Path(project_root_dir) / "patches" / project_name
        os.environ["ROCK_BUILDER_APP_SRC_DIR"] = str(project_src_dir)

    # private methods
    def __exec_subprocess_cmd(self, exec_cmd):
        ret = True
        if exec_cmd is not None:
            print("exec_cmd")
            # capture_output=True --> can print output after process exist, not possible to see the output during the build time
            # capture_output=False --> can print output only during build time
            #result = subprocess.run(exec_cmd, shell=True, capture_output=True, text=True)
            result = subprocess.run(exec_cmd, shell=True, capture_output=False, text=True)
            if result.returncode == 0:
                print(result.stdout)
                ret = False
            else:
                print(result.stdout)
                print(f"Error: {result.stderr}")
        return ret

    def __exec_subprocess_batch_file(self, batch_file):
        ret = True
        if batch_file is not None:
            print("batch_file: " + batch_file)
            # capture_output=True --> can print output after process exist, not possible to see the output during the build time
            # capture_output=False --> can print output only during build time
            #result = subprocess.run(exec_cmd, shell=True, capture_output=True, text=True)
            result = subprocess.run([batch_file], shell=True, capture_output=False, text=True)
            if result.returncode == 0:
                print(result.stdout)
                ret = False
            else:
                print(result.stdout)
                print(f"Error: {result.stderr}")
        return ret

    def __replace_env_variables(self, cmd_str):
        ret = os.path.expandvars(cmd_str)
        #print(ret)
        return ret

    def __get_latest_file(self, path, extension=None):
        ret = None;
        if path is not None:
            search_key = path.rstrip()
            if not ((search_key.endswith('\\')) or
                    (search_key.endswith('/')) or
                    (search_key.endswith('*'))):
                search_key = search_key + "/"
            if extension is not None:
                search_key = search_key + extension
            list_of_files = glob.glob(search_key)
            if (len(list_of_files) > 0):
                ret = max(list_of_files, key=os.path.getctime)
        print(ret)
        return ret

    #public methods
    def exec(self,
            args: list[str | Path],
            cwd: Path,
            *,
            stdout_devnull: bool = False):
        args = [str(arg) for arg in args]
        print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
        subprocess.check_call(
            args,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL if stdout_devnull else None,
        )


    def rev_parse(repo_path: Path, rev: str) -> str | None:
        """Parses a revision to a commit hash, returning None if not found."""
        try:
            raw_output = subprocess.check_output(
                ["git", "rev-parse", rev], cwd=str(repo_path), stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            return None
        return raw_output.decode().strip()


    def rev_list(repo_path: Path, revlist: str) -> list[str]:
        raw_output = subprocess.check_output(
            ["git", "rev-list", revlist], cwd=str(repo_path)
        )
        return raw_output.decode().splitlines()


    def list_submodules(self,
                        repo_path: Path,
                        *,
                        relative: bool = False,
                        recursive: bool = True
    ) -> list[Path]:
        """Gets paths of all submodules (recursively) in the repository."""
        recursive_args = ["--recursive"] if recursive else []
        raw_output = subprocess.check_output(
            ["git", "submodule", "status"] + recursive_args,
            cwd=str(repo_path),
        )
        lines = raw_output.decode().splitlines()
        relative_paths = [PurePosixPath(line.strip().split()[1]) for line in lines]
        if relative:
            return relative_paths
        return [repo_path / p for p in relative_paths]


    def list_status(self,
                    repo_path: Path) -> list[tuple[str, str]]:
        """Gets the status as a list of (status_type, relative_path)."""
        raw_output = subprocess.check_output(
            ["git", "status", "--porcelain", "-u", "--ignore-submodules"],
            cwd=str(repo_path),
        )
        lines = raw_output.decode().splitlines()
        return [tuple(line.strip().split()) for line in lines]


    def get_all_repositories(self,
                             root_path: Path) -> list[Path]:
        """Gets all repository paths, starting with the root and then including all
        recursive submodules."""
        all_paths = self.list_submodules(root_path)
        all_paths.insert(0, root_path)
        return all_paths


    def git_config_ignore_submodules(self, repo_path: Path):
        """Sets the `submodule.<name>.ignore = true` git config option for all submodules.
        This causes all submodules to not show up in status or diff reports, which is
        appropriate for our case, since we make arbitrary changes and patches to them.
        Note that pytorch seems to somewhat arbitrarily have some already set this way.
        We just set them all.
        """
        file_path = repo_path / ".gitmodules"
        if file_path.exists():
            try:
                config_names = (
                    subprocess.check_output(
                        [
                            "git",
                            "config",
                            "--file",
                            ".gitmodules",
                            "--name-only",
                            "--get-regexp",
                            "\\.path$",
                        ],
                        cwd=str(repo_path),
                    )
                    .decode()
                    .splitlines()
                )
                for config_name in config_names:
                    ignore_name = config_name.removesuffix(".path") + ".ignore"
                    self.exec(["git", "config", ignore_name, "all"], cwd=repo_path)
                submodule_paths = self.list_submodules(repo_path, relative=True, recursive=False)
                self.exec(["git", "update-index", "--skip-worktree"] + submodule_paths, cwd=repo_path)
            except Exception as e:
                # pytorch audio has empty .gitmodules file which can cause exception
                pass


    def save_repo_patches(self,
                          repo_path: Path,
                          patches_path: Path):
        """Updates the patches directory with any patches committed to the repository."""
        if patches_path.exists():
            shutil.rmtree(patches_path)
        # Get key revisions.
        upstream_rev = rev_parse(repo_path, TAG_UPSTREAM_DIFFBASE)
        hipify_rev = rev_parse(repo_path, TAG_HIPIFY_DIFFBASE)
        if upstream_rev is None:
            print(f"error: Could not find upstream diffbase tag {TAG_UPSTREAM_DIFFBASE}")
            sys.exit(1)
        hipified_count = 0
        if hipify_rev:
            hipified_revlist = f"{hipify_rev}..HEAD"
            base_revlist = f"{upstream_rev}..{hipify_rev}^"
            hipified_count = len(rev_list(repo_path, hipified_revlist))
        else:
            hipified_revlist = None
            base_revlist = f"{upstream_rev}..HEAD"
        base_count = len(rev_list(repo_path, base_revlist))
        if hipified_count == 0 and base_count == 0:
            return
        print(
            f"Saving {patches_path} patches: {base_count} base, {hipified_count} hipified"
        )
        if base_count > 0:
            base_path = patches_path / "base"
            base_path.mkdir(parents=True, exist_ok=True)
            self.exec(["git", "format-patch", "-o", base_path, base_revlist], cwd=repo_path)
        if hipified_count > 0:
            hipified_path = patches_path / "hipified"
            hipified_path.mkdir(parents=True, exist_ok=True)
            self.exec(["git", "format-patch", "-o",
                      hipified_path, hipified_revlist], cwd=repo_path)

    def apply_repo_patches(self,
                           repo_path: Path,
                           patches_path: Path):
        """Applies patches to a repository from the given patches directory."""
        patch_files = list(patches_path.glob("*.patch"))
        print("repo_path: " + str(repo_path) + ", patches_path: " + str(patches_path))
        if not patch_files:
            return
        patch_files.sort(key=lambda p: p.name)
        self.exec(
            [
                "git",
                "am",
                "--whitespace=nowarn",
                "--committer-date-is-author-date"
            ] + patch_files,
            cwd=repo_path)

    def apply_all_patches(self,
                          root_repo_path: Path,
                          patches_path: Path,
                          repo_name: str,
                          patchset_name: str):
        relative_sm_paths = self.list_submodules(root_repo_path, relative=True)
        # Apply base patches.
        self.apply_repo_patches(root_repo_path, patches_path / repo_name / patchset_name)
        for relative_sm_path in relative_sm_paths:
            self.apply_repo_patches(
                root_repo_path / relative_sm_path,
                patches_path / relative_sm_path / patchset_name,
            )


    # repo_hashtag_to_patches_dir_name('2.7.0-rc9') -> '2.7.0'
    def repo_hashtag_to_patches_dir_name(self,
                                         version_ref: str) -> str:
        pos = version_ref.find("-")
        if pos != -1:
            return version_ref[:pos]
        return version_ref

    def do_checkout(self,
					repo_fetch_depth=1,
                    repo_fetch_job_cnt=1,
                    apply_patches_enabled=1,
                    hipify_enabled=1,
                    repo_remote_name="origin"):
        dot_git_subdir = self.project_src_dir / ".git"
        if dot_git_subdir.exists():
            #print(f"Not cloning repository ({dot_git_subdir} exists)")
            pass
        else:
            print(f"Cloning repository at {self.project_version_hashtag}")
            self.project_src_dir.mkdir(parents=True, exist_ok=True)
            self.exec(["git", "init", "--initial-branch=main"], cwd=self.project_src_dir)
            self.exec(["git", "config", "advice.detachedHead", "false"], cwd=self.project_src_dir)
            self.exec(["git", "remote", "add", "origin", self.project_repo_url], cwd=self.project_src_dir)

        # fetch and checkout
        fetch_args = []
        if repo_fetch_depth:
            fetch_args.extend(["--depth", str(repo_fetch_depth)])
        if repo_fetch_job_cnt:
            fetch_args.extend(["-j", str(repo_fetch_job_cnt)])
        self.exec(["git", "fetch"] + fetch_args + ["origin", self.project_version_hashtag], cwd=self.project_src_dir)
        self.exec(["git", "checkout", "FETCH_HEAD"], cwd=self.project_src_dir)
        self.exec(["git", "tag", "-f", TAG_UPSTREAM_DIFFBASE], cwd=self.project_src_dir)
        try:
            self.exec(["git", "submodule", "update", "--init", "--recursive"] + fetch_args, cwd=self.project_src_dir)
        except subprocess.CalledProcessError:
            print("Failed to fetch git submodules")
            sys.exit(1)
        self.exec(
            [
                "git",
                "submodule",
                "foreach",
                "--recursive",
                f"git tag -f {TAG_UPSTREAM_DIFFBASE}",
            ],
            cwd=self.project_src_dir,
            stdout_devnull=True)
        self.git_config_ignore_submodules(self.project_src_dir)

        # base patches
        if apply_patches_enabled:
            self.apply_all_patches(self.project_src_dir,
                self.project_patch_dir_base / self.repo_hashtag_to_patches_dir_name(self.project_version_hashtag),
                self.project_name,
                "base",
            )

        # TODO: do_hipify execution and do_hipify patch execution after that needs to be handled via config
        # files in future (do_hipify could be an operation done on config phase but then applying patches again
        # does not suit well for traditional workflow. One solution would be to execute two specify in project config file
        # two different actions:
        # - execution of random code from the project (build_amd.py in pytorch)
        # - apply_patches after that requesting to pick patches from the hipify dir
        if hipify_enabled:
            self.do_hipify()

        # hipified patches
        if apply_patches_enabled:
            self.apply_all_patches(self.project_src_dir,
                self.project_patch_dir_base / self.repo_hashtag_to_patches_dir_name(self.project_version_hashtag),
                self.project_name,
                "hipified",
            )

    #TODO: This needs to be refactored to be configurable via project specific cfg files in rockbuilder
    def do_hipify(self):
        repo_dir: Path = self.project_src_dir
        print(f"Hipifying {repo_dir}")
        build_amd_path = repo_dir / "tools" / "amd_build" / "build_amd.py"
        if build_amd_path.exists():
            self.exec([sys.executable, build_amd_path], cwd=repo_dir)
        # Iterate over the base repository and all submodules. Because we process
        # the root repo first, it will not add submodule changes.
        all_paths = self.get_all_repositories(repo_dir)
        for module_path in all_paths:
            status = self.list_status(module_path)
            if not status:
                continue
            print(f"HIPIFY made changes to {module_path}: Committing")
            self.exec(["git", "add", "-A"], cwd=module_path)
            self.exec(["git", "commit", "-m", HIPIFY_COMMIT_MESSAGE], cwd=module_path)
            self.exec(["git", "tag", "-f", TAG_HIPIFY_DIFFBASE], cwd=module_path)

    def do_clean(self, clean_cmd):
        return self.__exec_subprocess_cmd(clean_cmd)

    def do_configure(self, configure_cmd):
        return self.__exec_subprocess_cmd(configure_cmd)

    def do_build(self, build_cmd):
        is_dos = any(platform.win32_ver())
        if is_dos:
            os.makedirs(self.project_build_dir, exist_ok=True)
            build_cmd_file = os.path.join(self.project_build_dir, "build_cmd.bat")
            with open(build_cmd_file, "w") as file:
                file.write(build_cmd)
            ret = self.__exec_subprocess_batch_file(str(build_cmd_file))
        else:
            ret = self.__exec_subprocess_cmd(build_cmd)
        return ret

    def do_install(self, install_cmd):
        ret = True
        if install_cmd is not None:
            if "ROCK_CONFIG_CMD__FIND_AND_INSTALL_LATEST_PYTHON_WHEEL" in install_cmd:
                install_cmd_arr = install_cmd.split()
                if len(install_cmd_arr) == 2:
                    wheel_search_path = install_cmd_arr[1]
                    wheel_search_path = self.__replace_env_variables(wheel_search_path)
                    print("wheel_search_path: " + wheel_search_path)
                    latest_whl = self.__get_latest_file(wheel_search_path, "*.whl")
                    print("latest_whl: " + latest_whl)
                    os.environ['PIP_BREAK_SYSTEM_PACKAGES'] = '1'
                    #res = subprocess.call([ "pip", "install", latest_whl])
                    inst_cmd = "pip uninstall -y " + latest_whl
                    res = subprocess.run(inst_cmd, shell=True, capture_output=False, text=True)
                    inst_cmd = "pip install " + latest_whl
                    res = subprocess.run(inst_cmd, shell=True, capture_output=False, text=True)
                    if res.returncode == 0:
                        print(res.stdout)
                    else:
                        ret = False
                        print(res.stdout)
                        print(f"Error: {result.stderr}")
            else:
                ret = self.__exec_subprocess_cmd(build_cmd)
        return ret

    def do_save_patches(self):
        patches_dir = self.project_patch_dir_base / self.repo_hashtag_to_patches_dir_name(self.project_version_hashtag)
        self.save_repo_patches(self.project_src_dir, patches_dir / self.project_name)
        relative_sm_paths = self.list_submodules(self.project_src_dir, relative=True)
        for relative_sm_path in relative_sm_paths:
            self.save_repo_patches(self.project_src_dir / relative_sm_path, patches_dir / relative_sm_path)
