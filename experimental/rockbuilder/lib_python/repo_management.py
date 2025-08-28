import argparse
import shlex
import shutil
import subprocess
import sys
import os
import glob
import platform
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse, urlunparse, quote

import subprocess

TAG_UPSTREAM_DIFFBASE = "THEROCK_UPSTREAM_DIFFBASE"
TAG_HIPIFY_DIFFBASE = "THEROCK_HIPIFY_DIFFBASE"
HIPIFY_COMMIT_MESSAGE = "DO NOT SUBMIT: HIPIFY"


class RockProjectRepo:
    def __init__(
        self,
        wheel_install_dir,
        project_name,
        project_root_dir: Path,
        project_src_dir: Path,
        project_build_dir: Path,
        project_exec_dir: Path,
        project_repo_url,
        project_version_hashtag,
        patch_dir_root_arr: Path,
    ):
        self.wheel_install_dir = wheel_install_dir
        self.project_name = project_name
        self.project_src_dir = Path(project_src_dir)
        self.project_build_dir = Path(project_build_dir)
        self.project_exec_dir = Path(project_exec_dir)
        self.project_repo_url = project_repo_url
        self.project_version_hashtag = project_version_hashtag
        self.patch_dir_root_arr = patch_dir_root_arr
        self.orig_env_variables_hashtable = dict()
        os.environ["ROCK_BUILDER_APP_SRC_DIR"] = project_src_dir.as_posix()
        os.environ["ROCK_BUILDER_APP_BUILD_DIR"] = project_build_dir.as_posix()

    # private methods
    def _exec_subprocess_cmd(self, exec_cmd, exec_dir):
        ret = True
        if exec_cmd is not None:
            exec_dir = self._replace_env_variables(exec_dir)
            print("exec_cmd: " + exec_cmd)
            # capture_output=True --> can print output after process exist, not possible to see the output during the build time
            # capture_output=False --> can print output only during build time
            # result = subprocess.run(exec_cmd, shell=True, capture_output=True, text=True)
            result = subprocess.run(
                exec_cmd, cwd=exec_dir, shell=True, capture_output=False, text=True
            )
            if result.returncode == 0:
                if result.stdout:
                    print(result.stdout)
            else:
                ret = False
                print(result.stdout)
                print(f"Error: {result.stderr}")
        return ret

    def _exec_subprocess_batch_file(self, batch_file):
        ret = True
        if batch_file is not None:
            print("batch_file: " + batch_file)
            # capture_output=True --> can print output after process exist, not possible to see the output during the build time
            # capture_output=False --> can print output only during build time
            # result = subprocess.run(exec_cmd, shell=True, capture_output=True, text=True)
            result = subprocess.run(
                [batch_file], shell=True, capture_output=False, text=True
            )
            if result.returncode == 0:
                print(result.stdout)
                ret = False
            else:
                print(result.stdout)
                print(f"Error: {result.stderr}")
        return ret

    def _replace_env_variables(self, cmd_str):
        ret = os.path.expandvars(cmd_str)
        # print("orig: " + cmd_str)
        # print("new: " + ret)
        return ret

    def _get_latest_file(self, path, extension=None):
        ret = None
        if path is not None:
            search_key = path.rstrip()
            if not (
                (search_key.endswith("\\"))
                or (search_key.endswith("/"))
                or (search_key.endswith("*"))
            ):
                search_key = search_key + "/"
            if extension is not None:
                search_key = search_key + extension
            list_of_files = glob.glob(search_key)
            if len(list_of_files) > 0:
                ret = max(list_of_files, key=os.path.getctime)
        print(ret)
        return ret

    # 1) search the latest wheel file from certain directory
    # 2) copy wheel to packages/wheel directory
    # 3) install wheel to current python environment
    def _handle_FIND_AND_HANDLE_LATEST_PYTHON_WHEEL_CMD(self, install_cmd):
        ret = True
        install_cmd_arr = install_cmd.split()
        print("len(install_cmd_arr): " + str(len(install_cmd_arr)))
        if len(install_cmd_arr) == 2:
            wheel_search_path = install_cmd_arr[1]
            wheel_search_path = self._replace_env_variables(wheel_search_path)
            print("wheel_search_path: " + wheel_search_path)
            # 1) search the wheel
            latest_whl = self._get_latest_file(wheel_search_path, "*.whl")
            if latest_whl:
                # shutil.copy will throw exception in error cases
                try:
                    print("latest_whl: " + latest_whl)
                    # 2) copy wheel
                    ret = self.wheel_install_dir.is_dir()
                    if not ret:
                        self.wheel_install_dir.mkdir(parents=True, exist_ok=True)
                    ret = self.wheel_install_dir.is_dir()
                    if ret:
                        shutil.copy2(latest_whl, self.wheel_install_dir)
                    # 3) install wheel
                    os.environ["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
                    # res = subprocess.call([ "pip", "install", latest_whl])
                    inst_cmd = "pip uninstall -y " + latest_whl
                    # we do not check the uninstall fails by purpose because the
                    # reason for failure is most likely that the previous version of wheel
                    # is not installed. But in cases that we do multiple builds for same
                    # wheel version with little changes, we need to do the uninstall first
                    # before we do the install for the package with same wheel version.
                    self._exec_subprocess_cmd(inst_cmd, self.project_exec_dir)
                    inst_cmd = "pip install " + latest_whl
                    ret = self._exec_subprocess_cmd(inst_cmd, self.project_exec_dir)
                    if not ret:
                        print("Install failed for " + self.project_name)
                        print("Failed command: " + install_cmd)
                        ret = False
                except:
                    print(
                        "Python wheel copy or install failed for project: "
                        + self.project_name
                    )
                    ret = False
            else:
                # no wheel found
                print("Failed to find python wheel from project: " + self.project_name)
                ret = False
        return ret

    def _handle_command_exec(self, exec_phase, exec_cmd, cmd_exec_dir):
        cmd_exec_dir = Path(os.path.expandvars(str(cmd_exec_dir)))
        if exec_cmd:
            exec_cmd = os.path.expandvars(exec_cmd)
        if cmd_exec_dir:
            cmd_exec_dir = os.path.expandvars(cmd_exec_dir)
        ret = Path(cmd_exec_dir).is_dir()
        # Handle first special API command or commands:
        #  - Special commands are keywords that will trigger the execution
        #    of internal python function.
        #  - Special commands needs to be the first commands executed
        while (
            (ret == True)
            and (exec_cmd is not None)
            and (exec_cmd.startswith("RCB_CMD__FIND_AND_INSTALL_LATEST_PYTHON_WHEEL"))
        ):
            line_arr = exec_cmd.splitlines(True)
            special_cmd = line_arr[0]
            # then concat rest of the lines for next command to be executed
            if len(line_arr) > 1:
                exec_cmd = "".join(line_arr[1:])
            else:
                exec_cmd = None
            ret = self._handle_FIND_AND_HANDLE_LATEST_PYTHON_WHEEL_CMD(special_cmd)
        # then handle regular command or multiple commands
        if (ret == True) and (exec_cmd is not None):
            is_multiline = self.is_multiline_text(exec_cmd)
            if is_multiline:
                is_windows = any(platform.win32_ver())
                if is_windows:
                    # subprocess.run does not handle the execution of multiple
                    # commands together in same in dos-prompt based command shell
                    # and instead they would each need to be separated with &&
                    # which would cause each of them to be run on own shell.
                    # Therefore in case of multiple commands, we need to write them
                    # first to bat-file and then execute that bat-file.
                    os.makedirs(self.project_build_dir, exist_ok=True)
                    build_cmd_file = os.path.join(
                        self.project_build_dir, exec_phase + ".bat"
                    )
                    with open(build_cmd_file, "w") as file:
                        file.write(exec_cmd)
                    ret = self._exec_subprocess_batch_file(str(build_cmd_file))
                else:
                    # bash can execute multiple commands in same subprocess.run process
                    print("------ " + exec_phase + " start ----------")
                    self._exec_subprocess_cmd("env", cmd_exec_dir)
                    print("------ " + exec_phase + " end ----------")
                    time.sleep(1)
                    ret = self._exec_subprocess_cmd(exec_cmd, cmd_exec_dir)
            else:
                # execute just a single command
                print("------ " + exec_phase + " start ----------")
                self._exec_subprocess_cmd("env", cmd_exec_dir)
                print("------ " + exec_phase + " end ----------")
                time.sleep(1)
                ret = self._exec_subprocess_cmd(exec_cmd, cmd_exec_dir)
        return ret

    # public methods
    def exec(self, args: list[str | Path], cwd: Path, *, stdout_devnull: bool = False):
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

    def list_submodules(
        self, repo_path: Path, *, relative: bool = False, recursive: bool = True
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

    def list_status(self, repo_path: Path) -> list[tuple[str, str]]:
        """Gets the status as a list of (status_type, relative_path)."""
        raw_output = subprocess.check_output(
            ["git", "status", "--porcelain", "-u", "--ignore-submodules"],
            cwd=str(repo_path),
        )
        lines = raw_output.decode().splitlines()
        return [tuple(line.strip().split()) for line in lines]

    def get_all_repositories(self, root_path: Path) -> list[Path]:
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
                submodule_paths = self.list_submodules(
                    repo_path, relative=True, recursive=False
                )
                self.exec(
                    ["git", "update-index", "--skip-worktree"] + submodule_paths,
                    cwd=repo_path,
                )
            except Exception as e:
                # pytorch audio has empty .gitmodules file which can cause exception
                pass

    def save_repo_patches(self, repo_path: Path, patches_path: Path):
        """Updates the patches directory with any patches committed to the repository."""
        if patches_path.exists():
            shutil.rmtree(patches_path)
        # Get key revisions.
        upstream_rev = rev_parse(repo_path, TAG_UPSTREAM_DIFFBASE)
        hipify_rev = rev_parse(repo_path, TAG_HIPIFY_DIFFBASE)
        if upstream_rev is None:
            print(
                f"error: Could not find upstream diffbase tag {TAG_UPSTREAM_DIFFBASE}"
            )
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
            self.exec(
                ["git", "format-patch", "-o", base_path, base_revlist], cwd=repo_path
            )
        if hipified_count > 0:
            hipified_path = patches_path / "hipified"
            hipified_path.mkdir(parents=True, exist_ok=True)
            self.exec(
                ["git", "format-patch", "-o", hipified_path, hipified_revlist],
                cwd=repo_path,
            )

    def apply_repo_patches(self, repo_path: Path, patches_path: Path):
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
                "--ignore-whitespace",
                "--committer-date-is-author-date",
                "--no-gpg-sign",
            ]
            + patch_files,
            cwd=repo_path,
        )


    def apply_main_repository_patches(
        self, root_repo_path: Path, patches_path: Path, repo_name: str, patchset_name: str
    ):
        # Apply patches to main repository.
        self.apply_repo_patches(root_repo_path, patches_path / repo_name / patchset_name)


    def apply_submodule_patches(
        self, root_repo_path: Path, patches_path: Path, repo_name: str, patchset_name: str
    ):
        relative_sm_paths = self.list_submodules(root_repo_path, relative=True)
        for relative_sm_path in relative_sm_paths:
            self.apply_repo_patches(
                root_repo_path / relative_sm_path,
                patches_path / relative_sm_path / patchset_name,
            )


    def apply_all_patches(
        self, root_repo_path: Path, patches_path: Path, repo_name: str, patchset_name: str
    ):
        self.apply_main_repository_patches(
            root_repo_path, patches_path, repo_name, patchset_name
        )
        self.apply_submodule_patches(root_repo_path, patches_path, repo_name, patchset_name)


    # repo_hashtag_to_patches_dir_name examples:
    #
    #     "2.7.0-rc9"    --> "2.7.0"
    #     "releases/2.7" --> "releases_2.7"
    def repo_hashtag_to_patches_dir_name(self, version_ref: str) -> str:
        version_ref = version_ref.replace("/", "_")
        pos = version_ref.find("-")
        if pos != -1:
            return version_ref[:pos]
        return version_ref

    # repo_hashtag_to_patches_dir_name('2.7.0-rc9') -> '2.7.0'
    def get_project_patch_dir_root(self,
                  patch_dir_root: Path,
                  project_name: str,
                  project_patch_dir_name: str) -> Path:
        project_patch_dir_name = self.repo_hashtag_to_patches_dir_name(project_patch_dir_name)
        return Path(patch_dir_root / project_name / project_patch_dir_name)

    def do_env_setup(self, env_setup_cmd_list):
        ret = True
        if env_setup_cmd_list:
            print(env_setup_cmd_list)
            for key_value_str in env_setup_cmd_list:
                print(key_value_str)
                key_value_arr = key_value_str.split("=", 1)
                if len(key_value_arr) == 2:
                    env_var_key = key_value_arr[0].strip()
                    env_var_new_value = key_value_arr[1].strip()
                    env_var_old_value = os.environ.get(env_var_key)
                    if env_var_new_value:
                        # replace ${ENV_SYNTAX} with it's real values on string
                        expanded_new_value = os.path.expandvars(env_var_new_value)
                        # print("key: " + env_var_key)
                        # print("new value: " + env_var_new_value)
                        # print("new expanded value: " + expanded_new_value)
                        os.environ[env_var_key] = expanded_new_value
                    self.orig_env_variables_hashtable[env_var_key] = env_var_old_value
                else:
                    print(
                        "Error, Invalid environment variable key-value pair in project: "
                        + self.project_name
                    )
                    print("Key: " + key_value_str)
                    ret = False
        else:
            print("No environment settings specified")
        print("------ env-settings start ----------")
        self._exec_subprocess_cmd("env", ".")
        print("------ env-settings end ----------")
        return ret

    def undo_env_setup(self, env_setup_cmd_list):
        ret = True
        print("undo_env_setup")
        for (
            env_var_key,
            orig_env_var_value,
        ) in self.orig_env_variables_hashtable.items():
            # print("stored restore key: " + env_var_key)
            if orig_env_var_value is None:
                # print("delete: " + env_var_key)
                del os.environ[env_var_key]
            else:
                # print("restore: " + env_var_key)
                # print("value: " + orig_env_var_value)
                os.environ[env_var_key] = orig_env_var_value
        return ret

    def is_multiline_text(self, exec_cmd):
        return len(exec_cmd.splitlines()) > 1

    def do_init(self, init_cmd):
        cur_p = Path(self.project_build_dir)
        cur_p.mkdir(parents=True, exist_ok=True)
        ret = cur_p.is_dir()
        if ret:
            ret = self._handle_command_exec("init", init_cmd, self.project_exec_dir)
        return ret

    def do_clean(self, clean_cmd):
        ret = True
        # we want to return true for clean command even if the project has not been checked out yet
        if self.project_src_dir.is_dir() == True:
            ret = self._handle_command_exec("clean", clean_cmd, self.project_exec_dir)
        return ret

    def do_checkout(
        self,
        repo_fetch_depth=1,
        repo_fetch_tags=None,
        repo_fetch_job_cnt=1,
        apply_patches_enabled=1,
        hipify_enabled=1,
        repo_remote_name="origin",
    ):
        ret = True
        print("do_checkout started")
        dot_git_subdir = self.project_src_dir / ".git"
        if dot_git_subdir.exists():
            # print(f"Not cloning repository ({dot_git_subdir} exists)")
            pass
        else:
            print(f"Cloning repository at {self.project_version_hashtag}")
            self.project_src_dir.mkdir(parents=True, exist_ok=True)
            self.exec(
                ["git", "init", "--initial-branch=main"], cwd=self.project_src_dir
            )
            self.exec(
                ["git", "config", "advice.detachedHead", "false"],
                cwd=self.project_src_dir,
            )
            self.exec(
                ["git", "remote", "add", "origin", self.project_repo_url],
                cwd=self.project_src_dir,
            )

        # fetch and checkout
        fetch_args = []
        fetch_args_main_prj_only = []
        if repo_fetch_depth:
            fetch_args.extend(["--depth", str(repo_fetch_depth)])
        if repo_fetch_tags:
            # can not go to submodule fetch
            # fetch also tags when full fetch is wanted because
            # full fetch may be wanted for projects that checkouts also tags
            fetch_args_main_prj_only.extend(["--tags"])
        if repo_fetch_job_cnt:
            fetch_args.extend(["-j", str(repo_fetch_job_cnt)])
        self.exec(["git", "reset", "--hard"], cwd=self.project_src_dir)
        try:
            self.exec(
                ["git", "fetch"]
                + fetch_args + fetch_args_main_prj_only
                + ["origin", "tag", self.project_version_hashtag],
                cwd=self.project_src_dir,
            )
            self.exec(
                ["git", "checkout", self.project_version_hashtag],
                cwd=self.project_src_dir,
            )
        except:
            # no git tag available, fetch and checkout other way
            self.exec(
                ["git", "fetch"]
                + fetch_args + fetch_args_main_prj_only
                + ["origin", self.project_version_hashtag],
                cwd=self.project_src_dir,
            )
            self.exec(["git", "checkout", "FETCH_HEAD"], cwd=self.project_src_dir)
        if apply_patches_enabled:
            # Apply base patches to main repository. Patches to
            # submodules will be applied later. This enables patches
            # to modify submodule version to be checked out.
            print("project_src_dir: " + str(self.project_src_dir))
            for ii, cur_patch_dir_root in enumerate(self.patch_dir_root_arr):
                full_patch_dir = self.get_project_patch_dir_root(cur_patch_dir_root,
                                                            self.project_name,
                                                            self.project_version_hashtag)
                print("full patch dir: " + str(full_patch_dir))
                if full_patch_dir.is_dir():
                    self.apply_main_repository_patches(
                        self.project_src_dir,
                        full_patch_dir,
                        self.project_name,
                        "base",
                    )
                    # apply patches only from the first directory that exist
                    break

        # add our own git tag to help with the create patches command
        self.exec(
            ["git", "tag", "-f", TAG_UPSTREAM_DIFFBASE, "--no-sign"],
            cwd=self.project_src_dir,
        )
        try:
            self.exec(
                ["git", "submodule", "update", "--init", "--recursive"] + fetch_args,
                cwd=self.project_src_dir,
            )
        except subprocess.CalledProcessError:
            print("Failed to fetch git submodules")
            sys.exit(1)
        self.exec(
            [
                "git",
                "submodule",
                "foreach",
                "--recursive",
                f"git tag -f {TAG_UPSTREAM_DIFFBASE} --no-sign",
            ],
            cwd=self.project_src_dir,
            stdout_devnull=True,
        )

        self.git_config_ignore_submodules(self.project_src_dir)

        if apply_patches_enabled:
            for ii, cur_patch_dir_root in enumerate(self.patch_dir_root_arr):
                full_patch_dir = self.get_project_patch_dir_root(cur_patch_dir_root,
                                                            self.project_name,
                                                            self.project_version_hashtag)
                print("apply submodule patches")
                print("full patch dir: " + str(full_patch_dir))
                if full_patch_dir.is_dir():
                    # Apply base patches to submodules.
                    self.apply_submodule_patches(
                        self.project_src_dir,
                        full_patch_dir,
                        self.project_name,
                        "base",
                    )
                    # apply patches only from the first directory that exist
                    break
        return ret

    def do_hipify(self, hipify_cmd):
        ret = True
        print("do_hipify started")
        if hipify_cmd:
            ret = self._exec_subprocess_cmd(hipify_cmd, self.project_exec_dir)
            # Iterate over the base repository and all submodules. Because we process
            # the root repo first, it will not add submodule changes.
            repo_dir: Path = self.project_src_dir
            all_paths = self.get_all_repositories(repo_dir)
            for module_path in all_paths:
                status = self.list_status(module_path)
                if not status:
                    # if no changes in repo, do not try to add and commit
                    continue
                print(f"HIPIFY made changes to {module_path}: Committing")
                self.exec(["git", "add", "-A"], cwd=module_path)
                self.exec(
                    ["git", "commit", "-m", HIPIFY_COMMIT_MESSAGE, "--no-gpg-sign"],
                    cwd=module_path,
                )
                self.exec(
                    ["git", "tag", "-f", TAG_HIPIFY_DIFFBASE, "--no-sign"],
                    cwd=module_path,
                )
            print("do_hipify, hipified files committed")
        # always apply the patches from hipified directory. (even if hipify_cmd was not specified in config file for project)

        for ii, cur_patch_dir_root in enumerate(self.patch_dir_root_arr):
            full_patch_dir = self.get_project_patch_dir_root(cur_patch_dir_root,
                                                        self.project_name,
                                                        self.project_version_hashtag)
            print("apply submodule patches")
            print("full patch dir: " + str(full_patch_dir))
            if full_patch_dir.is_dir():
                self.apply_all_patches(
                    self.project_src_dir,
                    full_patch_dir,
                    self.project_name,
                    "hipified",
                )
                print("do_hipify, hipified patches applied")
                # apply patches only from the first directory that exist
                break

        return ret

    def do_pre_config(self, pre_config_cmd):
        return self._handle_command_exec(
            "pre_config", pre_config_cmd, self.project_exec_dir
        )

    def do_config(self, config_cmd):
        return self._handle_command_exec("config", config_cmd, self.project_exec_dir)

    def do_cmake_config(self, cmake_config):
        ret = True
        if cmake_config:
            cmake_config = os.path.expandvars(str(cmake_config))
            cmake_config = "cmake -GNinja " + cmake_config
            ret = self._handle_command_exec(
                "cmake_config", cmake_config, self.project_build_dir
            )
        return ret

    def do_post_config(self, post_config_cmd):
        return self._handle_command_exec(
            "post_config", post_config_cmd, self.project_exec_dir
        )

    def do_cmake_build(self, cmake_config):
        ret = True
        if cmake_config:
            build_cmd = "cmake --build " + self.project_build_dir.as_posix()
            ret = self._handle_command_exec(
                "cmake build", build_cmd, self.project_build_dir
            )
        return ret

    def do_build(self, build_cmd):
        return self._handle_command_exec("build", build_cmd, self.project_exec_dir)

    def do_install(self, install_cmd):
        return self._handle_command_exec("install", install_cmd, self.project_exec_dir)

    def do_cmake_install(self, cmake_config):
        ret = True
        if cmake_config:
            install_cmd = "cmake --install " + self.project_build_dir.as_posix()
            ret = self._handle_command_exec(
                "cmake install", install_cmd, self.project_build_dir
            )
        return ret

    def do_post_install(self, post_install_cmd):
        return self._exec_subprocess_cmd(post_install_cmd, self.project_exec_dir)

    def do_save_patches(self):
        ret = True
        # even if there are multiple patch dirs from where to patches can be applied,
        # patches are always wanted to be saved to first dir in the list
        cur_patch_dir_root = self.patch_dir_root_arr[0]
        full_patch_dir = self.get_project_patch_dir_root(cur_patch_dir_root,
                                                    self.project_name,
                                                    self.project_version_hashtag)
        self.save_repo_patches(self.project_src_dir, full_patch_dir / self.project_name)
        relative_sm_paths = self.list_submodules(self.project_src_dir, relative=True)
        for relative_sm_path in relative_sm_paths:
            self.save_repo_patches(
                self.project_src_dir / relative_sm_path, full_patch_dir / relative_sm_path
            )
        return ret
