import ast
import configparser
import os
import platform
import sys
from lib_python.repo_management import RockProjectRepo
from pathlib import Path, PurePosixPath


class RockProjectBuilder(configparser.ConfigParser):

    # Read the value from the config-file's "project_info" section.
    #
    # Return value if it exist, otherwise return None
    def _get_project_info_config_value(self, config_key):
        try:
            ret = self.get("project_info", config_key)
        except:
            # just catch what ever exception is thrown by current python
            # env-version in case that the config-key/value is not specified
            # in the configuration file. (key/value pairs can be optional)
            ret = None
        return ret

    def __init__(self, rock_builder_root_dir, project_name, package_output_dir):
        super(RockProjectBuilder, self).__init__(allow_no_value=True)

        self.is_posix = not any(platform.win32_ver())
        self.rock_builder_root_dir = rock_builder_root_dir
        self.project_name = project_name
        self.cfg_file_path = (
            Path(rock_builder_root_dir) / "projects" / f"{project_name}.cfg"
        )
        self.package_output_dir = package_output_dir
        if self.cfg_file_path.exists():
            self.read(self.cfg_file_path)
        else:
            raise ValueError(
                "Could not find the configuration file: "
                + self.cfg_file_path.as_posix()
            )
        self.repo_url = self.get("project_info", "repo_url")
        self.project_version = self.get("project_info", "version")

        # environment setup can have common and os-specific sections that needs to be appended together
        if self.is_posix:
            self.skip_on_os = self._get_project_info_config_value("skip_linux")
        else:
            self.skip_on_os = self._get_project_info_config_value("skip_windows")
        self.env_setup_cmd = None
        value = self._get_project_info_config_value("env_common")
        if value:
            self.env_setup_cmd = list(
                filter(None, (x.strip() for x in value.splitlines()))
            )
        if self.is_posix:
            value = self._get_project_info_config_value("env_linux")
            if value:
                temp_env_list = list(
                    filter(None, (x.strip() for x in value.splitlines()))
                )
                if self.env_setup_cmd:
                    self.env_setup_cmd.extend(temp_env_list)
                else:
                    self.env_setup_cmd = temp_env_list
        else:
            value = self._get_project_info_config_value("env_windows")
            if value:
                temp_env_list = list(
                    filter(None, (x.strip() for x in value.splitlines()))
                )
                if self.env_setup_cmd:
                    self.env_setup_cmd.extend(temp_env_list)
                else:
                    self.env_setup_cmd = temp_env_list
        self.init_cmd = self._get_project_info_config_value("init_cmd")
        self.clean_cmd = self._get_project_info_config_value("clean_cmd")
        self.hipify_cmd = self._get_project_info_config_value("hipify_cmd")
        self.pre_config_cmd = self._get_project_info_config_value("pre_config_cmd")
        self.config_cmd = self._get_project_info_config_value("config_cmd")
        self.post_config_cmd = self._get_project_info_config_value("post_config_cmd")

        is_windows = any(platform.win32_ver())
        # here we want to check if window option is set
        # otherwise we use generic "build_cmd" option also on windows
        if is_windows and self.has_option("project_info", "build_cmd_windows"):
            self.build_cmd = self._get_project_info_config_value("build_cmd_windows")
        else:
            self.build_cmd = self._get_project_info_config_value("build_cmd")
        self.cmake_config = self._get_project_info_config_value("cmake_config")
        self.install_cmd = self._get_project_info_config_value("install_cmd")
        self.post_install_cmd = self._get_project_info_config_value("post_install_cmd")

        self.project_root_dir_path = Path(rock_builder_root_dir)
        self.project_src_dir_path = (
            Path(rock_builder_root_dir) / "src_projects" / self.project_name
        )
        self.project_build_dir_path = (
            Path(rock_builder_root_dir) / "builddir" / self.project_name
        )

        self.cmd_execution_dir = self._get_project_info_config_value("cmd_exec_dir")
        if self.cmd_execution_dir is None:
            # default value if not specified in the config-file
            self.cmd_execution_dir = self.project_src_dir_path
        self.project_patch_dir_root = (
            Path(rock_builder_root_dir)
            / "../../external-builds/pytorch/patches"
            / self.project_name
        )
        self.project_patch_dir_root = self.project_patch_dir_root.resolve()
        self.project_repo = RockProjectRepo(
            self.package_output_dir,
            self.project_name,
            self.project_root_dir_path,
            self.project_src_dir_path,
            self.project_build_dir_path,
            self.cmd_execution_dir,
            self.repo_url,
            self.project_version,
            self.project_patch_dir_root,
        )

    # printout project builder specific info for logging and debug purposes
    def printout(self, phase):
        print("Project build phase " + phase + ": -----")
        print("    Project_name: " + self.project_name)
        print("    Config_path: " + self.cfg_file_path.as_posix())
        print("    Version:     " + self.project_version)
        print("    Source_dir:  " + self.project_src_dir_path.as_posix())
        print("    Patch_dir:   " + self.project_patch_dir_root.as_posix())
        print("    Build_dir:   " + self.project_build_dir_path.as_posix())
        print("------------------------")

    def printout_error_and_terminate(self, phase):
        self.printout(phase)
        print(phase + " failed")
        sys.exit(1)

    # check whether operations should be skipped on current operating system
    def check_skip_on_os(self):
        ret = True
        if (self.skip_on_os is None) or (
            (self.skip_on_os != "1")
            and (str(self.skip_on_os).casefold() != str("y").casefold())
            and (str(self.skip_on_os).casefold() != str("yes").casefold())
            and (str(self.skip_on_os).casefold() != str("on").casefold())
        ):
            ret = False
        return ret

    def do_env_setup(self):
        res = self.project_repo.do_env_setup(self.env_setup_cmd)
        if not res:
            self.printout_error_and_terminate("env_setup")

    def undo_env_setup(self):
        res = self.project_repo.undo_env_setup(self.env_setup_cmd)
        if not res:
            self.printout_error_and_terminate("undo_env_setup")

    def init(self):
        res = self.project_repo.do_init(self.init_cmd)
        if not res:
            self.printout_error_and_terminate("init")

    def clean(self):
        res = self.project_repo.do_clean(self.clean_cmd)
        if not res:
            self.printout_error_and_terminate("clean")

    def checkout(self):
        res = self.project_repo.do_checkout()
        if not res:
            self.printout_error_and_terminate("checkout")

    def hipify(self):
        res = self.project_repo.do_hipify(self.hipify_cmd)
        if not res:
            self.printout_error_and_terminate("hipify")

    def pre_config(self):
        res = self.project_repo.do_pre_config(self.pre_config_cmd)
        if not res:
            self.printout_error_and_terminate("pre_config")

    def config(self):
        if self.cmake_config:
            # in case that project has cmake configure/build/install needs
            res = self.project_repo.do_cmake_config(self.cmake_config)
            if not res:
                self.printout_error_and_terminate("cmake_config")
        res = self.project_repo.do_config(self.config_cmd)
        if not res:
            self.printout_error_and_terminate("config")

    def post_config(self):
        res = self.project_repo.do_post_config(self.post_config_cmd)
        if not res:
            self.printout_error_and_terminate("post_config")

    def build(self):
        if self.cmake_config:
            # not all projects have things to build with cmake
            res = self.project_repo.do_cmake_build(self.cmake_config)
            if not res:
                self.printout_error_and_terminate("cmake_build")
        res = self.project_repo.do_build(self.build_cmd)
        if not res:
            self.printout_error_and_terminate("build")

    def install(self):
        if self.cmake_config:
            res = self.project_repo.do_cmake_install(self.cmake_config)
            if not res:
                self.printout_error_and_terminate("cmake_install")
        res = self.project_repo.do_install(self.install_cmd)
        if not res:
            self.printout_error_and_terminate("install")

    def post_install(self):
        res = self.project_repo.do_post_install(self.post_install_cmd)
        if not res:
            self.printout_error_and_terminate("post_install")


class RockExternalProjectListManager(configparser.ConfigParser):
    def __init__(self, rock_builder_root_dir, package_output_dir):
        # default application list to builds
        self.cfg_file_path = Path(rock_builder_root_dir) / "projects" / "core_apps.pcfg"
        self.rock_builder_root_dir = rock_builder_root_dir
        self.package_output_dir = package_output_dir
        super(RockExternalProjectListManager, self).__init__(allow_no_value=True)
        if self.cfg_file_path.exists():
            self.read(self.cfg_file_path)

    def get_external_project_list(self):
        value = self.get("projects", "project_list")
        # convert to list of project string names
        return list(filter(None, (x.strip() for x in value.splitlines())))

    def get_rock_project_builder(self, project_name):
        ret = None
        try:
            ret = RockProjectBuilder(
                self.rock_builder_root_dir, project_name, self.package_output_dir
            )
        except ValueError as e:
            print(str(e))
        return ret
