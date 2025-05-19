import ast
import configparser
import os
import platform
import sys
from lib_python.repo_management import RockProjectRepo
from pathlib import Path, PurePosixPath

class RockProjectBuilder(configparser.ConfigParser):
    def __init__(self, rock_builder_root_dir, project_name):
        super(RockProjectBuilder, self).__init__(allow_no_value=True)

        self.project_name  = project_name
        self.cfg_file_path = Path(rock_builder_root_dir) / "projects" / f"{project_name}.cfg"
        if self.cfg_file_path.exists():
            self.read(self.cfg_file_path)
        else:
            raise ValueError("Could not find the configuration file: " + str(self.cfg_file_path))
        self.repo_url = self.get('project_info', 'repo_url')
        self.project_version = self.get('project_info', 'version')
        try:
            self.clean_cmd = self.get('project_info', 'clean_cmd')
        except:
            self.clean_cmd = None
        try:
            self.configure_cmd = self.get('project_info', 'configure_cmd')
        except:
            self.configure_cmd = None
        try:
            is_dos = any(platform.win32_ver())
            if is_dos and self.has_option('project_info', 'build_cmd_dos'):
                self.build_cmd = self.get('project_info', 'build_cmd_dos')
            else:
                self.build_cmd = self.get('project_info', 'build_cmd')
            print("Build_cmd: ------------")
            print(self.build_cmd)
            print("------------------------")
        except Exception as ex1:
            print(ex1)
            self.build_cmd = None
        try:
            self.install_cmd = self.get('project_info', 'install_cmd')
        except:
            self.install_cmd = None
        self.project_root_dir_path = Path(rock_builder_root_dir)
        self.project_src_dir_path = Path(rock_builder_root_dir) / "src_projects" / self.project_name
        self.project_build_dir_path = Path(rock_builder_root_dir) / "builddir" / self.project_name
        self.patch_dir_path = Path(rock_builder_root_dir) / "patches" / self.project_name

    # printout project builder specific info for logging and debug purposes
    def printout(self):
        print("Project: ---------------")
        print("    Name: " + self.project_name)
        print("    Config path: " + str(self.cfg_file_path))
        print("    Version:     " + self.project_version)
        print("    Source_dir:  " + str(self.project_build_dir_path))
        print("    Patch_dir:   " + str(self.patch_dir_path))
        print("    Build dir:   " + self.project_version)
        print("------------------------")

    def checkout(self):
        project_repo = RockProjectRepo(self.project_name,
                                       self.project_root_dir_path,
                                       self.project_src_dir_path,
                                       self.project_build_dir_path,
                                       self.repo_url,
                                       self.project_version)
        project_repo.do_checkout()

    def clean(self):
        project_repo = RockProjectRepo(self.project_name,
                                       self.project_root_dir_path,
                                       self.project_src_dir_path,
                                       self.project_build_dir_path,
                                       self.repo_url,
                                       self.project_version)
        project_repo.do_clean(self.clean_cmd)

    def configure(self):
        project_repo = RockProjectRepo(self.project_name,
                                       self.project_root_dir_path,
                                       self.project_src_dir_path,
                                       self.project_build_dir_path,
                                       self.repo_url,
                                       self.project_version)
        project_repo.do_configure(self.configure_cmd)

    def build(self):
        if self.build_cmd is not None:
            project_repo = RockProjectRepo(self.project_name,
                                       self.project_root_dir_path,
                                       self.project_src_dir_path,
                                       self.project_build_dir_path,
                                       self.repo_url,
                                       self.project_version)
            project_repo.do_build(self.build_cmd)

    def install(self):
        project_repo = RockProjectRepo(self.project_name,
                                       self.project_root_dir_path,
                                       self.project_src_dir_path,
                                       self.project_build_dir_path,
                                       self.repo_url,
                                       self.project_version)
        project_repo.do_install(self.install_cmd)


class RockExternalProjectListManager(configparser.ConfigParser):
    def __init__(self, rock_builder_root_dir):
		# default application list to builds
        self.cfg_file_path = Path(rock_builder_root_dir) / "projects" / "core_apps.pcfg"
        self.rock_builder_root_dir = rock_builder_root_dir
        super(RockExternalProjectListManager, self).__init__(allow_no_value=True)
        if self.cfg_file_path.exists():
            self.read(self.cfg_file_path)
	
    def get_external_project_list(self):
        value = self.get('projects', 'project_list')
        # convert to list of project string names
        return list(filter(None, (x.strip() for x in value.splitlines())))
        
    def get_rock_project_builder(self, project_name):
        ret = None;
        try:
            ret = RockProjectBuilder(self.rock_builder_root_dir, project_name)
        except ValueError as e:
            print(str(e))
        return ret
