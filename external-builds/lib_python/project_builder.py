#! python

import ast
import configparser
import os
from lib_python.repo_management import RockProjectRepo
from pathlib import Path, PurePosixPath

class RockProjectBuilder(configparser.ConfigParser):
    def __init__(self, project_name):
        super(RockProjectBuilder, self).__init__(allow_no_value=True)

        self.project_name = project_name
        self.cfg_file_name = os.path.abspath('./' + project_name +  '.pcfg')        
        if os.path.exists(self.cfg_file_name):
            self.read(self.cfg_file_name)
        else:
            raise ValueError("Could not find the configuration file: " + self.cfg_file_name)
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
            self.build_cmd = self.get('project_info', 'build_cmd')
        except:
            self.build_cmd = None
        try:
            self.install_cmd = self.get('project_info', 'install_cmd')
        except:
            self.install_cmd = None
        ROCK_BUILDER_ROOT_DIR = Path(os.environ["ROCK_BUILDER_ROOT_DIR"])
        self.project_src_dir = ROCK_BUILDER_ROOT_DIR / "src_projects" / self.project_name
        self.patch_dir = os.path.abspath('./patches/' + self.project_name)

    # printout project builder specific info for logging and debug purposes
    def printout(self):
        print("project_name: " + self.project_name)
        print("cfg_file_name: " + self.cfg_file_name)
        print("version: " + self.project_version)
        print("patch_dir: " + self.patch_dir)
        
    def checkout(self):
        project_repo = RockProjectRepo(self.project_name,
                                       self.project_src_dir,
                                       self.repo_url,
                                       self.project_version)
        project_repo.do_checkout()

    def clean(self):
        project_repo = RockProjectRepo(self.project_name,
                                       self.project_src_dir,
                                       self.repo_url,
                                       self.project_version)
        project_repo.do_clean(self.clean_cmd)

    def configure(self):
        project_repo = RockProjectRepo(self.project_name,
                                       self.project_src_dir,
                                       self.repo_url,
                                       self.project_version)
        project_repo.do_configure(self.configure_cmd)

    def build(self):
        if self.build_cmd is not None:
            project_repo = RockProjectRepo(self.project_name,
                                           self.project_src_dir,
                                           self.repo_url,
                                           self.project_version)
            project_repo.do_build(self.build_cmd)

    def install(self):
        project_repo = RockProjectRepo(self.project_name,
                                       self.project_src_dir,
                                       self.repo_url,
                                       self.project_version)
        project_repo.do_install(self.install_cmd)


class RockExternalProjectListManager(configparser.ConfigParser):
    def __init__(self):
        self.cfg_file_name = os.path.abspath('./rockbuilder.cfg')
        super(RockExternalProjectListManager, self).__init__(allow_no_value=True)
        if os.path.exists(self.cfg_file_name):
            self.read(self.cfg_file_name)
	
    def get_external_project_list(self):
        value = self.get('projects', 'project_list')
        return list(filter(None, (x.strip() for x in value.splitlines())))
        
    def get_rock_project_builder(self, project_name):
        ret = None;
        try:
            ret = RockProjectBuilder(project_name)
        except ValueError as e:
            print(str(e))
        return ret
