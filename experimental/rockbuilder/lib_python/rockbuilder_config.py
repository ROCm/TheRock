import ast
import configparser
import os
import platform
import sys
import ast
import subprocess
from lib_python.repo_management import RockProjectRepo
from pathlib import Path, PurePosixPath


class RockBuilderConfig(configparser.ConfigParser):
    def __init__(self, rcb_root_dir: Path, rcb_build_dir: Path):
        super(RockBuilderConfig, self).__init__(allow_no_value=True)
        self.rcb_root_dir = rcb_root_dir
        self.rcb_build_dir = rcb_build_dir
        self.fname_cfg = rcb_root_dir / "rockbuilder.ini"
        self.gpu_target_list = None
        self.rock_sdk_whl_url = None
        self.rock_sdk_new_build_dir = None
        self.rock_sdk_old_build_dir = None
        self.cfg_last_mod_time_sec = 0

    def _replace_env_variables(self, cmd_str):
        ret = os.path.expandvars(cmd_str)
        return ret

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

    # if the rockbuilder.ini file is present, set some build options based on to it's values
    def read_cfg(self):
        ret = False
        if self.fname_cfg.exists():
            # get last modifued date
            f_stats = self.fname_cfg.stat()
            self.cfg_last_mod_time_sec = f_stats.st_mtime
            print("self.cfg_last_mod_time_sec:" + str(self.cfg_last_mod_time_sec))
            # read the config values
            self.read(self.fname_cfg)
            ret = True
        return ret

    # write stamp file used to verify whether the pip install has been done
    def _write_pip_install_stamp(self, fname_pip_done, python_home_dir, time_sec):
        ret = False
        try:
            dir_path = fname_pip_done.parent
            if not dir_path.is_dir():
                dir_path.mkdir(parents=True, exist_ok=True)
            config = configparser.ConfigParser()
            if fname_pip_done.exists():
                config.read(fname_pip_done)
            if "timestamps" not in config:
                config["timestamps"] = {}
            config["timestamps"][python_home_dir] = str(time_sec)
            with open(fname_pip_done, "w") as configfile:
                config.write(configfile)
            print(
                f"Timestamp "
                + str(time_sec)
                + " written to file: "
                + str(fname_pip_done)
            )
            ret = True
        except FileExistsError:
            print(f"Directory '{directory_name}' already exists.")
        except OSError as e:
            print(f"Error creating directory '{directory_name}': {e}")
        except IOError as e:
            print(f"Error writing to file: {e}")
        return ret

    def _check_pip_install_stamp(self, fname_pip_done, python_home_dir, time_sec):
        ret = False
        try:
            config = configparser.ConfigParser()
            if fname_pip_done.exists():
                config.read(fname_pip_done)
                if config.has_option("timestamps", python_home_dir):
                    time_read_str = config["timestamps"][python_home_dir]
                    if time_read_str == str(time_sec):
                        print(
                            f"Timestamp matches for "
                            + python_home_dir
                            + ": "
                            + str(time_sec)
                        )
                        ret = True
        except FileExistsError:
            print(f"Directory '{directory_name}' already exists.")
        except OSError as e:
            print(f"Error creating directory '{directory_name}': {e}")
        except IOError as e:
            print(f"Error writing to file: {e}")
        return ret

    def get_as_list(self, section_name_rocm_sdk, key_name_rocm_sdk_whl_url):
        # we get values as a string reporesenting a list of strings
        ret = self.get(section_name_rocm_sdk, key_name_rocm_sdk_whl_url)
        # convert it to real python list object
        ret = ast.literal_eval(ret)
        # ret = ret.split(", ")
        return ret

    def capture(self, args: list[str | Path], cwd: Path) -> str:
        args = [str(arg) for arg in args]
        try:
            return subprocess.check_output(args, cwd=str(cwd)).decode().strip()
        except subprocess.CalledProcessError as e:
            print(f"Error capturing output: {e}")
            return ""

    # this works only for rocm sdk's installed as a pip wheel which have rocm_sdk tool
    def get_rocm_python_wheel_sdk_path(self, path_name: str) -> Path:
        ret = None
        dir_str = self.capture(
            [sys.executable, "-m", "rocm_sdk", "path", f"--{path_name}"],
            cwd=Path.cwd(),
        )
        if len(dir_str) > 0:
            print("dir_str: " + str(len(dir_str)))
            dir_str.strip()
            ret = Path(dir_str)
        return ret

    # this works only for rocm sdk's installed as a pip wheel which have rocm_sdk tool
    def get_rocm_python_wheel_sdk_targets(self) -> str:
        # Run `rocm-sdk targets` to get the default architecture
        targets = self.capture(
            [sys.executable, "-m", "rocm_sdk", "targets"], cwd=Path.cwd()
        )
        if not targets:
            print("Warning: rocm-sdk targets returned empty or failed")
            return ""
        # convert space-separated targets to semicolon separated list
        # that can be used for most of the apps as a -DAMD_GPU_TARGETS parameter
        return targets.replace(" ", ";")

    def setup_build_env(self):
        section_name_rocm_sdk = "rocm_sdk"
        section_name_build_targets = "build_targets"
        key_name_rocm_sdk_whl_url = "rocm_sdk_whl_server_url"
        key_name_rocm_sdk_new_build = "rocm_sdk_build"
        key_name_rocm_sdk_old_build = "rocm_sdk_dir"
        key_name_gpus = "gpus"
        rocm_sdk_uninstall_cmd = (
            sys.executable
            + " -m pip cache remove rocm_sdk --cache-dir "
            + (self.rcb_root_dir / "pip").as_posix()
        )
        install_deps_cmd = (
            sys.executable
            + " -m pip install setuptools --cache-dir "
            + (self.rcb_root_dir / "pip").as_posix()
        )
        rocm_sdk_install_cmd_base = (
            sys.executable
            + " -m pip install rocm[libraries,devel] torch torchaudio torchvision --force-reinstall --cache-dir "
            + (self.rcb_root_dir / "pip").as_posix()
        )
        rocm_sdk_pytorch_install_cmd_base = (
            sys.executable
            + " -m pip install torch torchaudio torchvision --force-reinstall --cache-dir "
            + (self.rcb_root_dir / "pip").as_posix()
        )

        # check the selected target GPUs
        if self.has_option(section_name_build_targets, key_name_gpus):
            self.gpu_target_list = self.get_as_list(
                section_name_build_targets, key_name_gpus
            )
        # check option to use rocm sdk from the wheel
        if self.has_option(section_name_rocm_sdk, key_name_rocm_sdk_whl_url):
            self.rock_sdk_whl_url = self.get_as_list(
                section_name_rocm_sdk, key_name_rocm_sdk_whl_url
            )
            # there should be only one item in the list
            if len(self.rock_sdk_whl_url) == 1:
                self.rock_sdk_whl_url = self.rock_sdk_whl_url[0]
            else:
                print(
                    "rockbuilder.ini file error, wheel server url must be an an string array with with one server specified"
                )
                sys.exit(1)

        # check whether new rocm sdk build should be done
        if self.has_option(section_name_rocm_sdk, key_name_rocm_sdk_new_build):
            self.rock_sdk_new_build_dir = self.get_as_list(
                section_name_rocm_sdk, key_name_rocm_sdk_new_build
            )
            # there should be only one item in this list
            if len(self.rock_sdk_new_build_dir) == 1:
                self.rock_sdk_new_build_dir = self.rock_sdk_new_build_dir[0]
            else:
                print(
                    "rockbuilder.ini file error, wheel server url must be an an string array with with directory"
                )
                sys.exit(1)

        # check whether old rocm sdk build should be done
        if self.has_option(section_name_rocm_sdk, key_name_rocm_sdk_old_build):
            self.rock_sdk_old_build_dir = self.get_as_list(
                section_name_rocm_sdk, key_name_rocm_sdk_old_build
            )
            # there should be only one item in this list
            if len(self.rock_sdk_old_build_dir) == 1:
                self.rock_sdk_old_build_dir = self.rock_sdk_old_build_dir[0]
            else:
                print(
                    "rockbuilder.ini file error, wheel server url must be an an string array with with directory"
                )
                sys.exit(1)

        # rocm sdk from the pip wheel option
        if self.rock_sdk_whl_url and self.gpu_target_list:
            # get first the install cmd used to install the sdk itself
            # for now only 1 wheel can actually be installed, so rockbuilder_cfg.py selection tool
            # will allow user to select only one gpu target when this option is defined
            full_rocm_sdk_install_cmd = None
            for gpu_target in self.gpu_target_list:
                full_rock_sdk_whl_url = self.rock_sdk_whl_url + gpu_target
                full_rocm_sdk_install_cmd = (
                    rocm_sdk_install_cmd_base + " --index-url " + full_rock_sdk_whl_url
                )
                """
                full_rocm_sdk_pytorch_install_cmd = (
                    rocm_sdk_pytorch_install_cmd_base
                    + " --index-url "
                    + full_rock_sdk_whl_url
                )
                """
                break
            if full_rocm_sdk_install_cmd:
                print("rocm_sdk install cmd: " + full_rocm_sdk_install_cmd)
                # if rocm home is not found, we will install the sdk from the wheel
                rocm_home = self.get_rocm_python_wheel_sdk_path("root")
                print("rocm_home: " + str(rocm_home))
                sdk_installed = False
                fname_pip_done = self.rcb_build_dir / "rcb_sdk_pip_install.done"
                if rocm_home:
                    # even if the rocm sdk is found, we will check the stamp file
                    # whether to reinstall/update the sdk
                    sdk_installed = self._check_pip_install_stamp(
                        fname_pip_done, sys.prefix, self.cfg_last_mod_time_sec
                    )
                if not sdk_installed:
                    # install only once unless config-file date has been modified
                    # install first the pip wheel
                    # print("rocm_sdk_uninstall_cmd: " + rocm_sdk_uninstall_cmd)
                    print("root dir: " + self.rcb_root_dir.as_posix())
                    # dependencies
                    self._exec_subprocess_cmd(
                        install_deps_cmd, self.rcb_root_dir.as_posix()
                    )
                    # uninstall old
                    self._exec_subprocess_cmd(
                        rocm_sdk_uninstall_cmd, self.rcb_root_dir.as_posix()
                    )
                    # install rocm sdk and pytorch
                    self._exec_subprocess_cmd(
                        full_rocm_sdk_install_cmd, self.rcb_root_dir.as_posix()
                    )
                    """
                    # install pytorch separately is bad idea now
                    # -->
                    # this could lead to install where the rocm-sdk-devel version
                    #    is different version than other packages. --> causing broken sdk
                    self._exec_subprocess_cmd(
                        full_rocm_sdk_pytorch_install_cmd, self.rcb_root_dir.as_posix()
                    )
                    """
                res = self._write_pip_install_stamp(
                    fname_pip_done, sys.prefix, self.cfg_last_mod_time_sec
                )
                if not res:
                    print("Failed to write to file: " + str(fname_pip_done))
                    sys.exit(1)
            # set ROCM_HOME to point to the python venv which contains rocm_sdk
            if not rocm_home:
                rocm_home = self.get_rocm_python_wheel_sdk_path("root")
            os.environ["ROCM_HOME"] = rocm_home.as_posix()
            # set target GPUs to environment variable if not set earlier
            if not "THEROCK_AMDGPU_TARGETS" in os.environ:
                gpu_targets = self.get_rocm_python_wheel_sdk_targets()
                print("gpu_targets: " + gpu_targets)
                os.environ["THEROCK_AMDGPU_TARGETS"] = gpu_targets
            print("rocm_home: " + rocm_home.as_posix())

        # rocm sdk from the existing directory specified in the rockbuilder.init
        if self.rock_sdk_old_build_dir and self.gpu_target_list:
            os.environ["ROCM_HOME"] = self.rock_sdk_old_build_dir

        # set target GPUs to environment variable if not set earlier
        if not "THEROCK_AMDGPU_TARGETS" in os.environ:
            if self.gpu_target_list:
                for ii, gpu_target in enumerate(self.gpu_target_list):
                    if ii == 0:
                        gpu_targets = self.gpu_target_list[0]
                    else:
                        gpu_targets = gpu_targets + ";" + self.gpu_target_list[ii]
                os.environ["THEROCK_AMDGPU_TARGETS"] = gpu_targets
                print("gpu_targets: " + gpu_targets)

        # rocm sdk build for the new directory specified in the rockbuilder.init
        # if self.rock_sdk_old_build_dir and self.gpu_target_list:
        # A) Either do everything in python code
        # 1) create/init the rocm_sdk/.venv
        # 2) exec get_sources()
        # 3) cmake configure
        # 4) cmake --build build
        #    os.environ["ROCM_HOME"] = self.rock_sdk_old_build_dir
        # B) or generate setup_env and build_rocm_sdk scripts for linux and windows
