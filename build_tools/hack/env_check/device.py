from __future__ import annotations
from .find_tools import FindPython, FindMSVC
from .utils import *
import platform, re, sys, os
from pathlib import PureWindowsPath
import subprocess
from typing import List


class SystemInfo:
    """
    Provides system information in a nicely formatted output.

    Workflow:
    1. Have a function extracting the system information <X> you need
    2. Assign it in SystemInfo.__init__(self)
    3. Provide a property <X>_STATUS(self), that provides a list of strings,
       containing the lines of content stripped of whitespace.
    4. Add it to "states" in summary(self):
        ("<X>:", self.<X>_STATUS)

    Using it:
        summary: Print system info
        python_list: Print python list
        section_bar(<section>): Prints <section> centered with "==" surrounding it
    """

    def __init__(self):

        self._os = platform.system().capitalize()

        self._device_os_stat = self.device_os_status()

        # Define CPU configuration.
        self._device_cpu_stat = self.device_cpu_status()

        # Define GPU configuration list.
        self._device_gpu_list = self.device_gpu_list()

        # Define Device Memory status.
        self._device_dram_stat = self.device_dram_status()

        # Define Device Storage status.
        self._device_disk_stat = self.device_disk_status()

        # Define file system overview.
        self._device_file_system_list = self.device_file_system()

        # Define ccache status.
        self._device_ccache_stat = self.device_ccache_system()

        self._py = FindPython()

        # Define python package list
        self._py_list = self.device_python_list()

        if self.is_windows:
            self._cl = FindMSVC()

    # Define the device's system is Windows Operating System (Win32).
    @property
    def is_windows(self):
        return self._os == "Windows"

    @property
    def is_linux(self):
        return self._os == "Linux"

    @property
    def OS_NAME(self):
        if self.is_windows:
            return f"{self._device_os_stat[0]} {self._device_os_stat[1]} ({self._device_os_stat[2]}, Build {self._device_os_stat[3]})"
        elif self.is_linux:
            return f"{self._device_os_stat[0]} {self._device_os_stat[1]}"

    @property
    def OS_KERNEL(self):
        return self._device_os_stat[3]

    if is_windows:

        @property
        def is_cygwin(self):
            """
            Define the system environment is Cygwin.
            """
            return True if sys.platform == "cygwin" else False

        @property
        def is_msys2(self):
            """
            Define the system environment is MSYS2.
            """
            return True if sys.platform == "msys" else False

        @property
        def GPU_LIST(self):
            return self._device_gpu_list

        @property
        def VIRTUAL_MEMORY_AVAIL(self):
            return self._device_dram_stat[-1]

    if is_linux:

        @property
        def is_WSL2(self):
            with open("/proc/version", "r") as f:
                _f = f.read()
            return True if "microsoft-standard-WSL2" in _f else False

        @property
        def SWAP_MEMORY_AVAIL(self):
            return self._device_dram_stat[2]

    def device_os_status(self):
        """
        Returns Device's operating system status.
        - Windows: -> `(Windows, 10/11, 2_H_, XXXXX)`
        - Linux:   -> `(LINUX_DISTRO_NAME, LINUX_DISTRO_VERSION, "GNU/Linux", LINUX_KERNEL_VERSION)`
        - Others: -> `None`.
        """

        if self.is_windows:
            _os_major = platform.release()
            _os_build = platform.version()
            _os_update = get_regedit(
                "HKLM",
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
                "DisplayVersion",
            )

            return (platform.system(), _os_major, _os_update, _os_build)

        elif self.is_linux:
            kernel = subprocess.check_output(["uname", "-r"], text=True).strip()
            kernel_version = kernel.split("-")[0]

            with open("/etc/os-release") as f:
                _f = f.read().splitlines()
                for _line in _f:
                    _name_match = re.match(r'^NAME="?(.*?)"?$', _line)
                    _version_match = re.match(r'^VERSION_ID="?(.*?)"?$', _line)

                    if _name_match:
                        _LINUX_DISTRO_NAME = _name_match.group(1)
                    if _version_match:
                        _LINUX_DISTRO_VERSION = _version_match.group(1)

            return (
                _LINUX_DISTRO_NAME,
                _LINUX_DISTRO_VERSION,
                "GNU/Linux",
                kernel_version,
            )
        else:
            pass

    def device_cpu_status(self):
        """
        **Warning:** This function may broken in Cluster systems.
        Return CPU status, include its name, architecture, total cpu count.
        -> `(CPU_NAME, CPU_ARCH, CPU_CORES)`
        """

        import os, platform, subprocess, re

        if self.is_windows:
            _cpu_name = get_regedit(
                "HKLM",
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                "ProcessorNameString",
            )
            _cpu_arch = platform.machine()
            _cpu_core = os.cpu_count()

            return (_cpu_name, _cpu_core, _cpu_arch)

        elif self.is_linux:
            _cpu_name = (
                re.search(
                    r"^\s*Model name:\s*(.+)$",
                    subprocess.run(
                        ["lscpu"], capture_output=True, text=True, check=True
                    ).stdout,
                    re.MULTILINE,
                )
                .group(1)
                .strip()
            )
            _cpu_arch = platform.machine()
            _cpu_core = os.cpu_count()

            return (_cpu_name, _cpu_core, _cpu_arch)

        else:
            # <ADD BSD/Intel_MAC ???>
            ...
            pass

    def device_gpu_list(self):
        """
        Returns a list contains GPU info tuple on Windows platform.
        If on Linux, we skip test as return `None`.
        - Windows: `[(GPU_NUM, GPU_NAME, GPU_VRAM), (...), ...]` or `None`
        - Linux: `None`
        - Others: `None`
        """
        if self.is_windows:
            gpu_status_list = []

            gpu_result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "Name"],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            gpu_result_lines = [
                line.strip() for line in gpu_result.stdout.splitlines() if line.strip()
            ]
            gpu_count = len(gpu_result_lines[1:]) if len(gpu_result_lines) > 1 else []

            for i in range(0, gpu_count):
                _GPU_REG_KEY = str(
                    r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
                    + f"\\000{i}\\"
                )
                _GPU_CORE_NAME = RepoInfo.amdgpu_llvm_target(
                    get_regedit("HKLM", _GPU_REG_KEY, "DriverDesc")
                )
                if _GPU_CORE_NAME != "Microsoft Basic Display Adapter":
                    _GPU_VRAM = get_regedit(
                        "HKLM", _GPU_REG_KEY, "HardwareInformation.qwMemorySize"
                    )

                    if _GPU_VRAM == None:
                        gpu_status_list.append((i, f"{_GPU_VRAM}", None))
                    else:
                        gpu_status_list.append(
                            (i, f"{_GPU_CORE_NAME}", float(_GPU_VRAM / (1024**3)))
                        )
            return gpu_status_list
        else:
            return None

    def device_dram_status(self):
        """
        Analyze Device's DRAM Status. Both on Windows and Linux returns a tuple.
        - Windows: `(DRAM_PHYS_TOTAL, DRAM_PHYS_AVAIL, DRAM_VITURAL_AVAIL)`
        - Linux:   `(MEM_PHYS_TOTAL , MEM_PHYS_AVAIL , MEM_SWAP_AVAIL)`
        -  Others: `None`.
        """
        if self.is_windows:
            import ctypes

            class memSTAT(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem_status = memSTAT()
            mem_status.dwLength = ctypes.sizeof(memSTAT())

            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status))

            MEM_PHYS_TOTAL, MEM_PHYS_AVAIL, MEM_VITURAL_AVAIL = (
                float(mem_status.ullTotalPhys / (1024**3)),
                float(mem_status.ullAvailPhys / (1024**3)),
                float(mem_status.ullAvailPageFile / (1024**3)),
            )

            return (MEM_PHYS_TOTAL, MEM_PHYS_AVAIL, MEM_VITURAL_AVAIL)
        elif self.is_linux:
            import re

            with open("/proc/meminfo", "r") as f:
                _f = f.read().splitlines()
                for line in _f:
                    mem_tol = re.search(r"^MemTotal:\s+(\d+)\s+kB$", line)
                    mem_avl = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", line)
                    mem_swp = re.search(r"^SwapTotal:\s+(\d+)\s+kB$", line)

                    if mem_tol:
                        MEM_PHYS_TOTAL = float(mem_tol.group(1)) / (1024**2)
                    elif mem_avl:
                        MEM_PHYS_AVAIL = float(mem_avl.group(1)) / (1024**2)
                    elif mem_swp:
                        MEM_SWAP_AVAIL = float(mem_swp.group(1)) / (1024**2)

            return (MEM_PHYS_TOTAL, MEM_PHYS_AVAIL, MEM_SWAP_AVAIL)
        else:
            return None
        ...

    def device_disk_status(self):
        """
        Return a tuple with Disk Total/Usage messages.
        `(DISK_DEVICE, DISK_MOUNT_POINT, DISK_TOTAL_SPACE, DISK_USAGE_SPACE, DISK_AVAIL_SPACE, DISK_USAGE_RATIO)`
        - `DISK_DEVICE`: Returns `str`. The device "contains this repo" name and its mounting point.
         - Windows: Returns a Drive Letter. eg `F:/` or `F:`
         - Linux: Returns disk's mounted device name and its mounting point. eg `/dev/sdd at: /`
        - `DISK_REPO_POINT`: Returns `str`. RepoInfo current repo abs path.
        - `DISK_TOTAL_SPACE`: Returns `float`. Current repo stored disk's total space.
        - `DISK_USAGE_SPACE`: Returns `float`. Current repo stored disk's used space.
        - `DISK_AVAIL_SPACE`: Returns `float`. Current repo stored disk's avail space.
        - `DISK_USAGE_RATIO`: Returns `float`. Current repo stored disk's current usage percentage.
        """

        import os, subprocess
        from shutil import disk_usage

        if self.is_windows:
            repo_path = RepoInfo.repo()
            repo_disk = PureWindowsPath(__file__).drive

            DISK_TOTAL_SPACE, DISK_USAGE_SPACE, DISK_AVAIL_SPACE = disk_usage(repo_disk)

            DISK_USAGE_RATIO = float(DISK_USAGE_SPACE / DISK_TOTAL_SPACE) * 100.0
            DISK_TOTAL_SPACE = DISK_TOTAL_SPACE / (1024**3)
            DISK_USAGE_SPACE = DISK_USAGE_SPACE / (1024**3)
            DISK_AVAIL_SPACE = DISK_AVAIL_SPACE / (1024**3)

            return (
                repo_path,
                repo_disk,
                round(DISK_TOTAL_SPACE, 2),
                round(DISK_USAGE_SPACE, 2),
                round(DISK_AVAIL_SPACE, 2),
                round(DISK_USAGE_RATIO, 2),
            )

        elif self.is_linux:
            repo_path = RepoInfo.repo()
            DISK_STATUS_QUERY = (
                subprocess.run(
                    ["df", "-h", os.getcwd()],
                    capture_output=True,
                    check=True,
                    text=True,
                )
                .stdout.strip()
                .splitlines()[1]
                .split()
            )

            DISK_MOUNT_POINT, DISK_MOUNT_DEVICE = (
                DISK_STATUS_QUERY[-1],
                DISK_STATUS_QUERY[0],
            )
            DISK_TOTAL_SPACE, DISK_USAGE_SPACE, DISK_AVAIL_SPACE = disk_usage(
                DISK_MOUNT_POINT
            )
            DISK_USAGE_RATIO = DISK_USAGE_SPACE / DISK_TOTAL_SPACE * 100
            DISK_TOTAL_SPACE = DISK_TOTAL_SPACE / (1024**3)
            DISK_USAGE_SPACE = DISK_USAGE_SPACE / (1024**3)
            DISK_AVAIL_SPACE = DISK_AVAIL_SPACE / (1024**3)

            return (
                repo_path,
                f"{DISK_MOUNT_DEVICE} at: {DISK_MOUNT_POINT}",
                round(DISK_TOTAL_SPACE, 2),
                round(DISK_USAGE_SPACE, 2),
                round(DISK_AVAIL_SPACE, 2),
                round(DISK_USAGE_RATIO, 2),
            )

    def device_file_system(self):
        """
        Return a list of strings containing the output of `df -h`.
        `df -h` shows information about all file systems of the systems, including
        filesystem name, size, available and used storage, and where it is mounted
        """

        try:
            proc = subprocess.run(
                ["df", "-h"], capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError:
            return [cstring(f"[!] Command 'df' not found.", "warn")]

        return proc.stdout.splitlines()

    def device_ccache_system(self):
        """
        Returns a pair of string lists that contain information about the ccache on
        the system. If ccache is not installed, strings stating this are returned.

        CCACHE_STAT (= [0]) contains general status about ccache
        CCACHE_CONFIG ( = [1]) contains the ccache config
        """

        ccache = []
        try:
            proc = subprocess.run(
                ["ccache", "-s", "-v"], capture_output=True, text=True, check=True
            )

            ccache.append([proc.stdout.splitlines()])
        except subprocess.CalledProcessError:
            ccache.append(["Ccache not detected!"])
            ccache.append([""])
            return ccache

        proc = subprocess.run(
            ["ccache", "--show-config"], capture_output=True, text=True, check=True
        )
        ccache.append([proc.stdout.splitlines()])

        return ccache

    def device_python_list(self):
        """
        Return a list of strings containing the output of `df -h`.
        `df -h` shows information about all file systems of the systems, including
        filesystem name, size, available and used storage, and where it is mounted
        """

        proc = subprocess.run(
            ["pip", "list", "--format=freeze"],
            capture_output=True,
            text=True,
            check=True,
        )

        return proc.stdout.splitlines()

    @property
    def CCACHE_STAT(self):
        return self._device_ccache_stat[0]

    @property
    def CCACHE_CONFIG(self):
        return self._device_ccache_stat[1]

    @property
    def python(self):
        return self._py

    @property
    def CPU_NAME(self):
        return self._device_cpu_stat[0]

    @property
    def CPU_CORE(self):
        return self._device_cpu_stat[1]

    @property
    def CPU_ARCH(self):
        return self._device_cpu_stat[2]

    if is_windows:

        @property
        def cl(self):
            return self._cl

        @property
        def VSVER(self):
            """
            Define Visual Studio version.
            """
            _VSVER_NUM = os.getenv("VisualStudioVersion")
            return (
                None if (_VSVER_NUM is None or _VSVER_NUM == "") else float(_VSVER_NUM)
            )

        @property
        def VS20XX(self):
            """
            Define Visual Studio yearly version.
            """
            if self.VSVER is not None:
                match self.VSVER:
                    case 17.0:
                        return "VS2022"
                    case 16.0:
                        return "VS2019"
                    case 15.0:
                        return "VS2017"
                    case 14.0:
                        return "VS2015"
                    case _:
                        return "Legacy"
            else:
                return False

        @property
        def VC_VER(self):
            """Define MSVC build is v14X version."""
            _cl = self.cl.exe
            _vc_ver = os.getenv("VCToolsVersion")

            if _vc_ver == "14.43.34808":
                return "v143"
            elif _vc_ver == "14.29.30133":
                return "v142"
            elif _vc_ver == "14.16.27023":
                return "v141"
            elif (
                _cl
                == r"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\cl.exe"
            ):
                return "v140"
            else:
                return None

        @property
        def VS20XX_INSTALL_DIR(self):
            """Find Environment Variable `VSINSTALLDIR` to show the current installed VS20XX location."""
            _dir = os.getenv("VSINSTALLDIR")
            return _dir if _dir is not None else None

        @property
        def VC_SDK(self):
            """Define Visual Studio current used Windows SDK version."""
            _sdk = os.getenv("WindowsSDKVersion")
            return _sdk.replace("\\", "") if _sdk is not None else None

        @property
        def VC_HOST(self):
            """Find VC++ compiler host environment."""
            _host = os.getenv("VSCMD_ARG_HOST_ARCH")
            return _host if _host is not None else None

        @property
        def VC_TARGET(self):
            """Find VC++ compiler target environment."""
            _target = os.getenv("VSCMD_ARG_TGT_ARCH")
            return _target if _target is not None else None

        @property
        def MAX_PATH_LENGTH(self):
            """Find if Windows machine enabled Long PATHs."""
            if self.is_windows:
                _long_path = get_regedit(
                    "HKLM",
                    r"SYSTEM\CurrentControlSet\Control\FileSystem",
                    key="LongPathsEnabled",
                )
                return True if _long_path == 1 else False
            else:
                return None

    @property
    def ROCM_HOME(self):
        return os.getenv("ROCM_HOME")

    @property
    def HIP_PATH(self):
        return os.getenv("HIP_PATH")

    #
    # All _STATUS should return List[str]
    #

    # Define OS configuration.
    @property
    def OS_STATUS(self):
        if self.is_windows:
            return [self.OS_NAME]
        elif self.is_linux:
            return (
                [f"{self.OS_NAME}, GNU/Linux {self.OS_KERNEL} (WSL2)"]
                if self.is_WSL2
                else [f"{self.OS_NAME}, GNU/Linux {self.OS_KERNEL}"]
            )
        else:
            pass

    # Define CPU status.
    @property
    def CPU_STATUS(self):
        return [f"{self.CPU_NAME} ({self.CPU_ARCH})", f"Logical cores: {self.CPU_CORE}"]

    # Define GPU list status.
    @property
    def GPU_STATUS(self):
        if self._device_gpu_list is not None:
            _gpulist = []
            for _gpu_info in self._device_gpu_list:
                _gpu_num, _gpu_name, _gpu_vram = _gpu_info
                if _gpu_vram == None:
                    _gpulist += [f"GPU {_gpu_num}: \t{_gpu_name}"]
                else:
                    _gpulist += [
                        f"GPU {_gpu_num}: \t{_gpu_name} ({_gpu_vram:.2f}GB VRAM)"
                    ]
            return _gpulist

        elif self.is_linux:
            return [cstring(f"[!] Skip GPU detection on Linux.", "warn")]

    # Define Memory Device status.
    @property
    def MEM_STATUS(self):
        if self.is_windows:
            return [
                f"Total Physical Memory: {self._device_dram_stat[0]:.2f} GB",
                f"Avail Physical Memory: {self._device_dram_stat[1]:.2f} GB",
                f"Avail Virtual  Memory: {self._device_dram_stat[2]:.2f} GB",
            ]
        elif self.is_linux:
            return [
                f"Total Physical Memory: {self._device_dram_stat[0]:.2f} GB",
                f"Avail Physical Memory: {self._device_dram_stat[1]:.2f} GB",
                f"Avail   Swap   Memory: {self._device_dram_stat[2]:.2f} GB",
            ]
        else:
            pass

    # Define Disk Device status.
    @property
    def DISK_STATUS(self):
        return [
            f"Disk Total Space: {self._device_disk_stat[2]} GB",
            f"Disk Avail Space: {self._device_disk_stat[4]} GB",
            f"Disk Used  Space: {self._device_disk_stat[3]} GB",
            f"Disk Usage: {self._device_disk_stat[5]} %",
            f"Current Repo path: {self._device_disk_stat[0]}, Disk Device: {self._device_disk_stat[1]}",
        ]

    @property
    def FILE_SYSTEM_STATUS(self):
        return self._device_file_system_list

    @property
    def ENV_STATUS(self):
        if self.is_windows:
            return [
                f"Python ENV: {self.python.exe} ({self.python.ENV_TYPE})",
                f"Visual Studio: {self.VS20XX}",
                f"Cygwin: {self.is_cygwin}",
                f"MSYS2: {self.is_msys2}",
            ]
        elif self.is_linux:
            return [
                f"Python3 VENV: {self.python.exe} ({self.python.ENV_TYPE}) | WSL2: {self.is_WSL2}"
            ]
        else:
            return [f"Python3 VENV: {self.python.exe} ({self.python.ENV_TYPE}) "]

    @property
    def SDK_STATUS(self):
        if self.is_windows:

            _vs20xx_stat = self.VS20XX if self.VS20XX else "Not Detected"
            _vs20xx_sdk = self.VC_SDK if self.VC_SDK else "Not Detected"

            _hipcc_stat = self.HIP_PATH if self.HIP_PATH else "Not Detected"
            _rocm_stat = self.ROCM_HOME if self.ROCM_HOME else "Not Detected"

            return [
                f"Visual Studio:  {_vs20xx_stat} | Host/Target: {self.VC_HOST} --> {self.VC_TARGET}",
                f"VC++ Compiler:  {self.cl.version}",
                f"VC++ UCRT:      {_vs20xx_sdk}",
                f"AMD HIP SDK:    {_hipcc_stat}",
                f"AMD ROCm:       {_rocm_stat}",
            ]

    @property
    def CCACHE_STATUS(self):
        ccache_stats_cfg_list = [
            *self.CCACHE_STAT[0],
            f"-------------",
            f"Config:",
            f"-------------",
            *self.CCACHE_CONFIG[0],
        ]
        return ccache_stats_cfg_list

    @property
    def PYTHON_LIST(self):
        return self._py_list

    def format_status(self, title: str, content: List[str], longest_status=14) -> str:
        indent_title = lambda x: f"{' ':8}" + x.ljust(longest_status + 1)

        indent_str = ""

        for idx, line in enumerate(content):
            if idx == 0:
                indent_str += indent_title(title) + f"{line}\n"
            else:
                indent_str += indent_title("") + f"{line}\n"

        return indent_str

    def section_bar(self, title: str):
        longest_word = "TheRock build pre-diagnosis script completed in xxxx seconds"
        tabs = 3 * 4  # original amount of tabs in the center section

        # colored font needs extra treatement as len() returns length including ansi chars
        uncolored_string = cstring_strip_color(title)
        tabs += len(title) - len(uncolored_string)

        return (
            f"{' ':8}==================="
            + f"{title}".center(len(longest_word) + tabs)
            + "==================="
        )

    @property
    def summary(self):
        states = [
            ("OS:", self.OS_STATUS),
            ("CPU:", self.CPU_STATUS),
            ("GPU:", self.GPU_STATUS),
            ("RAM:", self.MEM_STATUS),
            ("STORAGE:", self.DISK_STATUS),
            ("FILE SYSTEM:", self.FILE_SYSTEM_STATUS),
        ]

        if self.is_windows:
            states_win = [
                ("ENV:", self.ENV_STATUS),
                ("SDK:", self.SDK_STATUS),
                ("MAX_PATH_ENABLED:", ["True"] if self.MAX_PATH_LENGTH else ["False"]),
            ]

            states += states_win

        # have ccache at the end as it has many lines of output
        states += [("CCACHE:", self.CCACHE_STATUS)]

        print(self.section_bar("Build Environment Summary"))
        print("")

        longest_state = 0

        for state, content in states:
            if len(state) > longest_state:
                longest_state = len(state)

        for state, content in states:
            line_end = "\n"
            if state == "OS:" or state == "GPU:":
                line_end = ""

            print(
                self.format_status(state, content, longest_status=longest_state),
                end=line_end,
            )

    @property
    def python_list(self):
        print(self.section_bar("Python List"))
        print()
        print(self.format_status("", self.PYTHON_LIST, longest_status=-1))
        print()
        print(self.section_bar("End Python List"))
