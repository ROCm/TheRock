#!/usr/bin/env python

#
#   Contributed by TaiXeflar and Scott Todd
#   TheRock Project building system pre-build diagnosis script
#   License follows TheRock project
#
#   Variables: "FULL_UPPER_VARIABLE" are been called to CMake-Like style.
#       *CMake                  *Python
#       WIN32 -->               WINDOWS
#       Linux -->               LINUX
#   CMAKE_MAJOR_VERSION
#
#   !  Hint: This script doesn't raise/throw back warnings/errors.
#   This script is for detecting environments use, We do a global scan on all requirements at once.
#   We do not want users have to fix its environment one-by-one and get frustrated,so the diagnosis won't throw errors on it.
#   If Users(Yes, It's you! maybe.) running this script have throwback errors, It must have bug in it.
#       PLZ! Please report it as new issue or open in a new disscus <3
#

import os, platform, subprocess, re, time
from typing import Literal
from shutil import disk_usage, which
import warnings
from os import environ as ENV

warnings.filterwarnings("ignore")

therock_detect_start = time.perf_counter()
os_type = platform.system()
if os_type == "Windows":
    WINDOWS, LINUX = True, False

elif os_type == "Linux":
    try:
        with open("/proc/version", "r") as f:
            version_info = f.read().lower()
            if "microsoft" in version_info:
                if "wsl2" in version_info:
                    LINUX, WSL, WINDOWS = True, True, False
                else:
                    raise f"WSL 1 not supported."
            else:
                LINUX, WSL, WINDOWS = True, False, False

    except FileNotFoundError:
        pass
    else:
        LINUX, WINDOWS = True, False
else:
    raise f"TheRock not support on this platform: {os_type}"

# Find_Program:
# By using where() to find executable and its version number. This works on most general/POSIX program.
# By using which() to find executable. MSVC/ML64/RC not able use `--version` do version query.
def where(exe: str):
    import shutil

    exe_path = shutil.which(exe)
    if exe_path is not None:
        VERSION_QUERY = subprocess.run(
            [f"{exe.lower()}", "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
        VERSION_VALUE = re.search(r"(\d+)\.(\d+)\.(\d+)", VERSION_QUERY)
        MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION = map(int, VERSION_VALUE.groups())
        # print(f"{exe} version: {MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION} ({exe_path})")
        return exe_path, MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION
    else:
        return False


def printc(text, rgb):
    if rgb is None:
        print(f"{text}")
    else:
        r, g, b = rgb
        print(f"\033[38;2;{r};{g};{b}m{text}\033[0m")


def hint(string: str):
    text = f"\033[38;2;150;255;255m{str(string)}\033[0m"
    return text


def warn(string: str):
    text = f"\033[38;2;255;230;66m{str(string)}\033[0m"
    return text


def err(string: str):
    text = f"\033[38;2;255;61;61m{str(string)}\033[0m"
    return text


Color_Hint, Color_Warning, Color_Error = (150, 255, 255), (255, 230, 66), (255, 61, 61)
Emoji_Pass, Emoji_Warning, Emoji_Error = "✅", "⚠️", "❌"

# Define get register key and win32 dll anyways.
RootKeys = Literal["HKEY_LOCAL_MACHINE", "HKLM", "HKEY_CURRENT_USER", "HKCU"]
if WINDOWS:

    def get_regkey(
        root_key: RootKeys = "HKEY_LOCAL_MACHINE", path: str = any, key: str = any
    ):
        """
        Function to get Key-Value in Windows Registry Editor.

        `root_key`: Root Keys or Predefined Keys.\nYou can type-in Regedit style or pwsh style as the choice below:\n
        - `HKEY_LOCAL_MACHINE` or `HKLM` \n
        - `HKEY_CURRENT_USER` or `HKCU` \n
        """
        from winreg import HKEY_LOCAL_MACHINE, HKEY_CURRENT_USER, QueryValueEx, OpenKey
        from typing import Literal

        if root_key in ("HKEY_LOCAL_MACHINE", "HKLM"):
            _ROOT_KEY = HKEY_LOCAL_MACHINE
        elif root_key in ("HKEY_CURRENT_USER", "HKCU"):
            _ROOT_KEY = HKEY_CURRENT_USER
        else:
            raise TypeError("Root Keys' type error.")

        try:
            regedit_val, _ = QueryValueEx(OpenKey(_ROOT_KEY, path), key)
        except FileNotFoundError as e:
            regedit_val = None
        return regedit_val

    def win32_dram_viewer():
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

        return (
            float(mem_status.ullTotalPhys / (1024**3)),
            float(mem_status.ullAvailPhys / (1024**3)),
            float(mem_status.ullAvailPageFile / (1024**3)),
        )


# THEROCK_SOURCE_DIR = f"{os.getcwd()}/../.."
# with open(f"{str(THEROCK_SOURCE_DIR)}/.git/refs/heads/main") as f:
#     THEROCK_GIT_HASH = f.read().strip()
#     THEROCK_GIT_HEAD = THEROCK_GIT_HASH[:7]

if WINDOWS:
    from win32com import client

    OS_TYPE = platform.system()
    OS_BUILD_VER = platform.version()
    OS_MAIN_VER = platform.release()
    OS_MIN_VER = get_regkey(
        "HKLM", r"SOFTWARE\Microsoft\Windows NT\CurrentVersion", "DisplayVersion"
    )

    CPU_NAME = get_regkey(
        "HKLM", r"HARDWARE\DESCRIPTION\System\CentralProcessor\0", "ProcessorNameString"
    )
    CPU_ARCH = platform.machine()
    CPU_CORE = ENV["NUMBER_OF_PROCESSORS"]

    DRAM_PHYSIC_TOTAL, DRAM_PHYSIC_AVAIL, DRAM_VIRTUAL_AVAIL = win32_dram_viewer()

    GPU_COUNT = len(client.GetObject("winmgmts:").InstancesOf("Win32_VideoController"))

    GPU_STATUS_LIST = ""
    for i in range(0, GPU_COUNT):
        _GPU_REG_KEY = str(
            r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
            + f"\\000{i}\\"
        )
        GPU_CORE_NAME = get_regkey("HKLM", _GPU_REG_KEY, "DriverDesc")

        if GPU_CORE_NAME != "Microsoft Basic Display Adapter":
            GPU_VRAM = get_regkey(
                "HKLM", _GPU_REG_KEY, "HardwareInformation.qwMemorySize"
            )
            GPU_STATUS_LIST += f"\n    GPU {i}:\n\tGPU_NAME:    {GPU_CORE_NAME}\n\tGPU_VRAM:    {float(int(GPU_VRAM)/(1024**3)):.2f}(GB)"
        else:
            pass

    repo_path = os.path.abspath(os.getcwd())
    repo_disk = os.path.splitdrive(repo_path)[0]

    DISK_TOTAL_SPACE, DISK_USAGE_SPACE, DISK_AVAIL_SPACE = disk_usage(repo_disk)

    DISK_USAGE_RATIO = float(DISK_USAGE_SPACE / DISK_TOTAL_SPACE) * 100.0
    DISK_TOTAL_SPACE = DISK_TOTAL_SPACE / (1024**3)
    DISK_USAGE_SPACE = DISK_USAGE_SPACE / (1024**3)
    DISK_AVAIL_SPACE = DISK_AVAIL_SPACE / (1024**3)

elif LINUX:
    LINUX_DISTRO_QUERY = subprocess.run(
        ["cat", "/etc/os-release"], capture_output=True, text=True, check=True
    ).stdout
    LINUX_DIST_NAME = (
        re.search(r'^NAME="?(.*?)"?$', LINUX_DISTRO_QUERY, re.MULTILINE)
        .group(1)
        .strip()
    )
    LINUX_DIST_VERSION = (
        re.search(r'^VERSION_ID="?(.*?)"?$', LINUX_DISTRO_QUERY, re.MULTILINE)
        .group(1)
        .strip()
    )
    LINUX_KERNEL_QUERY = subprocess.run(
        ["cat", "/proc/version"], check=True, capture_output=True, text=True
    ).stdout
    LINUX_KERNEL_VALUE = re.search(
        r"Linux version (\d+)\.(\d+)\.(\d+)\.(\d+)", LINUX_KERNEL_QUERY
    )
    LINUX_KERNEL_MAJOR_VER, LINUX_KERNEL_MINOR_VER, _, _ = map(
        int, LINUX_KERNEL_VALUE.groups()
    )
    WSL = True if "microsoft-standard-WSL2" in LINUX_KERNEL_QUERY.lower() else False

    CPU_LSCPU_QUERY = subprocess.run(
        ["lscpu"], capture_output=True, text=True, check=True
    ).stdout
    CPU_ARCH = (
        re.search(r"^\s*Architecture:\s*(.+)$", CPU_LSCPU_QUERY, re.MULTILINE)
        .group(1)
        .strip()
    )
    CPU_NAME = (
        re.search(r"^\s*Model name:\s*(.+)$", CPU_LSCPU_QUERY, re.MULTILINE)
        .group(1)
        .strip()
    )
    CPU_CORE = os.cpu_count()

    MEM_INFO_QUERY = subprocess.run(
        ["cat", "/proc/meminfo"], capture_output=True, text=True, check=True
    ).stdout
    MEM_PHYSIC_TOTAL = (
        re.search(r"^MemTotal:\s+(\d+)\s+kB", MEM_INFO_QUERY, re.MULTILINE)
        .group(1)
        .strip()
    )
    MEM_PHYSIC_AVAIL = (
        re.search(r"^MemAvailable:\s+(\d+)\s+kB", MEM_INFO_QUERY, re.MULTILINE)
        .group(1)
        .strip()
    )
    MEM_SWAP_TOTAL = (
        re.search(r"^SwapTotal:\s+(\d+)\s+kB", MEM_INFO_QUERY, re.MULTILINE)
        .group(1)
        .strip()
    )

    DISK_STATUS_QUERY = (
        subprocess.run(
            ["df", "-h", os.getcwd()], capture_output=True, check=True, text=True
        )
        .stdout.strip()
        .splitlines()[1]
        .split()
    )

    DISK_MOUNT_POINT, DISK_MOUNT_DEVICE = DISK_STATUS_QUERY[-1], DISK_STATUS_QUERY[0]
    DISK_TOTAL_SPACE, DISK_USAGE_SPACE, DISK_AVAIL_SPACE = disk_usage(DISK_MOUNT_POINT)
    DISK_USAGE_RATIO = DISK_USAGE_SPACE / DISK_TOTAL_SPACE * 100
    DISK_TOTAL_SPACE = DISK_TOTAL_SPACE / (1024**3)
    DISK_USAGE_SPACE = DISK_USAGE_SPACE / (1024**3)
    DISK_AVAIL_SPACE = DISK_AVAIL_SPACE / (1024**3)

# Read TheRock current repo commit hash.
THEROCK_SOURCE_DIR = f"{os.getcwd()}/../.."
try:
    with open(f"{str(THEROCK_SOURCE_DIR)}/.git/refs/heads/main") as f:
        THEROCK_GIT_HASH = f.read().strip()
        THEROCK_GIT_HEAD = THEROCK_GIT_HASH[:7]
except FileNotFoundError as e:
    THEROCK_GIT_HEAD = f"Unknown"

# Print AMD arrow logo.
print(
    f"""
{err(" ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪")}
{err("   ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪")}
{err("     ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪")}       {err("AMD TheRock Project")}
{err("                 ▪ ▪ ▪")}
{err("     ▪           ▪ ▪ ▪")}       Build Environment diagnosis script
{err("   ▪ ▪           ▪ ▪ ▪")}
{err(" ▪ ▪ ▪           ▪ ▪ ▪")}       Version TheRock ({err(THEROCK_GIT_HEAD)})
{err(" ▪ ▪ ▪ ▪ ▪ ▪ ▪   ▪ ▪ ▪")}
{err(" ▪ ▪ ▪ ▪ ▪ ▪       ▪ ▪")}
{err(" ▪ ▪ ▪ ▪ ▪           ▪")}
"""
)


# Print Local Machine status.
if WINDOWS:
    print(
        f"""

    ===========    Build Environment Summary    ===========

    OS Name: {OS_TYPE} {OS_MAIN_VER} {OS_MIN_VER} | Feature Experience Pack {OS_BUILD_VER}

    CPU_NAME:\t{CPU_NAME}({CPU_ARCH})
    CPU_ARCH:\t{CPU_ARCH}
    CPU_CORE:\t{CPU_CORE} Core(s)

    DRAM_Physical_Total_Volume:\t{DRAM_PHYSIC_TOTAL:.2f}(GB)
    DRAM_Physical_Avail_Volume:\t{DRAM_PHYSIC_AVAIL:.2f}(GB)
    DRAM_Virtual_Avail_Volume:\t{DRAM_VIRTUAL_AVAIL:.2f}(GB)
    {GPU_STATUS_LIST}

    Current Physical Disk: {repo_disk.capitalize()}
    Disk Total Space: {DISK_TOTAL_SPACE:.2f}GB
    Disk Avail Space: {DISK_AVAIL_SPACE:.2f}GB
    Disk Usage:       {DISK_USAGE_RATIO:.2f}%
    """,
        end="\n",
    )
elif LINUX:
    print(
        f"""

    OS Name: {LINUX_DIST_NAME} {LINUX_DIST_VERSION} (Linux {LINUX_KERNEL_MAJOR_VER}.{LINUX_KERNEL_MINOR_VER}), WSL2 = {WSL}

    CPU_Name:\t{CPU_NAME} ({CPU_ARCH})
    CPU_ARCH:\t{CPU_ARCH}
    CPU_CORE:\t{CPU_CORE} Core(s)

    DRAM_Physical_Total_Volume:\t{float(int(MEM_PHYSIC_TOTAL)/(1024**2)):.2f}(GB)
    DRAM_Physical_Avail_Volume:\t{float(int(MEM_PHYSIC_AVAIL)/(1024**2)):.2f}(GB)
    DRAM_SWAP_Avail_Volume:\t{float(int(MEM_SWAP_TOTAL)/(1024**2)):.2f}(GB)

    Disk Total Space: {DISK_TOTAL_SPACE:.2f}(GB)
    Disk Avail Space: {DISK_AVAIL_SPACE:.2f}(GB)
    Disk Usage Space: {DISK_USAGE_SPACE:.2f}(GB)
    Disk Usage Ratio:   {DISK_USAGE_RATIO:.2f}%
    Disk Current Reference:
        Disk Current Location: {os.getcwd()}
        Disk Mounted Device: {DISK_MOUNT_DEVICE}
        Disk Mounted Location: {DISK_MOUNT_POINT}
    """
    )


# Start Diagnosis Platform.

if CPU_ARCH.upper() in ("AMD64", "X86_64", "X86-64", "X64", "Intel 64"):
    print(f"{Emoji_Pass} [CPU] CPU supported architecture: {CPU_ARCH}")
else:
    print(f"{Emoji_Error} [CPU] CPU unsupported architecture: {err(CPU_ARCH)}")
    printc(
        f"""
    TheRock project is not supported for host CPU Architecture: {CPU_ARCH}.
    TheRock is supported for x86-64 arch host/target CPU (AMD64/Intel 64).
    Sorry for any inconvenience.

    traceback: Unsupported Host/Target archtecture: {CPU_ARCH}
    """,
        Color_Error,
    )

# Start check programs with required tools, toolchain, compiler infastructures.
## Git version control.

print(
    f"""
    ===========    Toolchain/Environment detection    ===========

    Starting Check your environment. Caution this will effect your
      build environment is stable or not.
"""
)

program = "Git".lower()
path = where(program)
if path:
    GIT_INSTALL_DIR, GIT_MAJOR_VERSION, GIT_MINOR_VERSION, GIT_PATCH_VERSION = where(
        program
    )
    print(
        f"{Emoji_Pass} [{program}] Found {program} {GIT_MAJOR_VERSION}.{GIT_MINOR_VERSION}.{GIT_PATCH_VERSION} at: {hint(GIT_INSTALL_DIR)}"
    )
else:
    print(f"{Emoji_Error} [{program}] Cannot found {err(program)} program.")
    printc(
        f"""
    We need a git program to clone TheRock and its side projects.
    On Windows platform: Please install it from Git for Windows installer (You can found at here: https://git-scm.com/downloads/win),
        or via chocolatey/winget package manager.
      PS > winget install --id Git.Git -e --source winget
      PS > choco install git
    On Linux platform: Please use your Linux distro's official package manager apt/dnf/pacman to install it.
      sh $ <pkg_manager> install git

    traceback: No available {program} installed or {program} not in PATH
    """,
        Color_Error,
    )

## Git Large File System.
program = "Git-LFS".lower()
path = where(program)
if path:
    (
        GITLFS_INSTALL_DIR,
        GITLFS_MAJOR_VERSION,
        GITLFS_MINOR_VERSION,
        GITLFS_PATCH_VERSION,
    ) = where(program)
    print(
        f"{Emoji_Pass} [{program}] Found {program} {GITLFS_MAJOR_VERSION}.{GITLFS_MINOR_VERSION}.{GITLFS_PATCH_VERSION} at: {hint(GITLFS_INSTALL_DIR)}"
    )
else:
    if WINDOWS:
        print(f"{Emoji_Warning} [{program}] Cannot found {warn(program)} program.")
        printc(
            f"""
    Through we dont know if this will effect or not, but we encourage you have Git Large Filesystem (Git-LFS).
    On Windows platform: You can install it from Git-LFS installer (You can found at here: https://git-lfs.com/),
        or via chocolatey/winget package manager.
    PS > winget install --id=GitHub.GitLFS  -e
    PS > choco install git.install -y --params "'/GitAndUnixToolsOnPath'"

    traceback: No available {program} installed or {program} not in PATH
    """,
            Color_Warning,
        )
    elif LINUX:
        print(f"{Emoji_Error} [{program}] Cannot found {err(program)} program.")
        printc(
            f"""
    Git Large File System is required tools on GNU/Linux platform.
    Please use Linux distro's official package manager apt/dnf/pacman to install it:
    sh $ <pkg_manager> install git-lfs

    traceback: No available {program} installed or {program} not in PATH
    """,
            Color_Error,
        )

## Python 3.
program = "Python".lower() if WINDOWS else "python3".lower()
path = where(program.lower())
if path:
    (
        PYTHON_INSTALL_DIR,
        PYTHON_MAJOR_VERSION,
        PYTHON_MINOR_VERSION,
        PYTHON_PATCH_VERSION,
    ) = where(program)
    if {PYTHON_MAJOR_VERSION} == 2:
        print(
            f"{Emoji_Error} [{program}] Found Python {PYTHON_INSTALL_DIR} is Python 2."
        )
        printc(
            f"""
    TheRock Project is not supported by using Python 2.
    Please use Python 3 version that is supported in bug-fix/security-fix status; We recommends Python 3 version >= 3.9.
    """,
            Color_Error,
        )
    if ({PYTHON_MAJOR_VERSION} == 3) and ({PYTHON_MINOR_VERSION} < 8):
        PYTHON_VERSION_NAME = (
            f"{PYTHON_MAJOR_VERSION}.{PYTHON_MINOR_VERSION}.{PYTHON_PATCH_VERSION}"
        )
        print(
            f"{Emoji_Warning} [{program}] Found Python {PYTHON_INSTALL_DIR} is version {warn(PYTHON_VERSION_NAME)}."
        )
        printc(
            f"""
    We found your {program} version is {PYTHON_MAJOR_VERSION}.{PYTHON_MINOR_VERSION}.{PYTHON_PATCH_VERSION} <3.8.
    We also used uv, which uv not support your version.
    Through {PYTHON_MAJOR_VERSION}.{PYTHON_MINOR_VERSION}.{PYTHON_PATCH_VERSION} version may usable, but we don't gurantee this version is stable for TheRock build and PyTorch.
    Please use Python 3 version that is supported in bug-fix/security-fix status; We recommends Python 3 version >= 3.9.
    """,
            Color_Warning,
        )

    if "VIRTUAL_ENV" in ENV:
        print(
            f"{Emoji_Pass} [{program}] Found {program} {PYTHON_MAJOR_VERSION}.{PYTHON_MINOR_VERSION}.{PYTHON_PATCH_VERSION} at: {PYTHON_INSTALL_DIR} (Virtual ENV)"
        )
    else:
        print(
            f"{Emoji_Warning} [{program}] Found {program} {PYTHON_MAJOR_VERSION}.{PYTHON_MINOR_VERSION}.{PYTHON_PATCH_VERSION} at: {PYTHON_INSTALL_DIR} (Global ENV)"
        )
        printc(
            f"""
    We found your have {program} program, but it seems like using Global env.
    Some python packages will include as dependices and might pollute your Global env.
    If you already generated venv, please activate it:
        PS > .venv/Scripts/activate.ps1
        sh $ source .venv/bin/activate
    If you not generate it, we recommend you use 'uv' to create a virtual env and activate it:
        PS > uv venv .venv --python <CPython_VERSION>
        PS > .venv/Scripts/activate.ps1

        sh $ uv venv .venv --python <CPython_VERSION>
        sh $ source .venv/bin/activate

    traceback: Detected {program} not using Virtual ENV or using Global ENV
    """,
            Color_Warning,
        )

else:
    print(f"{Emoji_Warning} [{program}] Cannot found {err(program)} program.")
    printc(
        f"""
    We cannot found {program} on your device.
    TheRock and PyTorch have some python dependices will be installed by requirements.txt.
    Create a Python with venv and activate it.

    traceback: No available {program} installed or {program} not in PATH
    """,
        Color_Warning,
    )

## Chocolatey (Unnecessary).
if WINDOWS:
    program = "choco".lower()
    path = where(program)
    if path:
        (
            CHOCO_INSTALL_DIR,
            CHOCO_MAJOR_VERSION,
            CHOCO_MINOR_VERSION,
            CHOCO_PATCH_VERSION,
        ) = where(program)
        print(
            f"{Emoji_Pass} [{program}] Found {program} {CHOCO_MAJOR_VERSION}.{CHOCO_MINOR_VERSION}.{CHOCO_PATCH_VERSION} at: {hint(CHOCO_INSTALL_DIR)}"
        )
    else:
        print(f"{Emoji_Warning} [{program}] Cannot found {warn(program)} program.")
    printc(
        f"""
    {Emoji_Warning} If you have already installed tools, or prefer manually install toolchain one-by-one with specified installers, you can ignore this message.

    We cannot find Chocolatey on your device.
    Chocolatey is a package manager for Windows platform, almost required packages here can install via choco command.
    You can install chocolatey via PowerShell command.
    PS > Set-ExecutionPolicy Bypass -Scope Process -Force
         [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
         iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

    traceback: No available {program} installed or {program} not in PATH
    """,
        Color_Warning,
    )

## Astral uv.
program = "uv".lower()
path = where(program)
if path:
    UV_INSTALL_DIR, UV_MAJOR_VERSION, UV_MINOR_VERSION, UV_PATCH_VERSION = where(
        program
    )
    print(
        f"{Emoji_Pass} [{program}] Found {program} {UV_MAJOR_VERSION}.{UV_MINOR_VERSION}.{UV_PATCH_VERSION} at: {hint(UV_INSTALL_DIR)}"
    )
else:
    print(f"{Emoji_Error} [{program}] Cannot found {err(program)} program.")
    printc(
        f"""
    We need uv to manage/accelerate generate Python venv for building TheRock and its side projects(PyTorch).
    On Windows platform: You can install it via your Global Env Python, chocolatey or Astral official powershell install script:
      PS > powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
      PS > pip install uv
      PS > choco install uv
    On Linux platform: Please use Astral official website shell script to install it.
      sh $ wget -qO- https://astral.sh/uv/install.sh | sh

    traceback: No available {program} installed or {program} not in PATH
    """,
        Color_Error,
    )

## Compilers.
# > Windows: MSVC toolchain with Visual Studio 20XX.
# > GNU/Linux/WSL2: GNU GCC toolchain.
if WINDOWS:

    # Find out what Visual Studio version did user installed.
    program = "Visual Studio"
    if "VSINSTALLDIR" in ENV:
        match ENV["VisualStudioVersion"]:
            case "17.0":
                VISUAL_STUDIO_VERSION = "VS2022"
            case "16.0":
                VISUAL_STUDIO_VERSION = "VS2019"
            case "15.0":
                VISUAL_STUDIO_VERSION = "VS2017"
            case "14.0":
                VISUAL_STUDIO_VERSION = "VS2015"
            case "12.0":
                VISUAL_STUDIO_VERSION = "VS2012"
        print(
            f"{Emoji_Pass} [{program}] {program} environment detected as {VISUAL_STUDIO_VERSION}"
        )
        if float(ENV["VisualStudioVersion"]) <= 14.0:
            printc(
                f"""
    {Emoji_Warning} We detected your Visual Studio version is {VISUAL_STUDIO_VERSION}.
    We may consider your Visual Studio version and its toolchain could be too old and may not have official support
    after TheRock official release.
    Please consider upgrade your Visual Studio version and toolchain.

    traceback: Visual Studio version {VISUAL_STUDIO_VERSION} too old
    """,
                Color_Warning,
            )

    else:
        print(f"{Emoji_Error} [{program}] {program} environment not detected. ")
        printc(
            f"""
    We detected you're running out of Visual Studio developer environment.
    MSVC compilers needs 'Developer prompt/PowerShell for Visual Studio 20XX' to load its tools and libraries.
    Please re-run a terminal or load ENV variable batch files to add MSVC toolchain.

    traceback: Developer for {program} ENV not detected/activated
    """,
            Color_Error,
        )

    VSCMD_ARG_HOST_ARCH = os.getenv("VSCMD_ARG_HOST_ARCH")
    if VSCMD_ARG_HOST_ARCH != "x64":
        print(
            f"{Emoji_Error} [{VISUAL_STUDIO_VERSION}/@Host] Detected {VISUAL_STUDIO_VERSION} env host is {VSCMD_ARG_HOST_ARCH}."
        )
        printc(
            f"""
    We found your {VISUAL_STUDIO_VERSION} host is {VSCMD_ARG_HOST_ARCH}.
    TheRock build requires host machine is x64(AMD64, Intel 64, x86-64/x86_64).
    Please re-configure your {VISUAL_STUDIO_VERSION} Host env.
    """,
            Color_Error,
        )
    else:
        print(
            f"{Emoji_Pass} [{VISUAL_STUDIO_VERSION}/@Host] Detected {VISUAL_STUDIO_VERSION} env host is {VSCMD_ARG_HOST_ARCH}."
        )

    VSCMD_ARG_TGT_ARCH = os.getenv("VSCMD_ARG_TGT_ARCH")
    if VSCMD_ARG_TGT_ARCH != "x64":
        print(
            f"{Emoji_Warning} [{VISUAL_STUDIO_VERSION}/@Target] Detected {VISUAL_STUDIO_VERSION} env target {VSCMD_ARG_TGT_ARCH}."
        )
        printc(
            f"""
    We found your {VISUAL_STUDIO_VERSION} is targeting to {VSCMD_ARG_TGT_ARCH}.
    Targeting to non x64/x86_64/x86-64/AMD64/Intel 64 platforms may causing the build binaries/runtimes/libraries in malfunctioning.
    Please re-configure your targeting device again.
    """,
            Color_Warning,
        )
    else:
        print(
            f"{Emoji_Pass} [{VISUAL_STUDIO_VERSION}/@Target] Detected {VISUAL_STUDIO_VERSION} env targeting {VSCMD_ARG_TGT_ARCH}."
        )

    if "VISUAL_STUDIO_VERSION" in globals():
        if "VCToolsInstallDir" and "VCToolsVersion" in ENV:
            VCToolsVersion = os.getenv("VCToolsVersion")
            VCToolsInstallDir = os.getenv("VCToolsInstallDir")
            print(
                f"{Emoji_Pass} [{VISUAL_STUDIO_VERSION}/MSVC] Detected MSVC version {VCToolsVersion}."
            )
            program = "cl".lower()
            path = which(f"{VCToolsInstallDir}/bin/hostx64/x64/{program}.exe")
            if path:
                print(
                    f"{Emoji_Pass} [{VISUAL_STUDIO_VERSION}/MSVC] Found MSVC C/C++ compiler driver at: {path}."
                )
            else:
                print(
                    f"{Emoji_Error} [{VISUAL_STUDIO_VERSION}/MSVC] Cannot found MSVC VC++ compiler."
                )
                printc(
                    f"""
    We found your MSVC install directory, but missing MSVC C/C++ compiler driver '{program}.exe'.
    It seems your VC++ toolchain installation is broken. Please re-install {VISUAL_STUDIO_VERSION}.
    traceback: No available {program} installed or {program} not in PATH
    """,
                    Color_Error,
                )
            program = "ml64".lower()
            path = which(f"{VCToolsInstallDir}/bin/hostx64/x64/{program}.exe")
            if path:
                print(
                    f"{Emoji_Pass} [{VISUAL_STUDIO_VERSION}/MSVC] Found MSVC Macro Assembler at: {path}."
                )
            else:
                print(
                    f"{Emoji_Error} [{VISUAL_STUDIO_VERSION}/MSVC] Cannot find MSVC Macro Assembler."
                )
                printc(
                    f"""
    We found your MSVC install directory, but missing MSVC Macro Assembler '{program}.exe'.
    This will effect on we compile TheRock sub-project [AMD-LLVM].
    It seems your VC++ toolchain installation is broken. Please re-install {VISUAL_STUDIO_VERSION}.

    traceback: No available {program} installed or {program} not in PATH
    """,
                    Color_Error,
                )
            program = "link".lower()
            path = which(f"{VCToolsInstallDir}/bin/hostx64/x64/{program}.exe")
            if path:
                print(
                    f"{Emoji_Pass} [{VISUAL_STUDIO_VERSION}/MSVC] Found MSVC Linker at: {path}."
                )
            else:
                print(
                    f"{Emoji_Error} [{VISUAL_STUDIO_VERSION}/MSVC] Cannot find MSVC Linker."
                )
                printc(
                    f"""
    We found your MSVC install directory, but missing MSVC Linker '{program}.exe'.
    It seems your VC++ toolchain installation is broken. Please re-install {VISUAL_STUDIO_VERSION}.

    traceback: No available {program} installed or {program} not in PATH
    """,
                    Color_Error,
                )
            program = "lib".lower()
            path = which(f"{VCToolsInstallDir}/bin/hostx64/x64/{program}.exe")
            if path:
                print(
                    f"{Emoji_Pass} [{VISUAL_STUDIO_VERSION}/MSVC] Found MSVC Archiver at: {path}."
                )
            else:
                print(
                    f"{Emoji_Error} [{VISUAL_STUDIO_VERSION}/MSVC] Cannot find MSVC Archiver."
                )
                printc(
                    f"""
    We found your MSVC install directory, but missing MSVC Archiver '{program}.exe'.
    It seems your VC++ toolchain installation is broken. Please re-install {VISUAL_STUDIO_VERSION}.

    traceback: No available {program} installed or {program} not in PATH
    """,
                    Color_Error,
                )
            WIN_SDK_QUERY = ENV.get("WindowsSDKVersion")
            WIN_SDK_VERSION = re.fullmatch(r"(\d+\.\d+\.\d+\.\d+)\\?", WIN_SDK_QUERY)
            if WIN_SDK_QUERY is not None:
                print(
                    f"{Emoji_Pass} [{VISUAL_STUDIO_VERSION}/WindowsSDK] Found Windows SDK version {WIN_SDK_VERSION.group(1)}."
                )
            else:
                print(
                    f"{Emoji_Error} [{VISUAL_STUDIO_VERSION}/WindowsSDK] No available Windows SDK installed."
                )
                printc(
                    f"""
    We can't found any available Windows SDK library. Your {VISUAL_STUDIO_VERSION} VC++ compoments installation may broken.
    At least need one Windows SDK installed on your device.
    Please re-check your Visual Studio install, ensure you have selected at least one suitable SDK version.

    traceback: ENV variable "WindowsSDKVersion" invalid or empty.
    """,
                    Color_Error,
                )
            program = "rc".lower()
            path = which(program)
            print
            if path:
                print(
                    f"{Emoji_Pass} [{VISUAL_STUDIO_VERSION}/WindowsSDK] Found Windows Resource Compiler at: {path}"
                )
            else:
                print(
                    f"{Emoji_Error} [{VISUAL_STUDIO_VERSION}/WindowsSDK] Cannot Find Windows Resource Compiler."
                )
                printc(
                    f"""
    We can'n find any availiable Windows Resource Compiler `rc.exe` in PATH.
    Resource Compiler is combined with Windows SDK. Please check if your Windows SDK is installed correct.
    """,
                    Color_Error,
                )
        else:
            print(
                f"{Emoji_Error} [{VISUAL_STUDIO_VERSION}/MSVC] MSVC Toolchain not found."
            )
            printc(
                f"""
        We cannot found a existed MSVC compiler toolchain anyways.
        Please re-check your {VISUAL_STUDIO_VERSION} installation or re-install {VISUAL_STUDIO_VERSION}.

        traceback: {VISUAL_STUDIO_VERSION} has no known existed MSVC compoments
        """,
                Color_Error,
            )

if LINUX:
    program = "gcc".lower()
    path = where(program)
    if path:
        (
            GCC_INSTALL_DIR,
            GCC_MAJOR_VERSION,
            GCC_MINOR_VERSION,
            GCC_PATCH_VERSION,
        ) = where(program)
        print(
            f"{Emoji_Pass} [GCC/{program}] Found GCC Compiler {program} {GCC_MAJOR_VERSION}.{GCC_MINOR_VERSION}.{GCC_PATCH_VERSION} at: {GCC_INSTALL_DIR}."
        )
    else:
        print(f"{Emoji_Error} [GCC/{program}] Cannot found GCC Compiler {program}.")
        printc(
            f"""
    We cannot found GNU C compiler `{program}`. Please check your compiler installation.
    If you not install it yet, please install it via your linux distro's package manager:
        sh $ apt install gcc g++
        sh $ dnf install gcc g++
        sh $ pacman -S gcc g++

    traceback: No available {program} installed or {program} not in PATH
    """,
            Color_Error,
        )

    program = "g++".lower()
    path = where(program)
    if path:
        (
            GXX_INSTALL_DIR,
            GXX_MAJOR_VERSION,
            GXX_MINOR_VERSION,
            GXX_PATCH_VERSION,
        ) = where(program)
        print(
            f"{Emoji_Pass} [GCC/{program}] Found GCC Compiler {program} {GXX_MAJOR_VERSION}.{GXX_MINOR_VERSION}.{GXX_PATCH_VERSION} at: {GXX_INSTALL_DIR}."
        )
    else:
        print(f"{Emoji_Error} [GCC/{program}] Cannot found GCC Compiler {program}.")
        printc(
            f"""
    We cannot found GNU C++ compiler `{program}`. Please check your compiler installation.
    If you not install it yet, please install it via your linux distro's package manager:
        sh $ apt install gcc g++
        sh $ dnf install gcc g++
        sh $ pacman -S gcc g++

    traceback: No available {program} installed or {program} not in PATH
    """,
            Color_Error,
        )

    program = "as".lower()
    path = which(program)
    if path:
        print(f"{Emoji_Pass} [GCC/{program}] Found GCC Assembler {program} at: {path}.")
    else:
        print(f"{Emoji_Error} [GCC/{program}] Cannot found GCC Assembler `{program}`.")
        printc(
            f"""
    We cannot found GNU Assembler `{program}`. Please check your compiler installation.
    If you not install it yet, please install it via your linux distro's package manager:
        sh $ apt install gcc g++
        sh $ dnf install gcc g++
        sh $ pacman -S gcc g++

    traceback: No available {program} installed or {program} not in PATH
    """,
            Color_Error,
        )

    program = "ar".lower()
    path = which(program)
    if path:
        print(f"{Emoji_Pass} [GCC/{program}] Found GCC archiver {program} at: {path}")
    else:
        print(f"{Emoji_Error} [GCC/{program}] Cannot found GCC archiver `{program}`.")
        printc(
            f"""
    We cannot found GNU Archiver `{program}`. Please check your compiler installation.
    If you not install it yet, please install it via your linux distro's package manager:
        sh $ apt install gcc g++
        sh $ dnf install gcc g++
        sh $ pacman -S gcc g++

    traceback: No available {program} installed or {program} not in PATH
    """,
            Color_Error,
        )

    program = "ld".lower()
    path = which(program)
    if path:
        print(f"{Emoji_Pass} [GCC/{program}] Found GCC linker {program} at: {path}")
    else:
        print(f"{Emoji_Error} [GCC/{program}] Cannot found GCC linker `{program}`.")
        printc(
            f"""
    We cannot found GNU Archiver `{program}`. Please check your compiler installation.
    If you not install it yet, please install it via your linux distro's package manager:
        sh $ apt install gcc g++
        sh $ dnf install gcc g++
        sh $ pacman -S gcc g++

    traceback: No available {program} installed or {program} not in PATH
    """,
            Color_Error,
        )
    # Assume we have installed binutils with gcc/g++ already.

program = "CMake".lower()
path = where(program)
if path:
    (
        CMAKE_INSTALL_DIR,
        CMAKE_MAJOR_VERSION,
        CMAKE_MINOR_VERSION,
        CMAKE_PATCH_VERSION,
    ) = where(program)
    if CMAKE_MAJOR_VERSION >= 4:
        print(
            f"{Emoji_Warning} [{program}] Found CMake version {CMAKE_MAJOR_VERSION}.{CMAKE_MINOR_VERSION}.{CMAKE_PATCH_VERSION} at: {hint(CMAKE_INSTALL_DIR)}"
        )
        printc(
            f"""
    We found your {program} is too new to TheRock project requires.
    Please downgrade it to CMake Major version less than 4 to avoid CMake behavior compatibility.
    You can use Visual Studio's CMake or Strawberry Perl's CMake program.

    traceback: {program} version too new may cause {program} behaviors have potential errors
    """,
            Color_Warning,
        )
    else:
        print(
            f"{Emoji_Pass} [{program}] Found CMake {CMAKE_MAJOR_VERSION}.{CMAKE_MINOR_VERSION}.{CMAKE_PATCH_VERSION} at: {hint(CMAKE_INSTALL_DIR)}"
        )

else:
    print(f"{Emoji_Error} [{program}] Cannot found CMake program.")
    printc(
        f"""
    We cannot found {program}.
    At here, we need CMake to configure TheRock (and PyTorch) build and generate build rules.
    On Windows platform:  Please check your {program} program is in PATH.
        If not installed, you can install it via Visual Studio Installer, check "CMake for Windows" option in VC++ install selection.
        Or install it via Strawberry Perl, or Chocolatey. We recommend not install it via pip or uv.
    On Linux platform: Please check your {program} program is in PATH. If not installed, please install it via your linux distro's package manager:
      sh $ apt install cmake
      sh $ dnf install cmake
      sh $ pacman -S cmake

    traceback: No available {program} installed or {program} not in PATH
    """,
        (255, 61, 61),
    )

program = "Ninja".lower()
path = where(program)
if path:
    (
        NINJA_INSTALL_DIR,
        NINJA_MAJOR_VERSION,
        NINJA_MINOR_VERSION,
        NINJA_PATCH_VERSION,
    ) = where(program)
    if (NINJA_MAJOR_VERSION, NINJA_MINOR_VERSION) == (1, 11):
        print(
            f"{Emoji_Warning} [{program}] Found {program} {NINJA_MAJOR_VERSION}.{NINJA_MINOR_VERSION}.{NINJA_PATCH_VERSION} at: {hint(NINJA_INSTALL_DIR)}"
        )
        printc(
            f"""
    We found your {program} is {NINJA_MAJOR_VERSION}.{NINJA_MINOR_VERSION}.
    TheRock contributors has confirmed a bug that CMake will endless re-run configuration by using this {program} version.
    Please upgrade your {program} version greater then 1.11 to avoid this problem.

    traceback: {program} version may hit unexpected errors
    """,
            Color_Warning,
        )
    else:
        print(
            f"{Emoji_Pass} [{program}] Found {program} {NINJA_MAJOR_VERSION}.{NINJA_MINOR_VERSION}.{NINJA_PATCH_VERSION} at: {hint(NINJA_INSTALL_DIR)}"
        )
else:
    print(f"{Emoji_Error} [{program}] Cannot found Generator {program}.")
    printc(
        f"""
    We cannot found {program} generator.
    Ninja is a small, focus on speed generator to build projects. At here, Ninja is the official supported generator.
    On Windows platform:  Please check your {program} program is in PATH.
        If not installed, you can install it via Visual Studio that comes with CMake support.
        Or install it via Strawberry Perl, or Chocolatey. We recommend not install it via pip or uv.
    On Linux platform: Please check your {program} program is in PATH. If not installed, please install it via your linux distro's package manager:
        sh $ apt install ninja-build
        sh $ dnf install ninja-build
        sh $ pacman -S ninja-build

    traceback: No available {program} installed or {program} not in PATH
    """,
        Color_Error,
    )

program = "CCache".lower()
path = where(program)
if path:
    (
        CCACHE_INSTALL_DIR,
        CCACHE_MAJOR_VERSION,
        CCACHE_MINOR_VERSION,
        CCACHE_PATCH_VERSION,
    ) = where(program)

    print(
        f"{Emoji_Pass} [{program}] Found CCache {CCACHE_MAJOR_VERSION}.{CCACHE_MINOR_VERSION}.{CCACHE_PATCH_VERSION} at: {hint(CCACHE_INSTALL_DIR)}"
    )
else:
    print(f"{Emoji_Warning} [{program}] Cannot found Compiler cache '{program}'.")
    printc(
        f"""
    We cannot found Compiler Launcher {program}.
    CCache is a external compiler tool, records compile cache and re-use to skip same build process.
    On Windows platform:  Please check your {program} program is in PATH.
        If not installed, you can install it via Strawberry Perl, or Chocolatey. We recommend not install it via pip or uv.
    On Linux platform: Please check your {program} program is in PATH. If not installed, please install it via your linux distro's package manager:
      sh $ apt install ccache
      sh $ dnf install ccache
      sh $ pacman -S ccache

    traceback: No available {program} installed or {program} not in PATH
    """,
        Color_Warning,
    )

if WINDOWS:
    MAX_PATH_LEN_STATUS = get_regkey(
        "HKLM", r"SYSTEM\CurrentControlSet\Control\FileSystem", key="LongPathsEnabled"
    )
    MAX_PATH_LEN_ENABLED = True if MAX_PATH_LEN_STATUS == 1 else False
    if MAX_PATH_LEN_ENABLED:
        print(
            f"{Emoji_Pass} [MAX_PATH_ENABLE] Status: Long PATH on Windows has already Enabled."
        )
    else:
        print(
            f"{Emoji_Warning} [MAX_PATH_ENABLE] Status: Long PATH on Windows has not disabled."
        )
        printc(
            f"""
    We found you have not enable Windows Long PATH support yet.
    Please enable this feature via one of these solution:
        > Using Registry Editor(regedit)
        > Using Group Policy, then restart your device.

    traceback: Windows Enable Long PATH support feature is Disabled
    \t Registry Key Hint: HKLM:/SYSTEM/CurrentControlSet/Control/FileSystem LongPathsEnabled = 0 (DWORD)
    """,
            Color_Warning,
        )

if WINDOWS:
    DISK_STATUS_MSG = f"\tDrive {repo_disk.capitalize()} | Total: {DISK_TOTAL_SPACE:.2f}(GB) | Used: {DISK_USAGE_SPACE:2f}(GB) | Avail: {DISK_AVAIL_SPACE:.2f}(GB) | Usage: {DISK_USAGE_RATIO:.2f}%"
    if (DISK_AVAIL_SPACE >= 128) and (DISK_USAGE_RATIO < 80):
        print(f"{Emoji_Pass} [DISK_SPACE] Disk space check passing.")
        printc(
            f"""
    We've checked the workspace disk {repo_disk.capitalize()} is available to build TheRock (and PyTorch).
    TheRock builds may needs massive storage for the build, and we recommends availiable disk space needs 128GB and usage not over 80%.

    {DISK_STATUS_MSG}
    """,
            Color_Hint,
        )
    else:
        print(f"{Emoji_Warning} [DISK_SPACE] Disk space ckeck attention.")
        printc(
            f"""
    We've checked the workspace disk {repo_disk.capitalize()} available space could be too small to build TheRock (and PyTorch).
    TheRock builds may needs massive storage for the build, and we recommends availiable disk space with 128GB and usage not over 80%.

    {DISK_STATUS_MSG}

    traceback: Disk space tool small or disk usage too high
    """,
            Color_Warning,
        )

elif LINUX:
    DISK_STATUS_MSG = f"\tDrive {DISK_MOUNT_DEVICE} (Mounting at: {DISK_MOUNT_POINT}) | Drive current location: {os.getcwd()} \n\tDrive Total: {DISK_TOTAL_SPACE:.2f}(GB) | Used: {DISK_USAGE_SPACE:2f}(GB) | Avail: {DISK_AVAIL_SPACE:.2f}(GB) | Usage: {DISK_USAGE_RATIO:.2f}%"
    if (DISK_AVAIL_SPACE >= 128) and (DISK_USAGE_RATIO < 80):
        print(f"{Emoji_Pass} [DISK_SPACE] Disk space check passing.")
        printc(
            f"""
    We've checked the workspace disk {DISK_MOUNT_DEVICE} is available to build TheRock (and PyTorch).
    TheRock builds may needs massive storage for the build, and we recommends availiable disk space needs 128GB and usage not over 80%.

    {DISK_STATUS_MSG}
    """,
            Color_Hint,
        )
    else:
        print(f"{Emoji_Warning} [DISK_SPACE] Disk space ckeck attention.")
        printc(
            f"""
    We've checked the workspace disk {DISK_MOUNT_DEVICE} available space could be too small to build TheRock (and PyTorch).
    TheRock builds may needs massive storage for the build, and we recommends availiable disk space with 128GB and usage not over 80%.

    {DISK_STATUS_MSG}

    traceback: Disk space tool small or disk usage too high
    """,
            Color_Warning,
        )

therock_detect_terminate = time.perf_counter()
therock_detect_time = therock_detect_terminate - therock_detect_start
print(
    f"""
    ===========    TheRock build pre-diagnosis script completed in {hint(f"{therock_detect_time:.2f}")} seconds    ===========
"""
)
