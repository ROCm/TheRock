from __future__ import annotations
from typing import Literal, Optional, Union, Tuple
import subprocess
import sys
import io
import platform

# Needed to be able to print the AMD logo (RepoInfo.__logo__())
if platform.system() == "Windows":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Define Color string print.
def cstring(
    msg: Union[str],
    color: Union[
        Optional[Literal["err", "warn", "hint", "pass"]],
        Tuple[int, int, int],
    ]
    | None = None,
) -> str:
    """
    ## Color String
    Returns with ANSI escape code formated string, with colors by (R, G, B).

    This feature passed on Linux Terminal, Windows Terminal, VSCode Terminal, and VSCode Jupyter Notebook.

    ### Usage
    `<STR_VAR> = cstring(string, color)`
    - msg: `str` type.
    - color: A user specified `tuple` with each value Ranged from `0` ~ `255` `(R, G, B)`.\t
    ```
    >>> your_text = cstring(msg="AMD RADEON RX 7800XT", color=(255, 0, 0))
    >>> your_text
    ```
    - If color's RGB not passed will be full white. Color also can be these keywords:
        - "err"
        - "warn"
        - "pass"
    """

    if isinstance(color, tuple):
        r, g, b = color
    else:
        match color:
            case "err":
                r, g, b = (255, 61, 61)
            case "warn":
                r, g, b = (184, 166, 48)
            case "hint":
                r, g, b = (115, 201, 201)
            case "pass":
                r, g, b = (6, 161, 60)
            case _:
                r, g, b = (255, 255, 255)

    return f"\033[38;2;{r};{g};{b}m{msg}\033[0m"


def cstring_strip_color(colored_string: str) -> str:
    """
    Strips color from a string and returns it.
    Immediately returns the string if no color is found.

    This is needed if one wants to know the correct length of a colored string,
    as len(colored_string) also counts the ansi characters describing the color.
    """
    import re

    while "\033[38" in colored_string:
        colored_string = re.sub(
            r"\033\[38;2;[0-9;]*;[0-9;]*;[0-9;]*m", "", colored_string
        )
        colored_string = re.sub(r"\033\[0m", "", colored_string)

    return colored_string


def get_regedit(
    root_key: Literal[
        "HKEY_LOCAL_MACHINE", "HKLM", "HKEY_CURRENT_USER", "HKCU"
    ] = "HKEY_LOCAL_MACHINE",
    path: str = any,
    key: str = any,
):
    """
    ## Get-Regedit
    Function to get Key-Value in Windows Registry Editor.
    `root_key`: Root Keys or Predefined Keys.You can type-in Regedit style or pwsh style as the choice below:
    - `HKEY_LOCAL_MACHINE` with pwsh alias `HKLM`
    - `HKEY_CURRENT_USER` with pwsh alias `HKCU`
    """

    from winreg import HKEY_LOCAL_MACHINE, HKEY_CURRENT_USER, QueryValueEx, OpenKey

    if root_key in ("HKEY_LOCAL_MACHINE", "HKLM"):
        _ROOT_KEY = HKEY_LOCAL_MACHINE
    elif root_key in ("HKEY_CURRENT_USER", "HKCU"):
        _ROOT_KEY = HKEY_CURRENT_USER
    else:
        raise TypeError("Unsupported Registry Root Key")

    try:
        regedit_val, _ = QueryValueEx(OpenKey(_ROOT_KEY, path), key)
    except FileNotFoundError as e:
        regedit_val = None
    return regedit_val


class Emoji:
    Pass = cstring("✓", "pass")
    Warn = cstring("!", "warn")
    Err = cstring("✗", "err")


class RepoInfo:
    """
    ## TheRock class
    AMD ROCm/TheRock project.

    - `head()`: `str`. Returns Repo cloned main's head.
    - `repo()`: `str`. Returns Repo's abs path.

    - `__logo__()`: Advanced Micro Devices Logo. Displays AMD Arrow Logo and current git HEAD.

    ![image](https://upload.wikimedia.org/wikipedia/commons/6/6a/AMD_Logo.png)
    """

    @staticmethod
    def head():
        _head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        return _head

    @staticmethod
    def repo():

        finder = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
        ).stdout.strip()
        return finder

    @staticmethod
    def __logo__(monospace=False):

        """
        ![image](https://upload.wikimedia.org/wikipedia/commons/6/6a/AMD_Logo.png)
        # Advanced Micro Devices Inc.
        """
        if monospace == True:
            print(
                f"""




        {cstring("\t\t\t    # # # # # # # # # # #","err")}
        {cstring("\t\t\t      # # # # # # # # # #","err")}
        {cstring("\t\t\t        # # # # # # # # #","err")}\t  {cstring("AMD ROCm/TheRock Project","err")}
        {cstring("\t\t\t                    # # #","err")}
        {cstring("\t\t\t        #           # # #","err")}\t  Build Environment diagnosis script
        {cstring("\t\t\t      # #           # # #","err")}
        {cstring("\t\t\t    # # #           # # #","err")}\t  Version TheRock (current HEAD: {cstring(RepoInfo.head(), "err")})
        {cstring("\t\t\t    # # # # # # #   # # #","err")}
        {cstring("\t\t\t    # # # # # #       # #","err")}
        {cstring("\t\t\t    # # # # #           #","err")}


        """
            )
        else:
            print(
                f"""




        {cstring("\t\t\t    ◼ ◼ ◼ ◼ ◼ ◼ ◼ ◼ ◼ ◼ ◼","err")}
        {cstring("\t\t\t      ◼ ◼ ◼ ◼ ◼ ◼ ◼ ◼ ◼ ◼","err")}
        {cstring("\t\t\t        ◼ ◼ ◼ ◼ ◼ ◼ ◼ ◼ ◼","err")}\t  {cstring("AMD ROCm/TheRock Project","err")}
        {cstring("\t\t\t                       ◼ ◼ ◼","err")}
        {cstring("\t\t\t        ◼             ◼ ◼ ◼","err")}\t  Build Environment diagnosis script
        {cstring("\t\t\t      ◼ ◼            ◼ ◼ ◼","err")}
        {cstring("\t\t\t    ◼ ◼ ◼           ◼ ◼ ◼","err")}\t  Version TheRock (current HEAD: {cstring(RepoInfo.head(), "err")})
        {cstring("\t\t\t    ◼ ◼ ◼ ◼ ◼ ◼ ◼  ◼ ◼ ◼","err")}
        {cstring("\t\t\t    ◼ ◼ ◼ ◼ ◼ ◼      ◼ ◼","err")}
        {cstring("\t\t\t    ◼ ◼ ◼ ◼ ◼          ◼","err")}


        """
            )

    @staticmethod
    def amdgpu_llvm_target(GPU):
        # Information from https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
        from . import AMDGPU_LLVM_TARGET

        name_to_gfx = {}
        for gfx, names in AMDGPU_LLVM_TARGET._amdgpu.items():
            for name in names:
                name_to_gfx[name] = gfx

        gpu_llvm = f"{GPU} ({name_to_gfx[GPU]})" if GPU in name_to_gfx else GPU

        return gpu_llvm
