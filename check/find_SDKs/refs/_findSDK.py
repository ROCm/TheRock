from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Literal, Any, Tuple, Union, overload

from utils import *
from utils.status import _NotDefinedType, _NotFoundType
from pathlib import Path
import re, json, subprocess as subproc


class FindSDK(ABC):
    """
    FindSDK
    ==================================================

    FindSDK provide a template for any FindSDKs to help TheRock find required/optional system tools,
        and can based on different OS Types to have their detect behavior.

    FindSDK have implemented `__init__()` to automatically gain SDK information. By using subclassing
        method `super().__init__()` to do next dispach_map for POSTing name figure.

     - By define several `__WINDOWS__()`, `__LINUX__()`, `__MACOS__()`, `__BSD__()` explictly to define
        different OS based systems SDK behavior on `__init__()`.
        You can define same methods if program info gripping is same/similar on cross os platform
        program like `git`, `cmake`, `ninja` etc.


     - POST

        We use 'POST' nounce from BIOS/UEFI level 'Self Posting', meaning as SDK info test self feature process.

        Method `__post__()` is a control center to gain `post(Feature)` and its keyword arguments to self
            specified `_post_Feature()`.

        Implementing a `self._dispatch_map` to establish a dict for calling `post(...)`. `post("Feture")`
            will call `__post__()` to find `_post_FEATURE()` via `self._dispatch_map`.

    By re-design the base class to make all Subclassing will be more flexable to future SDK
        requirements if need add into this diagnose system.

    ## Example
    ```
    class FindSubSDK(FindSDK):
        def __init__(self) -> None:
            super().__init__()

            self._dispatch_map: dict[str, Callable] = {
                "post1": self._post_feat1,
                "post2": self._post_feat2,
                ...}

        @property
        def Prop_1(self): ...

        @property
        def Prop_2(self): ...

        def __WINDOWS__(self) -> dict[str, Any]:
            exe = self._find_program(...)
            ver = self._find_version(exe, args=..., vertemp="...", pattern=r"...")
            ...

            return {
                "prop1": ...
                "prop2": ...
                ...
            }

        def __LINUX__(self) -> dict[str, Any]:
            exe = self._find_program(...)
            ver = self._find_version(exe, args=..., vertemp="...", pattern=r"...")
            ...

            return {
                "prop1": ...
                "prop2": ...
                ...
            }

        def _sdk_fn1(self): ...
        def _sdk_fn2(self): ...

        def _post_feat1(self, ..., /, *, options:Literal["foo", "bar"]):  ...
        def _post_feat2(self, ..., /, *, flags:bool=False):  ...

        @overload
        def post(self,
                 kwd:Literal["feat1"],
                 condition:Literal["REQUIRED", "OPTIONAL"]="...",
                 /,
                 *,
                 optionA:Literal["foo", "bar"]=...,
                 optionB:Literal["foo", "bar"]=...,
                 flagX:type=...,
                 flagY:type=...,
                 ...
                 fail_level:Literal["HINT", "WARNING", "ERROR"]=...): ...
        @overload
        def post(self, kwd:Literal["feat2"], ...): ...

        def post(self, kwd:str, condition:str, **kwargs: Any):
            return super().post(kwd, condition=condition, **kwargs)

    ```
    """

    os: Literal["Windows", "Linux", "BSD", "macOS"] = os_type()

    name: str = None

    @abstractmethod
    def __init__(self):

        os = self.os

        info: dict[str, Any]

        message("REPRINT", f"Status: Initializing module {self.__class__.__name__}")

        if os == "Windows":
            info = self.__WINDOWS__()
        if os == "Linux":
            info = self.__LINUX__()
        if os == "BSD":
            info = self.__BSD__()
        if os == "macOS":
            info = self.__MACOS__()

        if os not in ["Windows", "Linux", "BSD", "macOS"]:
            message(
                "FATAL_ERROR",
                f"Found Invalid Platform {os}. If this platform is supported, Please contact to TheRock dev team.",
            )

        if not isinstance(info, dict):
            message(
                "FATAL_ERROR",
                f"self.info initialization recieved type {type(info)}, not dict",
                ValueError,
            )

        self.info = info

        self._dispatch_map: dict[str, Callable]

    def __repr__(self):

        if self.exe is NOTFOUND:
            return f"{self.Version:<10} --> NOTFOUND"
        else:
            return f"{self.Version:<10} --> {self.exe.resolve().as_posix()}"

    def __post__(self, kwd: str, condition: str, **kwargs: Any) -> Any:

        if "_dispatch_map" not in self.__dict__:
            message(
                "FATAL_ERROR",
                f"{self.__name__} have not define self._dispatch_map yet.",
            )

        target = self._dispatch_map.get(kwd)
        if not target:
            message("FATAL_ERROR", f"SDK cannot find '{kwd}' post options.")
        return target(**kwargs)

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return self.__post__(kwd, condition=condition, **kwargs)

    @abstractmethod
    def __WINDOWS__(self) -> dict[str, Any]:
        """
        Specialized method for Windows (Windows NT) only.

        ## Inplemention
        >>> def __WINDOWS__(self):
                ...
        """
        ...

    @abstractmethod
    def __LINUX__(self) -> dict[str, Any]:
        """
        Specialized method for Linux (manyLinux) only.

        ## Inplemention
        >>> def __LINUX__(self):
                ...
        """
        ...

    # @abstractmethod
    def __BSD__(self) -> dict[str, Any]:
        """
        Specialized method for BSD systems only.

        ## Inplemention
        >>> def __BSD__(self):
                ...
        """
        ...

    # @abstractmethod
    def __MACOS__(self) -> dict[str, Any]:
        """
        Specialized method for macOS X (Darwin) only.

        ## Inplemention
        >>> def __MACOS__(self):
                ...
        """
        ...

    @property
    def exe(self) -> Path:
        return self.info["exe"]

    @property
    def Version(self) -> str | VersionNum | _NotDefinedType:
        return self.info["Version"]

    def _find_program(self, *args) -> Path | _NotDefinedType:

        if not args or NOTDEFINED in args or NOTFOUND in args:
            return NOTFOUND

        executable = Path(*args)

        if executable is NOTDEFINED or executable is NOTFOUND:
            return NOTFOUND

        _phony = where(str(executable))
        _dir = Path(_phony).resolve() if _phony else NOTFOUND
        return _dir

    def _find_version(
        self,
        executable: Path | str = NOTFOUND,
        /,
        *,
        args: Literal["--version"] | list[str] | str = "--version",
        input: str = None,
        vertemp: Literal["X.Y.Z", "X.Y"] = None,
        pattern: str = None,
    ):

        if executable is NOTFOUND:
            return NOTDEFINED

        # 1. 定義樣板 Regex (使用 raw string)
        templates = {"X.Y.Z": r"(\d+)\.(\d+)\.(\d+)", "X.Y": r"(\d+)\.(\d+)"}

        # 2. 建立候選策略清單 (Strategy List)
        candidates = []

        # 規則 A: 如果 vertemp 存在，優先加入
        if vertemp is not None:
            if vertemp in templates:
                candidates.append(templates[vertemp])
            else:
                raise ValueError(f"Unknown vertemp: {vertemp}")

        # 規則 B: 如果 pattern 存在，加入 (作為 vertemp 的後備，或是主要策略)
        if pattern is not None:
            candidates.append(pattern)

        # 規則 C: 如果兩者都沒設 (Defaults)，採用自動降級策略：先 X.Y.Z 再 X.Y
        if not candidates:
            candidates.append(templates["X.Y.Z"])
            candidates.append(templates["X.Y"])

        # 3. 執行指令取得輸出
        if isinstance(args, str):
            args = args.split(" ")
        cmd = [executable, *args] if args else [executable]

        try:
            # 這裡建議加上 encoding='utf-8' 或是 errors='ignore' 以防亂碼炸裂
            query_run = subproc.run(
                cmd, input=input, text=True, capture_output=True, check=True
            ).stdout.strip()
        except subproc.SubprocessError:
            # 執行失敗 (例如 flag 不支援)，這邊視需求決定要 raise 還是回傳 NOTDEFINED
            # 根據你的 "raise" 精神，這裡可能也值得丟出錯誤，或是讓它自然 failed
            print(f"Error executing {executable}")
            return NOTDEFINED

        # 4. 依序嘗試匹配
        for pat in candidates:
            # 使用 re.search 尋找
            match = re.search(pat, query_run)
            if match:
                # 抓到了！
                # 優先使用 groups (精確捕獲)，如果沒有 groups 則使用整個字串
                val = match.groups() if match.groups() else match.group(0)
                return VersionNum(val)

        # 5. 都沒抓到 -> Raise Error
        # 這會直接噴出 Traceback，讓你知道是誰、輸出了什麼導致解析失敗
        raise RuntimeError(
            f"Failed to parse version for '{executable}'. \n"
            f"Output was: {query_run[:100]}... \n"
            f"Tried patterns: {candidates}"
        )

    def _find_header(self, header: str, /, hint: re.Pattern | str):
        """Find information with headers and json, etc."""

        file = Path(header).resolve()

        if not file.exists():
            message(
                "FATAL_ERROR",
                f"""File missing during reading.
TheRock cannot find file {file}.
    traceback: Missing required finding file""",
            )

        file_ext = file.as_posix().split("/")[-1].split(".")[-1]

        with open(file, "r", encoding="utf-8") as f:
            content = ""
            if file_ext == "json":
                content = json.loads(f)
            elif file_ext in ("h", "hpp"):
                content = f.loads()

        pat = re.compile(hint)

        ...

    def _compoment_fail(
        self,
        condition: Literal["REQUIRED", "OPTIONAL"],
        status: Literal["HINT", "WARNING", "ERROR"],
        /,
    ):

        if condition == "REQUIRED":
            status = "ERROR"
        elif condition == "OPTIONAL":
            status = "WARNING" if status == "FATAL" else status
        else:
            raise ValueError(f'Invalid compoment check level "{condition}"')

        if status == "ERROR":
            res = FATAL
        elif status == "WARNING":
            res = WARNING
        elif status == "HINT":
            res = HINT
        else:
            raise ValueError(f'Invalid check failed return type "{status}"')

        return res
