from __future__ import annotations
from typing import Iterable, Tuple, Union, Optional, Literal, Any, overload
import re, os
from itertools import zip_longest
from collections.abc import Iterable as ABCIterable
from pathlib import Path
import shutil
from .status import NOTDEFINED, NOTFOUND

import re


class VersionNum:
    def __init__(self, version_input):
        self.valid = False
        self.major = 0
        self.minor = 0
        self.patch = 0
        self.suffix = ""
        self.fullname = str(version_input)

        if isinstance(version_input, str):
            self._parse_string(version_input)
        elif isinstance(version_input, (list, tuple)):
            self._parse_sequence(version_input)
        elif isinstance(version_input, VersionNum):
            self.valid = version_input.valid
            self.major, self.minor, self.patch, self.suffix = (
                version_input.major,
                version_input.minor,
                version_input.patch,
                version_input.suffix,
            )
        else:
            self.fullname = str(version_input)

        if not self.valid:
            # 如果正則匹配失敗，至少保留完整名稱，不要變成 0.0.0
            self.major = self.minor = self.patch = 0

    def _parse_string(self, v_str):
        # 讓 minor, patch 與 suffix 都變成可選，且支援 X.Y
        match = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?(.*)", v_str)
        if match:
            self.valid = True
            self.major = int(match.group(1) or 0)
            self.minor = int(match.group(2) or 0)
            self.patch = int(match.group(3) or 0)
            self.suffix = match.group(4).strip(".- ")
        self.fullname = v_str

    def _parse_sequence(self, seq):
        # 支援 tuple 長度不足 3 的情況 (例如來自 re.groups() 的 ('0', '28'))
        if not seq:
            return
        self.valid = True
        self.major = int(seq[0]) if len(seq) > 0 and seq[0] is not None else 0
        self.minor = int(seq[1]) if len(seq) > 1 and seq[1] is not None else 0
        self.patch = int(seq[2]) if len(seq) > 2 and seq[2] is not None else 0

    @property
    def verTuple(self):
        return (self.major, self.minor, self.patch)

    # 核心比較邏輯
    def __lt__(self, other):
        other = VersionNum(other)  # 自動轉換輸入
        return self.verTuple < other.verTuple

    def __eq__(self, other):
        other = VersionNum(other)
        return self.verTuple == other.verTuple

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self) -> str:
        # 這會印出 VersionNum('1.2.3')，引號會自動處理
        return self.__str__()

    def __format__(self, format_spec: str) -> str:
        # 如果使用者沒有指定格式 (例如 f"{v}")，format_spec 會是空字串
        if not format_spec:
            return str(self)

        # 這裡直接使用 python 內建的 format 函數
        # 它會自動看懂 "<10" 這種語法
        return format(str(self), format_spec)


def VERSION(
    obj: Union[str, VersionNum],
    op: Literal["=", "≠", "!=", "<", "<=", ">", ">="],
    compare: Union[str, VersionNum],
    /,
    *,
    blacklist: Iterable[str | VersionNum] = None,
    fuzzy: bool = False,
) -> bool:

    if obj is NOTDEFINED or obj is NOTFOUND:
        return False

    v1, v2 = VersionNum(obj), VersionNum(compare)
    ops = {
        "=": v1 == v2,
        "≠": v1 != v2,
        "!=": v1 != v2,
        "<": v1 < v2,
        "<=": v1 <= v2,
        ">": v1 > v2,
        ">=": v1 >= v2,
    }

    # 1. 基礎比較
    result = ops.get(op, False)

    # 2. 如果基礎比較通過，且有黑名單，則進一步檢查
    if result and blacklist is not None:
        return VERSION_BLACKLIST(v1, blacklist, fuzzy=fuzzy)

    return result


def VERSION_IN_RANGE(
    min: str | VersionNum,
    op1: Literal["<", "<="],
    v: str | VersionNum,
    op2: Literal["<", "<="],
    max: str | VersionNum,
    /,
    *,  # 強制關鍵字參數，避免混淆
    blacklist: Iterable[str | VersionNum] = None,
    fuzzy: bool = False,
) -> bool:
    """
    判斷 v 是否在 min 與 max 之間，並可選排除黑名單內的版本。
    """

    if v is NOTDEFINED or v is NOTFOUND:
        return False

    # 1. 先做原本的範圍檢查
    # 注意：這裡直接利用短路邏輯，如果範圍不合直接回傳 False
    left = VERSION(min, op1, v)
    if not left:
        return False

    right = VERSION(v, op2, max)
    if not right:
        return False

    # 2. 如果有設定黑名單，再檢查是否命中黑名單
    if blacklist is not None:
        # VERSION_BLACKLIST 回傳 True 代表「安全 (Pass)」，False 代表「命中黑名單 (Fail)」
        return VERSION_BLACKLIST(v, blacklist, fuzzy=fuzzy)

    return True


def VERSION_EXCLUDE_RANGE(
    v: str | VersionNum,
    op1: Literal["<", "<="],
    min_v,
    op2: Literal[">", ">="],
    max_v,
    /,
    *,
    blacklist: Iterable[str | VersionNum] = None,
    fuzzy: bool = False,
):

    if v is NOTDEFINED or v is NOTFOUND:
        return False
    # (v, ">=", max, "<=", min) -> v > b and v < a
    # 1. 原本邏輯：在 min 之下 或 在 max 之上
    in_safe_zone = VERSION(v, op1, min_v) or VERSION(v, op2, max_v)

    if not in_safe_zone:
        return False

    # 2. 黑名單過濾
    if blacklist is not None:
        return VERSION_BLACKLIST(v, blacklist, fuzzy=fuzzy)

    return True


def VERSION_WHITELIST(
    v: str | VersionNum,
    find_list: Iterable[str],
    /,
    *,
    compatibility: Literal["STRICT", "MINOR", "MAJOR", "FUZZY"] = "STRICT",
) -> bool:

    if v is NOTDEFINED or v is NOTFOUND:
        return False

    target = VersionNum(v)

    for raw_item in find_list:
        if compatibility.upper() == "FUZZY":
            if str(raw_item) in target.fullname:
                return True
            continue

        item = VersionNum(raw_item)

        if compatibility.upper() == "STRICT":
            if target == item:
                return True

        elif compatibility.upper() == "MINOR":
            if target.major == item.major and target.minor == item.minor:
                return True

        elif compatibility.upper() == "MAJOR":
            if target.major == item.major:
                return True

    return False


def VERSION_BLACKLIST(
    v: str | VersionNum,
    prohibited: Iterable[str | VersionNum],  # 擴充型別提示
    /,
    *,
    fuzzy: bool = False,
):

    if v is NOTDEFINED or v is NOTFOUND:
        return False

    target = VersionNum(v)

    for item in prohibited:
        if fuzzy:
            # Fuzzy 模式通常比較適合字串比對
            if str(item) in target.fullname:
                return False
        else:
            # 利用 VersionNum 的 __eq__ 來處理 (str vs VersionNum) 的自動轉型
            # 這樣就算 blacklist 傳入 ["3.11.0", VersionNum("3.12.1")] 也能通
            if target == item:
                return False
    return True


@overload
def STREQUAL(obj1: str, obj2: str) -> bool: ...
@overload
def STREQUAL(obj1: VersionNum, obj2: VersionNum) -> bool: ...


def STREQUAL(obj1: Union[str, VersionNum], obj2: Union[str, VersionNum]):

    if obj1 is NOTDEFINED or obj1 is NOTFOUND:
        return False

    if isinstance(obj1, VersionNum) or isinstance(obj2, VersionNum):
        return VersionNum(obj1) == VersionNum(obj2)
    return str(obj1) == str(obj2)


@overload
def STRMATCH(obj: str, find: str) -> bool: ...
@overload
def STRMATCH(obj: str, find: Iterable[str] | list[str] | Tuple[str]) -> bool: ...


def STRMATCH(obj: str, find: str | Iterable[str] | list[str] | Tuple[str]) -> bool:
    target_str = str(obj)
    if isinstance(find, (list, tuple)):
        return any(str(keyword) in target_str for keyword in find)
    return str(find) in target_str


#   ================================================================
#       Experimental EXISTS()


class _ExistsResult:
    def __init__(self, value: Any, found: bool, fallback: Any):
        self._value = value
        self._found = found
        self._fallback = fallback

    def __bool__(self) -> bool:
        return self._found

    def __enter__(self) -> Any:
        return self._value if self._found else self._fallback

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@overload
def EXISTS(*, file: Union[str, Path]) -> _ExistsResult: ...


@overload
def EXISTS(*, env: str) -> _ExistsResult: ...


@overload
def EXISTS(obj: Any, attr: str, /) -> _ExistsResult: ...


def EXISTS(
    obj: Any = None,
    attr: Optional[str] = None,
    /,
    *,
    file: Optional[Union[str, Path]] = None,
    env: Optional[str] = None,
) -> _ExistsResult:

    if obj is not None and attr is not None:
        if hasattr(obj, attr):
            return _ExistsResult(getattr(obj, attr), True, NOTDEFINED)
        return _ExistsResult(None, False, NOTDEFINED)

    if env is not None:
        val = os.environ.get(env)
        if val is not None:
            return _ExistsResult(val, True, NOTDEFINED)
        return _ExistsResult(None, False, NOTDEFINED)

    if file is not None:
        f_str = str(file)

        path_res = (
            Path(f_str).as_posix()
            if os.path.exists(f_str)
            else Path(shutil.which(f_str)).resolve().as_posix()
        )
        if path_res is not None:
            return _ExistsResult(path_res, True, NOTFOUND)
        return _ExistsResult(None, False, NOTFOUND)

    return _ExistsResult(None, False, NOTDEFINED)
