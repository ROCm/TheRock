from __future__ import annotations
from collections.abc import Iterable
from typing import Optional, Union, Literal, overload, Any, TYPE_CHECKING
import re
from warnings import warn as warning
from textwrap import dedent
import sys

import utils.config as config
from .status import HINT, WARNING, FATAL

# =========================================================================
# 關鍵修改 1: 使用 TYPE_CHECKING 引入 status 類別
# 這樣可以讓 IDE 知道型別，但在執行時不會跑這段，避免循環引用。
# =========================================================================
if TYPE_CHECKING:
    from .status import _SuccessType, _HintType, _WarningType, _FatalType


# ... (ColorString class 定義保持不變) ...
class ColorString:
    def __init__(self, text: str, rgb: Optional[tuple[int, int, int]] = None) -> None:
        self.content = text
        self.rgb = rgb

        if rgb:
            r, g, b = rgb
            self._formatted = f"\033[38;2;{r};{g};{b}m{text}\033[0m"
        else:
            self._formatted = text

    def __str__(self) -> str:
        return self._formatted if not config.DISABLE_ANSI_COLOR else self.content

    def __repr__(self) -> str:
        return f"ColorString(text={self.content!r}, rgb={self.rgb})"

    def __add__(self, other: Any) -> str:
        return str(self) + str(other)

    def __radd__(self, other: Any) -> str:
        return str(other) + str(self)


_NAMED_COLORS = {
    "ERROR": (255, 61, 61),
    "WARNING": (184, 166, 48),
    "SUCCESS": (6, 171, 80),
    "HINT": (115, 201, 201),
}

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def _parse_color(v: Any) -> tuple[int, int, int]:
    """
    解析顏色輸入。
    支援:
    1. 字串 Key ("SUCCESS", "ERROR"...) 或 Hex ("#RRGGBB")
    2. RGB Tuple (255, 255, 255)
    3. [NEW] Status 物件 (透過 repr 對應 Key)
    """

    # 1. 字串處理
    if isinstance(v, str):
        s = v.strip()
        # 先比對 Key (Case-insensitive 處理建議統一，這裡維持原樣)
        # 假設你的 key 都是全大寫，若輸入有小寫需求可 s.upper()
        s_upper = s.upper()
        if s_upper in _NAMED_COLORS:
            return _NAMED_COLORS[s_upper]

        if _HEX_RE.match(s):
            s = s.lstrip("#")
            if len(s) == 3:
                s = "".join(c * 2 for c in s)
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))

    # 2. Iterable (Tuple/List) 處理
    elif isinstance(v, Iterable) and not isinstance(v, (str, bytes)):
        # 這裡要小心，status 物件可能沒有 __iter__，所以這個檢查通常會跳過 status 物件
        # 除非 status 物件也實作了 Iterable
        try:
            rgb = tuple(map(int, v))
            if len(rgb) == 3 and all(0 <= c <= 255 for c in rgb):
                return rgb
        except (ValueError, TypeError):
            pass

    # =====================================================================
    # 關鍵修改 2: 物件對應處理 (Duck Typing)
    # 不 import status 類別，而是檢查傳入物件的 repr() 是否對應有名稱的顏色。
    # status.py 裡的 SUCCESS repr 是 "SUCCESS"，剛好對應 _NAMED_COLORS 的 key。
    # =====================================================================
    try:
        # 使用 repr() 是安全的，因為 status.py 明確定義了 __repr__ 回傳字串
        # 且不會觸發 status.__str__ (避免 status.__str__ 呼叫 cstring 的無限迴圈風險)
        key = repr(v)
        if key in _NAMED_COLORS:
            return _NAMED_COLORS[key]
    except Exception:
        pass

    raise ValueError(f"Invalid color: {v!r}")


# =========================================================================
# 關鍵修改 3: 更新 Overload
# 這裡可以使用上面 TYPE_CHECKING 引入的型別，IDE 會懂，Runtime 不會爆。
# =========================================================================
@overload
def cstring(
    text: Union[str, ColorString],
    color: Literal["SUCCESS", "WARNING", "ERROR", "HINT"],
    /,
) -> ColorString: ...


@overload
def cstring(
    text: Union[str, ColorString], color: Union[Iterable[int], str], /
) -> ColorString: ...


@overload
def cstring(
    text: Union[str, ColorString],
    color: Union[_SuccessType, _HintType, _WarningType, _FatalType],
    /,
) -> ColorString: ...


@overload
def cstring(text: Union[str, ColorString], color: None = None, /) -> str: ...


def cstring(text: Any, color: Any = None, /) -> Union[ColorString, str]:

    raw_text = text.content if hasattr(text, "content") else str(text)

    if color is None:
        return raw_text

    try:
        rgb = _parse_color(color)
        return ColorString(raw_text, rgb)
    except (ValueError, TypeError) as e:
        # 這裡建議保留原始錯誤堆疊或是單純 raise
        raise e


@overload
def message(text: str, /) -> None: ...
@overload
def message(mode: Literal["NOTICE"], /, text: str) -> None: ...
@overload
def message(mode: Literal["STATUS"], /, text: str, latency: float = 0.05) -> None: ...
@overload
def message(
    mode: Literal["CHECK"],
    /,
    text: str,
    check_result: Union[_HintType, _WarningType, _FatalType],
    latency: float = 0.05,
) -> None: ...
@overload
def message(mode: Literal["REPRINT"], /, text: str) -> None: ...
@overload
def message(mode: Literal["HINT", "WARNING", "ERROR"], /, text: str) -> None: ...
@overload
def message(mode: Union[_HintType, _WarningType, _FatalType], /, text: str) -> None: ...
@overload
def message(mode: Literal["DEPRECATED"], /, text: str) -> None: ...
@overload
def message(mode: Literal["FATAL_ERROR"], /, text: str) -> None: ...


def message(
    mode: Union[
        Literal[
            "NOTICE",
            "REPRINT",
            "STATUS",
            "CHECK",
            "HINT",
            "WARNING",
            "ERROR",
            "DEPRECATED",
            "FATAL_ERROR",
        ],
        str,
        Union[_HintType, _WarningType, _FatalType],
    ],
    text: str = None,
    latency: float = 0.05,
    *,
    check_result: Union[_HintType, _WarningType, _FatalType] = None,
) -> None:

    if text is None:
        text = mode
        mode = "NOTICE"

    if mode in (HINT, WARNING, FATAL):
        if mode is FATAL:
            mode = "ERROR"
        if mode is WARNING:
            mode = "WARNING"
        if mode is HINT:
            mode = "HINT"

    from time import sleep

    if mode == "NOTICE":
        print(text)
    elif mode == "REPRINT":
        print(f"\r{text}", end="\033[K", flush=True)
    elif mode == "STATUS":
        sleep(latency)
        print(f"{f" -- {text}:":<60}", end="", flush=True)
        sleep(latency)
    elif mode == "CHECK":
        _prefx = f"{text:<25}"
        print(f"{_prefx} -- {check_result}", flush=True)
    elif mode in ("HINT", "WARNING", "ERROR"):
        if not config.TOO_LONG_DIDNT_READ:
            print(cstring(text, mode))
    elif mode == "DEPRECATED":
        warning(cstring(text, "WARNING"), DeprecationWarning)
    elif mode == "FATAL_ERROR":
        print(cstring(text, "ERROR"))
        print(cstring("Progress Terminated.", "ERROR"))
        sys.exit(1)
    else:
        unexpect = f"Invalid message mode {mode}."
        raise ValueError(cstring(unexpect, "ERROR"))
