from typing import overload, Literal, List, Optional, Union, Any

from utils.status import NOTDEFINED
from .color_string import cstring
from .system_function import os_type

if os_type() == "Windows":
    import winreg


@overload
def regedit(
    root_key: Literal["HKLM", "HKCU"], path: str, /, *, key_name: str
) -> str: ...


@overload
def regedit(
    root_key: Literal["HKLM", "HKCU"],
    path: str,
    /,
) -> List[str]: ...


def regedit(
    root_key: Literal["HKLM", "HKCU"] = "HKLM",
    path: str = "",
    /,
    *,
    key_name: Optional[
        str
    ] = None,  # 預設改為 None 以觸發 List 模式，或保留 "" 為 Default Value
) -> Union[str, List[str], Any]:  # Any for NOTDEFINED
    """
    ## Get-Regedit (Enhanced)
    - 若指定 `key_name` (str): 回傳該數值 (Value)。
    - 若 `key_name` 為 None: 回傳該路徑下的所有子機碼名稱 (Subkeys List)。
    """

    # 對應 Root Key
    if root_key in ("HKEY_LOCAL_MACHINE", "HKLM"):
        _root = winreg.HKEY_LOCAL_MACHINE
    elif root_key in ("HKEY_CURRENT_USER", "HKCU"):
        _root = winreg.HKEY_CURRENT_USER
    else:
        raise TypeError(f"Unsupported Registry Root Key: {root_key}.")

    try:
        with winreg.OpenKey(_root, path) as hkey:

            # --- 分歧點：讀取數值 vs 列舉機碼 ---

            if key_name is not None:
                # 模式 A: 讀取數值 (Original)
                value, _ = winreg.QueryValueEx(hkey, key_name)
                return str(value)

            else:
                # 模式 B: 列舉子機碼 (List Subkeys)
                num_subkeys, _, _ = winreg.QueryInfoKey(hkey)
                subkeys_list = [winreg.EnumKey(hkey, i) for i in range(num_subkeys)]
                return subkeys_list

    except (FileNotFoundError, OSError) as e:
        # 這裡可以根據需求決定是否要 print log，或是直接回傳錯誤狀態
        # print(cstring(f"Hit {e} during runtime process.", "warn"))
        return NOTDEFINED
    except PermissionError:
        print(cstring(f"Hit No enough permission error.", "warn"))
        return NOTDEFINED
