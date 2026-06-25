from pathlib import Path
import argparse
import sys

from . import config
from .utility_functions import *


def args_update():
    class CustomFormatter(argparse.HelpFormatter):
        def __init__(self, prog):
            # 增加 max_help_position 可以讓標籤佔據更多空間而不換行
            super().__init__(prog, max_help_position=35, width=150)

    parser = argparse.ArgumentParser(
        description="TheRock Build Environment pre-Diagnosis system",
        add_help=False,
        prefix_chars="-/",
        formatter_class=CustomFormatter,
    )

    # 這裡的 dest 必須與 config.py 中的變數名稱「完全一致」
    parser.add_argument(
        "--help",
        "-help",
        "-?",
        "/?",
        "/help",
        action="store_true",  # <--- 關鍵修改：攔截系統預設的 help 行為
        help="Show this help message and exit",
    )
    parser.add_argument(
        "--build", type=str, required=True, dest="DEFAULT_TASK", help="Build Tasks"
    )
    parser.add_argument(
        "--ROCM_HOME",
        type=Path,
        required=False,
        dest="ROCM_HOME",
        help="Manually define ROCm install dir",
    )
    parser.add_argument(
        "--tldr",
        "-tldr",
        action="store_true",
        dest="TOO_LONG_DIDNT_READ",
        default=argparse.SUPPRESS,
        help="Disable Test fail error",
    )
    parser.add_argument(
        "--cls",
        action="store_true",
        dest="CLEAR_HOST",
        help="Enable clear terminal and output",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        dest="DISABLE_ANSI_COLOR",
        default=argparse.SUPPRESS,
        help="Disable ANSI escape coded colot string output",
    )
    parser.add_argument(
        "--ccache",
        type=str,
        choices=["sv", "pv", "xv", "verbose"],
        dest="DISPLAY_CCACHE_MODE",
        default=argparse.SUPPRESS,
        help="Display CCache status",
    )
    parser.add_argument(
        "--sccache",
        action="store_true",
        dest="DISPLAY_SCCACHE_MODE",
        default=argparse.SUPPRESS,
        help="Display SCCache status",
    )
    parser.add_argument(
        "--llvm",
        action="store_true",
        dest="DISPLAY_LLVM",
        help="Display LLVM configuration",
    )
    parser.add_argument(
        "--df",
        action="store_true",
        dest="DISPLAY_DISKFILESYSTEM",
        help="Display Disk FileSystem",
    )
    parser.add_argument(
        "--no-diag",
        action="store_true",
        dest="DISABLE_TEST",
        help="System analysisonly",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="JSON_OUTPUT",
        help="dump a JSON format diagnosis result",
    )

    help_flags = {"--help", "-help", "-?", "/?", "/help"}

    if any(flag in sys.argv for flag in help_flags):
        # 1. 提早手動覆寫 config 變數
        if "--cls" in sys.argv:
            config.CLEAR_HOST = True  # <--- 關鍵：先讓 config 知道要清空畫面

        if "--no-color" in sys.argv:
            config.DISABLE_ANSI_COLOR = True

        # 2. 執行動作
        if config.CLEAR_HOST:
            clear()  # 現在 clear() 內部的判斷式就能順利通過了

        logo()
        print(parser.format_help())
        sys.exit(0)

    args = parser.parse_args()

    # 走訪解析出的參數，如果有輸入值，就覆寫到記憶體中的 config 模組
    for key, value in vars(args).items():
        if value is not None:
            # 這裡只會修改記憶體中的 config 狀態，絕對不會改動 config.py 檔案
            setattr(config, key, value)
