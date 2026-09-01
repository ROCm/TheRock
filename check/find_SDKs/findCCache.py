from typing import Literal, Any, overload, Dict
import subprocess
from textwrap import dedent


from utils import *
from utils.compare_functions import *
from utils.status import *
from . import FindSDK


class FindCCache(FindSDK):

    ccache = "ccache"

    def __init__(self):
        super().__init__()

        self.info: dict[Literal["exe", "Version", "Stats"], Any]

        self._dispatch_map = {
            "AVAIL": self._post_ccache_avail,
        }

    @property
    def stats(self):
        stats_content = f"CCache statistics\n"

        stat: dict = self.info["Stats"]

        if stat is NOTDEFINED:
            return ""

        for title, dic_t in stat.items():
            stats_content += f"    [{title}]\n"

            dic_t: dict
            for k, v in dic_t.items():
                k_prefix = f"\t{k}:"
                stats_content += f"{k_prefix:<40} {v}\n"
        return stats_content if config.DISPLAY_CCACHE_MODE else ""

    def __WINDOWS__(self):
        return self._find_ccache()

    def __LINUX__(self):
        return self._find_ccache()

    def _find_ccache(self):
        exe = self._find_program(self.ccache)
        ver = self._find_version(exe)

        stats = self._parse_ccache_output() if exe else NOTDEFINED

        return {"exe": exe, "Version": ver, "Stats": stats}

    def _parse_ccache_output(self):

        match config.DISPLAY_CCACHE_MODE:
            case "sv":
                cmd = ["ccache", "-s", "-v"]
            case "xv":
                cmd = ["ccache", "-x", "-v"]
            case "pv":
                cmd = ["ccache", "-p", "-v"]
            case "verbose":
                cmd = ["ccache", "-x", "-s", "-p", "--print-stats", "-v"]
            case _:
                cmd = ["ccache", "--version"]

        ccache_query = (
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            .stdout.replace("\\", "/")
            .replace('"', "'")
        )

        result = {
            "info": {},
            "config": {},
            "stats": {},
        }

        config_pattern = re.compile(r"^\(([^)]+)\)\s+([a-zA-Z_]+)\s*=\s*(.*)$")
        stats_pattern = re.compile(r"^([a-z_]+)\s+(\d+)$")
        info_pattern = re.compile(r"^\s*([^:]+):\s*(.+)$")

        for line in ccache_query.strip().split("\n"):

            if not line or line.startswith("Scanning...") or line.startswith("(base)"):
                continue

            match_config = config_pattern.match(line)
            if match_config:
                source, key, value = match_config.groups()
                result["config"][key.strip()] = value.strip()
                continue

            match_stats = stats_pattern.match(line)
            if match_stats:
                key, value = match_stats.groups()
                result["stats"][key.strip()] = int(value)
                continue

            match_info = info_pattern.match(line)
            if match_info:
                key, value = match_info.groups()
                result["info"][key.strip()] = value.strip()
                continue

        return result

    @overload
    def post(
        self,
        kwd: Literal["AVAIL"],
        condition: Literal["REQUIRED", "OPTIONAL", "DISABLE"] = "OPTIONAL",
        /,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "HINT",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd.upper(), condition=condition, **kwargs)

    def _post_ccache_avail(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "OPTIONAL",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "HINT",
    ):
        message("STATUS", "Check for Compiler Cache program 'ccache'")

        if condition == "DISABLE":
            result = FATAL if self.exe else SUCCESS
        else:
            if self.exe:
                result = SUCCESS
            else:
                result = self._compoment_fail(condition, fail_level)

        message("CHECK", self.Version, check_result=result)
        if result is not SUCCESS:
            message(result, "Optional ccache not found.")

        return result


class FindSCCache(FindSDK):

    name = "sccache"

    def __init__(self):
        super().__init__()

        self.info: dict[Literal["exe", "Version", "Stats"], Any]

        self._dispatch_map = {"AVAIL": self._post_sccache_avail}

    @property
    def stats(self):

        stats_content = f"SCCache statistics\n"

        stats_dict: dict = self.info["Stats"]

        if stats_dict is NOTDEFINED:
            return ""

        for k, v in stats_dict.items():
            k_prefix = f"\t{k}: "
            stats_content += f"{k_prefix:<40}{v}\n"

        return stats_content if config.DISPLAY_SCCACHE_MODE else ""

    def __WINDOWS__(self):
        return self._find_sccache()

    def __LINUX__(self):
        return self._find_sccache()

    def _find_sccache(self):
        exe = self._find_program("sccache")
        ver = self._find_version(exe, args="--version", vertemp="X.Y.Z")

        if exe:
            query = (
                subprocess.run(
                    ["sccache", "-s"], capture_output=True, text=True, check=True
                )
                .stdout.replace("\\", "/")
                .replace('"', "'")
            )
            conf = self._parse_sccache_stats(query)
        else:
            conf = NOTDEFINED

        return {"exe": exe, "Version": ver, "Stats": conf}

    def _parse_sccache_stats(self, text: str) -> Dict:
        data = {}

        pattern = re.compile(r"^(.*?)\s{2,}(.*)$")

        for line in text.strip().splitlines():
            match = pattern.match(line)
            if match:
                key_raw, val_raw = match.groups()

                key = key_raw.strip().lower().replace(" ", "_").replace("?", "")

                val_str = val_raw.strip()

                if val_str == "-":
                    value = None

                elif val_str.endswith(" s"):
                    try:
                        value = float(val_str.replace(" s", ""))
                    except ValueError:
                        value = val_str

                elif val_str.endswith(" GiB"):
                    try:
                        value = float(val_str.replace(" GiB", ""))
                    except ValueError:
                        value = val_str

                elif val_str.isdigit():
                    value = int(val_str)

                elif val_str.lower() == "yes":
                    value = True
                elif val_str.lower() == "no":
                    value = False

                else:
                    value = val_str.replace(r"\\", "/")

                data[key] = value

        return data

    @overload
    def post(
        self,
        kwd: Literal["AVAIL"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "OPTIONAL",
        /,
        *,
        fail_level: Literal["HINT", "WARNING"] = "HINT",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    def _post_sccache_avail(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "OPTIONAL",
        fail_level: Literal["HINT", "WARNING"] = "HINT",
    ):
        message("STATUS", "Check for Compiler Cache program 'sccache'")
        if self.exe:
            result = SUCCESS
        else:
            result = self._compoment_fail(condition, fail_level)
        message("CHECK", self.Version, check_result=result)

        if result is not SUCCESS:
            message(result, "Optional sccache not found.")
        return result
