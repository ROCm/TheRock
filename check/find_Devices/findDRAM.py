from typing import Literal, Any, overload
from textwrap import dedent

from find_SDKs import FindSDK
from utils.status import *
from utils import *


class FindDRAM(FindSDK):

    def __init__(self):
        super().__init__()

        self.info: dict[
            Literal[
                "MEM_PHYS_TOTAL",
                "MEM_PHYS_AVAIL",
                "MEM_VITURAL_AVAIL",
                "MEM_SWAP_AVAIL",
                "MEM_USE_RATIO",
                "MEM_ECC_LEVEL",
            ],
            Any,
        ]

        self._dispatch_map = {
            "VOLUME": self._post_dram_volume,
            "TYPE": self._post_ecc_memory,
        }

    def __repr__(self):
        return f"{self.MEM_PHYS_TOTAL}GB {self.MEM_ECC_LEVEL} in Total, {self.MEM_PHYS_AVAIL}GB Avail, {self.MEM_VITURAL_AVAIL}GB Virtual, {self.MEM_SWAP_AVAIL}GB Swap"

    @property
    def MEM_PHYS_TOTAL(self) -> float:
        return self.info["MEM_PHYS_TOTAL"]

    @property
    def MEM_PHYS_AVAIL(self) -> float:
        return self.info["MEM_PHYS_AVAIL"]

    @property
    def MEM_VITURAL_AVAIL(self) -> float:
        return self.info["MEM_VITURAL_AVAIL"]

    @property
    def MEM_SWAP_AVAIL(self) -> float:
        return self.info["MEM_SWAP_AVAIL"]

    @property
    def MEM_USE_RATIO(self) -> float:
        return self.info["MEM_USE_RATIO"]

    @property
    def MEM_ECC_LEVEL(self) -> Literal[f"Non-ECC (UDIMM)", f"ECC", f"ECC-REG (RDIMM)"]:
        """DRAM Memory device ECC equipment level."""
        return self.info["MEM_ECC_LEVEL"]

    def __WINDOWS__(self) -> dict[str, Any]:

        #   1: Grab out Memory volumes
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

        MEM_PHYS_TOTAL, MEM_PHYS_AVAIL, MEM_VITURAL_AVAIL, MEM_SWAP_AVAIL = (
            float(mem_status.ullTotalPhys / (1024**3)),
            float(mem_status.ullAvailPhys / (1024**3)),
            float(mem_status.ullAvailPageFile / (1024**3)),
            float((mem_status.ullTotalPageFile - mem_status.ullTotalPhys) / (1024**3)),
        )

        MEM_USE_RATIO = MEM_PHYS_AVAIL / MEM_PHYS_TOTAL

        #   2: Find out Memory Types

        MEM_ECC_LEVEL = self._win_get_ecc()

        return {
            "MEM_PHYS_TOTAL": round(MEM_PHYS_TOTAL, 2),
            "MEM_PHYS_AVAIL": round(MEM_PHYS_AVAIL, 2),
            "MEM_VITURAL_AVAIL": round(MEM_VITURAL_AVAIL, 2),
            "MEM_SWAP_AVAIL": round(MEM_SWAP_AVAIL, 2),
            "MEM_USE_RATIO": round(MEM_USE_RATIO, 2),
            "MEM_ECC_LEVEL": MEM_ECC_LEVEL,
        }

    def __LINUX__(self):
        import re

        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            _f = f.read().splitlines()

        for line in _f:
            mem_tol = re.search(r"^MemTotal:\s+(\d+)\s+kB$", line)
            mem_avl = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", line)
            mem_swp = re.search(r"^SwapTotal:\s+(\d+)\s+kB$", line)
            mem_vir = re.search(r"^VmallocTotal:\s+(\d+)\s+kB$", line)

            if mem_tol:
                MEM_PHYS_TOTAL = float(mem_tol.group(1)) / (1024**2)
            elif mem_avl:
                MEM_PHYS_AVAIL = float(mem_avl.group(1)) / (1024**2)
            elif mem_swp:
                MEM_SWAP_AVAIL = float(mem_swp.group(1)) / (1024**2)
            elif mem_vir:
                MEM_VITURAL_AVAIL = float(mem_vir.group(1)) / (1024**2)

        MEM_USE_RATIO = MEM_PHYS_AVAIL / MEM_PHYS_TOTAL
        MEM_ECC_LEVEL = self._linux_get_ecc()

        return {
            "MEM_PHYS_TOTAL": round(MEM_PHYS_TOTAL, 2),
            "MEM_PHYS_AVAIL": round(MEM_PHYS_AVAIL, 2),
            "MEM_VITURAL_AVAIL": round(MEM_VITURAL_AVAIL, 2),
            "MEM_SWAP_AVAIL": round(MEM_SWAP_AVAIL, 2),
            "MEM_USE_RATIO": round(MEM_USE_RATIO, 2),
            "MEM_ECC_LEVEL": MEM_ECC_LEVEL,
        }

    def _win_get_ecc(self):
        import subprocess, json

        cmd = [
            "powershell",
            "-Command",
            "Get-CimInstance Win32_MemoryArray | Select-Object ErrorInfo | ConvertTo-Json",
        ]
        cmd_output = subprocess.run(
            cmd, text=True, capture_output=True, check=True
        ).stdout
        output_json = json.loads(cmd_output)

        ecc_info: int = output_json["ErrorInfo"]

        if ecc_info == 3 or ecc_info is None:
            ecc = f"Non-ECC (UDIMM)"
        elif ecc_info == 5:
            ecc = f"ECC"
        elif ecc_info == 6:
            ecc = f"ECC-REG (RDIMM)"
        else:
            message(
                "FATAL_ERROR", f"ECC mem level read with value {ecc_info}", RuntimeError
            )

        return ecc

    def _linux_get_ecc(self):
        import subprocess, os

        is_sudo = True if os.geteuid() == 0 else False

        if is_sudo:
            try:
                output = subprocess.check_output(["dmidecode", "-t", "16"]).decode()

                res = {"ECC": "No", "REG": "No"}

                if "Multi-bit ECC" in output:
                    res = "REG"
                elif "Single-bit ECC" in output:
                    res = "ECC"
                else:
                    res = "UDIMM"

                return res
            except Exception as e:
                return FAILED
        else:
            try:
                output = subprocess.check_output(
                    ["lshw", "-C", "memory"], stderr=subprocess.DEVNULL
                ).decode()
                if "ECC" in output:
                    return "ECC"
                return "UDIMM"
            except:
                return FAILED

    @overload
    def post(
        self,
        kwd: Literal["VOLUME"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        required: int = 16,
        recommend: int = 32,
        fail_level: Literal["HINT", "WARNING", "FATAL"] = "FATAL",
    ): ...

    @overload
    def post(
        self,
        kwd: Literal["TYPE"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "OPTIONAL",
        /,
        *,
        memory: Literal["UDIMM", "ECC", "REG"] = "UDIMM",
        fail_level: Literal["HINT", "WARNING", "FATAL"] = "HINT",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    def _post_dram_volume(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["HINT", "WARNING", "FATAL"] = "FATAL",
        required: int = 16,
        recommend: int = 32,
    ):

        message("STATUS", "Check for DRAM avail volume")
        if self.MEM_PHYS_TOTAL < required or self.MEM_PHYS_AVAIL < required:
            result = self._compoment_fail(condition, fail_level)
        else:
            result = SUCCESS

        message("CHECK", f"{self.MEM_PHYS_AVAIL} GB Avail", check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Found System have low physical DRAM total volume.
                            TheRock requires large DRAM volume, which will have large DRAM usage incrementally
                            during linking time. Small volume of DRAM will make system lag, frozen or crash.
                            Please consider add DRAM volume over 32GB.
                            TheRock requires {required} GB system DRAM on your device.

                             >>> traceback: Low DRAM volume: {self.MEM_PHYS_TOTAL} GB Total, Required: {required} GB, recommended: {recommend}"""
            )
            message(fail_level, reason)

        return result

    def _post_ecc_memory(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "OPTIONAL",
        memory: Literal["UDIMM", "ECC", "REG"] = None,
        fail_level: Literal["HINT", "WARNING", "FATAL"] = "HINT",
    ):
        message("STATUS", "Check for DRAM Memory Type")

        if memory is not None:
            if memory == "UDIMM":
                result = SUCCESS
            elif memory == "ECC":
                result = (
                    SUCCESS
                    if self.MEM_ECC_LEVEL in ["ECC", "REG"]
                    else self._compoment_fail(condition, fail_level)
                )
            elif memory == "REG":
                result = (
                    SUCCESS
                    if self.MEM_ECC_LEVEL == memory
                    else self._compoment_fail(condition, fail_level)
                )

        elif memory is None:
            result = SUCCESS

        else:
            message(
                "FATAL_ERROR",
                f"Unsupported DRAM Type {memory}. Expected with UDIMM, ECC and REG.",
            )

        message("CHECK", self.MEM_ECC_LEVEL, check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Cannot confirm or Found unexpected DRAM Type.
                            This pre-build diagnose system has been set to require {memory} Type of DRAM on your device.
                            Please contact to your related administrator, or contact to TheRock devs for further DRAM support.

                             >>> traceback: Required system need {memory} type DRAM but failed due to incorrect type or confirm not sudo previlage"""
            )
            message(result, reason)

        return result
