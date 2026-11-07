import re
import platform
import os
import subprocess
from textwrap import dedent
from typing import Literal, Any, overload

from find_SDKs import FindSDK
from utils.compare_functions import *
from utils.Registry import regedit
from utils import *


class FindCPU(FindSDK):

    required_cpu_arch = [
        "AMD64",
        "EM64T",
        "Intel64",
        "Intel 64",
        "x86_64",
        "x86-64",
        "x64",
    ]

    def __init__(self):
        super().__init__()

        self.info: dict[Literal["CPU_NAME", "CPU_ARCH", "CPU_CORE"], Any]

        self._dispatch_map = {
            "ARCH": self._post_arch,
            "CORES": self._post_cores,
        }

    def __repr__(self):
        return f"{self.CPU_NAME} {self.CPU_CORE} cores ({self.CPU_ARCH})"

    @property
    def CPU_NAME(self):
        return self.info["CPU_NAME"]

    @property
    def CPU_ARCH(self):
        return self.info["CPU_ARCH"]

    @property
    def CPU_CORE(self):
        return self.info["CPU_CORE"]

    def __WINDOWS__(self):

        _cpu_name = regedit(
            "HKLM",
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            key_name="ProcessorNameString",
        )
        _cpu_arch = platform.machine()
        _cpu_core = os.cpu_count()

        return {
            "CPU_NAME": _cpu_name.replace("            ", ""),
            "CPU_ARCH": _cpu_arch,
            "CPU_CORE": _cpu_core,
        }

    def __LINUX__(self):
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

        return {"CPU_NAME": _cpu_name, "CPU_ARCH": _cpu_arch, "CPU_CORE": _cpu_core}

    def __BSD__(self): ...

    @overload
    def post(
        self,
        kwd: Literal["ARCH"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...

    @overload
    def post(
        self,
        kwd: Literal["CORES"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "OPTIONAL",
        /,
        *,
        require: int = 4,
        recommend: int = 16,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "WARNING",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd.upper(), condition=condition, **kwargs)

    def _post_arch(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):
        message("STATUS", "Check for CPU Architecture")
        result = (
            self._compoment_fail(condition, fail_level)
            if self.CPU_ARCH not in self.required_cpu_arch
            else SUCCESS
        )
        message("CHECK", self.CPU_ARCH, check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""TheRock only supports AMD64/EM64T CPU targets.

                                >>> traceback: Found unsupported CPU Architecture: {self.CPU_ARCH}"""
            )
            message(reason, result)

        return result

    def _post_cores(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "OPTIONAL",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "WARNING",
        require: int = 4,
        recommend: int = 16,
    ):

        message("STATUS", "Check for CPU core(s) count")
        if self.CPU_CORE >= recommend:
            result = SUCCESS
        elif self.CPU_CORE < recommend and self.CPU_CORE >= require:
            result = self._compoment_fail(condition, fail_level)
        else:
            result = FATAL
        message("CHECK", f"{self.CPU_CORE} core(s)", check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""TheRock would needs a powerful machine to compile.
                                TheRock is a CMake super project includes AMD-LLVM and various roc-X/hip-X libraries, which
                                will have over 10k build objects to compile/link. TheRock also using Ninja-Build as CMake
                                generator to invoke all CPU cores to build at default. So This CPU is able to compile but
                                will be very slow to finish due to CPU performance bottlenecks.

                                >>> traceback: CPU cores: {self.CPU_CORE}, required: {require}, recommended: {recommend}"""
            )

            message(reason, result)

        return result
