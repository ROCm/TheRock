from typing import Literal, Any, overload, Union, Mapping
from pathlib import Path
from textwrap import dedent

from find_SDKs import FindSDK
from utils import *
from ._AMDGPU import amdgpu_gfx_name, therock_avail_status
from ._NVPTX import nvptx_sm_name

##############################################################################################
#
#   GPU Archtecture representation. For example:
#
#   {
#       "GPU0": {"Name": "AMD Radeon W7900 Pro", "LLVM_TARGET": "gfx1100", "VRAM": 47.98}
#       "GPU1": {"Name": "AMD Radeon RX 7800XT", "LLVM_TARGET": "gfx1101", "VRAM": 15.92}
#        ...
#       "GPUX": {"Name": "ROG Astral 5080 Hatsune Miku Edition", "LLVM_TARGET": "sm_120", "VRAM": 15.92}
#   }
#
#   GPU Avail List will read from cmake/therock_amdgpu_targets.cmake,
#       with a dict item contains avail gfx target and its possiable
#       excluded roc-X/hip-X libraries.
#   The possiable excluded build libraries will hint/warn users/developers
#       build from source can know each gfx target with those libraries
#       will not be built.
#

_exclude_igpu = Literal[
    "Microsoft Basic Display Driver",
    "AMD Radeon Graphics",
    "Intel UHD Graphics",
    "Intel Iris Xe Graphics",
]
_amd_apu = Literal[
    "AMD Radeon 680M",
    "AMD Radeon 780M",
    "AMD Strix Point",
    "AMD Strix Halo",
    "AMD Krackan 1",
    "AMD Radeon 820M",
]


class GPUDevice(Mapping):

    def __init__(
        self,
        gpu_index: str,
        gpu_name: str,
        gpu_type: Literal["iGPU", "dGPU", "APU"],
        /,
        *,
        vram_byte: int = None,
        vram_unit: Literal["B", "KB", "MB", "GB", "TB"] = "B",
        vram_unit_round: int = 2,
    ):
        self.vram_unit = vram_unit

        if vram_byte is None:
            self._original_vram_bytes = 0
        elif isinstance(vram_byte, int):
            self._original_vram_bytes = vram_byte
        else:
            try:
                self._original_vram_bytes = int(vram_byte)
            except:
                message(
                    "FATAL_ERROR",
                    f"Invalid vram_byte value {vram_byte} with type {type(vram_byte)}.",
                )

        # vram volume will be (int, "B") at info greping.
        self.info: dict[
            Literal[
                "GPU_INDEX",
                "GPU_NAME",
                "GPU_TYPE",
                "GPU_VRAM",
                "GPU_VRAM_UNIT",
                "GPU_LLVM_TARGET",
            ],
            Any,
        ] = {
            "GPU_INDEX": gpu_index,
            "GPU_NAME": gpu_name,
            "GPU_TYPE": gpu_type,
            "GPU_VRAM": self._vram_calc(
                self._original_vram_bytes, vram_unit, vram_unit_round
            ),
            "GPU_VRAM_UNIT": vram_unit,
            "GPU_LLVM_TARGET": self._find_llvm_target_ISA(gpu_name),
        }

    @property
    def GPU_INDEX(self):
        return self.info["GPU_INDEX"]

    @property
    def GPU_NAME(self):
        return self.info["GPU_NAME"]

    @property
    def GPU_TYPE(self):
        return self.info["GPU_TYPE"]

    @property
    def GPU_VRAM(self):
        return self.info["GPU_VRAM"]

    @property
    def GPU_LLVM_TARGET(self):
        return self.info["GPU_LLVM_TARGET"]

    def __getitem__(self, key: str):
        return self.info[key]

    def __iter__(self):
        return iter(self.info)

    def __len__(self):
        return len(self.info)

    def to_dict(self):
        return {
            "GPU_NAME": self.GPU_NAME,
            "GPU_TYPE": self.GPU_TYPE,
            "GPU_VRAM": self.GPU_VRAM,
            "GPU_LLVM_TARGET": self.GPU_LLVM_TARGET,
        }

    def _vram_calc(
        self,
        volume: Union[int, float],
        unit: Literal["B", "KB", "MB", "GB", "TB"] = "B",
        /,
        rnd: int = 2,
    ):

        units = {"B": 0, "KB": 1, "MB": 2, "GB": 3, "TB": 4}
        if unit not in units:
            message("FATAL_ERROR", f"VRAM volume unit not support {unit}.")

        if isinstance(volume, str):
            volume = int(volume)

        power = units[unit]
        vol: Union[int, float] = round(volume / (1024**power), rnd)
        return vol

    def _find_llvm_target_ISA(self, gpu_name: str, /):
        # Find GPU ISA name.
        if gpu_name in _exclude_igpu:
            target = "N/A"

        elif "NVIDIA" in gpu_name:
            target = nvptx_sm_name(gpu_name)

        elif "AMD" in gpu_name:
            target = amdgpu_gfx_name(gpu_name)

        # elif "Intel", "Qualcomm" in gpuname:
        #    ...

        else:
            target = "N/A"

        if target is NOTDEFINED:
            target = "N/A"

        return target

    def __repr__(self):
        return f"{self.GPU_INDEX}: {self.GPU_NAME} ({self.GPU_LLVM_TARGET}) @{self.GPU_VRAM} {self.vram_unit}"

    def __str__(self):
        return f" {self.GPU_INDEX}: {self.GPU_NAME} ({self.GPU_LLVM_TARGET}) @{self.GPU_VRAM} {self.vram_unit}"


class FindGPU(FindSDK):

    amdgpu_support_stat = therock_avail_status()
    GPU_CLASS_PATH = (
        r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    )

    def __init__(self):
        super().__init__()

        self.info: dict[str, GPUDevice]

        self._dispatch_map = {
            "ROCm": self._post_ROCm,
        }

    def __repr__(self):
        content = ""

        gpus = len(self.info)

        for i in range(0, gpus):
            gpu = self.info[f"GPU{i}"]
            gpu_info_prefix = f"{gpu.GPU_NAME:50}{gpu.GPU_VRAM:>7} GB VRAM"
            if i == gpus - 1:
                content += f"    {gpu.GPU_INDEX}:       {gpu_info_prefix}, LLVM Target ISA: {gpu.GPU_LLVM_TARGET:>8}"
            else:
                content += f"    {gpu.GPU_INDEX}:       {gpu_info_prefix}, LLVM Target ISA: {gpu.GPU_LLVM_TARGET:>8} \n    "
        return content

    def __WINDOWS__(self):

        # The GPU info registration is unique and can be found in regedit.
        # Deprecates wmic and PowerShell Get-CimInstance.

        gpu_subreg_lst = regedit("HKLM", self.GPU_CLASS_PATH)
        gpu_subkey_lst = [subkey for subkey in gpu_subreg_lst if subkey.startswith("0")]
        gpu_reg_dict = {}

        for i in gpu_subkey_lst:
            gpux_dict_key_index = f"GPU{gpu_subkey_lst.index(i)}"
            gpu_direct_regkey = rf"{self.GPU_CLASS_PATH}\{i}"
            name = regedit("HKLM", gpu_direct_regkey, key_name="DriverDesc")
            vram = regedit(
                "HKLM", gpu_direct_regkey, key_name="HardwareInformation.qwMemorySize"
            )

            if vram is NOTDEFINED:
                vram = 0

            if name in _exclude_igpu:
                gpu_type = "iGPU"
            elif name in _amd_apu:
                gpu_type = "APU"
            else:
                gpu_type = "dGPU"

            gpu_reg_dict[gpux_dict_key_index] = GPUDevice(
                gpux_dict_key_index,
                name,
                gpu_type,
                vram_byte=vram,
                vram_unit="GB",
                vram_unit_round=2,
            )

        return gpu_reg_dict

    def __LINUX__(self):
        import subprocess
        import re
        import os

        gpu_dict = {}
        gpu_counter = 0

        try:

            output = subprocess.check_output(["lspci", "-mm"], text=True)
            vga_lines = [
                line
                for line in output.splitlines()
                if any(keyword in line for keyword in ["VGA", "3D", "Display"])
            ]

            for line in vga_lines:
                parts = [p.strip('"') for p in line.split('" "')]

                if len(parts) >= 4:
                    pci_slot = parts[0]
                    vendor_raw = parts[2]
                    device_raw = parts[3]

                    match = re.findall(r"\[(.*?)\]", device_raw)
                    clean_name = match[-1] if match else device_raw

                    if "AMD" in vendor_raw or "ATI" in vendor_raw:
                        clean_name = (
                            f"AMD {clean_name}"
                            if "AMD" not in clean_name
                            else clean_name
                        )
                    elif "NVIDIA" in vendor_raw:
                        clean_name = (
                            f"NVIDIA {clean_name}"
                            if "NVIDIA" not in clean_name
                            else clean_name
                        )
                    elif "Intel" in vendor_raw:
                        clean_name = (
                            f"Intel {clean_name}"
                            if "Intel" not in clean_name
                            else clean_name
                        )

                    if (
                        clean_name in _exclude_igpu
                        or "Integrated" in clean_name
                        or "UHD" in clean_name
                    ):
                        gpu_type = "iGPU"
                    elif clean_name in _amd_apu:
                        gpu_type = "APU"
                    else:
                        gpu_type = "dGPU"

                    vram_bytes = 0
                    sysfs_pci_id = (
                        f"0000:{pci_slot}"
                        if len(pci_slot.split(":")) == 2
                        else pci_slot
                    )
                    sysfs_vram_path = (
                        f"/sys/bus/pci/devices/{sysfs_pci_id}/mem_info_vram_total"
                    )

                    if os.path.exists(sysfs_vram_path):
                        try:
                            with open(sysfs_vram_path, "r") as f:
                                vram_bytes = int(f.read().strip())
                        except Exception:
                            pass

                    dict_key = f"GPU{gpu_counter}"
                    gpu_dict[dict_key] = GPUDevice(
                        dict_key,
                        clean_name,
                        gpu_type,
                        vram_byte=vram_bytes,
                        vram_unit="GB",
                        vram_unit_round=2,
                    )
                    gpu_counter += 1

        except FileNotFoundError:
            message(
                "WARNING",
                "'lspci' command not found. Cannot detect Linux GPUs natively.",
            )
        except Exception as e:
            message("WARNING", f"Error parsing Linux GPU info: {e}")

        return gpu_dict

    @overload
    def post(
        self,
        kwd: Literal["ROCm"],
        /,
    ): ...

    def post(self, kwd: str, **kwargs: Any):
        return super().post(kwd, condition=None, **kwargs)

    def _post_ROCm(self):

        res_lst = {}

        for gpu in self.info:

            this_gpu_device: GPUDevice = self.info[gpu]
            gpu_index = this_gpu_device.GPU_INDEX
            gpu_name = this_gpu_device.GPU_NAME
            gpu_llvm = this_gpu_device.GPU_LLVM_TARGET

            message("STATUS", f"Check for {gpu_index} support on ROCm")
            if gpu_llvm not in self.amdgpu_support_stat:
                result = WARNING
                message("CHECK", gpu_name, check_result=result)
                message(
                    "WARNING",
                    f"Warning: {gpu_index} {gpu_name} ({gpu_llvm}) is not supported to build ROCm.\n",
                )

            else:
                exclude_rocLIB_dict = self.amdgpu_support_stat[gpu_llvm]
                if not exclude_rocLIB_dict:
                    result = SUCCESS
                    message("CHECK", gpu_name, check_result=result)
                else:
                    result = HINT
                    message("CHECK", gpu_name, check_result=result)
                    cmakefile = Path(
                        gitrepo / "cmake/therock_amdgpu_targets.cmake"
                    ).as_posix()

                    formatted_excludes = "\n        ".join(
                        [
                            f"{k:<15} (Issue: https://github.com/ROCm/TheRock/issues/{v})"
                            for k, v in exclude_rocLIB_dict.items()
                        ]
                    )

                    reason = dedent(
                        f"""\
                        Found available GPU but with build disabled options. TheRock have found defined excluded
                            roc-X libraries build support on {gpu_index} {gpu_name} (GPU Target ISA: {gpu_llvm}).
                        If you need furthur roc-X libraries build support from TheRock, please trace on commented
                            issue URLs, and contact to TheRock dev team.

                        Defined in {cmakefile}:
                            therock_add_amdgpu_target(gfx... "..." FAMILY ... ... ...
                         -> EXCLUDE_TARGET_PROJECTS(
                                __EXCLUDES_PLACEHOLDER__
                            )

                         >>> traceback: Found defined excluded roc-X libraries support on {gpu_index}: {gpu_name}"""
                    )
                    reason = reason.replace(
                        "__EXCLUDES_PLACEHOLDER__", formatted_excludes
                    )

                    message("HINT", reason)

            gk = f"{gpu_index} ({gpu_llvm})"
            res_lst[gk] = result

        return res_lst
