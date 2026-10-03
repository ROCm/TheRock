from pathlib import PureWindowsPath
from shutil import disk_usage
import subprocess as subproc, os
from typing import Literal, overload, Any, Mapping, Union
from textwrap import dedent
import json

from find_SDKs import FindSDK
from utils import *
from utils.status import *


class DiskDrive(Mapping):
    def __init__(self, raw_data: dict[str, Any]):
        """
        初始化 DiskDrive 物件

        : param raw_data: 從 PowerShell 傳回來的單一磁碟 JSON 資料 (dict)
        """
        # 1. 先處理原始資料
        self.info: dict[
            Literal[
                "DRIVE_LETTER",
                "DRIVE_INDEX",
                "DRIVE_MODEL",
                "DRIVE_PHYS_TYPE",
                "DRIVE_BUS_TYPE",
                "DRIVE_HEALTH",
                "DRIVE_PART_STYLE",
                "DRIVE_VOL_FILESYSTEM",
                "DRIVE_TOTAL_SPACE",
                "DRIVE_USAGE_SPACE",
                "DRIVE_AVAIL_SPACE",
                "DRIVE_USAGE_RATIO",
                "DRIVE_AVAIL_RATIO",
            ],
            Any,
        ] = raw_data
        self._processed_media = raw_data.get("DRIVE_PHYS_TYPE", "Unknown")
        bus_type = raw_data.get("DRIVE_BUS_TYPE", "Unknown")

        if self._processed_media == "Unspecified" and bus_type in ["SAS", "SCSI"]:
            self._processed_media = "VHD/VHDX"

        if self._processed_media == "Unspecified" and bus_type in ["USB"]:
            self._processed_media = "USB"

        drive_letter = raw_data["DRIVE_LETTER"]
        DRIVE_TOTAL_SPACE, DRIVE_USAGE_SPACE, DRIVE_AVAIL_SPACE = disk_usage(
            drive_letter
        )

        DRIVE_USAGE_RATIO = round(
            float(DRIVE_USAGE_SPACE / DRIVE_TOTAL_SPACE) * 100.0, 2
        )
        DRIVE_AVAIL_RATIO = round(100 - DRIVE_USAGE_RATIO, 2)
        DRIVE_TOTAL_SPACE = round(DRIVE_TOTAL_SPACE / (1024**3), 2)
        DRIVE_USAGE_SPACE = round(DRIVE_USAGE_SPACE / (1024**3), 2)
        DRIVE_AVAIL_SPACE = round(DRIVE_AVAIL_SPACE / (1024**3), 2)

        self.info["DRIVE_TOTAL_SPACE"] = DRIVE_TOTAL_SPACE
        self.info["DRIVE_USAGE_SPACE"] = DRIVE_USAGE_SPACE
        self.info["DRIVE_AVAIL_SPACE"] = DRIVE_AVAIL_SPACE
        self.info["DRIVE_USAGE_RATIO"] = DRIVE_USAGE_RATIO
        self.info["DRIVE_AVAIL_RATIO"] = DRIVE_AVAIL_RATIO

    def __getitem__(self, key: str):

        if key == "DRIVE_PHYS_TYPE":
            return self.DRIVE_PHYS_TYPE
        if key == "DRIVE_TOTAL_PHYS":
            return self.DRIVE_TOTAL_SPACE
        return self.info.get(key)

    def __iter__(self):
        key = self.info.keys()
        return iter(key)

    def __len__(self):
        return len(self.info.keys())

    @property
    def DRIVE_TOTAL_SPACE(self):
        return self.info["DRIVE_TOTAL_SPACE"]

    @property
    def DRIVE_USAGE_SPACE(self):
        return self.info["DRIVE_USAGE_SPACE"]

    @property
    def DRIVE_AVAIL_SPACE(self):
        return self.info["DRIVE_AVAIL_SPACE"]

    @property
    def DRIVE_USAGE_RATIO(self):
        return self.info["DRIVE_USAGE_RATIO"]

    @property
    def DRIVE_AVAIL_RATIO(self):
        return self.info["DRIVE_AVAIL_RATIO"]

    @property
    def DRIVE_LETTER(self) -> str:
        """Drive Letter. eg: C:, D:,"""
        return self.info.get("DRIVE_LETTER")

    @property
    def DRIVE_INDEX(self) -> int:
        """Disk Index Number"""
        return self.info.get("DRIVE_INDEX")

    @property
    def DRIVE_MODEL(self) -> str:
        """Disk Friendly Name"""
        return self.info.get("DRIVE_MODEL")

    @property
    def DRIVE_PHYS_TYPE(self) -> str:
        """Physical Type (SSD, HDD, VHD/VHDX)"""
        return self._processed_media

    @property
    def DRIVE_BUS_TYPE(self) -> str:
        """BUS Interface (NVMe, SATA, USB, SAS...)"""
        return self.info.get("DRIVE_BUS_TYPE")

    @property
    def DRIVE_HEALTH(self) -> str:
        """Health Status"""
        return self.info.get("DRIVE_HEALTH")

    @property
    def DRIVE_PART_STYLE(self) -> str:
        """return GPT / MBR."""
        return self.info.get("DRIVE_PART_STYLE")

    @property
    def DRIVE_VOL_FILESYSTEM(self) -> str:
        """檔案系統 (NTFS, FAT32)"""
        return self.info.get("DRIVE_VOL_FILESYSTEM")

    def __repr__(self):
        return (
            f"{self.DRIVE_LETTER} {self.DRIVE_MODEL:<30} {self.DRIVE_PHYS_TYPE} (Disk {self.DRIVE_INDEX}) "
            + f"{self.DRIVE_TOTAL_SPACE:>7} GB {self.DRIVE_VOL_FILESYSTEM} Total, "
            + f"{self.DRIVE_AVAIL_SPACE:>7} GB ({self.DRIVE_AVAIL_RATIO:>5}%) Avail, "
            + f"{self.DRIVE_USAGE_SPACE:>7} GB ({self.DRIVE_USAGE_RATIO:>5}%) Used"
        )


class DiskFileSystem(FindSDK, Mapping):

    def __init__(self):
        super().__init__()

        self.info: dict[
            Union[
                Literal[
                    "C:",
                    "D:",
                    "E:",
                    "F:",
                    "G:",
                    "H:",
                    "I:",
                    "J:",
                    "K:",
                    "L:",
                    "M:",
                    "N:",
                    "O:",
                    "P:",
                    "Q:",
                    "R:",
                    "S:",
                    "T:",
                    "U:",
                    "V:",
                    "W:",
                    "X:",
                    "Y:",
                    "Z:",
                    "A:",
                    "B:",
                ],
                str,
            ],
            DiskDrive,
        ]

        self._dispatch_map = {}

    def __getitem__(self, key: str) -> DiskDrive:
        return self.info[key]

    def __iter__(self):
        return iter(self.info)

    def __len__(self):
        return len(self.info)

    def __repr__(self):

        if config.DISPLAY_DISKFILESYSTEM:

            content = "Disk File System\n"

            for drive in self.info.values():
                content += f"    {drive.__repr__()}\n"
        else:
            content = ""

        return content

    def __WINDOWS__(self):
        ps_script = """
    Get-Partition | Where-Object { $_.DriveLetter } | ForEach-Object {
        $vol = Get-Volume -DriveLetter $_.DriveLetter -ErrorAction SilentlyContinue
        $disk = Get-Disk -Number $_.DiskNumber -ErrorAction SilentlyContinue
        $phys = $disk | Get-PhysicalDisk | Select-Object -First 1

        [PSCustomObject]@{
            DRIVE_LETTER            = [string]$_.DriveLetter
            DRIVE_INDEX             = [int]$_.DiskNumber
            DRIVE_MODEL             = [string]$disk.FriendlyName
            DRIVE_PHYS_TYPE         = [string]$phys.MediaType
            DRIVE_BUS_TYPE          = [string]$phys.BusType
            DRIVE_HEALTH            = [string]$phys.HealthStatus
            DRIVE_PART_STYLE        = [string]$disk.PartitionStyle
            DRIVE_VOL_FILESYSTEM    = [string]$vol.FileSystem
        }
    } | ConvertTo-Json -Compress
    """
        try:
            cmd = (
                f"[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; {ps_script}"
            )

            result = subproc.run(
                ["powershell.exe", "-NoProfile", "-Command", cmd],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )

            if not result.stdout.strip():
                return FAILED

            data = json.loads(result.stdout)

            # 處理單一結果 (dict) vs 多個結果 (list)
            if isinstance(data, dict):
                data = [data]

            drive_objects = {}
            for d in data:
                drive_tag = f"{d['DRIVE_LETTER']}:"
                d["DRIVE_LETTER"] = drive_tag  # 你在這裡修改了原始 dict
                drive_objects[drive_tag] = DiskDrive(d)

            return drive_objects

        except Exception as e:
            message("FATAL_ERROR", f"Disk Filesystem Initialization Failed with {e}.")

    def __LINUX__(self) -> dict:
        import subprocess as subproc
        import json
        import os

        try:
            cmd = [
                "lsblk",
                "-J",
                "-l",
                "-o",
                "KNAME,PKNAME,TYPE,FSTYPE,MOUNTPOINT,MODEL,ROTA,TRAN,PTTYPE",
            ]
            result = subproc.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            devices = data.get("blockdevices", [])

            all_devs = {dev.get("kname"): dev for dev in devices}

            def get_physical_disk(dev_kname):
                curr = all_devs.get(dev_kname)
                while curr and curr.get("type") != "disk" and curr.get("pkname"):
                    curr = all_devs.get(curr.get("pkname"))
                return curr if curr and curr.get("type") == "disk" else {}

            drive_objects = {}
            for dev in devices:
                mountpoint = dev.get("mountpoint")

                if (
                    mountpoint
                    and mountpoint.startswith("/")
                    and os.path.exists(mountpoint)
                ):
                    phys_disk = get_physical_disk(dev.get("kname"))

                    model = phys_disk.get("model") or dev.get("model") or "Unknown"
                    rota = phys_disk.get("rota") or dev.get("rota")

                    if rota == "1":
                        phys_type = "HDD"
                    elif rota == "0":
                        phys_type = "SSD"
                    else:
                        phys_type = "Unknown"

                    if any(
                        v in str(model).upper()
                        for v in ["VBOX", "VMWARE", "QEMU", "VIRTUAL"]
                    ):
                        phys_type = "VHD/VHDX"

                    bus_type = phys_disk.get("tran") or dev.get("tran") or "Unknown"
                    if bus_type.lower() == "nvme":
                        bus_type = "NVMe"
                    elif bus_type.lower() == "sata":
                        bus_type = "SATA"

                    pttype = dev.get("pttype") or phys_disk.get("pttype") or "Unknown"
                    if pttype == "dos":
                        pttype = "MBR"
                    elif pttype == "gpt":
                        pttype = "GPT"

                    raw_data = {
                        "DRIVE_LETTER": mountpoint,
                        "DRIVE_INDEX": dev.get("kname"),
                        "DRIVE_MODEL": str(model).strip(),
                        "DRIVE_PHYS_TYPE": phys_type,
                        "DRIVE_BUS_TYPE": (
                            str(bus_type).upper()
                            if bus_type != "Unknown"
                            else "Unknown"
                        ),
                        "DRIVE_HEALTH": "Unknown",
                        "DRIVE_PART_STYLE": (
                            str(pttype).upper() if pttype != "Unknown" else "Unknown"
                        ),
                        "DRIVE_VOL_FILESYSTEM": dev.get("fstype") or "Unknown",
                    }

                    drive_objects[mountpoint] = DiskDrive(raw_data)

            return drive_objects

        except Exception as e:
            from utils.color_string import message

            message(
                "FATAL_ERROR", f"Linux Disk Filesystem Initialization Failed with {e}."
            )
            return {}


class FindDisk(FindSDK):

    def __init__(self):
        super().__init__()

        self.info: dict[
            Literal[
                "DRIVE_LETTER",
                "DRIVE_INDEX",
                "DRIVE_MODEL",
                "DRIVE_PHYS_TYPE",
                "DRIVE_BUS_TYPE",
                "DRIVE_HEALTH",
                "DRIVE_PART_STYLE",
                "DRIVE_VOL_FILESYSTEM",
                "DRIVE_TOTAL_SPACE",
                "DRIVE_USAGE_SPACE",
                "DRIVE_AVAIL_SPACE",
                "DRIVE_USAGE_RATIO",
                "DRIVE_AVAIL_RATIO",
                "DISK_FILESYSTEM",
            ],
            Any,
        ]

        self.disk = self.info["REPO_DISK"]

        self._dispatch_map = {
            "AVAIL": self._post_disk_space,
            "TYPE": self._post_disk_type,
        }

    def __repr__(self):

        return (
            f"{self.DRIVE_LETTER} {self.DRIVE_MODEL} {self.DRIVE_PHYS_TYPE} (Disk {self.DRIVE_INDEX}) "
            + f"{self.DRIVE_TOTAL_SPACE:>7} GB {self.DRIVE_VOL_FILESYSTEM} Total, "
            + f"{self.DRIVE_AVAIL_SPACE:>7} GB ({self.DRIVE_AVAIL_RATIO:>5}%) Avail, "
            + f"{self.DRIVE_USAGE_SPACE:>7} GB ({self.DRIVE_USAGE_RATIO:>5}%) Used"
        )

    @property
    def REPO_PATH(self):
        return self.info["REPO_PATH"]

    @property
    def DRIVE_TOTAL_SPACE(self):
        return self.info["DRIVE_TOTAL_SPACE"]

    @property
    def DRIVE_USAGE_SPACE(self):
        return self.info["DRIVE_USAGE_SPACE"]

    @property
    def DRIVE_AVAIL_SPACE(self):
        return self.info["DRIVE_AVAIL_SPACE"]

    @property
    def DRIVE_USAGE_RATIO(self):
        return self.info["DRIVE_USAGE_RATIO"]

    @property
    def DRIVE_AVAIL_RATIO(self):
        return self.info["DRIVE_AVAIL_RATIO"]

    @property
    def DRIVE_INDEX(self) -> str:
        """Drive tag"""
        return self.info.get("DRIVE_INDEX")

    @property
    def DRIVE_LETTER(self) -> str:
        """Drive Letter. eg: C:, D:,"""
        return self.info.get("DRIVE_LETTER")

    @property
    def DRIVE_MODEL(self) -> str:
        """Disk Friendly Name"""
        return self.info.get("DRIVE_MODEL")

    @property
    def DRIVE_PHYS_TYPE(self) -> str:
        """Physical Type (SSD, HDD, VHD/VHDX)"""
        return self.info.get("DRIVE_PHYS_TYPE")

    @property
    def DRIVE_BUS_TYPE(self) -> str:
        """BUS Interface (NVMe, SATA, USB, SAS...)"""
        return self.info.get("DRIVE_BUS_TYPE")

    @property
    def DRIVE_HEALTH(self) -> str:
        """Health Status"""
        return self.info.get("DRIVE_HEALTH")

    @property
    def DRIVE_PART_STYLE(self) -> str:
        """return GPT / MBR."""
        return self.info.get("DRIVE_PART_STYLE")

    @property
    def DRIVE_VOL_FILESYSTEM(self) -> str:
        """檔案系統 (NTFS, FAT32)"""
        return self.info.get("DRIVE_VOL_FILESYSTEM")

    @property
    def DISK_FILESYSTEM(self) -> str:
        return self.info["DISK_FILESYSTEM"]

    def __WINDOWS__(self):

        dfs: dict[
            Literal[
                "C:",
                "D:",
                "E:",
                "F:",
                "G:",
                "H:",
                "I:",
                "J:",
                "K:",
                "L:",
                "M:",
                "N:",
                "O:",
                "P:",
                "Q:",
                "R:",
                "S:",
                "T:",
                "U:",
                "V:",
                "W:",
                "X:",
                "Y:",
                "Z:",
                "A:",
                "B:",
            ],
            DiskDrive,
        ] = DiskFileSystem()

        repo_path = gitrepo
        repo_drive_tag = PureWindowsPath(__file__).drive
        repo_disk = dfs[repo_drive_tag]

        return {
            "REPO_PATH": repo_path,
            "REPO_DISK": repo_disk,
            "DRIVE_TOTAL_SPACE": repo_disk.DRIVE_TOTAL_SPACE,
            "DRIVE_AVAIL_SPACE": repo_disk.DRIVE_AVAIL_SPACE,
            "DRIVE_AVAIL_RATIO": repo_disk.DRIVE_AVAIL_RATIO,
            "DRIVE_USAGE_SPACE": repo_disk.DRIVE_USAGE_SPACE,
            "DRIVE_USAGE_RATIO": repo_disk.DRIVE_USAGE_RATIO,
            "DRIVE_INDEX": repo_disk.DRIVE_INDEX,
            "DRIVE_LETTER": repo_disk.DRIVE_LETTER,
            "DRIVE_MODEL": repo_disk.DRIVE_MODEL,
            "DRIVE_PHYS_TYPE": repo_disk.DRIVE_PHYS_TYPE,
            "DRIVE_BUS_TYPE": repo_disk.DRIVE_BUS_TYPE,
            "DRIVE_HEALTH": repo_disk.DRIVE_HEALTH,
            "DRIVE_PART_STYLE": repo_disk.DRIVE_PART_STYLE,
            "DRIVE_VOL_FILESYSTEM": repo_disk.DRIVE_VOL_FILESYSTEM,
            "DISK_FILESYSTEM": dfs,
        }

    def __LINUX__(self):
        import subprocess as subproc
        import os

        dfs = DiskFileSystem()
        repo_path = gitrepo

        try:
            target_mount = (
                subproc.run(
                    ["df", "--output=target", repo_path],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                .stdout.strip()
                .split("\n")[-1]
                .strip()
            )
        except Exception:
            target_mount = "/"

        repo_disk = dfs.get(target_mount)

        if not repo_disk:
            if os.path.exists(target_mount):
                try:
                    fstype = (
                        subproc.run(
                            ["df", "-T", "--output=fstype", repo_path],
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        .stdout.strip()
                        .split("\n")[-1]
                        .strip()
                    )
                except Exception:
                    fstype = "Unknown"

                fallback_data = {
                    "DRIVE_LETTER": target_mount,
                    "DRIVE_INDEX": "N/A",
                    "DRIVE_MODEL": "WSL/Network Mount",
                    "DRIVE_PHYS_TYPE": "Virtual",
                    "DRIVE_BUS_TYPE": "Net/FS",
                    "DRIVE_HEALTH": "Unknown",
                    "DRIVE_PART_STYLE": "Unknown",
                    "DRIVE_VOL_FILESYSTEM": fstype,
                }
                repo_disk = DiskDrive(fallback_data)
            else:
                from utils.color_string import message

                message(
                    "FATAL_ERROR",
                    f"Failed to map repo path {repo_path} to any known Linux mount point.",
                )

        return {
            "REPO_PATH": repo_path,
            "REPO_DISK": repo_disk,
            "DRIVE_TOTAL_SPACE": repo_disk.DRIVE_TOTAL_SPACE,
            "DRIVE_AVAIL_SPACE": repo_disk.DRIVE_AVAIL_SPACE,
            "DRIVE_AVAIL_RATIO": repo_disk.DRIVE_AVAIL_RATIO,
            "DRIVE_USAGE_SPACE": repo_disk.DRIVE_USAGE_SPACE,
            "DRIVE_USAGE_RATIO": repo_disk.DRIVE_USAGE_RATIO,
            "DRIVE_INDEX": repo_disk.DRIVE_INDEX,
            "DRIVE_LETTER": repo_disk.DRIVE_LETTER,
            "DRIVE_MODEL": repo_disk.DRIVE_MODEL,
            "DRIVE_PHYS_TYPE": repo_disk.DRIVE_PHYS_TYPE,
            "DRIVE_BUS_TYPE": repo_disk.DRIVE_BUS_TYPE,
            "DRIVE_HEALTH": repo_disk.DRIVE_HEALTH,
            "DRIVE_PART_STYLE": repo_disk.DRIVE_PART_STYLE,
            "DRIVE_VOL_FILESYSTEM": repo_disk.DRIVE_VOL_FILESYSTEM,
            "DISK_FILESYSTEM": dfs,
        }

    @overload
    def post(
        self,
        kwd: Literal["AVAIL"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        required: int = 200,
        ratio: float = 30.0,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    def _post_disk_space(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        required: int = 200,
        ratio: float = 30.0,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):

        if ratio > 1 and ratio < 100:
            avail_ratio = ratio / 100
            per = ratio
        elif ratio > 0 and ratio < 1:
            avail_ratio = ratio
            per = ratio * 100
        else:
            raise TypeError("Invalid assumed Disk Used Ratio.")

        message("STATUS", "Check for Disk usage status")

        if self.DRIVE_TOTAL_SPACE < required:
            result = FATAL
        elif self.DRIVE_AVAIL_SPACE < required:
            result = FATAL
        elif (self.DRIVE_AVAIL_SPACE - required) / self.DRIVE_TOTAL_SPACE < avail_ratio:
            result = self._compoment_fail(condition, fail_level)
        else:
            result = SUCCESS

        message("CHECK", f"{self.DRIVE_AVAIL_SPACE} GB Avail", check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""Found Disk Storage space is low or High usage.
                                TheRock will need a large avail space to store source code and build programs, like compiled
                                objects, Shared Objects/DLLs, Static Libraries, Executables, roc-X/hip-X Libraries etc.
                                TheRock will need avail space over {required} GB for the builds. And also recommends the
                                Disk storage usage not over {per}% to slow down SSD Wear/Endurance and its TBW.

                                >>> traceback:  Required Space: {required} GB, Avail: {self.DRIVE_AVAIL_SPACE} GB, Disk usage: {self.DRIVE_USAGE_RATIO}%"""
            )
            message(result, reason)
        return result

    def _post_disk_type(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"],
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):
        message("STATUS", "Check for Disk Physical Type")

        if self.DRIVE_PHYS_TYPE in ["SSD", "VHD/VHDX"]:
            result = SUCCESS
        else:
            result = self._compoment_fail(condition, fail_level)
