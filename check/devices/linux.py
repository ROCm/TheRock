from __future__ import annotations
from os import environ
from typing import Literal, Union

from find_SDKs import *
from find_Devices import FindLinux, FindCPU, FindGPU, FindDRAM, FindDisk


class ManyLinux:

    NT = "Linux"

    def __init__(self):

        self.info: dict[
            Literal[
                "os",
                "cpu",
                "gpu",
                "ram",
                "ssd",
                "env",
                "python",
                "dvc",
                "cmake",
                "ninja",
                "git",
                "ccache",
                "sccache",
                "conda",
                "uv",
                "gcc",
                "gxx",
                "gfort",
                "gcc_ar",
                "ar",
                "as",
                "ld",
                "glibc",
                "pkgconf",
            ],
            Union[
                FindLinux,
                FindDisk,
            ],
        ] = {
            "os": FindLinux(),
            "cpu": FindCPU(),
            "gpu": FindGPU(),
            "ram": FindDRAM(),
            "ssd": FindDisk(),
            "python": FindPython(),
            "dvc": FindDVC(),
            "uv": FindUV(),
            "git": FindGit(),
            "conda": FindConda(),
            "cmake": FindCMake(),
            "ninja": FindNinja(),
            "ccache": FindCCache(),
            "sccache": FindSCCache(),
            "gcc": FindGCC(),
            "g++": FindGXX(),
            "gfort": FindGFortran(),
            "gcc_ar": FindGCC_AR(),
            "ar": FindAR(),
            "as": FindAS(),
            "ld": FindLD(),
            "glibc": FindGLIBC(),
            "pkgconf": FindPkgConfig(),
            "rocm": FindROCm(),
            "env": dict(environ),
        }

        self.info["df"] = self.info["ssd"].DISK_FILESYSTEM
        self.info["llvm"] = self.info["rocm"].info["__LLVM__"]

    def __repr__(self):
        return f"""
    System configuration:
        OS:         {self.OS}
        CPU:        {self.CPU}
        {self.GPU}
        RAM:        {self.RAM}
        Disk:       {self.ROM}

    {self.DiskFileSystem}

    Software configuration:
        Python:     {self.Python}
        Astral uv:  {self.UV}
        DVC:        {self.DVC}
        CMake:      {self.CMake}
        Ninja:      {self.Ninja}
        CCache:     {self.CCache}
        SCCache:    {self.SCCache}
        PkgConfig:  {self.PkgConfig}

        gcc:        {self.gcc}
        g++:        {self.gxx}
        gfortran:   {self.gfortran}
        gcc-ar:     {self.gcc_ar}
        ar:         {self.bin_ar}
        as:         {self.bin_as}
        ld:         {self.ld}
        glibc:      {self.glibc}

        {self.CCache.stats}
        {self.SCCache.stats}
            {self.llvm.__stat__()}
        {self.ROCm.stats}
                            """

    def __info__(self):
        return {
            "System configuration": {
                "OS": self.OS,
                "CPU": self.CPU,
                "GPU": self.GPU,
                "RAM": self.RAM,
                "Disk": self.ROM,
            },
            "Disk FileSystem": self.DiskFileSystem.info,
            "Software configuration": {
                "Python": self.Python,
                "Astral uv": self.UV,
                "Conda": self.Conda,
                "DVC": self.DVC,
                "Git": self.Git,
                "CMake": self.CMake,
                "Ninja": self.Ninja,
                "CCache": self.CCache,
                "SCCache": self.SCCache,
                "PkgConfig": self.PkgConfig,
            },
            "GCC compiler": {
                "gcc": self.gcc,
                "g++": self.gxx,
                "gfortran": self.gfortran,
                "gcc-ar": self.gcc_ar,
                "ar": self.bin_ar,
                "as": self.bin_as,
                "ld": self.ld,
            },
            "CCache stats": self.CCache.info,
            "SCCache stats": self.SCCache.info,
            "LLVM Configuration": self.llvm.info,
            "ROCM Configuration": self.ROCm.info,
        }

    @property
    def OS(self) -> FindLinux:
        return self.info["os"]

    @property
    def CPU(self) -> FindCPU:
        return self.info["cpu"]

    @property
    def GPU(self) -> FindGPU:
        return self.info["gpu"]

    @property
    def RAM(self) -> FindDRAM:
        return self.info["ram"]

    @property
    def ROM(self) -> FindDisk:
        return self.info["ssd"]

    @property
    def DiskFileSystem(self):
        return self.info["df"]

    @property
    def Python(self) -> FindPython:
        return self.info["python"]

    @property
    def Conda(self) -> FindConda:
        return self.info["conda"]

    @property
    def UV(self) -> FindUV:
        return self.info["uv"]

    @property
    def DVC(self) -> FindDVC:
        return self.info["dvc"]

    @property
    def Git(self) -> FindGit:
        return self.info["git"]

    @property
    def CMake(self) -> FindCMake:
        return self.info["cmake"]

    @property
    def Ninja(self) -> FindNinja:
        return self.info["ninja"]

    @property
    def CCache(self) -> FindCCache:
        return self.info["ccache"]

    @property
    def SCCache(self) -> FindSCCache:
        return self.info["sccache"]

    @property
    def PkgConfig(self) -> FindPkgConfigLite | FindPkgConfig:
        return self.info["pkgconf"]

    @property
    def gcc(self) -> FindGCC:
        return self.info["gcc"]

    @property
    def gxx(self) -> FindGXX:
        return self.info["g++"]

    @property
    def gfortran(self) -> FindGFortran:
        return self.info["gfort"]

    @property
    def gcc_ar(self) -> FindGCC_AR:
        return self.info["gcc_ar"]

    @property
    def bin_ar(self) -> FindAR:
        return self.info["ar"]

    @property
    def bin_as(self) -> FindAS:
        return self.info["as"]

    @property
    def ld(self) -> FindLD:
        return self.info["ld"]

    @property
    def glibc(self) -> FindGLIBC:
        return self.info["glibc"]

    @property
    def llvm(self) -> FindAMDLLVM:
        return self.info["llvm"]

    @property
    def ROCm(self) -> FindROCm:
        return self.info["rocm"]
