from __future__ import annotations
from os import environ
from typing import Literal, Any

from find_SDKs import *
from find_Devices import FindWindows, FindCPU, FindGPU, FindDRAM, FindDisk


class WindowsNT:

    NT = "Windows"

    def __init__(self):

        self.info: dict[
            Literal[
                "os",
                "cpu",
                "gpu",
                "ram",
                "ssd",
                "env",
                "vs",
                "python",
                "dvc",
                "git",
                "cmake",
                "ninja",
                "pkgconf",
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
                "llvm",
                "rocm",
            ],
            Any,
        ] = {
            "os": FindWindows(),
            "cpu": FindCPU(),
            "gpu": FindGPU(),
            "ram": FindDRAM(),
            "ssd": FindDisk(),
            "vs": FindVS20XX(),
            "python": FindPython(),
            "dvc": FindDVC(),
            "uv": FindUV(),
            "conda": FindConda(),
            "git": FindGit(),
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
            "pkgconf": FindPkgConfigLite(),
            "rocm": FindROCm(),
            "env": dict(environ),
        }
        self.info["llvm"] = self.info["rocm"].info["__LLVM__"]
        self.info["df"] = self.info["ssd"].DISK_FILESYSTEM

    def __repr__(self):
        return f"""\
    System configuration:
        OS:         {self.OS}
        CPU:        {self.CPU}
    {self.GPU}
        RAM:        {self.RAM}
        Disk:       {self.ROM}

    {self.DiskFileSystem}
    Software configuration:
        Python:     {self.Python.Version:<10} --> {self.Python.exe.as_posix()}
        Astral uv:  {self.UV.Version:<10} --> {self.UV.exe.as_posix()}
        Conda:      {self.Conda.Version:<10} --> {self.Conda.exe.as_posix()}
        DVC:        {self.DVC.Version:<10} --> {self.DVC.exe.as_posix()}
        Git:        {self.Git.Version:<10} --> {self.Git.exe.as_posix()}
        CMake:      {self.CMake.Version:<10} --> {self.CMake.exe.as_posix()}
        Ninja:      {self.Ninja.Version:<10} --> {self.CMake.exe.as_posix()}
        CCache:     {self.CCache.Version:<10} --> {self.CCache.exe.as_posix()}
        SCCache:    {self.SCCache.Version:<10} --> {self.SCCache.exe.as_posix()}
        PkgConfig:  {self.PkgConfig.Version:<10} --> {self.PkgConfig.exe.as_posix()}

    Python 3 Status
        Python3:    {self.Python.exe.as_posix()}
        Version:    {self.Python.Interpreter} {self.Python.Version} {self.Python.Release}
        Environ:    {self.Python.VENV}
        GIL:        {self.Python.GIL}
        JIT:        {self.Python.JIT}

    Visual Studio Status
        VS20XX:     {self.VS20XX.VS20XX:<10} --> {self.VS20XX.VS_DIR}
        MSVC:       {self.VS20XX.cl.as_posix()}
        ML64:       {self.VS20XX.ml64.as_posix()}
        LIB:        {self.VS20XX.lib.as_posix()}
        LINK:       {self.VS20XX.link.as_posix()}
        MSVC Ver:   {self.VS20XX.VC_VERSION}
        MSVC Host:  {self.VS20XX.VC_HOST}
        MSVC Arch:  {self.VS20XX.VC_ARCH}
        MSVC ATL:   {self.VS20XX.VC_ATLMFC}
        UCRT:       {self.VS20XX.UCRT_VERSION}
        RC:         {self.VS20XX.rc.as_posix()}
        MT:         {self.VS20XX.mt.as_posix()}
        MSBuild:    {self.VS20XX.MSBuild.as_posix()}
        CMake:      {self.VS20XX.cmake.as_posix()}

    GCC compiler Status
        gcc:        {self.gcc.Version:<10} --> {self.gcc.exe.as_posix()}
        g++:        {self.gxx.Version:<10} --> {self.gxx.exe.as_posix()}
        gfortran:   {self.gfortran.Version:<10} --> {self.gfortran.exe.as_posix()}
        gcc-ar:     {self.gcc_ar.Version:<10} --> {self.gcc_ar.exe.as_posix()}
        ar:         {self.bin_ar.Version:<10} --> {self.bin_ar.exe.as_posix()}
        as:         {self.bin_as.Version:<10} --> {self.bin_as.exe.as_posix()}
        ld:         {self.ld.Version:<10} --> {self.ld.exe.as_posix()}

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
                "GPU": self.GPU.info,
                "RAM": self.RAM,
                "Disk": self.ROM,
            },
            "Disk FileSystem": self.DiskFileSystem.info,
            "Software configuration": {
                "Python": self.Python.info,
                "Astral uv": self.UV.info,
                "Conda": self.Conda.info,
                "DVC": self.DVC.info,
                "Git": self.Git.info,
                "CMake": self.CMake.info,
                "Ninja": self.Ninja.info,
                "CCache": self.CCache.info,
                "SCCache": self.SCCache.info,
                "PkgConfig": self.PkgConfig.info,
            },
            "Visual Studio": {
                "VS20XX": self.VS20XX.VS20XX,
                "VS_DIR": self.VS20XX.VS_DIR,
                "MSVC": self.VS20XX.cl,
                "MSVC Ver": self.VS20XX.VC_VERSION,
                "MSVC Host": self.VS20XX.VC_HOST,
                "MSVC Arch": self.VS20XX.VC_ARCH,
                "MSVC ATL": self.VS20XX.VC_ATLMFC,
                "UCRT": self.VS20XX.UCRT_VERSION,
                "MSBuild": self.VS20XX.MSBuild,
                "CMake": self.VS20XX.cmake,
            },
            "GCC compiler": {
                "gcc": self.gcc.info,
                "g++": self.gxx.info,
                "gfortran": self.gfortran.info,
                "gcc-ar": self.gcc_ar.info,
                "ar": self.bin_ar.info,
                "as": self.bin_as.info,
                "ld": self.ld.info,
            },
            "CCache stats": self.CCache.info["Stats"],
            "SCCache stats": self.SCCache.info["Stats"],
            "LLVM Configuration": self.llvm.info,
            "ROCM Configuration": self.ROCm.info,
        }

    @property
    def OS(self) -> FindWindows:
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
    def VS20XX(self) -> FindVS20XX:
        return self.info["vs"]

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
    def llvm(self) -> FindAMDLLVM:
        return self.info["llvm"]

    @property
    def ROCm(self) -> FindROCm:
        return self.info["rocm"]
