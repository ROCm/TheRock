#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Provides a common abstract base class for all ROCm package managers
(such as installers and uninstallers).

This class is not executed directly.
Instead, other scripts inherit from it and implement the 'execute()' method.

Example of how to extend this class:
```
    from packaging_base_manager import PackageManagerBase

    class PackageUninstaller(PackageManagerBase):
    def execute(self):
    for pkg in self.packages:
    print("Handling package:", pkg.package)

    Usage:
    uninstaller = PackageUninstaller(package_list)
    uninstaller.execute()
```
"""

from abc import ABC, abstractmethod
from typing import List
from native_package_info import PackageInfo


class PackageManagerBase(ABC):
    """
    Abstract base class defining the interface for package management.
    """

    def __init__(self, package_list: List[PackageInfo]):
        self.packages = package_list

    @abstractmethod
    def execute(self):
        """Perform the package management operation."""
        pass

    def get_package(self, name: str) -> PackageInfo:
        for pkg in self.packages:
            if pkg.package == name:
                return pkg
        raise ValueError(f"Package '{name}' not found.")
