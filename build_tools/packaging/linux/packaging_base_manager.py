# base_manager.py

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
