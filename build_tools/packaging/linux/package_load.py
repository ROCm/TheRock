#!/usr/bin/env python3
import json
import os
import re
import subprocess
import platform
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("package_load")


class LoadPackages:
    def __init__(self, package_json_path: str, version: bool, amdgpu_family: str = None, rocm_version: str = None):
        self.package_json_path = package_json_path
        self.amdgpu_family = amdgpu_family
        self.gfx_suffix = self.amdgpu_family.split("-")[0].lower()  # e.g., gfx94x-dcgpu 
        self.rocm_version = rocm_version
        self.version = version
        self.packages = self._load_packages()
        self.os_family = self.detect_os_family()
        self.pkg_map = {pkg["Package"]: pkg for pkg in self.packages}

    # ---------------------------------------------------------------------
    # Core JSON and Package Utilities
    # ---------------------------------------------------------------------
    def _load_packages(self):
        with open(self.package_json_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Expected a list of package objects in JSON.")
        return data

    def arch_suffix_flag(self, package_name):
        """
        Given a list of package dicts, returns a list of package names.
        Rules:
          - If 'Gfxarch' is True and amdgpu_family is provided, append '-<amdgpu_family>'.
          - Do not append if package name contains 'devel'.
        """
        pkg = self.pkg_map.get(package_name)
        if not pkg:
            # fallback: if package not found in self.packages, leave as is
            return package_name
        gfx_arch_flag = str(pkg.get("Gfxarch", "False")).lower() == "true"

        return gfx_arch_flag

    def list_composite_packages(self):
        """Return (non_composite, composite) package lists."""
        composite = []
        non_composite = []
        for pkg in self.packages:
            name = pkg.get("Package", "").strip()
            if not name:
                continue
            if str(pkg.get("Composite", "")).strip().lower() == "yes":
                composite.append(name)
            else:
                non_composite.append(name)
        return non_composite, composite

    def _build_dependency_graph(self, packages, use_rpm=False):
        dep_key = "RPMRequires" if use_rpm else "DEBDepends"
        graph = {}
        pkg_names = {p["Package"] for p in packages if "Package" in p}

        for pkg in packages:
            name = pkg["Package"]
            deps = [d for d in pkg.get(dep_key, []) if d in pkg_names]
            graph[name] = deps
        return graph

    def _dfs_sort(self, graph):
        visited, sorted_list = set(), []

        def visit(pkg):
            if pkg in visited:
                return
            visited.add(pkg)
            for dep in graph.get(pkg, []):
                visit(dep)
            sorted_list.append(pkg)

        for pkg in graph:
            visit(pkg)
        return sorted_list

    def sort_packages_by_dependencies(self, pacakge_names, use_rpm=False):

        packages = [pkg for pkg in self.packages if pkg["Package"] in pacakge_names]

        sorted_pacakges = self._dfs_sort(
            self._build_dependency_graph(packages, use_rpm)
        )

        if self.os_family == "debian":
            sorted_pacakges = [
                re.sub("-devel$", "-dev", word) for word in sorted_pacakges
            ]

        return sorted_pacakges

    # ---------------------------------------------------------------------
    # OS & Installation Helpers
    # ---------------------------------------------------------------------
    def detect_os_family(self):
        """Detect OS family (debian/redhat/suse/unknown)."""
        os_release = {}
        try:
            with open("/etc/os-release", "r") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        os_release[k] = v.strip('"')
        except FileNotFoundError:
            system_name = platform.system().lower()
            return "linux" if "linux" in system_name else "unknown"

        os_id = os_release.get("ID", "").lower()
        os_like = os_release.get("ID_LIKE", "").lower()

        if "ubuntu" in os_id or "debian" in os_like:
            return "debian"
        elif (
            any(x in os_id for x in ["rhel", "centos"])
            or "redhat" in os_like
        ):
            return "redhat"
        elif "suse" in os_id or "sles" in os_id:
            return "suse"
        else:
            return "unknown"

    def derive_package_name(self, base, version_flag):
        """
        Derive full package name using base and flags.
        Example:
          base=hipfft, version_flag=True  -> hipfft7.0.0-gfx94x
          base=hipfft, version_flag=False -> hipfft-gfx94x
        """
        gfx_arch_flag = self.arch_suffix_flag(base)


        # Case 1: GFX architecture enabled
        if gfx_arch_flag and self.gfx_suffix and "devel" not in base.lower() and "dev" not in base.lower():
            if version_flag:
                return f"{base}{self.rocm_version}-{self.gfx_suffix}"
            else:
                return f"{base}-{self.gfx_suffix}"

        # Case 2: No GFX architecture, but versioning enabled
        elif version_flag:
            return f"{base}{self.rocm_version}"

        # Case 3: Plain base name (no version, no gfx)
        return base


    def find_packages_for_base(self, dest_dir, base, version_flag,use_repo):
        """
        Look up packages in local directory or return derived name for repo installation.
        """

        derived_name = self.derive_package_name(base, version_flag)
        if use_repo:
            return derived_name
        else:
            # If local directory has .deb/.rpm files → return matches
            all_files = [f for f in os.listdir(dest_dir) if f.endswith((".deb", ".rpm"))]
            matched = [os.path.join(dest_dir, f) for f in all_files if f.startswith(derived_name)]
            if matched:
                return matched
            else:
                logger.error(f"No matching package found for: {derived_name}")

    def _run_install_command(self, pkg_name, use_repo, pkg_path=None):
        """
        Build and run OS-specific install command for a package.
        
        :param pkg_name: Name of the package (base name)
        :param pkg_path: Full path for local install (required for local)
        :param source_type: 'local' or 'repo'
        """
        os_family = self.detect_os_family()
        cmd = None

        # Determine command based on source type and OS
        if not use_repo:
            if not pkg_path:
                logger.error(f"Local pkg_path must be provided for {pkg_name}")
                return

            if os_family == "debian":
                cmd = ["sudo", "dpkg", "-i", pkg_name]
            elif os_family == "redhat":
                cmd = ["sudo", "rpm", "-ivh", "--replacepkgs", pkg_name]
            elif os_family == "suse":
                cmd = ["sudo", "zypper", "--non-interactive", "install", "--replacepkgs", pkg_name]
            else:
                logger.error(f"Unsupported OS for local install: {pkg_name}")
                return

        else:
            if os_family == "debian":
                cmd = ["sudo", "apt-get", "install", "-y", pkg_name]
            elif os_family == "redhat":
                cmd = ["sudo", "yum", "install", "-y", pkg_name]
            elif os_family == "suse":
                cmd = ["sudo", "zypper", "--non-interactive", "install", pkg_name]
            else:
                logger.error(f"Unsupported OS for repo install: {pkg_name}")
                return

        # Execute command
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if result.returncode != 0:
                logger.error(f"Failed to install {pkg_name}:\n{result.stdout}")
            else:
                logger.info(f"Installed {pkg_name}")
        except Exception as e:
            logger.exception(f"Exception installing {pkg_name}: {e}")

    def _run_uninstall_command(self, pkg_name):
        """
        Build and run OS-specific install command for a package.

        :param pkg_name: Name of the package (base name)
        :param pkg_path: Full path for local install (required for local)
        :param source_type: 'local' or 'repo'
        """
        os_family = self.detect_os_family()
        cmd = None


        if os_family == "debian":
            cmd = ["sudo", "apt-get", "autoremove", "-y", pkg_name]
        elif os_family == "redhat":
            cmd = ["sudo", "yum", "remove", "-y", pkg_name]
        elif os_family == "suse":
            cmd = ["sudo", "zypper", "remove", pkg_name]
        else:
            logger.error(f"Unsupported OS for repo uninstall: {pkg_name}")
            return

        # Execute command
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if result.returncode != 0:
                logger.error(f"Failed to uninstall {pkg_name}:\n{result.stdout}")
            else:
                logger.info(f"Uninstalled {pkg_name}")
        except Exception as e:
            logger.exception(f"Exception uninstalling {pkg_name}: {e}")


    def uninstall_packages(self,pkg_list,composite_flag):
        """Uninstall the given list of packages based on OS."""

        if not pkg_list:
            logger.info("No packages provided for uninstallation.")
            return

        logger.info(f"Preparing to uninstall {len(pkg_list)} package(s): {pkg_list}")
        if composite_flag:
            for pkg in reversed(pkg_list):
                pkg = pkg.strip()
                if pkg:
                    derived_name = self.derive_package_name(pkg, True)
                    self._run_uninstall_command(derived_name)
        else:
            derived_name = self.derive_package_name("rocm-core", True)
            self._run_uninstall_command(derived_name)


    # ---------------------------------------------------------------------
    # Install Logic
    # ---------------------------------------------------------------------
    def install_packages(self, dest_dir, sorted_packages, version_flag, use_repo=False):
        os_family = self.detect_os_family()
        logger.info(f"Detected OS family: {os_family}")

        if not use_repo:
            if not os.path.isdir(dest_dir):
                logger.error(f"Artifacts directory not found: {dest_dir}")
                return

            final_install_list = []
            for base in sorted_packages:
                if version_flag:
                    pkgs = self.find_packages_for_base(dest_dir, base, version_flag,use_repo)
                    final_install_list.extend(pkgs or [])
                else:
                    pkgs = self.find_packages_for_base(dest_dir, base, True, use_repo)
                    final_install_list.extend(pkgs or [])
                    pkgs = self.find_packages_for_base(dest_dir, base, False, use_repo)
                    final_install_list.extend(pkgs or [])

            logger.info(f"Final install list count: {len(final_install_list)}")

            # logger.info(f"sorted_packages: {(final_install_list)}")
            if not final_install_list:
                logger.warning("No packages to install based on filters.")
                return

            # logger.info(f"sorted_packages: {(final_install_list)}")

            for pkg_name in final_install_list:
                try:
                    logger.info(f"Installing from local dir: {pkg_name}")
                    self._run_install_command(pkg_name, use_repo, dest_dir)
                except Exception as e:
                    logger.exception(f"Exception installing {pkg_name} from repo: {e}")
        else:
            self.populate_repo_file(dest_dir)
            # Post-upload: install via system repo
            final_install_list = []
            for base in sorted_packages:
                if version_flag:
                    pkgs = self.find_packages_for_base(dest_dir, base, version_flag,use_repo)
                    final_install_list.append(pkgs or [])
                else:
                    pkgs = self.find_packages_for_base(dest_dir, base, True, use_repo)
                    final_install_list.append(pkgs or [])
                    pkgs = self.find_packages_for_base(dest_dir, base, False, use_repo)
                    final_install_list.append(pkgs or [])
            for pkg_name in final_install_list:
                try:
                    logger.info(f"Installing from repo: {pkg_name}")
                    self._run_install_command(pkg_name, use_repo, None)
                except Exception as e:
                    logger.exception(f"Exception installing {pkg_name} from repo: {e}")

    # ---------------------------------------------------------------------
    # Repo Population
    # ---------------------------------------------------------------------
    def populate_repo_file(self, dest_dir: str):
        """
        Populate a repo file for post-upload installation.
        - Debian: creates /etc/apt/sources.list.d/rocm.list
        - RPM-based: placeholder (to be implemented)
        """
        os_family = self.detect_os_family()
        logger.info(f"Populating repo file for OS: {os_family}")

        try:
            base_url = f"https://therock-deb-rpm-test.s3.us-east-2.amazonaws.com/{self.amdgpu_family}_{dest_dir}"

            if os_family == "debian":
                repo_file_path = "/etc/apt/sources.list.d/rocm.list"
                repo_entry = f"deb [trusted=yes] {base_url}/deb stable main\n"

                logger.info(f"Writing Debian repo entry to {repo_file_path}")

                cmd = f'echo "{repo_entry.strip()}" | sudo tee {repo_file_path} > /dev/null'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                if result.returncode != 0:
                    logger.error(f"Failed to populate repo file: {result.stderr.strip()}")
                    raise RuntimeError(f"Error populating repo file: {result.stderr.strip()}")

                logger.info("Running apt-get update...")
                subprocess.run(["sudo", "apt-get", "update"], check=False)

            elif os_family == "redhat":
                logger.info("Detected RPM-based system. Placeholder for repo setup.")
                repo_file_path = "/etc/yum.repos.d/rocm.repo"
                repo_entry = (
                     f"[rocm]\nname=ROCm Repo\nbaseurl={base_url}/rpm\n"
                     "enabled=1\ngpgcheck=0\n"
                )
                with open(repo_file_path, "w") as f:
                     f.write(repo_entry)
                subprocess.run(["sudo", "yum", "clean", "all"], check=False)
                subprocess.run(["sudo", "yum", "makecache"], check=False)
            else:
                logger.warning(f"Unsupported OS family for repo population: {os_family}")

        except Exception as e:
            logger.error(f"Error populating repo file: {e}")
            raise

