# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Utilities for producing Python packages."""

from typing import Callable, Sequence

import importlib.util
import re
import os
from pathlib import Path
import platform
import shlex
import subprocess
import shutil
import sys
import tarfile

from .artifacts import ArtifactCatalog, ArtifactName
from .exe_stub_gen import generate_exe_link_stub

is_windows = platform.system() == "Windows"

if not is_windows:
    # Used on Linux to check file types. Buggy/broken on Windows, but file
    # types are generally known from file extensions there.
    import magic

BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
PYTHON_PACKAGING_DIR = BUILD_TOOLS_DIR / "packaging" / "python" / "templates"
DIST_INFO_PATH = PYTHON_PACKAGING_DIR / "rocm" / "src" / "rocm_sdk" / "_dist_info.py"

assert BUILD_TOOLS_DIR.exists()
assert PYTHON_PACKAGING_DIR.exists()
assert DIST_INFO_PATH.exists()

ENABLED_VLOG_LEVEL: int = 5


def log(*args, vlog: int = 0, **kwargs):
    if vlog > ENABLED_VLOG_LEVEL:
        return
    file = sys.stdout
    print(*args, **kwargs, file=file)
    file.flush()


class PopulatedFiles:
    """Tracks all populated files from the artifact catalog."""

    def __init__(self):
        self.materialized_relpaths: dict[str, tuple["PopulatedDistPackage", Path]] = {}
        # Mapping of relpath of shared library aliases to the soname
        self.soname_aliases: dict[str, str] = {}

    def has(self, relpath: str) -> bool:
        return relpath in self.materialized_relpaths

    def mark_populated(
        self, package: "PopulatedDistPackage", relpath: str, dest_path: Path
    ):
        assert (
            not relpath in self.materialized_relpaths
        ), f"File already populated {relpath}"
        self.materialized_relpaths[relpath] = (package, dest_path)


class Parameters:
    """Stores all parameters needed for package generation."""

    def __init__(
        self,
        dest_dir: Path,
        version: str,
        version_suffix: str,
        artifacts: ArtifactCatalog,
        kpack_split: bool = False,
    ):
        self.dest_dir = dest_dir
        self.version = version
        self.version_suffix = version_suffix
        self.artifacts = artifacts
        self.kpack_split = kpack_split
        self.all_target_families = artifacts.all_target_families
        _sorted_families = sorted(self.all_target_families)
        self.default_target_family: str | None = (
            _sorted_families[0] if _sorted_families else None
        )
        self.populated_packages: list["PopulatedDistPackage"] = []
        self.runtime_artifact_names: set[str] = set()

        # Load and interpolate the _dist_info.py template.
        # Base: version and nonce only — no family lines. Used as the starting
        # point for restrict_families packages so they can write clean family
        # content without a .clear() dance.
        dist_info_base = DIST_INFO_PATH.read_text()
        dist_info_base += f"__version__ = '{version}'\n"
        dist_info_base += f"PY_PACKAGE_SUFFIX_NONCE = '{version_suffix}'\n"
        self.dist_info_base_contents = dist_info_base

        # Full: base extended with all families. Used by most packages and by
        # the dynamically loaded self.dist_info module below.
        dist_info_contents = dist_info_base

        # In kpack-split mode, the libraries package is arch-neutral (no family
        # suffix). Override its dist_package_template so that
        # get_py_package_name(target_family=None) works for both the libraries
        # wheel itself and the device wheel overlay.
        if kpack_split:
            dist_info_contents += (
                'ALL_PACKAGES["libraries"].dist_package_template = '
                '"rocm-sdk-libraries"\n'
            )

        if self.default_target_family is not None:
            dist_info_contents += (
                f"DEFAULT_TARGET_FAMILY = '{self.default_target_family}'\n"
            )
        for target_family in self.all_target_families:
            dist_info_contents += (
                f"AVAILABLE_TARGET_FAMILIES.append('{target_family}')\n"
            )
        self.dist_info_contents = dist_info_contents

        # And dynamically load it so that we have access to the same config during
        # populate as will be loaded at setup and run time.
        spec = importlib.util.spec_from_loader("rocm_sdk_dist_info", loader=None)
        self.dist_info = importlib.util.module_from_spec(spec)
        exec(self.dist_info_contents, self.dist_info.__dict__)

    def filter_artifacts(
        self,
        filter: Callable[[ArtifactName], bool] = lambda _: True,
        includes: tuple[str] = (),
        excludes: tuple[str] = (),
    ) -> ArtifactCatalog:
        """Filters the global view of artifacts to only include a subset."""
        return ArtifactCatalog(
            self.artifacts.artifact_dir,
            filter=filter,
            includes=includes,
            excludes=excludes,
        )


class PopulatedDistPackage:
    """Represents a single populated dist package, bootstrapped from a template."""

    def __init__(
        self,
        params: Parameters,
        *,
        logical_name: str,
        target_family: str | None = None,
        restrict_families: bool = False,
    ):
        self.params = params
        self.logical_name = logical_name
        self.target_family = target_family
        try:
            self.entry = self.params.dist_info.ALL_PACKAGES[logical_name]
        except KeyError:
            raise KeyError(
                f"Logical package name {logical_name} not found (of {self.params.dist_info.ALL_PACKAGES.keys()})"
            )

        self.rpath_deps: list[tuple["PopulatedDistPackage", str]] = []
        self.files = PopulatedFiles()

        # restrict_families packages start from the base (no family lines) so
        # the per-family overrides below write clean content without a .clear()
        # dance. All other packages start from the full dist_info_contents.
        if restrict_families and target_family is not None:
            dist_info_contents = self.params.dist_info_base_contents
        else:
            dist_info_contents = self.params.dist_info_contents
        dist_info_contents += f"THIS_TARGET_FAMILY = {repr(target_family)}\n"
        dist_info_contents += (
            f"THIS_PACKAGE_ENTRY = ALL_PACKAGES[{repr(logical_name)}]\n"
        )

        # For per-family packages (e.g. meta/rocm in multi-arch builds), restrict
        # DEFAULT_TARGET_FAMILY and AVAILABLE_TARGET_FAMILIES to only this family
        # so that determine_target_family() at install time only resolves to this
        # family's packages. No .clear() needed — the base has an empty list.
        if restrict_families and target_family is not None:
            dist_info_contents += f"DEFAULT_TARGET_FAMILY = '{target_family}'\n"
            dist_info_contents += (
                f"AVAILABLE_TARGET_FAMILIES.append('{target_family}')\n"
            )

        # Device packages need to know the libraries package name so their
        # setup.py can declare the correct install_requires and overlay dir.
        if logical_name == "device":
            libraries_entry = self.params.dist_info.ALL_PACKAGES["libraries"]
            libraries_dist_name = libraries_entry.get_dist_package_name(
                target_family=None
            )
            libraries_py_package_name = libraries_entry.get_py_package_name(
                target_family=None
            )
            dist_info_contents += f"LIBRARIES_DIST_NAME = {repr(libraries_dist_name)}\n"
            dist_info_contents += (
                f"LIBRARIES_PY_PACKAGE_NAME = {repr(libraries_py_package_name)}\n"
            )

        # Populate from template.
        self.path = self._copy_package_template(
            self.params.dest_dir,
            self.entry.template_directory,
            dist_info_relpath=f"src/{self.entry.pure_py_package_name}/_dist_info.py",
            dist_info_contents=dist_info_contents,
            dest_name=self.entry.get_dist_package_name(target_family=target_family),
        )

        # Device packages overlay into the libraries package's platform dir
        # so .kpack files land alongside host .so files in site-packages.
        if logical_name == "device":
            libraries_entry = self.params.dist_info.ALL_PACKAGES["libraries"]
            self._platform_dir = (
                self.path
                / "platform"
                / libraries_entry.get_py_package_name(target_family=None)
            )
        else:
            self._platform_dir = (
                self.path
                / "platform"
                / self.entry.get_py_package_name(self.target_family)
            )

    @property
    def pure_dir(self) -> Path:
        """Gets the path of the pure package directory."""
        return self.path / "src" / self.entry.pure_py_package_name

    @property
    def platform_dir(self) -> Path:
        """Directory that contains platform files. Created on access."""
        self._platform_dir.mkdir(parents=True, exist_ok=True)
        (self._platform_dir / "__init__.py").touch()
        return self._platform_dir

    def rpath_dep(
        self, dep_package: "PopulatedDistPackage", rpath: str
    ) -> "PopulatedDistPackage":
        """Marks this package as needing to depend on RPATH from another package."""
        self.rpath_deps.append((dep_package, rpath))
        return self

    def _copy_package_template(
        self,
        dest_dir: Path,
        template_name: str,
        *,
        dist_info_relpath: str,
        dist_info_contents: str,
        dest_name: str | None = None,
    ) -> Path:
        """Copies a named template to the dest_dir and returns the Path.

        The template_name names a sub-directory of packaging/python. If no dest_name
        is specified, then a same-named sub-directory will be created in dest_dir
        and returned. Otherwise, it will be named dest_name.
        """
        if dest_name is None:
            dest_name = template_name
        template_path = PYTHON_PACKAGING_DIR / template_name
        if not template_path.exists():
            raise ValueError(f"Python package template {template_path} does not exist")
        package_dest_dir = dest_dir / dest_name
        if package_dest_dir.exists():
            shutil.rmtree(package_dest_dir)
        log(f"::: Creating package '{template_name}': {package_dest_dir}")
        shutil.copytree(
            template_path, package_dest_dir, symlinks=True, dirs_exist_ok=True
        )

        # Replace the _dist_info.py file.
        dist_info_path = package_dest_dir / dist_info_relpath
        log(f"  Writing dist info: {dist_info_path}")
        dist_info_path.parent.mkdir(parents=True, exist_ok=True)
        dist_info_path.write_text(dist_info_contents)

        return package_dest_dir

    def populate_runtime_files(
        self, artifacts: ArtifactCatalog
    ) -> "PopulatedDistPackage":
        """Populates runtime files in an artifact catalog to the platform directory.

        All populated files are tracked in `files`, and any that were previously
        populated to another package are skipped.

        In this context, "runtime" means that we are only populating a subset of files
        needed by runtime packages. Specifically, this impacts shared library symlinks,
        which only populate the soname variant in the link farm.
        """
        log(
            f"::: Populating runtime files {self.logical_name}[{self.target_family}]: "
            f"{self.path}"
        )
        for an, an_path in artifacts.artifact_basedirs:
            log(f"  + {an}: {an_path}")
            # Accumulate the artifact names we have created runtime packages for.
            # This will be used later to restrict devel packages to only these.
            self.params.runtime_artifact_names.add(an.name)

        package_dest_dir = self.platform_dir
        for relpath, dir_entry in artifacts.pm.matches():
            if self.files.has(relpath):
                continue
            dest_path = package_dest_dir / relpath
            if dir_entry.is_symlink():
                # Chase the symlink.
                self._populate_runtime_symlink(relpath, dest_path, dir_entry)
            else:
                # Copy the file.
                file_type = get_file_type(dir_entry)
                if file_type == "so":
                    # We only populate runtime shared libraries that correspond
                    # with their soname (or that don't have one).
                    soname = get_soname(dir_entry.path)
                    if soname:
                        if soname == dir_entry.name:
                            self._populate_file(
                                relpath, dest_path, dir_entry, resolve_src=True
                            )
                        else:
                            self.files.soname_aliases[relpath] = soname
                        continue
                # Otherwise, just copy the file.
                self._populate_file(relpath, dest_path, dir_entry, resolve_src=True)
        self.params.populated_packages.append(self)
        return self

    def populate_device_files(
        self, artifacts: ArtifactCatalog
    ) -> "PopulatedDistPackage":
        """Populates device files into the platform directory.

        Device artifacts are a mix of opaque binary data (.kpack archives,
        .co/.dat/.hsaco kernels, MIOpen DBs) and per-arch ELF shared
        libraries (e.g. libMIOpenCKGroupedConv_gfx1201.so). Data files
        are copied verbatim; ELF .so/exe files go through the same
        RPATH-patching path as populate_runtime_files() so their
        dynamic deps resolve across the kpack-split wheel layout.
        """
        log(
            f"::: Populating device files {self.logical_name}[{self.target_family}]: "
            f"{self.path}"
        )
        for an, an_path in artifacts.artifact_basedirs:
            log(f"  + {an}: {an_path}")

        package_dest_dir = self.platform_dir
        for relpath, dir_entry in artifacts.pm.matches():
            if self.files.has(relpath):
                continue
            dest_path = package_dest_dir / relpath
            self._populate_file(relpath, dest_path, dir_entry, resolve_src=True)
        self.params.populated_packages.append(self)
        return self

    def _populate_runtime_symlink(
        self, relpath: str, dest_path: Path, src_entry: os.DirEntry[str]
    ):
        # We can't have any symlinks in a runtime tree.
        # Here is what we do based on what it points to:
        #   1. Dangling or directory symlink: drop
        #   2. Shared library symlink: materialize if the symlink name is the SONAME
        #   3. Executable: Build a little executable launcher in place of the symlink
        #   4. Copy it into place (this should work for scripts and such -- hopefully).
        link_target = Path(src_entry.path).resolve()
        # Case 1: Drop dangling or directory symlink.
        if link_target.is_dir() or not link_target.exists():
            return
        file_type = get_file_type(link_target)
        # Case 2: Shared library.
        if file_type == "so" and (soname := get_soname(link_target)):
            if soname == src_entry.name:
                self._populate_file(relpath, dest_path, src_entry, resolve_src=True)
            else:
                self.files.soname_aliases[relpath] = soname
            return
        # Case 3: Executable.
        if file_type == "exe":
            # Compile a standalone executable that dynamically emulates the symlink.
            raw_link_target = os.readlink(src_entry.path)
            log(f"  EXESTUB: {relpath} (from {raw_link_target})", vlog=2)
            generate_exe_link_stub(dest_path, raw_link_target)
            self.files.mark_populated(self, relpath, dest_path)
            return
        # Case 4: Copy.
        self._populate_file(relpath, dest_path, src_entry, resolve_src=True)

    def _populate_file(
        self,
        relpath: str,
        dest_path: Path,
        src_entry: os.DirEntry[str],
        *,
        resolve_src: bool,
    ):
        is_symlink = src_entry.is_symlink()
        src_path = Path(src_entry.path)
        if resolve_src and is_symlink:
            src_path = src_path.resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # If it is a directory entry, just mkdir it.
        if src_entry.is_dir():
            dest_path.mkdir(parents=False, exist_ok=True)
            return

        # It is a regular file of some kind.
        if dest_path.exists():
            os.unlink(dest_path)
        # We have to patch many files, so we do not hard-link: always copy.
        log(f"  MATERIALIZE: {relpath} (from {src_path})", vlog=2)
        shutil.copy2(src_path, dest_path)
        if self.files.has(relpath):
            log(f"WARNING: Path already materialized: {relpath}")
        else:
            self.files.mark_populated(self, relpath, dest_path)

        if not is_windows:
            # Update RPATHs on Linux.
            file_type = get_file_type(dest_path)
            if file_type == "exe" or file_type == "so":
                self._extend_rpath(dest_path)
                self._normalize_rpath(dest_path)

    def _extend_rpath(self, file_path: Path):
        for dep_project, rpath in self.rpath_deps:
            parent_relpath = self._platform_dir.parent.relative_to(
                file_path.parent, walk_up=True
            )
            dep_py_package_name = dep_project.entry.get_py_package_name(
                self.target_family
            )
            addl_rpath = f"$ORIGIN/{parent_relpath}/{dep_py_package_name}/{rpath}"
            log(f"  ADD_RPATH: {file_path}: {addl_rpath}")
            patchelf_cl = [
                "patchelf",
                "--add-rpath",
                addl_rpath,
                str(file_path),
            ]
            subprocess.check_call(patchelf_cl)

    def _normalize_rpath(self, file_path: Path):
        existing_rpath = (
            subprocess.check_output(
                [
                    "patchelf",
                    "--print-rpath",
                    str(file_path),
                ]
            )
            .decode()
            .strip()
        )
        if not existing_rpath:
            return

        # Possibly in the future, do manual normalization of the RPATH.
        norm_rpath = existing_rpath

        log(f"  NORMALIZE_RPATH: {file_path}: {norm_rpath}")
        subprocess.check_call(
            [
                "patchelf",
                "--set-rpath",
                norm_rpath,
                # Forces the use of RPATH vs RUNPATH, which is more appropriate
                # for hermetic libraries like these since it does not allow
                # LD_LIBRARY_PATH to interfere.
                "--force-rpath",
                str(file_path),
            ]
        )

    def populate_devel_files(
        self,
        *,
        addl_artifact_names: Sequence[str] = (),
        exclude_components: Sequence[str] = (),
        tarball_compression: bool = True,
    ):
        """Populates all files that have not yet been materialized and symlink the rest."""
        package_path = self.platform_dir
        # Set up for what artifacts to include in the devel package, including
        # any emitted runtime artifacts plus additional requested.
        devel_artifact_names = set(self.params.runtime_artifact_names)
        devel_artifact_names.update(addl_artifact_names)
        # Exclude profiler-owned artifacts from the devel package.
        #
        # The profiler runtime (rocprofiler-compute and rocprofiler-systems)
        # is now packaged in the separate `rocm-profiler` wheel. However,
        # devel packaging automatically includes all runtime artifacts via
        # `runtime_artifact_names`, which would otherwise pull these profiler
        # artifacts back into the devel package.
        #
        # This leads to CI failures where devel tests attempt to load profiler
        # shared libraries without their full dependency closure (e.g. missing
        # rocprofiler-sdk or libomp resolution).
        #
        # Explicitly removing them here ensures correct package ownership:
        #   - rocm-profiler → owns profiler runtime
        #   - rocm-sdk-devel → does NOT include profiler runtime
        devel_artifact_names.difference_update(
            {
                "rocprofiler-compute",
                "rocprofiler-systems",
            }
        )
        excluded = set(exclude_components)
        log(f":: Devel artifact inclusions: {devel_artifact_names}")

        def _devel_artifact_filter(an: ArtifactName) -> bool:
            if an.name not in devel_artifact_names:
                # We didn't generate a runtime artifact for it, so no devel
                # artifact.
                return False
            if an.component in excluded:
                return False
            if an.target_family != "generic" and an.target_family != self.target_family:
                # Only include artifacts for this devel package's target family.
                return False
            return True

        artifacts = self.params.filter_artifacts(_devel_artifact_filter)
        log(f"::: Populating devel package {package_path}")
        for an, an_path in artifacts.artifact_basedirs:
            log(f"  + {an}: {an_path}")
        for relpath, dir_entry in artifacts.pm.matches():
            dest_path = package_path / relpath
            self._populate_devel_file(
                relpath,
                dest_path,
                dir_entry,
            )

        # For packaging, the devel platform/ contents are not wheel safe, so we
        # store them into their own tarball and dynamically decompress at runtime.
        # The tarball will contain as its first path component the top level
        # python package name that contains the platform files.
        tar_suffix = ".tar.xz" if tarball_compression else ".tar"
        tar_mode = "w:xz" if tarball_compression else "w"
        tar_path = self.pure_dir / f"_devel{tar_suffix}"
        log(f"::: Building secondary devel tarball: {tar_path}")
        with tarfile.open(tar_path, mode=tar_mode) as tf:
            for root, dirnames, files in os.walk(package_path):
                for file in list(files) + list(dirnames):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, package_path.parent)
                    log(f"Adding {arcname}", vlog=2)
                    tf.add(file_path, arcname=arcname, recursive=False)
        shutil.rmtree(package_path)

    def _find_populated(
        self, relpath: str
    ) -> "tuple[PopulatedDistPackage, Path] | None":
        """Search all populated runtime packages for a materialized relpath.

        When this devel package is target-family-specific, skip runtime packages
        that belong to a different (non-None) target family so that symlinks point
        into the correct arch's libraries package.
        """
        for pkg in self.params.populated_packages:
            if not pkg.files.has(relpath):
                continue
            if (
                self.target_family is not None
                and pkg.target_family is not None
                and pkg.target_family != self.target_family
            ):
                continue
            return pkg.files.materialized_relpaths[relpath]
        return None

    def _find_soname_alias(self, relpath: str) -> "str | None":
        """Search all populated runtime packages for a soname alias.

        When this devel package is target-family-specific, skip runtime packages
        that belong to a different (non-None) target family.
        """
        for pkg in self.params.populated_packages:
            alias = pkg.files.soname_aliases.get(relpath)
            if alias is None:
                continue
            if (
                self.target_family is not None
                and pkg.target_family is not None
                and pkg.target_family != self.target_family
            ):
                continue
            return alias
        return None

    def _populate_devel_file(
        self, relpath: str, dest_path: Path, src_entry: os.DirEntry[str]
    ):
        if src_entry.is_dir(follow_symlinks=False):
            dest_path.mkdir(parents=True, exist_ok=False)
            return

        # Re-add soname aliases.
        soname_alias = self._find_soname_alias(relpath)
        if soname_alias is not None:
            # This file is an alias to the proper soname. Just emit a relative
            # link.
            dest_path.symlink_to(soname_alias)
            return

        populated = self._find_populated(relpath)
        if populated is not None:
            # Already materialized: Link to it.
            populated_package, populated_path = populated
            # Materialize as a symlink to the original placement. This is tricky
            # because the symlink needs to be correct with respect to the install
            # placement, which is something like:
            #   _rocm_sdk_core/path/to/foo.txt
            #   _rocm_sdk_devel/path/to/foo.txt
            # In this example, relpath would be "path/to/foo.txt" and the proper
            # symlink:
            #   ../../../_rocm_sdk_core/path/to/foo.txt
            # Conveniently, the root sibling is len(relpath_segments) up from the
            # original_path.
            # This is admittedly ugly, but there is not a super convenient way to
            # do it given that the relative path is computed for an installation
            # layout that is different from the packaging input layout.
            relpath_segment_count = len(Path(relpath).parts)
            backtrack_path = Path(*([".."] * relpath_segment_count))
            link_target = backtrack_path / populated_package.platform_dir.name / relpath
            log(f"DEVLINK: {relpath} -> {link_target}", vlog=2)
            if dest_path.exists(follow_symlinks=False):
                dest_path.unlink()
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.symlink_to(link_target)
            return

        # If the source is a symlink, faithfully transcribe it.
        if src_entry.is_symlink():
            if dest_path.exists(follow_symlinks=False):
                dest_path.unlink()
            target_path = os.readlink(src_entry.path)
            log(f"LINK: {relpath} (to {target_path})", vlog=2)
            os.symlink(target_path, dest_path)
            return

        # Otherwise, no one else has emitted it, so just materialize verbatim.
        log(f"MATERIALIZE: {relpath} (from {src_entry.path})", vlog=2)
        if dest_path.exists(follow_symlinks=False):
            dest_path.unlink()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_entry.path, dest_path)

        if not is_windows:
            # Update RPATHs on Linux.
            file_type = get_file_type(dest_path)
            if file_type == "exe" or file_type == "so":
                self._normalize_rpath(dest_path)


MAGIC_AR_MATCH = re.compile("ar archive")
MAGIC_EXECUTABLE_MATCH = re.compile("ELF[^,]+executable,")
MAGIC_SO_MATCH = re.compile("ELF[^,]+shared object,")


def get_file_type(dir_entry: os.DirEntry[str] | Path) -> str:
    if isinstance(dir_entry, os.DirEntry):
        path = Path(dir_entry.path)
    else:
        path = Path(dir_entry)

    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "dir"

    # We only care about finding certain needles in the haystack, so exclude
    # text-like files from the get-go.
    path = str(path)
    if path.endswith(".txt") or path.endswith(".h") or path.endswith(".hpp"):
        return "text"
    elif path.endswith(".hsaco") or path.endswith(".co"):
        # These read as shared libraries.
        return "hsaco"
    elif path.endswith(".lib"):
        return "ar"
    elif path.endswith(".exe"):
        return "exe"

    if is_windows:
        # Don't try to use 'magic' on Windows, since it is buggy/broken.
        # Hopefully the file type was covered by an extension check above.
        return "other"

    desc = magic.from_file(path)
    if MAGIC_EXECUTABLE_MATCH.search(desc):
        return "exe"
    if MAGIC_SO_MATCH.search(desc):
        return "so"
    if MAGIC_AR_MATCH.search(desc):
        return "ar"
    return "other"


def get_soname(sofile: Path) -> str:
    return (
        subprocess.check_output(["patchelf", "--print-soname", str(sofile)])
        .decode()
        .strip()
    )


def build_packages(
    dest_dir: Path,
    *,
    wheel_compression: bool = True,
    package_dirs: list[Path] | None = None,
    dist_dir: Path | None = None,
):
    effective_dist_dir = dist_dir or (dest_dir / "dist")
    effective_dist_dir.mkdir(parents=True, exist_ok=True)
    if package_dirs is None:
        package_dirs = [
            p
            for p in dest_dir.iterdir()
            if p.is_dir() and (p / "pyproject.toml").exists()
        ]
    for child_path in package_dirs:
        child_name = child_path.name

        # Some of our packages build as sdists and some as wheels.
        # Contrary to documented wisdom, we invoke setuptools directly. This is
        # because the "build frontends" have an impossible compatibility matrix
        # and opinions about how to pass arguments to the backends. So we skip
        # the frontends for such a closed case as this.
        setuppy_path = child_path / "setup.py"
        build_args = [
            sys.executable,
            str(setuppy_path.resolve()),
        ]
        if child_name in ["rocm"]:
            build_args.append("sdist")
        else:
            build_args.append("bdist_wheel")
            if not wheel_compression:
                build_args.append("--compression")
                build_args.append("stored")
        build_args.extend(
            [
                "-v",
                "--dist-dir",
                str(effective_dist_dir.resolve()),
            ]
        )

        log(f"::: Building python package {child_name}: {shlex.join(build_args)}")
        subprocess.check_call(build_args, cwd=child_path, stderr=subprocess.STDOUT)
