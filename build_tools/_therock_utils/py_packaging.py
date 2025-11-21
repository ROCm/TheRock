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
    ):
        self.dest_dir = dest_dir
        self.version = version
        self.version_suffix = version_suffix
        self.artifacts = artifacts
        self.all_target_families = artifacts.all_target_families
        self.default_target_family = sorted(self.all_target_families)[0]
        self.files = PopulatedFiles()
        self.runtime_artifact_names: set[str] = set()

        # Load and interpolate the _dist_info.py template.
        dist_info_contents = DIST_INFO_PATH.read_text()
        dist_info_contents += f"__version__ = '{version}'\n"
        dist_info_contents += f"PY_PACKAGE_SUFFIX_NONCE = '{version_suffix}'\n"
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

        # Augment the dist_info with THIS_TARGET_FAMILY and THIS_PACKAGE_ENTRY
        dist_info_contents = self.params.dist_info_contents
        dist_info_contents += f"THIS_TARGET_FAMILY = {repr(target_family)}\n"
        dist_info_contents += (
            f"THIS_PACKAGE_ENTRY = ALL_PACKAGES[{repr(logical_name)}]\n"
        )

        # Populate from template.
        self.path = self._copy_package_template(
            self.params.dest_dir,
            self.entry.template_directory,
            dist_info_relpath=f"src/{self.entry.pure_py_package_name}/_dist_info.py",
            dist_info_contents=dist_info_contents,
            dest_name=self.entry.get_dist_package_name(target_family=target_family),
        )

        self._platform_dir = (
            self.path / "platform" / self.entry.get_py_package_name(self.target_family)
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

        files = self.params.files
        package_dest_dir = self.platform_dir
        for relpath, dir_entry in artifacts.pm.matches():
            if files.has(relpath):
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
                            self.params.files.soname_aliases[relpath] = soname
                        continue
                # Otherwise, just copy the file.
                self._populate_file(relpath, dest_path, dir_entry, resolve_src=True)
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
                self.params.files.soname_aliases[relpath] = soname
            return
        # Case 3: Executable.
        if file_type == "exe":
            # Compile a standalone executable that dynamically emulates the symlink.
            raw_link_target = os.readlink(src_entry.path)
            log(f"  EXESTUB: {relpath} (from {raw_link_target})", vlog=2)
            generate_exe_link_stub(dest_path, raw_link_target)
            self.params.files.mark_populated(self, relpath, dest_path)
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
        if self.params.files.has(relpath):
            log(f"WARNING: Path already materialized: {relpath}")
        else:
            self.params.files.mark_populated(self, relpath, dest_path)

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
        try:
            existing_rpath = (
                subprocess.check_output(
                    [
                        "patchelf",
                        "--print-rpath",
                        str(file_path),
                    ],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError:
            # patchelf failed (for wmma it was a binary with too many sections), skip normalization
            log(
                f"  SKIP_RPATH: {file_path.name}: patchelf failed, skipping RPATH normalization"
            )
            return

        if not existing_rpath:
            return

        # Possibly in the future, do manual normalization of the RPATH.
        norm_rpath = existing_rpath

        log(f"  NORMALIZE_RPATH: {file_path}: {norm_rpath}")
        try:
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
                ],
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            log(f"  SKIP_RPATH_SET: {file_path.name}: patchelf could not modify RPATH")

    def populate_devel_files(
        self,
        *,
        addl_artifact_names: Sequence[str] = (),
        tarball_compression: bool = True,
    ):
        """Populates all files that have not yet been materialized and symlink the rest."""
        package_path = self.platform_dir
        # Set up for what artifacts to include in the devel package, including
        # any emitted runtime artifacts plus additional requested.
        devel_artifact_names = set(self.params.runtime_artifact_names)
        devel_artifact_names.update(addl_artifact_names)
        log(f":: Devel artifact inclusions: {devel_artifact_names}")

        def _devel_artifact_filter(an: ArtifactName) -> bool:
            if an.name not in devel_artifact_names:
                # We didn't generate a runtime artifact for it, so no devel
                # artifact.
                return False
            if (
                an.target_family != "generic"
                and an.target_family != self.params.default_target_family
            ):
                # We only materialize the default target family for devel packages.
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

    def _populate_devel_file(
        self, relpath: str, dest_path: Path, src_entry: os.DirEntry[str]
    ):
        if src_entry.is_dir(follow_symlinks=False):
            dest_path.mkdir(parents=True, exist_ok=False)
            return

        # Re-add soname aliases.
        soname_alias = self.params.files.soname_aliases.get(relpath)
        if soname_alias is not None:
            # This file is an alias to the proper soname. Just emit a relative
            # link.
            dest_path.symlink_to(soname_alias)
            return

        if self.params.files.has(relpath):
            # Already materialized: Link to it.
            populated_package, populated_path = self.params.files.materialized_relpaths[
                relpath
            ]
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


def build_packages(dest_dir: Path, *, wheel_compression: bool = True):
    dist_dir = dest_dir / "dist"
    for child_path in dest_dir.iterdir():
        if not child_path.is_dir():
            continue
        if not (child_path / "pyproject.toml").exists():
            continue
        child_name = child_path.name

        # Some of our packages build as sdists and some as wheels.
        # Contrary to documented wisdom, we invoke setuptools directly. This is
        # because the "build frontends" have an impossible compatibility matrix
        # and opinions about how to pass arguments to the backends. So we skip
        # the frontends for such a closed case as this.
        build_args = [sys.executable, "-m", "build", "-v", "--outdir", str(dist_dir)]
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
                str(dist_dir.resolve()),
            ]
        )

        log(f"::: Building python package {child_name}: {shlex.join(build_args)}")
        subprocess.check_call(build_args, cwd=child_path, stderr=subprocess.STDOUT)
