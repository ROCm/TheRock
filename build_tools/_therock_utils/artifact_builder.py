"""Builds artifacts from a descriptor.

See `artifacts` for a general description of artifacts and utilities for processing
them once built.
"""

import os
from pathlib import Path
import platform

from _therock_utils.pattern_match import PatternMatcher, MatchPredicate


class ComponentDefaults:
    """Defaults for to apply to artifact merging by component name."""

    ALL: dict[str, "ComponentDefaults"] = {}

    def __init__(self, name: str = "", includes=(), excludes=(), extends=()):
        self.includes = list(includes)
        self.excludes = list(excludes)
        self.extends = list(extends)
        if name:
            if name in ComponentDefaults.ALL:
                raise KeyError(f"ComponentDefaults {name} already defined")
            ComponentDefaults.ALL[name] = self

    @staticmethod
    def get(name: str) -> "ComponentDefaults":
        return ComponentDefaults.ALL.get(name) or ComponentDefaults(name)


# Lib components include shared libraries, dlls and any assets needed for use
# of shared libraries at runtime. Files are included by name pattern and
# descriptors should include/exclude non-standard variations.
ComponentDefaults(
    "lib",
    includes=[
        "**/*.dll",
        "**/*.dylib",
        "**/*.dylib.*",
        "**/*.so",
        "**/*.so.*",
    ],
    excludes=[],
)
# Run components layer on top of 'lib' components and also include executables
# and tools that are not needed by library consumers. Descriptors should
# explicitly include "bin" directory contents as needed.
ComponentDefaults("run", extends=["lib"])


# Debug components collect all platform specific dbg file patterns.
ComponentDefaults(
    "dbg",
    includes=[
        # Linux build-id based debug files.
        ".build-id/**/*.debug",
    ],
    extends=["run"],
)

# Dev components include all static library based file patterns and
# exclude file name patterns implicitly included for "run" and "lib".
# Descriptors should explicitly include header file any package file
# sub-trees that do not have an explicit "cmake" or "include" path components
# in them.
ComponentDefaults(
    "dev",
    includes=[
        "**/*.a",
        "**/*.lib",
        "**/cmake/**",
        "**/include/**",
        "**/share/modulefiles/**",
        "**/pkgconfig/**",
    ],
    excludes=[],
    extends=["dbg"],
)
ComponentDefaults("doc", includes=["**/share/doc/**"], extends=["dbg"])

# To help layering, we make lib/dev/run default patterns exclude patterns
# that the others define. This makes it easier for one of these to do directory
# level includes and have the files sorted into the proper component.
# ComponentDefaults.get("dev").excludes.extend(ComponentDefaults.get("lib").includes)
# ComponentDefaults.get("dev").excludes.extend(ComponentDefaults.get("run").includes)
# ComponentDefaults.get("dev").excludes.extend(ComponentDefaults.get("doc").includes)
# ComponentDefaults.get("lib").excludes.extend(ComponentDefaults.get("dev").includes)
# ComponentDefaults.get("lib").excludes.extend(ComponentDefaults.get("run").includes)
# ComponentDefaults.get("lib").excludes.extend(ComponentDefaults.get("doc").includes)
# ComponentDefaults.get("run").excludes.extend(ComponentDefaults.get("dev").includes)
# ComponentDefaults.get("run").excludes.extend(ComponentDefaults.get("lib").includes)
# ComponentDefaults.get("run").excludes.extend(ComponentDefaults.get("doc").includes)


class ArtifactDescriptor:
    """An artifact descriptor is typically loaded from a TOML file with records like:

        "components" : dict of covered component names
            "{component_name}": dict of build/ relative paths to materialize
                "{stage_directory}":
                    "default_patterns": bool (default True) whether component default
                        patterns are used
                    "include": str or list[str] of include patterns
                    "exclude": str or list[str] of exclude patterns
                    "force_include": str or list[str] of include patterns that if
                        matched, force inclusion, regardless of whether they match
                        an exclude pattern.
                    "optional": if true and the directory does not exist, it
                      is not an error. Use for optionally built projects. This
                      can also be either a string or array of strings, which
                      are interpreted as a platform name. If the case-insensitive
                      `platform.system()` equals one of them, then it is
                      considered optional.
    Most sections can typically be blank because by default they use
    component specific include/exclude patterns (see `COMPONENT_DEFAULTS` above)
    that cover most common cases. Local deviations must be added explicitly
    in the descriptor.
    """

    ALLOWED_KEYS = set(["components"])

    def __init__(self, record: dict):
        _check_allowed_keys(record, ArtifactDescriptor.ALLOWED_KEYS)
        self.components: dict[str, "ComponentDescriptor"] = {}
        # Populate components record.
        try:
            components_record = record["components"]
        except KeyError:
            # No components.
            pass
        else:
            for name, component_record in components_record.items():
                component = ComponentDescriptor(name, component_record)
                self.components[name] = component

        # Add an empty component for each default component, since they form a
        # chain of extensions that must exist.
        for default_name in ComponentDefaults.ALL.keys():
            if default_name not in self.components:
                self.components[default_name] = ComponentDescriptor(default_name, {})

    @staticmethod
    def load_toml_file(p: Path) -> "ArtifactDescriptor":
        try:
            import tomllib
        except ModuleNotFoundError:
            # Python <= 3.10 compatibility (requires install of 'tomli' package)
            import tomli as tomllib
        with open(p, "rb") as f:
            kwdict = tomllib.load(f)
        try:
            return ArtifactDescriptor(kwdict or {})
        except Exception as e:
            raise ValueError(f"Error while loading descriptor from {p}") from e


class ComponentDescriptor:
    ALLOWED_KEYS = set(["extends"])

    def __init__(self, name, record: dict):
        self.name = name
        self.basedirs: dict[str, ComponentBasedirDescriptor] = {}
        # All dict-valued fields are basedir_records. Others are fields.
        basedir_records: dict[str, dict] = {}
        fields = {}
        for key, value in record.items():
            if isinstance(value, dict):
                basedir_records[key] = value
            else:
                fields[key] = value
        _check_allowed_keys(fields, ComponentDescriptor.ALLOWED_KEYS)

        # Resolve extends if explicit or get it from defaults.
        optional_extends = fields.get("extends")
        if optional_extends is None:
            self.extends = list(self.defaults.extends)
        else:
            self.extends = _dup_list_or_str(optional_extends)

        # Instantiate all children.
        for basedir_relpath, basedir_record in basedir_records.items():
            self.basedirs[basedir_relpath] = ComponentBasedirDescriptor(
                self, basedir_relpath, basedir_record
            )

    @staticmethod
    def empty(self, name: str) -> "ComponentDescriptor":
        return ComponentDescriptor(name, {})

    @property
    def defaults(self) -> ComponentDefaults:
        found = ComponentDefaults.ALL.get(self.name)
        if not found:
            return ComponentDefaults()
        return found


class ComponentBasedirDescriptor:
    ALLOWED_KEYS = set(
        [
            "default_patterns",
            "exclude",
            "force_include",
            "include",
            "optional",
        ]
    )

    def __init__(
        self, component: ComponentDescriptor, basedir_relpath: str, record: dict
    ):
        _check_allowed_keys(record, ComponentBasedirDescriptor.ALLOWED_KEYS)
        self.basedir_relpath = basedir_relpath
        self.optional = _evaluate_optional(record.get("optional"))
        use_default_patterns = record.get("default_patterns", True)
        defaults = component.defaults

        includes = _dup_list_or_str(record.get("include"))
        if use_default_patterns:
            includes.extend(defaults.includes)
        excludes = _dup_list_or_str(record.get("exclude"))
        if use_default_patterns:
            excludes.extend(defaults.excludes)
        force_includes = _dup_list_or_str(record.get("force_include"))

        self.predicate = MatchPredicate(
            includes=includes,
            excludes=excludes,
            force_includes=force_includes,
        )


class ComponentContents:
    def __init__(self, cd: ComponentDescriptor):
        self.component_descriptor = cd
        # The set of all transitive extended paths. These are ignored when
        # populating this component.
        self.transitive_relpaths: set[str] = set()

        # We poke any files that we want in this component directly into this
        # pattern matcher.
        self.basedir_contents: dict[str, PatternMatcher] = dict()

    def write_artifact(self, destdir: Path):
        for basedir_relpath, pm in self.basedir_contents.items():
            pm.copy_to(
                destdir=destdir, destprefix=basedir_relpath + "/", remove_dest=False
            )
        # Write a manifest containing relative paths of all base directories.
        manifest_path = destdir / "artifact_manifest.txt"
        manifest_path.write_text("\n".join(self.basedir_contents.keys()) + "\n")


class ComponentScanner:
    """Takes an ArtifactDescriptor and sorts all files into a component."""

    def __init__(self, root_dir: Path, ad: ArtifactDescriptor):
        self.root_dir = root_dir

        # Each distinct basedir gets one PatternMatcher, so that we only scan
        # each directory once.
        self.basedir_cache: dict[str, PatternMatcher] = dict()

        # As we resolve a component, store it here.
        self.components: dict[str, ComponentContents] = dict()

        # Map of the first relpath to DirEntry that we have seen across all
        # components.
        self.all_entries: dict[str, os.DirEntry[str]] = dict()

        # Set of all relpaths that have been consumed by some component.
        # Together with all_entries, this can be used to lint for unmatched paths.
        self.matched_relpaths: set[str] = set()

        # Now process each component descriptor in a worklist fashion, populating
        # it when it has no extends that are unresolved.
        worklist: list[ComponentDescriptor] = list(ad.components.values())
        next_worklist: list[ComponentDescriptor] = []
        forward_progress = True
        while worklist:
            # Error if we did not make forward progress.
            if not forward_progress:
                illegal_names = ", ".join(
                    [f"{cd.name} -> {'|'.join(cd.extends)}" for cd in worklist]
                )
                raise ValueError(
                    f"The following components have non existing or circular extends: "
                    f"{illegal_names}"
                )
            forward_progress = False

            # Make one pass over the worklist.
            for cd in worklist:
                # If all extends have been populated, then we can process.
                extends = self._resolve_extends(cd.extends)
                if extends is None:
                    # Indicates that one or more extends are not yet resolved.
                    # Will try again on the next pass.
                    next_worklist.append(cd)
                    continue

                self._populate_component(cd, extends)
                forward_progress = True

            # Swap worklist.
            worklist = next_worklist
            next_worklist = []

    @property
    def unmatched_files(self) -> list[str, os.DirEntry[str]]:
        return [
            relpath
            for relpath, direntry in self.all_entries.items()
            if relpath not in self.matched_relpaths and not direntry.is_dir()
        ]

    @property
    def all_basedirs(self) -> list[str]:
        return list(self.basedir_cache.keys())

    def _populate_component(
        self, cd: ComponentDescriptor, extends: set[ComponentContents]
    ):
        print(f"*** Populate component: {cd.name}")
        contents = ComponentContents(cd)

        # Anything that we are extending goes in the hitlist that we will skip
        # as part of this one.
        for extend_contents in extends:
            contents.transitive_relpaths.update(extend_contents.transitive_relpaths)

        # Process each basedir.
        for bd in cd.basedirs.values():
            pm = self._get_basedir(bd.basedir_relpath)
            dest_pm = PatternMatcher()
            contents.basedir_contents[bd.basedir_relpath] = dest_pm
            for relpath, direntry in pm.matches():
                if relpath in contents.transitive_relpaths:
                    # Ignore - implicit no match because we have already seen it.
                    continue
                if relpath not in self.all_entries:
                    self.all_entries[relpath] = direntry
                if bd.predicate.matches(relpath, direntry):
                    # Match. Add it.
                    self.matched_relpaths.add(relpath)
                    dest_pm.add_entry(relpath, direntry)
                    contents.transitive_relpaths.add(relpath)
                    print(f"** ADD: {relpath} -> {direntry}")

        # Memorize that we have processed this descriptor.
        self.components[cd.name] = contents

    def _resolve_extends(self, extends: list[str]) -> set[ComponentDescriptor] | None:
        cds: set[ComponentDescriptor] = set()
        for name in extends:
            cd = self.components.get(name)
            if cd is None:
                return None
            cds.add(cd)
        return cds

    def _get_basedir(self, basedir: str) -> PatternMatcher:
        pm = self.basedir_cache.get(basedir)
        if pm is None:
            pm = PatternMatcher()
            pm.add_basedir(self.root_dir / basedir)
            self.basedir_cache[basedir] = pm
        return pm


def _check_allowed_keys(record: dict, allowed_keys: set[str]):
    for key in record.keys():
        if key not in allowed_keys:
            raise ValueError(
                f"Descriptor contains illegal key: '{key}' (keys: {record.keys()}) "
                f"(allowed: {allowed_keys})"
            )


def _evaluate_optional(optional_value) -> bool:
    """Returns true if the given value should be considered optional on this platform.

    It can be either a str, list of str, or a truthy value. If a str/list, then it will
    return true if any of the strings match the case insensitive
    `platform.system()`.
    """
    if optional_value is None:
        return False
    if isinstance(optional_value, str):
        optional_value = [optional_value]
    if isinstance(optional_value, list):
        system_name = platform.system().lower()
        for v in optional_value:
            if str(v).lower() == system_name:
                return True
        return False
    return bool(optional_value)


def _dup_list_or_str(v: list[str] | str | None) -> list[str]:
    if not isinstance(v, (list, str, type(None))):
        raise ValueError(f"Expected list, str or None but got: {v}")
    if not v:
        return []
    if isinstance(v, str):
        return [v]
    return [str(it) for it in v]
