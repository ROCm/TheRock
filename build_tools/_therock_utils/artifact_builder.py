"""Builds artifacts from a descriptor.

See `artifacts` for a general description of artifacts and utilities for processing
them once built.
"""

from pathlib import Path
import platform


class ComponentDefaults:
    """Defaults for to apply to artifact merging by component name."""

    ALL: dict[str, "ComponentDefaults"] = {}

    def __init__(self, name: str = "", includes=(), excludes=()):
        self.includes = list(includes)
        self.excludes = list(excludes)
        if name:
            if name in ComponentDefaults.ALL:
                raise KeyError(f"ComponentDefaults {name} already defined")
            ComponentDefaults.ALL[name] = self

    @staticmethod
    def get(name: str) -> "ComponentDefaults":
        return ComponentDefaults.ALL.get(name) or ComponentDefaults(name)


# Debug components collect all platform specific dbg file patterns.
ComponentDefaults(
    "dbg",
    includes=[
        # Linux build-id based debug files.
        ".build-id/**/*.debug",
    ],
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
)
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
ComponentDefaults("run")
ComponentDefaults("doc", includes=["**/share/doc/**"])

# To help layering, we make lib/dev/run default patterns exclude patterns
# that the others define. This makes it easier for one of these to do directory
# level includes and have the files sorted into the proper component.
ComponentDefaults.get("dev").excludes.extend(ComponentDefaults.get("lib").includes)
ComponentDefaults.get("dev").excludes.extend(ComponentDefaults.get("run").includes)
ComponentDefaults.get("dev").excludes.extend(ComponentDefaults.get("doc").includes)
ComponentDefaults.get("lib").excludes.extend(ComponentDefaults.get("dev").includes)
ComponentDefaults.get("lib").excludes.extend(ComponentDefaults.get("run").includes)
ComponentDefaults.get("lib").excludes.extend(ComponentDefaults.get("doc").includes)
ComponentDefaults.get("run").excludes.extend(ComponentDefaults.get("dev").includes)
ComponentDefaults.get("run").excludes.extend(ComponentDefaults.get("lib").includes)
ComponentDefaults.get("run").excludes.extend(ComponentDefaults.get("doc").includes)


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
    ALLOWED_KEYS = set()

    def __init__(self, name, record: dict):
        self.name = name
        self.basedirs: dict[str, ComponentBasedirDescriptor] = {}
        for basedir_relpath, basedir_record in record.items():
            self.basedirs[basedir_relpath] = ComponentBasedirDescriptor(
                basedir_relpath, basedir_record
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

    def __init__(self, basedir_relpath: str, record: dict):
        _check_allowed_keys(record, ComponentBasedirDescriptor.ALLOWED_KEYS)
        self.basedir_relpath = basedir_relpath
        self.use_default_patterns = record.get("default_patterns", True)
        self.optional = _evaluate_optional(record.get("optional"))
        self.force_includes = _dup_list_or_str(record.get("force_include"))
        self.includes = _dup_list_or_str(record.get("include"))
        self.excludes = _dup_list_or_str(record.get("exclude"))


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


def _dup_list_or_str(v: list[str] | str) -> list[str]:
    if not v:
        return []
    if isinstance(v, str):
        return [v]
    return list(v)
