# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Maps monorepo project paths to artifact names for granular reuse.

This module enables artifact-level impact analysis within monorepo submodules
like rocm-libraries. When a change is detected in a specific project directory,
this mapping allows the CI system to rebuild only the affected artifacts while
reusing unaffected artifacts from a baseline run.

Example:
    A change to rocm-libraries/projects/rocblas/ only impacts the "blas" artifact,
    not the "fft", "prim", or "rand" artifacts which can be reused.
"""

from __future__ import annotations

from typing import Optional


# Mapping from rocm-libraries project directories to artifact names.
# Each key is a path prefix (relative to the rocm-libraries root) that maps
# to an artifact name defined in BUILD_TOPOLOGY.toml.
ROCM_LIBRARIES_PROJECT_MAP: dict[str, str] = {
    # BLAS libraries
    "projects/rocblas": "blas",
    "projects/hipblas": "blas",
    "projects/hipblas-common": "blas",
    "projects/hipblaslt": "blas",
    "projects/hipsparselt": "blas",  # Part of BLAS artifact per BUILD_TOPOLOGY
    # Solver libraries
    "projects/hipsolver": "solver",
    "projects/rocsolver": "solver",
    # Sparse libraries
    "projects/hipsparse": "sparse",
    "projects/rocsparse": "sparse",
    # FFT libraries
    "projects/rocfft": "fft",
    "projects/hipfft": "fft",
    # Primitives libraries
    "projects/rocprim": "prim",
    "projects/hipcub": "prim",
    "projects/rocthrust": "prim",
    # Random number libraries
    "projects/rocrand": "rand",
    "projects/hiprand": "rand",
    # Composable kernel and tensor libraries
    "projects/composablekernel": "composable-kernel",
    "projects/hiptensor": "hiptensor",
    # Other math libraries
    "projects/rocwmma": "rocwmma",
    "projects/rocalution": "rocalution",
    # ML libraries (ml-libs artifact group)
    "projects/miopen": "miopen",
    "projects/hipdnn": "hipdnn",
    # RPP (ROCm Performance Primitives) - part of support
    "projects/rpp": "support",
    # Threading library - part of prim
    "projects/hipthreads": "prim",
}


# Paths within rocm-libraries that affect ALL artifacts in the source set.
# Any change to these paths triggers a conservative rebuild of all artifacts
# that depend on rocm-libraries.
ROCM_LIBRARIES_GLOBAL_PATHS: tuple[str, ...] = (
    "shared/",
    "cmake/",
    "CMakeLists.txt",
    ".github/",
    ".gitmodules",
    ".gitattributes",
    ".clang-format",
    ".cmake-format.py",
    ".pre-commit-config.yaml",
    # dnn-providers affects multiple ml-libs artifacts
    "dnn-providers/",
    # tools affect build process
    "tools/",
    # test infrastructure affects multiple artifacts
    "test/",
    # docs changes might be coupled with code changes
    "docs/",
)


def get_artifact_for_path(submodule: str, path: str) -> Optional[str]:
    """Map a path within a submodule to a specific artifact name.

    This function enables granular artifact-level impact analysis by mapping
    changed file paths to the specific artifacts they affect.

    Args:
        submodule: The submodule name (e.g., "rocm-libraries").
        path: The path within the submodule (e.g., "projects/rocblas/src/foo.cpp").

    Returns:
        The artifact name if a specific mapping exists, or None if the path
        affects all artifacts (conservative fallback). None indicates the
        caller should treat this as impacting all artifacts in the source set.

    Example:
        >>> get_artifact_for_path("rocm-libraries", "projects/rocblas/src/foo.cpp")
        "blas"
        >>> get_artifact_for_path("rocm-libraries", "shared/common.hpp")
        None  # affects all artifacts
        >>> get_artifact_for_path("llvm-project", "llvm/lib/foo.cpp")
        None  # no granular mapping for llvm-project
    """
    # Only rocm-libraries has granular project-to-artifact mapping
    if submodule != "rocm-libraries":
        return None

    # Check if it's a global path that affects all artifacts
    for global_prefix in ROCM_LIBRARIES_GLOBAL_PATHS:
        if global_prefix.endswith("/"):
            if path.startswith(global_prefix) or path == global_prefix.rstrip("/"):
                return None  # Conservative: affects all
        else:
            if path == global_prefix:
                return None  # Conservative: affects all

    # Try to match a specific project directory
    for project_prefix, artifact in ROCM_LIBRARIES_PROJECT_MAP.items():
        # Match exact directory or files within the directory
        if path.startswith(project_prefix + "/") or path == project_prefix:
            return artifact

    # Unknown path within rocm-libraries - conservative fallback
    # This could be a new project directory or miscellaneous file
    return None


def get_all_rocm_libraries_artifacts() -> frozenset[str]:
    """Return all artifact names that can be produced from rocm-libraries.

    This is used for the conservative fallback when a global path changes
    and all rocm-libraries artifacts need to be rebuilt.

    Returns:
        A frozenset of all artifact names mapped from rocm-libraries projects.
    """
    return frozenset(ROCM_LIBRARIES_PROJECT_MAP.values())


def parse_changed_path(changed_path: str) -> tuple[str | None, str | None]:
    """Parse a repository-relative changed path into submodule and subpath.

    Args:
        changed_path: A repository-relative path (e.g., "rocm-libraries/projects/rocblas/foo.cpp").

    Returns:
        A tuple of (submodule, subpath) if the path is within a submodule,
        or (None, None) if the path cannot be parsed.

    Example:
        >>> parse_changed_path("rocm-libraries/projects/rocblas/foo.cpp")
        ("rocm-libraries", "projects/rocblas/foo.cpp")
        >>> parse_changed_path("build_tools/foo.py")
        ("build_tools", "foo.py")
    """
    parts = changed_path.split("/", 1)
    if len(parts) < 2:
        # Single component path - treat the whole path as the submodule
        return (changed_path, "")
    return (parts[0], parts[1])


def resolve_artifacts_for_paths(
    changed_paths: list[str],
    source_set_submodules: dict[str, list[str]],
) -> tuple[set[str], bool]:
    """Resolve changed paths to specific artifacts within rocm-libraries.

    This is the main entry point for granular artifact analysis. Given a list
    of changed paths and the source set configuration, it determines which
    specific artifacts are impacted.

    Args:
        changed_paths: List of repository-relative changed paths.
        source_set_submodules: Mapping of source set name to list of submodules.
            Used to determine if a path is within a known monorepo submodule.

    Returns:
        A tuple of (impacted_artifacts, is_conservative).
        - impacted_artifacts: Set of artifact names that need to be rebuilt.
        - is_conservative: True if any path triggered a conservative fallback
          (meaning all artifacts from that source set should be rebuilt).

    Example:
        >>> resolve_artifacts_for_paths(
        ...     ["rocm-libraries/projects/rocblas/src/foo.cpp"],
        ...     {"rocm-libraries": ["rocm-libraries"]}
        ... )
        ({"blas"}, False)
    """
    impacted: set[str] = set()
    is_conservative = False

    # Build reverse mapping: submodule -> source set
    submodule_to_source_set: dict[str, str] = {}
    for source_set, submodules in source_set_submodules.items():
        for submodule in submodules:
            submodule_to_source_set[submodule] = source_set

    for path in changed_paths:
        submodule, subpath = parse_changed_path(path)
        if submodule is None:
            continue

        # Only process paths in rocm-libraries
        if submodule != "rocm-libraries":
            continue

        # Try to resolve to a specific artifact
        artifact = get_artifact_for_path(submodule, subpath)
        if artifact is not None:
            impacted.add(artifact)
        else:
            # Path triggers conservative fallback
            is_conservative = True
            impacted.update(get_all_rocm_libraries_artifacts())

    return (impacted, is_conservative)
