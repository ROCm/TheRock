# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate TheRock's consumer graph by statically parsing the super-project CMake.

Parses the Git-tracked ``CMakeLists.txt`` / ``*.cmake`` files without running
CMake or checking out submodules, and extracts the reverse-dependency
("consumer") graph from ``therock_cmake_subproject_declare()`` calls. The result
is a conservative *may-depend* graph: both sides of every conditional are
analyzed and unioned, so it may contain edges no single configure activates but
never omits an edge a real configure produces.

The committed graph lives at ``test_tools/therock_consumer_graph.json`` (the
reverse-edge ``dependency -> {consumers}`` schema).

**How it works**

- Lists Git-tracked ``CMakeLists.txt`` / ``*.cmake`` (a submodule is a single
  gitlink path, so the file set is stable whether or not submodules are checked
  out) and traverses from the root listfile.
- Extracts each ``therock_cmake_subproject_declare()``'s name, ``BUILD_DEPS``,
  ``RUNTIME_DEPS``, and ``COMPILER_TOOLCHAIN``.
- Models the dependency-relevant CMake subset: ``set``/``unset``,
  ``list(APPEND/PREPEND/REMOVE_ITEM)``, ``${var}`` and ``;``-list expansion,
  ``if()`` branch union, ``foreach()`` (incl. ``IN LISTS``/``IN ITEMS``/
  ``RANGE``), and ``add_subdirectory()`` child scope vs. ``include()`` caller
  scope.
- An unresolved variable in a dependency argument raises rather than being
  dropped.

**Known limitations**

- Correlations between separate conditions are ignored, so combinations that can
  never co-occur may still contribute edges.
- Only the dependency-relevant CMake subset is modeled; unsupported dependency
  expressions should raise and gain a test.
- Only literal ``therock_cmake_subproject_declare()`` / ``add_subdirectory()``
  calls are seen — those reached through a wrapping function or macro are
  invisible.
- A *defined-but-empty* dependency variable (incl. a ``foreach(... IN LISTS x)``
  where ``x`` is defined but empty) expands to nothing without being flagged,
  silently dropping its edge. An *undefined* ``IN LISTS`` source is treated
  differently: it is reported unresolved (fail-loud), diverging from CMake, where
  an undefined ``IN LISTS`` variable is simply an empty list.
- ``set(... PARENT_SCOPE)`` is modeled add-only: the assignment is unioned into the
  enclosing scope rather than replacing it.
- Sources wired through ``therock_enable_external_source()`` do not resolve
  statically.
- Subtree mapping assumes the default ``rocm-libraries`` / ``rocm-systems``
  layout and ignores a ``THEROCK_ROCM_*_SOURCE_DIR`` cache override.
"""

import copy
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from cmake_parser import CMakeParseError
from cmake_parser import ast as cmake_ast
from cmake_parser.lexer import Token
from cmake_parser.parser import parse_tree


_VARIABLE_REFERENCE_PATTERN = re.compile(r"\$\{([^}]+)\}")

_DECLARATION_FLAGS = {
    "ACTIVATE",
    "USE_DIST_AMDGPU_TARGETS",
    "USE_TEST_AMDGPU_TARGETS",
    "DISABLE_AMDGPU_TARGETS",
    "EXCLUDE_FROM_ALL",
    "BACKGROUND_BUILD",
    "NO_MERGE_COMPILE_COMMANDS",
    "OUTPUT_ON_FAILURE",
    "NO_INSTALL_RPATH",
    "FPRINT_SOURCE_HASH",
}
_DECLARATION_ONE_VALUE_ARGS = {
    "EXTERNAL_SOURCE_DIR",
    "BINARY_DIR",
    "DIR_PREFIX",
    "INSTALL_DESTINATION",
    "COMPILER_TOOLCHAIN",
    "INTERFACE_PROGRAM_DIRS",
    "CMAKE_LISTS_RELPATH",
    "INTERFACE_PKG_CONFIG_DIRS",
    "INSTALL_RPATH_EXECUTABLE_DIR",
    "INSTALL_RPATH_LIBRARY_DIR",
    "LOGICAL_TARGET_NAME",
    "FPRINT_SOURCE_DIR",
}
_DECLARATION_MULTI_VALUE_ARGS = {
    "BUILD_DEPS",
    "RUNTIME_DEPS",
    "CMAKE_ARGS",
    "CMAKE_INCLUDES",
    "INTERFACE_INCLUDE_DIRS",
    "INTERFACE_LINK_DIRS",
    "IGNORE_PACKAGES",
    "EXTRA_DEPENDS",
    "INSTALL_RPATH_DIRS",
    "INTERFACE_INSTALL_RPATH_DIRS",
    "DEFAULT_GPU_TARGETS",
    "FPRINT_FILE_GLOBS",
    "INSTALL_OPTIONAL_COMPONENTS",
}
_DECLARATION_KEYWORDS = (
    _DECLARATION_FLAGS | _DECLARATION_ONE_VALUE_ARGS | _DECLARATION_MULTI_VALUE_ARGS
)

# Keyword vocabulary of therock_provide_artifact(slice_name ...); mirrors the
# cmake_parse_arguments in cmake/therock_artifacts.cmake. Used to split out the
# SUBPROJECT_DEPS list for the source-dir map.
_PROVIDE_ARTIFACT_KEYWORDS = frozenset(
    {"TARGET_NEUTRAL", "DESCRIPTOR", "DISTRIBUTION", "COMPONENTS", "SUBPROJECT_DEPS"}
)

# TODO: mirrors CMake's therock_compiler_toolchain_subproject() (the CMake-side
# source); share one definition via a data file if a third toolchain is added.
_TOOLCHAIN_SUBPROJECTS = {
    "amd-hip": "hip-clr",
    "amd-llvm": "amd-llvm",
}


class AnalysisError(RuntimeError):
    """Raised when the analyzer cannot conservatively analyze relevant code."""


@dataclass(frozen=True)
class SourceLocation:
    """Location of a CMake command in the analyzed repository."""

    path: Path
    line: int

    def format(self) -> str:
        """Format the location as ``path:line`` for diagnostics."""
        return f"{self.path.as_posix()}:{self.line}"


@dataclass
class Subproject:
    """Conservative union of declarations for one subproject."""

    name: str
    build_deps: set[str] = field(default_factory=set)
    runtime_deps: set[str] = field(default_factory=set)
    compiler_toolchains: set[str] = field(default_factory=set)
    external_source_dirs: set[str] = field(default_factory=set)
    locations: set[SourceLocation] = field(default_factory=set)

    @property
    def all_deps(self) -> set[str]:
        """Return the explicit (build/runtime) and implicit (toolchain) dependencies."""
        result = self.build_deps | self.runtime_deps
        for toolchain in self.compiler_toolchains:
            toolchain_subproject = _TOOLCHAIN_SUBPROJECTS.get(toolchain)
            if toolchain_subproject is None:
                raise AnalysisError(
                    f"Unsupported COMPILER_TOOLCHAIN value {toolchain!r} "
                    f"on subproject {self.name!r}"
                )
            result.add(toolchain_subproject)
        return result


@dataclass(frozen=True)
class SkippedPath:
    """An add_subdirectory/include path excluded from tracked traversal."""

    location: SourceLocation
    expression: str
    reason: str


@dataclass
class AnalysisResult:
    """Results and diagnostics from one repository analysis."""

    tracked_cmake_files: set[Path]
    parsed_cmake_files: set[Path]
    reachable_cmake_files: set[Path]
    tracked_declaration_files: set[Path]
    declaration_files: set[Path]
    subprojects: dict[str, Subproject]
    skipped_paths: list[SkippedPath]
    repository_root: Path
    # artifact slice name -> set of SUBPROJECT_DEPS subproject names, from
    # therock_provide_artifact(). Empty when no provide_artifact calls were reached.
    artifacts: dict[str, set[str]] = field(default_factory=dict)

    @property
    def declaration_count(self) -> int:
        """Return the number of declaration call sites that were reached."""
        return sum(
            len(subproject.locations) for subproject in self.subprojects.values()
        )

    @property
    def unreachable_declaration_files(self) -> set[Path]:
        """Return tracked files with declarations missed by traversal."""
        return self.tracked_declaration_files - self.declaration_files

    def build_consumer_graph(self) -> dict[str, dict[str, list[str]]]:
        """Build the reverse-dependency graph (``dependency -> {consumers}``)."""
        consumers: dict[str, set[str]] = {
            name.lower(): set() for name in self.subprojects
        }
        for subproject in self.subprojects.values():
            consumer = subproject.name.lower()
            for dependency in subproject.all_deps:
                dependency_key = dependency.lower()
                if dependency_key in consumers and dependency_key != consumer:
                    consumers[dependency_key].add(consumer)
        return {
            name: {"consumers": sorted(consumers[name])} for name in sorted(consumers)
        }

    def build_subtree_map(self) -> dict[str, list[str]]:
        """Map each external-repo source subtree to the graph key(s) built from it.

        EXTERNAL_SOURCE_DIR is made relative to the rocm-libraries / rocm-systems
        roots to yield a ``category/name`` subtree (e.g. ``projects/clr``); one
        subtree may back several keys. Sources outside those roots, or that did not
        resolve (see ``_resolve_source_dir_section``), get no entry.
        """
        # Default source roots (THEROCK_ROCM_{LIBRARIES,SYSTEMS}_SOURCE_DIR); the
        # parser assumes the default layout and does not honor a cache override.
        roots = (
            self.repository_root / "rocm-libraries",
            self.repository_root / "rocm-systems",
        )
        subtree_to_keys: dict[str, set[str]] = {}
        for subproject in self.subprojects.values():
            key = subproject.name.lower()
            # A subproject with several (conditional) source dirs fans into each.
            for source_dir in subproject.external_source_dirs:
                source_path = Path(source_dir)
                for root in roots:
                    # source_dir and root share the resolved THEROCK_SOURCE_DIR
                    # prefix, so a textual relative_to is enough.
                    try:
                        relative = source_path.relative_to(root)
                    except ValueError:
                        continue
                    subtree_to_keys.setdefault(relative.as_posix(), set()).add(key)
                    break
        return {
            subtree: sorted(keys) for subtree, keys in sorted(subtree_to_keys.items())
        }

    def build_source_dir_map(self) -> dict[str, list[str]]:
        """Map each artifact to the source subtree(s) its SUBPROJECT_DEPS build from.

        For every ``therock_provide_artifact()`` slice, join each ``SUBPROJECT_DEPS``
        subproject to its ``EXTERNAL_SOURCE_DIR`` subtree (relativized against the
        rocm-libraries / rocm-systems roots, as in ``build_subtree_map``). This map is
        informational; it does not replace ``BUILD_TOPOLOGY.toml`` ``source_paths``,
        which are hand-curated and only partly overlap. Artifacts whose deps resolve to
        no statically-captured source dir are omitted.
        """
        roots = (
            self.repository_root / "rocm-libraries",
            self.repository_root / "rocm-systems",
        )

        def subtrees_for(subproject_key: str) -> set[str]:
            subproject = self.subprojects.get(subproject_key)
            if subproject is None:
                return set()
            result: set[str] = set()
            for source_dir in subproject.external_source_dirs:
                source_path = Path(source_dir)
                for root in roots:
                    try:
                        relative = source_path.relative_to(root)
                    except ValueError:
                        continue
                    result.add(relative.as_posix())
                    break
            return result

        artifact_to_subtrees: dict[str, set[str]] = {}
        for artifact, deps in self.artifacts.items():
            subtrees: set[str] = set()
            for dep in deps:
                subtrees |= subtrees_for(dep.lower())
            if subtrees:
                artifact_to_subtrees[artifact] = subtrees
        return {
            artifact: sorted(subtrees)
            for artifact, subtrees in sorted(artifact_to_subtrees.items())
        }

    def dangling_dependencies(self) -> dict[str, list[str]]:
        """Return dependencies that do not name any discovered subproject."""
        known = {name.lower() for name in self.subprojects}
        dangling: dict[str, list[str]] = {}
        for subproject in self.subprojects.values():
            missing = sorted(
                dependency
                for dependency in subproject.all_deps
                if dependency.lower() not in known
            )
            if missing:
                dangling[subproject.name] = missing
        return dangling


@dataclass(frozen=True)
class GraphComparison:
    """Direct node and edge comparison between two consumer graphs."""

    common_nodes: list[str]
    generated_only_nodes: list[str]
    reference_only_nodes: list[str]
    common_edges: list[str]
    generated_only_edges: list[str]
    reference_only_edges: list[str]


@dataclass
class Expansion:
    """Result of expanding CMake token(s): resolved values and any unresolved names.

    ``values`` holds the concrete strings a token expanded to. ``unresolved`` holds
    the names of ``${var}`` references that are not defined in the environment; when
    it is non-empty the expansion could not be completed and ``values`` is empty.
    """

    values: set[str]
    unresolved: set[str]


class Environment:
    """CMake variable scope: names to the set of values they may hold.

    Copy-on-write: ``copy()`` shares the value sets, and every mutator replaces a
    name's set rather than mutating it in place, so a child scope never disturbs
    its parent. Sets returned by ``get()`` must therefore be treated as read-only.
    """

    def __init__(self, variables: dict[str, set[str]] | None = None) -> None:
        self._variables: dict[str, set[str]] = (
            variables if variables is not None else {}
        )

    def __contains__(self, name: str) -> bool:
        return name in self._variables

    def get(self, name: str, default: set[str] | None = None) -> set[str] | None:
        """Return the value set for ``name``, or ``default`` if it is unset."""
        return self._variables.get(name, default)

    def assign(self, name: str, values: set[str]) -> None:
        """Set ``name`` to a copy of ``values`` (replacing any existing set)."""
        self._variables[name] = set(values)

    def append(self, name: str, values: set[str]) -> None:
        """Union ``values`` into ``name`` (list(APPEND)/list(PREPEND))."""
        self._variables[name] = self._variables.get(name, set()) | set(values)

    def remove(self, name: str, values: set[str]) -> None:
        """Remove ``values`` from ``name`` (list(REMOVE_ITEM))."""
        self._variables[name] = self._variables.get(name, set()) - set(values)

    def discard(self, name: str) -> None:
        """Unset ``name`` if present."""
        self._variables.pop(name, None)

    def copy(self) -> Self:
        """Return a copy-on-write child scope sharing this scope's value sets."""
        return Environment(dict(self._variables))

    def merge_from(self, other: Self) -> None:
        """Union another scope's values in, never dropping (folds a loop body back)."""
        for name, values in other._variables.items():
            self._variables[name] = self._variables.get(name, set()) | values

    def replace_with(self, other: Self) -> None:
        """Adopt another scope's variables in place (used for the if()/else() merge)."""
        self._variables = other._variables

    @classmethod
    def branch_union(cls, first: Self, second: Self) -> Self:
        """Union two scopes key-by-key (the conservative if()/else() merge)."""
        result = cls()
        for name in first._variables.keys() | second._variables.keys():
            result._variables[name] = first._variables.get(
                name, set()
            ) | second._variables.get(name, set())
        return result


def list_tracked_cmake_files(repository_root: Path) -> set[Path]:
    """List CMake files tracked by the super-project, excluding gitlinks."""
    command = [
        "git",
        "-c",
        f"safe.directory={repository_root.as_posix()}",
        "-C",
        str(repository_root),
        "ls-files",
        "-z",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as error:
        raise AnalysisError("git executable not found on PATH") from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.decode("utf-8", errors="replace").strip()
        raise AnalysisError(
            f"git ls-files failed in {repository_root}: {stderr}"
        ) from error
    tracked_paths = {
        Path(raw_path.decode("utf-8"))
        for raw_path in result.stdout.split(b"\0")
        if raw_path
    }
    return {
        path
        for path in tracked_paths
        if path.name == "CMakeLists.txt" or path.suffix == ".cmake"
    }


def compare_graphs(
    generated: dict[str, dict[str, list[str]]],
    reference: dict[str, dict[str, list[str]]],
) -> GraphComparison:
    """Compare two graphs in the consumer graph schema by node and edge sets."""
    generated_nodes = set(generated)
    reference_nodes = set(reference)
    generated_edges = _graph_edges(generated)
    reference_edges = _graph_edges(reference)
    return GraphComparison(
        common_nodes=sorted(generated_nodes & reference_nodes),
        generated_only_nodes=sorted(generated_nodes - reference_nodes),
        reference_only_nodes=sorted(reference_nodes - generated_nodes),
        common_edges=sorted(generated_edges & reference_edges),
        generated_only_edges=sorted(generated_edges - reference_edges),
        reference_only_edges=sorted(reference_edges - generated_edges),
    )


def load_consumer_graph(path: Path) -> dict[str, dict[str, list[str]]]:
    """Load and minimally validate a consumer graph JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AnalysisError(f"Expected a JSON object in {path}")
    for name, entry in data.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("consumers"), list):
            raise AnalysisError(
                f"Malformed consumer graph entry {name!r} in {path}: "
                "expected an object with a 'consumers' list"
            )
    return data


def _graph_edges(graph: dict[str, dict[str, list[str]]]) -> set[str]:
    """Return the set of ``dependency -> consumer`` edge strings in a graph."""
    return {
        f"{dependency} -> {consumer}"
        for dependency, entry in graph.items()
        for consumer in entry.get("consumers", [])
    }


class RepositoryAnalyzer:
    """Analyze tracked, reachable CMake files without entering submodules."""

    def __init__(
        self,
        repository_root: Path,
        tracked_cmake_files: set[Path] | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.tracked_cmake_files = (
            tracked_cmake_files
            if tracked_cmake_files is not None
            else list_tracked_cmake_files(self.repository_root)
        )
        self._ast_cache: dict[Path, list[cmake_ast.AstNode]] = {}
        self._parsed_files: set[Path] = set()
        self._reachable_files: set[Path] = set()
        self._tracked_declaration_files: set[Path] = set()
        self._declaration_files: set[Path] = set()
        self._subprojects: dict[str, Subproject] = {}
        self._artifacts: dict[str, set[str]] = {}
        self._skipped_paths: list[SkippedPath] = []
        self._active_files: set[Path] = set()

    def analyze(self) -> AnalysisResult:
        """Parse the tracked inventory and traverse from the root listfile."""
        self._parse_tracked_inventory()
        root_listfile = Path("CMakeLists.txt")
        if root_listfile not in self.tracked_cmake_files:
            raise AnalysisError(
                f"Root CMakeLists.txt is not tracked under {self.repository_root}"
            )
        initial_environment = Environment(
            {"THEROCK_SOURCE_DIR": {str(self.repository_root)}}
        )
        self._process_file(root_listfile, initial_environment)
        return AnalysisResult(
            tracked_cmake_files=set(self.tracked_cmake_files),
            parsed_cmake_files=set(self._parsed_files),
            reachable_cmake_files=set(self._reachable_files),
            tracked_declaration_files=set(self._tracked_declaration_files),
            declaration_files=set(self._declaration_files),
            subprojects=copy.deepcopy(self._subprojects),
            skipped_paths=list(self._skipped_paths),
            repository_root=self.repository_root,
            artifacts=copy.deepcopy(self._artifacts),
        )

    def _parse_tracked_inventory(self) -> None:
        """Parse every tracked file and record which ones declare a subproject."""
        for relative_path in sorted(self.tracked_cmake_files):
            nodes = self._parse_file(relative_path)
            if _contains_declaration(nodes):
                self._tracked_declaration_files.add(relative_path)

    def _parse_file(self, relative_path: Path) -> list[cmake_ast.AstNode]:
        """Parse one tracked CMake file to an AST, caching the result."""
        cached = self._ast_cache.get(relative_path)
        if cached is not None:
            return cached
        absolute_path = self.repository_root / relative_path
        if not absolute_path.is_file():
            raise AnalysisError(f"Tracked CMake file does not exist: {absolute_path}")
        try:
            source = absolute_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise AnalysisError(
                f"Tracked CMake file is not valid UTF-8: {relative_path.as_posix()}"
            ) from error
        try:
            nodes = list(parse_tree(source, skip_comments=True))
        except CMakeParseError as error:
            raise AnalysisError(f"Failed to parse {relative_path}: {error}") from error
        self._ast_cache[relative_path] = nodes
        self._parsed_files.add(relative_path)
        return nodes

    def _process_file(
        self,
        relative_path: Path,
        inherited_environment: Environment,
        *,
        inherit_scope: bool = False,
        parent_environment: Environment | None = None,
    ) -> None:
        """Execute a listfile's nodes, modelling CMake variable scoping.

        add_subdirectory() (and the root) open a fresh child scope; include() runs
        in the caller's scope so variables set() in the included file persist to the
        includer. Recursive re-entry is recorded as a skipped path.

        ``parent_environment`` is the enclosing directory scope that a
        ``set(... PARENT_SCOPE)`` writes through to (``None`` at the root).
        """
        if relative_path in self._active_files:
            location = SourceLocation(path=relative_path, line=1)
            self._skipped_paths.append(
                SkippedPath(
                    location=location,
                    expression=relative_path.as_posix(),
                    reason="recursive include/add_subdirectory cycle",
                )
            )
            return
        self._active_files.add(relative_path)
        self._reachable_files.add(relative_path)
        source_directory = (self.repository_root / relative_path).parent.resolve()
        try:
            if inherit_scope:
                environment = inherited_environment
                saved_list_dir = environment.get("CMAKE_CURRENT_LIST_DIR")
                environment.assign("CMAKE_CURRENT_LIST_DIR", {str(source_directory)})
                try:
                    self._execute_nodes(
                        nodes=self._parse_file(relative_path),
                        environment=environment,
                        relative_path=relative_path,
                        parent_environment=parent_environment,
                    )
                finally:
                    if saved_list_dir is None:
                        environment.discard("CMAKE_CURRENT_LIST_DIR")
                    else:
                        environment.assign("CMAKE_CURRENT_LIST_DIR", saved_list_dir)
            else:
                environment = inherited_environment.copy()
                environment.assign("CMAKE_CURRENT_SOURCE_DIR", {str(source_directory)})
                environment.assign("CMAKE_CURRENT_LIST_DIR", {str(source_directory)})
                self._execute_nodes(
                    nodes=self._parse_file(relative_path),
                    environment=environment,
                    relative_path=relative_path,
                    parent_environment=parent_environment,
                )
        finally:
            self._active_files.remove(relative_path)

    def _execute_nodes(
        self,
        nodes: list[cmake_ast.AstNode],
        environment: Environment,
        relative_path: Path,
        parent_environment: Environment | None = None,
    ) -> None:
        """Execute a list of AST nodes against the given environment."""
        for node in nodes:
            if isinstance(node, cmake_ast.Set):
                self._execute_set(node, environment, relative_path, parent_environment)
            elif isinstance(node, cmake_ast.Unset):
                self._execute_unset(node, environment)
            elif isinstance(node, cmake_ast.If):
                self._execute_if(node, environment, relative_path, parent_environment)
            elif isinstance(node, cmake_ast.Block):
                self._execute_nodes(
                    node.body, environment, relative_path, parent_environment
                )
            elif isinstance(node, cmake_ast.ForEach):
                self._execute_foreach(
                    node, environment, relative_path, parent_environment
                )
            elif isinstance(node, cmake_ast.Command):
                self._execute_command(node, environment, relative_path)
            elif isinstance(node, cmake_ast.Include):
                self._execute_include(
                    node, environment, relative_path, parent_environment
                )

    def _execute_set(
        self,
        node: cmake_ast.Set,
        environment: Environment,
        relative_path: Path,
        parent_environment: Environment | None,
    ) -> None:
        """Model ``set(VAR ...)``, honoring PARENT_SCOPE and stopping at CACHE."""
        if not node.args:
            return
        variable_name = node.args[0].value
        parent_scope = False
        value_tokens = []
        for token in node.args[1:]:
            if token.value == "CACHE":
                break
            if token.value == "PARENT_SCOPE":
                parent_scope = True
                break
            value_tokens.append(token)
        values = _expand_tokens(value_tokens, environment).values
        if not parent_scope:
            environment.assign(variable_name, values)
            return
        # set(... PARENT_SCOPE) writes the enclosing directory scope, not this
        # one; the root has no parent to write.
        if parent_environment is None:
            self._skipped_paths.append(
                SkippedPath(
                    location=SourceLocation(path=relative_path, line=node.args[0].line),
                    expression=f"set({variable_name} ... PARENT_SCOPE)",
                    reason="PARENT_SCOPE at root has no enclosing scope",
                )
            )
            return
        parent_environment.append(variable_name, values)

    def _execute_unset(self, node: cmake_ast.Unset, environment: Environment) -> None:
        """Model ``unset(VAR)``."""
        if node.args:
            environment.assign(node.args[0].value, set())

    def _execute_if(
        self,
        node: cmake_ast.If,
        environment: Environment,
        relative_path: Path,
        parent_environment: Environment | None,
    ) -> None:
        """Execute both branches and union their environments (conservative)."""
        true_environment = environment.copy()
        false_environment = environment.copy()
        self._execute_nodes(
            node.if_true, true_environment, relative_path, parent_environment
        )
        if node.if_false is not None:
            self._execute_nodes(
                node.if_false, false_environment, relative_path, parent_environment
            )
        environment.replace_with(
            Environment.branch_union(true_environment, false_environment)
        )

    def _execute_foreach(
        self,
        node: cmake_ast.ForEach,
        environment: Environment,
        relative_path: Path,
        parent_environment: Environment | None,
    ) -> None:
        """Execute a foreach() body once with the loop variable over-approximated."""
        loop_environment = environment.copy()
        if node.args:
            loop_variable = node.args[0].value
            loop_expansion = self._foreach_loop_values(node.args[1:], environment)
            if loop_expansion.unresolved:
                # Leave the loop variable unset rather than empty, so a dependency
                # built from it raises via the unresolved guard instead of being
                # dropped.
                loop_environment.discard(loop_variable)
            else:
                loop_environment.assign(loop_variable, loop_expansion.values)
        self._execute_nodes(
            node.body, loop_environment, relative_path, parent_environment
        )
        # Fold the loop body back into the enclosing scope (add-only).
        environment.merge_from(loop_environment)

    def _foreach_loop_values(
        self, tokens: list[Token], environment: Environment
    ) -> Expansion:
        """Values the foreach() loop variable may take, across all iterations.

        Handles the keyword forms: ``IN LISTS <var>...`` dereferences each named
        list variable, ``IN ITEMS <val>...`` takes the values directly, and
        ``RANGE ...`` yields integers (never names) so it contributes nothing.
        A bare ``foreach(var a b c)`` expands the operands as literal items.

        An ``IN LISTS`` operand naming an *undefined* variable is reported in
        ``unresolved`` so the caller can leave the loop variable unset; ``RANGE``
        and empty ``IN ITEMS`` stay legitimately empty (resolved, no values).
        """
        if not tokens:
            return Expansion(values=set(), unresolved=set())
        if tokens[0].value == "RANGE":
            return Expansion(values=set(), unresolved=set())
        if tokens[0].value != "IN":
            return _expand_tokens(tokens, environment)
        values: set[str] = set()
        unresolved: set[str] = set()
        mode: str | None = None
        for token in tokens[1:]:
            if token.value in {"LISTS", "ITEMS"}:
                mode = token.value
                continue
            operand = _expand_token(token, environment)
            unresolved.update(operand.unresolved)
            if mode == "LISTS":
                for list_name in operand.values:
                    list_values = environment.get(list_name)
                    if list_values is None:
                        # Diverges from CMake, where an undefined IN LISTS variable
                        # is an empty list (zero iterations); reporting it unresolved
                        # makes a dependency built from the loop variable raise
                        # rather than resolve to nothing.
                        unresolved.add(list_name)
                    else:
                        values.update(list_values)
            elif mode == "ITEMS":
                values.update(operand.values)
        return Expansion(values=values, unresolved=unresolved)

    def _execute_command(
        self,
        node: cmake_ast.Command,
        environment: Environment,
        relative_path: Path,
    ) -> None:
        """Dispatch the CMake commands the analyzer models."""
        identifier = node.identifier.lower()
        if identifier == "list":
            self._execute_list(node, environment)
        elif identifier == "add_subdirectory":
            self._execute_add_subdirectory(node, environment, relative_path)
        elif identifier == "therock_cmake_subproject_declare":
            self._record_declaration(node, environment, relative_path)
        elif identifier == "therock_provide_artifact":
            self._record_artifact(node, environment, relative_path)

    def _execute_list(self, node: cmake_ast.Command, environment: Environment) -> None:
        """Model ``list(APPEND|PREPEND|REMOVE_ITEM VAR ...)``."""
        if len(node.args) < 2:
            return
        operation = node.args[0].value.upper()
        variable_name = node.args[1].value
        values = _expand_tokens(node.args[2:], environment).values
        if operation in {"APPEND", "PREPEND"}:
            environment.append(variable_name, values)
        elif operation == "REMOVE_ITEM":
            environment.remove(variable_name, values)

    def _execute_add_subdirectory(
        self,
        node: cmake_ast.Command,
        environment: Environment,
        relative_path: Path,
    ) -> None:
        """Traverse into a tracked ``add_subdirectory()`` target (fresh scope)."""
        if not node.args:
            return
        child_paths = self._resolve_listfile_paths(
            token=node.args[0],
            environment=environment,
            relative_path=relative_path,
            is_subdirectory=True,
        )
        for child_path in child_paths:
            # The child directory opens a fresh scope; this directory's scope is
            # its parent for set(... PARENT_SCOPE) write-through.
            self._process_file(child_path, environment, parent_environment=environment)

    def _execute_include(
        self,
        node: cmake_ast.Include,
        environment: Environment,
        relative_path: Path,
        parent_environment: Environment | None,
    ) -> None:
        """Traverse into a tracked ``include()`` target (caller's scope)."""
        if not node.args:
            return
        include_paths = self._resolve_listfile_paths(
            token=node.args[0],
            environment=environment,
            relative_path=relative_path,
            is_subdirectory=False,
        )
        for include_path in include_paths:
            # include() runs in the caller's scope, so PARENT_SCOPE writes through
            # to the caller's own parent.
            self._process_file(
                include_path,
                environment,
                inherit_scope=True,
                parent_environment=parent_environment,
            )

    def _resolve_listfile_paths(
        self,
        token: Token,
        environment: Environment,
        relative_path: Path,
        is_subdirectory: bool,
    ) -> set[Path]:
        """Resolve an add_subdirectory/include argument to tracked listfile paths.

        Unresolved variables and non-tracked (external/generated/built-in) targets
        are recorded as skipped diagnostics and contribute no paths.
        """
        expansion = _expand_token(token, environment)
        location = SourceLocation(path=relative_path, line=token.line)
        if expansion.unresolved:
            self._skipped_paths.append(
                SkippedPath(
                    location=location,
                    expression=token.value,
                    reason=f"unresolved variables: {', '.join(sorted(expansion.unresolved))}",
                )
            )
            return set()
        result: set[Path] = set()
        for value in expansion.values:
            candidates = self._path_candidates(
                value=value,
                relative_path=relative_path,
                is_subdirectory=is_subdirectory,
            )
            tracked_candidates = {
                candidate
                for candidate in candidates
                if candidate in self.tracked_cmake_files
            }
            if tracked_candidates:
                result.update(tracked_candidates)
            else:
                self._skipped_paths.append(
                    SkippedPath(
                        location=location,
                        expression=token.value,
                        reason="path is external, generated, built-in, or not tracked",
                    )
                )
        return result

    def _path_candidates(
        self, value: str, relative_path: Path, is_subdirectory: bool
    ) -> set[Path]:
        """Return repo-relative candidate listfiles for a resolved path value."""
        value_path = Path(value)
        current_directory = (self.repository_root / relative_path).parent
        absolute_path = (
            value_path if value_path.is_absolute() else current_directory / value_path
        ).resolve()
        absolute_candidates = []
        if is_subdirectory:
            absolute_candidates.append(absolute_path / "CMakeLists.txt")
        else:
            absolute_candidates.append(absolute_path)
            if absolute_path.suffix != ".cmake":
                absolute_candidates.append(absolute_path.with_suffix(".cmake"))
                absolute_candidates.append(
                    (self.repository_root / "cmake" / value).with_suffix(".cmake")
                )
        result = set()
        for candidate in absolute_candidates:
            try:
                result.add(candidate.resolve().relative_to(self.repository_root))
            except ValueError:
                continue
        return result

    def _record_declaration(
        self,
        node: cmake_ast.Command,
        environment: Environment,
        relative_path: Path,
    ) -> None:
        """Extract one ``therock_cmake_subproject_declare()`` into a Subproject."""
        if not node.args:
            raise AnalysisError(
                f"{relative_path.as_posix()}:{node.line}: declaration has no name"
            )
        name_expansion = _expand_token(node.args[0], environment)
        if name_expansion.unresolved or len(name_expansion.values) != 1:
            raise AnalysisError(
                f"{relative_path.as_posix()}:{node.line}: cannot resolve exactly one "
                f"subproject name from {node.args[0].value!r}"
            )
        name = next(iter(name_expansion.values))
        sections = _declaration_sections(node.args[1:])
        build_deps = self._resolve_dependency_section(
            sections.get("BUILD_DEPS", []), environment, relative_path, node.line
        )
        runtime_deps = self._resolve_dependency_section(
            sections.get("RUNTIME_DEPS", []), environment, relative_path, node.line
        )
        toolchains = self._resolve_dependency_section(
            sections.get("COMPILER_TOOLCHAIN", []),
            environment,
            relative_path,
            node.line,
        )
        source_dirs = self._resolve_source_dir_section(
            sections.get("EXTERNAL_SOURCE_DIR", []), environment
        )
        key = name.lower()
        subproject = self._subprojects.setdefault(key, Subproject(name=name))
        subproject.build_deps.update(build_deps)
        subproject.runtime_deps.update(runtime_deps)
        subproject.compiler_toolchains.update(toolchains)
        subproject.external_source_dirs.update(source_dirs)
        subproject.locations.add(SourceLocation(path=relative_path, line=node.line))
        self._declaration_files.add(relative_path)

    def _record_artifact(
        self,
        node: cmake_ast.Command,
        environment: Environment,
        relative_path: Path,
    ) -> None:
        """Extract one ``therock_provide_artifact()`` slice's SUBPROJECT_DEPS.

        Feeds the informational ``source_dir_map`` only, not the consumer graph.
        Resolution failures are non-fatal: the artifact is omitted with a SkippedPath
        diagnostic instead of aborting the analysis.
        """
        location = SourceLocation(path=relative_path, line=node.line)
        if not node.args:
            self._skipped_paths.append(
                SkippedPath(
                    location=location,
                    expression="therock_provide_artifact",
                    reason="provide_artifact has no name",
                )
            )
            return
        name_expansion = _expand_token(node.args[0], environment)
        if name_expansion.unresolved or len(name_expansion.values) != 1:
            self._skipped_paths.append(
                SkippedPath(
                    location=location,
                    expression=node.args[0].value,
                    reason="cannot resolve exactly one artifact name",
                )
            )
            return
        name = next(iter(name_expansion.values))
        sections = _declaration_sections(node.args[1:], _PROVIDE_ARTIFACT_KEYWORDS)
        expansion = _expand_tokens(sections.get("SUBPROJECT_DEPS", []), environment)
        if expansion.unresolved:
            self._skipped_paths.append(
                SkippedPath(
                    location=location,
                    expression=name,
                    reason=(
                        "unresolved SUBPROJECT_DEPS variables: "
                        f"{', '.join(sorted(expansion.unresolved))}"
                    ),
                )
            )
            return
        deps = {value for value in expansion.values if value}
        self._artifacts.setdefault(name, set()).update(deps)

    def _resolve_dependency_section(
        self,
        tokens: list[Token],
        environment: Environment,
        relative_path: Path,
        declaration_line: int,
    ) -> set[str]:
        """Resolve a dependency argument list; an unresolved variable is fatal."""
        expansion = _expand_tokens(tokens, environment)
        if expansion.unresolved:
            raise AnalysisError(
                f"{relative_path.as_posix()}:{declaration_line}: unresolved "
                f"dependency variables: {', '.join(sorted(expansion.unresolved))}"
            )
        return {value for value in expansion.values if value}

    def _resolve_source_dir_section(
        self, tokens: list[Token], environment: Environment
    ) -> set[str]:
        """Resolve an EXTERNAL_SOURCE_DIR value; unresolved -> {} (skipped silently).

        Sources wired through therock_enable_external_source() do not resolve
        statically and get no subtree_map entry (handled as reader-side overrides).
        """
        expansion = _expand_tokens(tokens, environment)
        if expansion.unresolved:
            return set()
        return {value for value in expansion.values if value}


def _declaration_sections(
    tokens: list[Token],
    keywords: frozenset[str] | set[str] = _DECLARATION_KEYWORDS,
) -> dict[str, list[Token]]:
    """Group an argument list's tokens by their keyword (BUILD_DEPS, ...)."""
    sections: dict[str, list[Token]] = {}
    current_keyword: str | None = None
    for token in tokens:
        if token.kind == "COMMENT":
            continue
        if token.value in keywords:
            current_keyword = token.value
            sections.setdefault(current_keyword, [])
        elif current_keyword is not None:
            sections[current_keyword].append(token)
    return sections


def _expand_tokens(tokens: list[Token], environment: Environment) -> Expansion:
    """Expand several tokens, unioning their resolved values and unresolved names."""
    values: set[str] = set()
    unresolved: set[str] = set()
    for token in tokens:
        if token.kind == "COMMENT":
            continue
        expansion = _expand_token(token, environment)
        values.update(expansion.values)
        unresolved.update(expansion.unresolved)
    return Expansion(values=values, unresolved=unresolved)


def _expand_token(token: Token, environment: Environment) -> Expansion:
    """Expand one token's ``${var}`` references against the environment.

    Returns the concrete values (semicolon-separated CMake lists split into
    elements) with an empty ``unresolved`` set. If any reference names an *undefined*
    variable, returns empty values and those names in ``unresolved`` so the caller
    can raise. A reference to a *defined but empty* variable resolves to nothing
    (empty values, empty unresolved) — its edge is dropped rather than flagged.
    """
    references = _VARIABLE_REFERENCE_PATTERN.findall(token.value)
    unresolved = {name for name in references if name not in environment}
    if unresolved:
        return Expansion(values=set(), unresolved=unresolved)
    expanded_values = {token.value}
    for variable_name in references:
        replacements = environment.get(variable_name, set())
        if not replacements:
            return Expansion(values=set(), unresolved=set())
        next_values = set()
        reference = "${" + variable_name + "}"
        for value in expanded_values:
            for replacement in replacements:
                next_values.add(value.replace(reference, replacement))
        expanded_values = next_values
    split_values = {
        item for value in expanded_values for item in value.split(";") if item
    }
    return Expansion(values=split_values, unresolved=set())


def _contains_declaration(nodes: list[cmake_ast.AstNode]) -> bool:
    """Return whether any node (recursively) is a subproject declaration call."""
    for node in nodes:
        if (
            isinstance(node, cmake_ast.Command)
            and node.identifier.lower() == "therock_cmake_subproject_declare"
        ):
            return True
        child_lists = []
        for attribute in ("body", "if_true", "if_false"):
            children = getattr(node, attribute, None)
            if children:
                child_lists.append(children)
        if any(_contains_declaration(children) for children in child_lists):
            return True
    return False
