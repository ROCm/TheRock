# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Conservative static analysis of TheRock subproject declarations."""

import copy
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

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

# TODO: mirrors CMake's therock_compiler_toolchain_subproject() (the CMake-side
# source); share one definition via a data file if a third toolchain is added.
_TOOLCHAIN_SUBPROJECTS = {
    "amd-hip": "hip-clr",
    "amd-llvm": "amd-llvm",
}


class AnalysisError(RuntimeError):
    """Raised when the prototype cannot conservatively analyze relevant code."""


@dataclass(frozen=True)
class SourceLocation:
    """Location of a CMake command in the analyzed repository."""

    path: Path
    line: int

    def format(self) -> str:
        """Format the location for diagnostics."""
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
        """Return all explicit and implicit direct dependencies."""
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
        """Build the existing reverse-dependency JSON schema."""
        graph = {
            name.lower(): {"consumers": []}
            for name in sorted(self.subprojects, key=str.lower)
        }
        consumers: dict[str, set[str]] = {name: set() for name in graph}
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
        roots to yield a `category/name` subtree (e.g. `projects/clr`); one subtree
        may back several keys. Sources outside those roots, or that did not resolve
        (see _resolve_source_dir_section), get no entry.
        """
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

    def to_json_data(self) -> dict[str, object]:
        """Return stable JSON-compatible comparison data."""
        return {
            "summary": {
                "common_nodes": len(self.common_nodes),
                "generated_only_nodes": len(self.generated_only_nodes),
                "reference_only_nodes": len(self.reference_only_nodes),
                "common_edges": len(self.common_edges),
                "generated_only_edges": len(self.generated_only_edges),
                "reference_only_edges": len(self.reference_only_edges),
            },
            "common_nodes": self.common_nodes,
            "generated_only_nodes": self.generated_only_nodes,
            "reference_only_nodes": self.reference_only_nodes,
            "common_edges": self.common_edges,
            "generated_only_edges": self.generated_only_edges,
            "reference_only_edges": self.reference_only_edges,
        }


Environment = dict[str, set[str]]


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
    result = subprocess.run(command, check=True, capture_output=True)
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
    """Compare two graphs in the existing consumer graph schema."""
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
    """Load and minimally validate an existing consumer graph."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AnalysisError(f"Expected a JSON object in {path}")
    return data


def _graph_edges(graph: dict[str, dict[str, list[str]]]) -> set[str]:
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
        initial_environment: Environment = {
            "THEROCK_SOURCE_DIR": {str(self.repository_root)},
        }
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
        )

    def _parse_tracked_inventory(self) -> None:
        for relative_path in sorted(self.tracked_cmake_files):
            nodes = self._parse_file(relative_path)
            if _contains_declaration(nodes):
                self._tracked_declaration_files.add(relative_path)

    def _parse_file(self, relative_path: Path) -> list[cmake_ast.AstNode]:
        cached = self._ast_cache.get(relative_path)
        if cached is not None:
            return cached
        absolute_path = self.repository_root / relative_path
        if not absolute_path.is_file():
            raise AnalysisError(f"Tracked CMake file does not exist: {absolute_path}")
        source = absolute_path.read_text(encoding="utf-8")
        try:
            nodes = list(parse_tree(source, skip_comments=True))
        except CMakeParseError as error:
            raise AnalysisError(f"Failed to parse {relative_path}: {error}") from error
        self._ast_cache[relative_path] = nodes
        self._parsed_files.add(relative_path)
        return nodes

    def _process_file(
        self, relative_path: Path, inherited_environment: Environment
    ) -> None:
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
        environment = copy.deepcopy(inherited_environment)
        source_directory = (self.repository_root / relative_path).parent.resolve()
        environment["CMAKE_CURRENT_SOURCE_DIR"] = {str(source_directory)}
        environment["CMAKE_CURRENT_LIST_DIR"] = {str(source_directory)}
        try:
            self._execute_nodes(
                nodes=self._parse_file(relative_path),
                environment=environment,
                relative_path=relative_path,
            )
        finally:
            self._active_files.remove(relative_path)

    def _execute_nodes(
        self,
        nodes: list[cmake_ast.AstNode],
        environment: Environment,
        relative_path: Path,
    ) -> None:
        for node in nodes:
            if isinstance(node, cmake_ast.Set):
                self._execute_set(node, environment)
            elif isinstance(node, cmake_ast.Unset):
                self._execute_unset(node, environment)
            elif isinstance(node, cmake_ast.If):
                self._execute_if(node, environment, relative_path)
            elif isinstance(node, cmake_ast.Block):
                self._execute_nodes(node.body, environment, relative_path)
            elif isinstance(node, cmake_ast.ForEach):
                self._execute_foreach(node, environment, relative_path)
            elif isinstance(node, cmake_ast.Command):
                self._execute_command(node, environment, relative_path)
            elif isinstance(node, cmake_ast.Include):
                self._execute_include(node, environment, relative_path)

    def _execute_set(self, node: cmake_ast.Set, environment: Environment) -> None:
        if not node.args:
            return
        variable_name = node.args[0].value
        value_tokens = []
        for token in node.args[1:]:
            if token.value in {"CACHE", "PARENT_SCOPE"}:
                break
            value_tokens.append(token)
        values, _ = _expand_tokens(value_tokens, environment)
        environment[variable_name] = values

    def _execute_unset(self, node: cmake_ast.Unset, environment: Environment) -> None:
        if node.args:
            environment[node.args[0].value] = set()

    def _execute_if(
        self,
        node: cmake_ast.If,
        environment: Environment,
        relative_path: Path,
    ) -> None:
        true_environment = copy.deepcopy(environment)
        false_environment = copy.deepcopy(environment)
        self._execute_nodes(node.if_true, true_environment, relative_path)
        if node.if_false is not None:
            self._execute_nodes(node.if_false, false_environment, relative_path)
        environment.clear()
        for variable_name in true_environment.keys() | false_environment.keys():
            environment[variable_name] = true_environment.get(
                variable_name, set()
            ) | false_environment.get(variable_name, set())

    def _execute_foreach(
        self,
        node: cmake_ast.ForEach,
        environment: Environment,
        relative_path: Path,
    ) -> None:
        loop_environment = copy.deepcopy(environment)
        if node.args:
            loop_variable = node.args[0].value
            loop_values, _ = _expand_tokens(node.args[1:], environment)
            loop_environment[loop_variable] = loop_values
        self._execute_nodes(node.body, loop_environment, relative_path)
        # Deliberately over-approximate: merge the loop environment back (loop
        # variable included). Only ever adds values, never drops — the safe direction.
        for variable_name, values in loop_environment.items():
            environment.setdefault(variable_name, set()).update(values)

    def _execute_command(
        self,
        node: cmake_ast.Command,
        environment: Environment,
        relative_path: Path,
    ) -> None:
        identifier = node.identifier.lower()
        if identifier == "list":
            self._execute_list(node, environment)
        elif identifier == "add_subdirectory":
            self._execute_add_subdirectory(node, environment, relative_path)
        elif identifier == "therock_cmake_subproject_declare":
            self._record_declaration(node, environment, relative_path)

    def _execute_list(self, node: cmake_ast.Command, environment: Environment) -> None:
        if len(node.args) < 2:
            return
        operation = node.args[0].value.upper()
        variable_name = node.args[1].value
        values, _ = _expand_tokens(node.args[2:], environment)
        if operation in {"APPEND", "PREPEND"}:
            environment.setdefault(variable_name, set()).update(values)
        elif operation == "REMOVE_ITEM":
            environment.setdefault(variable_name, set()).difference_update(values)

    def _execute_add_subdirectory(
        self,
        node: cmake_ast.Command,
        environment: Environment,
        relative_path: Path,
    ) -> None:
        if not node.args:
            return
        child_paths = self._resolve_listfile_paths(
            token=node.args[0],
            environment=environment,
            relative_path=relative_path,
            is_subdirectory=True,
        )
        for child_path in child_paths:
            self._process_file(child_path, environment)

    def _execute_include(
        self,
        node: cmake_ast.Include,
        environment: Environment,
        relative_path: Path,
    ) -> None:
        if not node.args:
            return
        include_paths = self._resolve_listfile_paths(
            token=node.args[0],
            environment=environment,
            relative_path=relative_path,
            is_subdirectory=False,
        )
        for include_path in include_paths:
            self._process_file(include_path, environment)

    def _resolve_listfile_paths(
        self,
        token: Token,
        environment: Environment,
        relative_path: Path,
        is_subdirectory: bool,
    ) -> set[Path]:
        values, unresolved = _expand_token(token, environment)
        location = SourceLocation(path=relative_path, line=token.line)
        if unresolved:
            self._skipped_paths.append(
                SkippedPath(
                    location=location,
                    expression=token.value,
                    reason=f"unresolved variables: {', '.join(sorted(unresolved))}",
                )
            )
            return set()
        result: set[Path] = set()
        for value in values:
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
        if not node.args:
            raise AnalysisError(
                f"{relative_path.as_posix()}:{node.line}: declaration has no name"
            )
        names, unresolved_names = _expand_token(node.args[0], environment)
        if unresolved_names or len(names) != 1:
            raise AnalysisError(
                f"{relative_path.as_posix()}:{node.line}: cannot resolve exactly one "
                f"subproject name from {node.args[0].value!r}"
            )
        name = next(iter(names))
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

    def _resolve_dependency_section(
        self,
        tokens: list[Token],
        environment: Environment,
        relative_path: Path,
        declaration_line: int,
    ) -> set[str]:
        values, unresolved = _expand_tokens(tokens, environment)
        if unresolved:
            raise AnalysisError(
                f"{relative_path.as_posix()}:{declaration_line}: unresolved "
                f"dependency variables: {', '.join(sorted(unresolved))}"
            )
        return {value for value in values if value}

    def _resolve_source_dir_section(
        self, tokens: list[Token], environment: Environment
    ) -> set[str]:
        """Resolve an EXTERNAL_SOURCE_DIR value; unresolved -> {} (skipped silently)."""
        # Sources wired through therock_enable_external_source() do not resolve
        # statically and get no subtree_map entry (handled as reader-side overrides).
        values, unresolved = _expand_tokens(tokens, environment)
        if unresolved:
            return set()
        return {value for value in values if value}


def _declaration_sections(tokens: list[Token]) -> dict[str, list[Token]]:
    sections: dict[str, list[Token]] = {}
    current_keyword: str | None = None
    for token in tokens:
        if token.kind == "COMMENT":
            continue
        if token.value in _DECLARATION_KEYWORDS:
            current_keyword = token.value
            sections.setdefault(current_keyword, [])
        elif current_keyword is not None:
            sections[current_keyword].append(token)
    return sections


def _expand_tokens(
    tokens: list[Token], environment: Environment
) -> tuple[set[str], set[str]]:
    values: set[str] = set()
    unresolved: set[str] = set()
    for token in tokens:
        if token.kind == "COMMENT":
            continue
        token_values, token_unresolved = _expand_token(token, environment)
        values.update(token_values)
        unresolved.update(token_unresolved)
    return values, unresolved


def _expand_token(token: Token, environment: Environment) -> tuple[set[str], set[str]]:
    references = _VARIABLE_REFERENCE_PATTERN.findall(token.value)
    unresolved = {name for name in references if name not in environment}
    if unresolved:
        return set(), unresolved
    expanded_values = {token.value}
    for variable_name in references:
        replacements = environment.get(variable_name, set())
        if not replacements:
            # A variable that is *defined* but empty is not flagged unresolved, so in
            # a dependency arg its edge drops silently (an under-selection). The
            # differential-configure superset check is the guard; a unit test pins it.
            return set(), set()
        next_values = set()
        reference = "${" + variable_name + "}"
        for value in expanded_values:
            for replacement in replacements:
                next_values.add(value.replace(reference, replacement))
        expanded_values = next_values
    split_values = {
        item for value in expanded_values for item in value.split(";") if item
    }
    return split_values, set()


def _contains_declaration(nodes: list[cmake_ast.AstNode]) -> bool:
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
