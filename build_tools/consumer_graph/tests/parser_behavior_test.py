# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Characterization tests for cmake-parser 0.9.2 behavior."""

from cmake_parser import CMakeParseError
from cmake_parser import ast as cmake_ast
from cmake_parser.parser import parse_tree
import pytest


def test_command_in_comment_or_quoted_value_is_not_false_positive() -> None:
    source = """
# therock_cmake_subproject_declare(commented BUILD_DEPS fake)
set(example "therock_cmake_subproject_declare(in-a-string)")
therock_cmake_subproject_declare(real BUILD_DEPS actual)
"""
    nodes = list(parse_tree(source, skip_comments=True))
    declarations = [
        node
        for node in nodes
        if isinstance(node, cmake_ast.Command)
        and node.identifier == "therock_cmake_subproject_declare"
    ]
    assert len(declarations) == 1
    assert declarations[0].args[0].value == "real"


def test_nested_if_has_structured_branches_and_source_locations() -> None:
    source = """if(WIN32)
  list(APPEND deps windows-dep)
else()
  list(APPEND deps linux-dep)
endif()
"""
    nodes = list(parse_tree(source, skip_comments=True))
    assert len(nodes) == 1
    conditional = nodes[0]
    assert isinstance(conditional, cmake_ast.If)
    assert conditional.line == 1
    assert conditional.if_true[0].line == 2
    assert conditional.if_false is not None
    assert conditional.if_false[0].line == 4


def test_skip_comments_keeps_comment_tokens_inside_command_arguments() -> None:
    source = """
therock_cmake_subproject_declare(example
  COMPILER_TOOLCHAIN
    # This comment remains an argument token.
    amd-hip)
"""
    nodes = list(parse_tree(source, skip_comments=True))
    declaration = nodes[0]
    assert isinstance(declaration, cmake_ast.Command)
    assert [token.kind for token in declaration.args] == [
        "RAW",
        "RAW",
        "COMMENT",
        "RAW",
    ]


def test_malformed_cmake_fails_instead_of_returning_partial_tree() -> None:
    with pytest.raises(CMakeParseError):
        list(parse_tree("if(WIN32)\nset(x y)\n", skip_comments=True))
