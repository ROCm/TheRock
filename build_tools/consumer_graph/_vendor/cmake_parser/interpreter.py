# CMake Parser
# Copyright 2023 Timo Röhling <timo@gaussglocke.de>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Helper functions for executing CMake code in Python.
"""

import re
import os
from attrs import define, evolve
from functools import partial
from typing import Dict, List, Tuple, Union, Callable, Any, Type
from .ast import *
from .lexer import Token, TokenGenerator
from .error import CMakeExprError


try:
    from typing import Self
except ImportError:
    from typing import TypeVar

    Self = TypeVar("Self", bound="Context")


@define
class Context:
    """
    Execution context for CMake code.

    :param parent: the parent of a scoped context or :const:`None` for the top-level context.
    :param var: defined CMake variables
    :param env: available environment variables
    :param cache: defined CMake cache variables
    :param functions: defined functions
    :param macros: defined macros
    """

    parent: Self = None
    var: Dict[str, str] = {}
    env: Dict[str, str] = {}
    cache: Dict[str, str] = {}
    functions: Dict[str, Function] = {}
    macros: Dict[str, Macro] = {}

    def exists(self, f: str) -> bool:
        """
        Helper function to check if a file or directory exists

        :param f: a path name
        :return: :const:`True` if the path name refers to an existing file or directory.
        """
        return os.path.exists(f)


_TOKEN_SPEC = [
    ("ESCAPE", r"\\."),
    ("SEMICOLON", r";"),
    ("VAR_BEGIN", r"\$(?P<VAR_TYPE>[A-Za-z]*)\{"),
    ("VAR_END", r"\}"),
    ("LITERAL", r"[^\\\$\{\}]+"),
]

_NEXT_TOKEN = re.compile(
    "|".join(f"(?P<{name}>{expr})" for name, expr in _TOKEN_SPEC)
).match


_ESC_SEQS = {
    "r": "\r",
    "n": "\n",
    "t": "\t",
    ";": "\\;",  # Semicolons get special treatment in _split()
}


def _resolve_vars(ctx: Context, token: Token) -> str:
    def scan(s: str, pos: int) -> Tuple[str, int]:
        result = ""
        mo = _NEXT_TOKEN(s, pos)
        while mo:
            kind = mo.lastgroup
            val = mo.group(kind)
            if kind == "VAR_BEGIN":
                var_type = mo.group("VAR_TYPE")
                identifier, end_pos = scan(s, mo.end(kind))
                if var_type == "":
                    result += ctx.var.get(identifier, "")
                elif var_type == "ENV":
                    result += ctx.env.get(identifier, "")
                elif var_type == "CACHE":
                    result += ctx.cache.get(identifier, "")
                mo = _NEXT_TOKEN(s, end_pos)
                continue
            if kind == "VAR_END" and pos > 0:
                return result, mo.end(kind)
            if kind == "ESCAPE":
                result += _ESC_SEQS.get(val[1], val[1])
            else:
                result += val
            mo = _NEXT_TOKEN(s, mo.end(kind))
        if pos > 0:
            raise CMakeExprError(
                f"variable reference without terminating '}}' at line {token.line}, column {token.column}: {s[pos:]!r}"
            )
        return result, len(s)

    return scan(token.value, 0)[0]


def _split(s: str) -> List[str]:
    result = []
    for item in re.finditer(r"(?:\\.|[^;\\])+", s):
        result.append(item.group(0).replace("\\;", ";"))
    return result


def resolve_args(ctx: Context, args: List[Token]) -> List[Token]:
    """
    Resolve variable references in argument lists.

    Due to the semantics of unquoted variable references, this
    function may also change the number and type of arguments,
    which means that the returned argument list can have more or
    fewer items than the input, depending on the execution context.

    The returned token list will have a proper 1:1 relation between
    function arguments and tokens. Boolean expressions are suitable
    input for :func:`eval_expr` after they have been processed by this
    function.

    :param ctx: the current execution context
    :param args: the argument tokens as they were parsed by
        :func:`~cmake_parser.parser.parse_raw` or
        :func:`~cmake_parser.parser.parse_tree`.
    :return: a new argument list with all variable references resolved
    """
    result = []
    for token in args:
        if token.kind == "RAW":
            value = _resolve_vars(ctx, token)
            result.extend(evolve(token, value=item) for item in _split(value))
        else:
            if token.kind == "QUOTED":
                value = _resolve_vars(ctx, token)
            else:
                value = token.value
            result.append(evolve(token, value=value))
    return result


class LookAheadIterator:
    def __init__(self, iterable):
        self._iterable = iter(iterable)
        self._exhausted = False
        self._lookahead = None
        try:
            self._lookahead = next(self._iterable)
        except StopIteration:
            self._exhausted = True

    def peek(self):
        return self._lookahead

    def __iter__(self):
        return self

    def __next__(self):
        if self._exhausted:
            raise StopIteration
        value = self._lookahead
        try:
            self._lookahead = next(self._iterable)
        except StopIteration:
            self._exhausted = True
        return value


def _is_true_constant(s: str) -> bool:
    try:
        number = float(s)
        return number != 0
    except ValueError:
        pass
    return s.upper() in ["ON", "YES", "TRUE", "Y"]


def _is_false_constant(s: str) -> bool:
    try:
        number = float(s)
        return number == 0
    except ValueError:
        pass
    su = s.upper()
    return su in ["", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND"] or su.endswith(
        "-NOTFOUND"
    )


def _eval_value(ctx: Context, value: Union[bool, Token, str]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Token):
        if value.kind == "QUOTED":
            return _is_true_constant(value.value)
        value = value.value
    if _is_true_constant(value):
        return True
    if _is_false_constant(value):
        return False
    value = ctx.var.get(value, "")
    return not _is_false_constant(value)


def _get_argument(op: str, G: TokenGenerator) -> Token:
    token = next(G, None)
    if token is None or token.kind not in ["RAW", "QUOTED"]:
        raise CMakeExprError(f"missing argument for {op} operator")
    return token


def _eval_defined(ctx: Context, arg: Token) -> bool:
    return arg.value in ctx.var


def _eval_exists(ctx: Context, arg: Token) -> bool:
    return ctx.exists(arg.value)


def _eval_command(ctx: Context, arg: Token) -> bool:
    return arg.value in ctx.functions or arg.value in ctx.macros


def _path(s: str) -> str:
    return re.sub(r"([\\/])\1+", r"\1", s)


def _version(s: str) -> Tuple[int, int, int, int]:
    def _try_int(s: str) -> int:
        try:
            return int(s)
        except ValueError:
            return 0

    return tuple(_try_int(v) for v in (s.split(".") + ["0", "0", "0", "0"])[:4])


def _eval_compare(
    coerce: Type[Any],
    compare: Callable[[Any, Any], bool],
    ctx: Context,
    arg1: Token,
    arg2: Token,
) -> bool:
    value1 = (
        ctx.var[arg1.value]
        if arg1.kind == "RAW" and arg1.value in ctx.var
        else arg1.value
    )
    value2 = (
        ctx.var[arg2.value]
        if arg2.kind == "RAW" and arg2.value in ctx.var
        else arg2.value
    )
    try:
        c1 = coerce(value1)
        c2 = coerce(value2)
        return compare(c1, c2)
    except ValueError:
        return False


_PRECEDENCES = {
    "AND": 1,
    "OR": 1,
    "NOT": 2,
}

_UNARY_OPS = {
    "DEFINED": _eval_defined,
    "EXISTS": _eval_exists,
    "COMMAND": _eval_command,
}

_BINARY_OPS = {
    "EQUAL": partial(_eval_compare, int, lambda x, y: x == y),
    "LESS": partial(_eval_compare, int, lambda x, y: x < y),
    "LESS_EQUAL": partial(_eval_compare, int, lambda x, y: x <= y),
    "GREATER": partial(_eval_compare, int, lambda x, y: x > y),
    "GREATER_EQUAL": partial(_eval_compare, int, lambda x, y: x >= y),
    "STREQUAL": partial(_eval_compare, str, lambda x, y: x == y),
    "STRLESS": partial(_eval_compare, str, lambda x, y: x < y),
    "STRLESS_EQUAL": partial(_eval_compare, str, lambda x, y: x <= y),
    "STRGREATER": partial(_eval_compare, str, lambda x, y: x > y),
    "STRGREATER_EQUAL": partial(_eval_compare, str, lambda x, y: x >= y),
    "VERSION_EQUAL": partial(_eval_compare, _version, lambda x, y: x == y),
    "VERSION_LESS": partial(_eval_compare, _version, lambda x, y: x < y),
    "VERSION_LESS_EQUAL": partial(_eval_compare, _version, lambda x, y: x <= y),
    "VERSION_GREATER": partial(_eval_compare, _version, lambda x, y: x > y),
    "VERSION_GREATER_EQUAL": partial(_eval_compare, _version, lambda x, y: x >= y),
    "PATH_EQUAL": partial(_eval_compare, _path, lambda x, y: x == y),
}


def _eval_expr(ctx: Context, G: LookAheadIterator, precedence: int) -> bool:
    token = next(G, None)
    if token is None or token.kind == "RPAREN":
        return False
    stack = []
    while token is not None:
        if token.kind == "LPAREN":
            stack.append(_eval_expr(ctx, G, precedence=1))
        elif token.kind == "RPAREN":
            if len(stack) != 1:
                raise CMakeExprError("Malformed expression")
            return _eval_value(ctx, stack.pop())
        elif token.kind == "RAW":
            if token.value == "NOT":
                stack.append(not _eval_expr(ctx, G, precedence=2))
            elif token.value == "AND":
                first_arg = _eval_value(ctx, stack.pop())
                second_arg = _eval_expr(ctx, G, precedence=2)
                stack.append(first_arg and second_arg)
            elif token.value == "OR":
                first_arg = _eval_value(ctx, stack.pop())
                second_arg = _eval_expr(ctx, G, precedence=2)
                stack.append(first_arg or second_arg)
            else:
                unary_op = _UNARY_OPS.get(token.value, None)
                if unary_op:
                    arg = _get_argument(token.value, G)
                    stack.append(unary_op(ctx, arg))
                else:
                    binary_op = _BINARY_OPS.get(token.value, None)
                    if binary_op:
                        if not stack:
                            raise CMakeExprError(
                                f"missing argument for {token.value} operator"
                            )
                        first_arg = stack.pop()
                        second_arg = _get_argument(token.value, G)
                        stack.append(binary_op(ctx, first_arg, second_arg))
                    else:
                        stack.append(token)
        else:
            stack.append(token)
        lookahead = G.peek()
        if lookahead is None:
            break
        if lookahead.kind == "RAW":
            lookahead_precendence = _PRECEDENCES.get(lookahead.value, precedence)
            if lookahead_precendence < precedence:
                break
        token = next(G, None)
    if len(stack) != 1:
        raise CMakeExprError("Malformed expression")
    return _eval_value(ctx, stack.pop())


def eval_expr(ctx: Context, args: List[Token]) -> bool:
    """
    Evaluate boolean expressions.

    Some CMake command such as ``if()`` have a boolean expression as argument.
    This function will evaluate such expressions to :const:`False` or :const:`True`
    given the current execution context.

    .. note:: This function does not resolve variable references, so you need to
        pass the argument list through :func:`resolve_args` first.

    :param ctx: the current execution context
    :param args: the boolean expression with all variable references resolved
    :return: the boolean value of the expression
    :raise: :exc:`~cmake_parser.error.CMakeExprError` if the expression is malformed.
    """
    G = LookAheadIterator(args)
    return _eval_expr(ctx, G, precedence=1)
