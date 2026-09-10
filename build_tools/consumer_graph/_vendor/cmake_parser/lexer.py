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
Basic functionality for tokenizing CMake code into its syntactic constituents.
"""

import re
from attrs import define
from typing import Generator, List, Tuple

_token_spec = [
    ("SP", r"\s+"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("SEMICOLON", r";"),
    ("QUOTED", r'"(?:\\.|[^\\"])*"'),
    ("BRACKET_COMMENT", r"#\[(?P<COMMENT_FILL>=*)\[.*?\](?P=COMMENT_FILL)\]"),
    ("BRACKETED", r"\[(?P<BRACKET_FILL>=*)\[.*?\](?P=BRACKET_FILL)\]"),
    ("LINE_COMMENT", r"#(?!\[).*?$"),
    ("RAW", r'(?:\\.|[^\\\(\)"# \t\r\n;])+'),
]

_next_token = re.compile(
    "|".join(f"(?P<{name}>{expr})" for name, expr in _token_spec),
    re.MULTILINE | re.DOTALL,
).match


@define
class Token:
    """
    Parser token.

    Instances of this class are yielded by :func:`tokenize` and represent the
    syntactic primitives of CMake code.

    :param kind: the token type. This attribute can take one of the string
        constants ``COMMENT``, ``RAW``, ``QUOTED``, ``BRACKETED``, ``LPAREN``,
        ``RPAREN``, or ``SEMICOLON``. Two additional token types, ``UNMATCHED_BRACKET``
        and ``UNPARSEABLE``, can occur in malformed CMake code.

    :param value: the literal text that comprises the token, without the enclosing
        quotes or brackets (if applicable).

    :param span: the token location in the tokenized string.

    :param line: the line number where the token begins. Some tokens may span multiple
        lines.

    :param column: the column where the token begins.
    """

    kind: str
    value: str
    span: slice
    line: int
    column: int


TokenGenerator = Generator[Token, None, None]


def line_ranges(data: str) -> Generator[Tuple[int, int], None, None]:
    start = 0
    for mo in re.finditer(r"\r?\n", data):
        yield (start, mo.start())
        start = mo.end()
    yield (start, len(data))


def normalize_lineendings(data: str) -> str:
    return data.replace("\r\n", "\n")


def tokenize(data: str) -> TokenGenerator:
    """
    Split CMake code into parser tokens.

    Usually, you will not call this function directly but through
    :func:`~cmake_parser.parser.parse_raw` or :func:`~cmake_parser.parser.parse_tree`.
    :func:`tokenize` splits the input string into meaningful chunks for the parser.
    It will not resolve variable references nor split arguments yet; the dynamic nature of
    the CMake language requires this be handled at a later stage.

    :param data: a string containing CMake code.
    :return: a generator that yields the parser tokens as they occur in the code.

    >>> list((t.kind, t.value) for t in tokenize('foo("bar")'))
    [('RAW', 'foo'), ('LPAREN', '('), ('QUOTED', 'bar'), ('RPAREN', ')')]
    """
    pos = 0
    LR = line_ranges(data)
    L = next(LR, None)
    line = 1
    mo = _next_token(data)
    while mo is not None:
        kind = mo.lastgroup
        if kind != "SP":
            val = mo.group(kind)
            if kind == "QUOTED":
                val = val[1:-1]
                val = re.sub(r"\\\r?\n", "", val)
            elif kind == "BRACKETED":
                val = re.sub(
                    r"^\[(=*)\[(?:\r?\n)?(.*)\]\1\]$", r"\2", val, flags=re.DOTALL
                )
            elif kind == "BRACKET_COMMENT":
                val = re.sub(
                    r"^.\[(=*)\[(?:\r?\n)?(.*)\]\1\]$", r"\2", val, flags=re.DOTALL
                )
                kind = "COMMENT"
            elif kind == "LINE_COMMENT":
                val = val[1:]
                kind = "COMMENT"
            elif kind == "RAW":
                if re.match(r"#?\[=*\[", val):
                    kind = "UNMATCHED_BRACKET"
            yield Token(
                kind=kind,
                value=val,
                span=slice(mo.start(), mo.end()),
                line=line,
                column=mo.start() + 1 - L[0],
            )
        pos = mo.end()
        while L and L[1] < pos:
            L = next(LR, None)
            line += 1
        mo = _next_token(data, pos)
    if pos < len(data):
        yield Token(
            kind="UNPARSEABLE",
            value=None,
            span=slice(pos, len(data)),
            line=line,
            column=pos + 1 - L[0],
        )
