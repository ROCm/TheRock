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
Exception which are raised by :mod:`cmake_parser` functions.
"""


class CMakeError(RuntimeError):
    """
    Exception base class for :mod:`cmake_parser`.
    """


class CMakeParseError(CMakeError):
    """
    Exception that indicates a parsing error.

    This exception is raised by :func:`~.parser.parse_raw` and
    :func:`~parser.parse_tree`.
    """


class CMakeExprError(CMakeError):
    pass
