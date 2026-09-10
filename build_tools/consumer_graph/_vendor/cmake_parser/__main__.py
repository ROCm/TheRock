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


def prepare_args():
    p = argparse.ArgumentParser()
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument(
        "--parse-raw",
        metavar="FILE",
        type=argparse.FileType("r"),
        help="use raw parser on CMake file",
    )
    m.add_argument(
        "--parse",
        metavar="FILE",
        type=argparse.FileType("r"),
        help="use tree parser on CMake file",
    )
    m.add_argument(
        "--resolve-args",
        metavar="ARG",
        nargs="*",
        help="parse and resolve CMake arguments",
    )
    m.add_argument(
        "--eval-expr",
        metavar="ARG",
        nargs="*",
        help="parse and evaluate boolean expression",
    )
    p.add_argument(
        "-D",
        metavar="NAME=VALUE",
        action="append",
        default=[],
        help="set CMake variable",
    )
    p.add_argument(
        "--skip-comments",
        action="store_true",
        default=False,
        help="filter out comments while parsing",
    )
    return p


if __name__ == "__main__":
    import argparse
    import sys
    from pprint import pprint

    parser = prepare_args()
    args = parser.parse_args()

    vars = {}
    for name_value in args.D:
        name, value = name_value.split("=", 1)
        vars[name] = value

    if args.parse_raw:
        from .parser import parse_raw

        pprint(list(parse_raw(args.parse_raw.read(), skip_comments=args.skip_comments)))
        sys.exit(0)

    if args.parse:
        from .parser import parse_tree

        pprint(list(parse_tree(args.parse.read(), skip_comments=args.skip_comments)))
        sys.exit(0)

    if args.resolve_args:
        from .parser import parse_raw
        from .interpreter import Context, resolve_args

        cmd = next(parse_raw(f"_({' '.join(args.resolve_args)})"))
        ctx = Context(var=vars)
        pprint(resolve_args(ctx, cmd.args))
        sys.exit(0)

    if args.eval_expr:
        from .parser import parse_raw
        from .interpreter import Context, resolve_args, eval_expr

        cmd = next(parse_raw(f"_({' '.join(args.eval_expr)})"))
        ctx = Context(var=vars)
        expr = resolve_args(ctx, cmd.args)
        print(eval_expr(ctx, expr))
        sys.exit(0)

    sys.exit(1)
