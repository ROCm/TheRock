#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the rocgdb-emulate wrapper script.

The argument parsing helpers are tested directly. The end-to-end tests install
the wrapper as bin/rocgdb-emulate in a fake ROCm tree whose mirage, rocgdb and
llvm-objdump are small shell scripts, so no GPU, emulator or build is needed.
The fake mirage prints the profile it was given and runs the command after
`--`; the fake rocgdb prints its arguments and exits with 3. The fake
llvm-objdump reports one offload bundle per line of the program file, so a
"program" is just a text file listing gfx targets.
"""

import importlib.util
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = THEROCK_ROOT / "debug-tools" / "rocgdb" / "rocgdb-emulate.py"

_spec = importlib.util.spec_from_file_location("rocgdb_emulate", WRAPPER)
rocgdb_emulate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rocgdb_emulate)

FAKE_MIRAGE = """\
#!/bin/sh
[ "$1" = run ] && [ "$2" = --profile ] && [ "$4" = -- ] || exit 99
echo "mirage profile: $3"
shift 4
exec "$@"
"""

FAKE_ROCGDB = """\
#!/bin/sh
echo "rocgdb args: $*"
exit 3
"""

FAKE_OBJDUMP = """\
#!/bin/sh
[ "$1" = --offloading ] || exit 99
while read -r target; do
    echo "Extracting offload bundle: $2.0.hip-amdgcn-amd-amdhsa--$target"
done < "$2"
"""


def write_executable(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class ParseWrapperArgsTest(unittest.TestCase):
    def setUp(self):
        saved = os.environ.pop("ROCGDB_EMULATE_PROFILE", None)
        if saved is not None:
            self.addCleanup(os.environ.__setitem__, "ROCGDB_EMULATE_PROFILE", saved)

    def test_leading_options_only(self):
        args = rocgdb_emulate.parse_wrapper_args(
            ["--profile", "mi350x", "--args", "./app", "--profile", "x", "-h"]
        )
        self.assertEqual(args.profile, "mi350x")
        self.assertFalse(args.show_help)
        self.assertEqual(args.rocgdb_args, ["--args", "./app", "--profile", "x", "-h"])

    def test_double_dash_ends_wrapper_options(self):
        args = rocgdb_emulate.parse_wrapper_args(["--", "--profile", "mi350x"])
        self.assertIsNone(args.profile)
        self.assertEqual(args.rocgdb_args, ["--profile", "mi350x"])

    def test_empty_profile_means_none(self):
        args = rocgdb_emulate.parse_wrapper_args(["--profile=", "./app"])
        self.assertIsNone(args.profile)
        self.assertEqual(args.rocgdb_args, ["./app"])

    def test_missing_profile_value(self):
        with self.assertRaisesRegex(rocgdb_emulate.EmulateError, "needs a value"):
            rocgdb_emulate.parse_wrapper_args(["--profile"])


class FindProgramTest(unittest.TestCase):
    def test_find_program(self):
        cases = [
            (["./app", "1"], "./app"),
            (["-q", "-ex", "break main", "./app"], "./app"),
            (["--eval-command=run", "./app"], "./app"),
            (["-x", "cmds.gdb", "--args", "./app", "-x"], "./app"),
            (["-batch", "--args"], None),
            (["-p", "1234"], None),
            ([], None),
        ]
        for args, program in cases:
            with self.subTest(args=args):
                self.assertEqual(rocgdb_emulate.find_program(args), program)


class TreeTargetsTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.tree = Path(self._temp_dir.name)
        self.info = self.tree / "share" / "therock" / "dist_info.json"
        self.info.parent.mkdir(parents=True)

    def test_separators(self):
        self.info.write_text('{"dist_amdgpu_targets": "gfx942;gfx950, gfx1250"}')
        self.assertEqual(
            rocgdb_emulate.tree_targets(self.tree), ["gfx942", "gfx950", "gfx1250"]
        )

    def test_missing_file(self):
        self.info.unlink(missing_ok=True)
        self.assertEqual(rocgdb_emulate.tree_targets(self.tree), [])

    def test_invalid_json(self):
        self.info.write_text("not json")
        with self.assertRaisesRegex(rocgdb_emulate.EmulateError, "cannot read"):
            rocgdb_emulate.tree_targets(self.tree)

    def test_unexpected_json_shape(self):
        for content in ('["gfx950"]', '{"dist_amdgpu_targets": ["gfx950"]}'):
            with self.subTest(content=content):
                self.info.write_text(content)
                with self.assertRaisesRegex(
                    rocgdb_emulate.EmulateError, "is not a string"
                ):
                    rocgdb_emulate.tree_targets(self.tree)


@unittest.skipIf(sys.platform == "win32", "rocgdb-emulate runs POSIX tools")
class RocgdbEmulateTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.work_dir = Path(self._temp_dir.name)
        self.tree = self.work_dir / "rocm"
        self.wrapper = self.tree / "bin" / "rocgdb-emulate"
        write_executable(self.wrapper, WRAPPER.read_text(encoding="utf-8"))
        write_executable(self.tree / "bin" / "mirage", FAKE_MIRAGE)
        write_executable(self.tree / "bin" / "rocgdb", FAKE_ROCGDB)
        write_executable(
            self.tree / "lib" / "llvm" / "bin" / "llvm-objdump", FAKE_OBJDUMP
        )
        self.set_tree_targets("gfx942;gfx950")

    def set_tree_targets(self, targets: str) -> None:
        info = self.tree / "share" / "therock" / "dist_info.json"
        info.parent.mkdir(parents=True, exist_ok=True)
        info.write_text(
            f'{{\n  "dist_amdgpu_targets" : "{targets}"\n}}\n', encoding="utf-8"
        )

    def make_program(self, *targets: str) -> str:
        program = self.work_dir / "app"
        program.write_text("".join(f"{t}\n" for t in targets), encoding="utf-8")
        return "./app"

    def run_wrapper(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        run_env = {k: v for k, v in os.environ.items() if not k.startswith("ROCGDB_")}
        run_env.update(env)
        return subprocess.run(
            [str(self.wrapper), *args],
            cwd=self.work_dir,
            env=run_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )

    def assert_launched(
        self, result: subprocess.CompletedProcess, profile: str, args: str
    ) -> None:
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [f"mirage profile: {profile}", f"rocgdb args: {args}"],
        )

    def assert_fails(self, result: subprocess.CompletedProcess, message: str) -> None:
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn(message, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_profile_option(self):
        program = self.make_program("gfx950")
        for option in (
            ["--profile", "mi350x"],
            ["--profile", "gfx950"],
            ["--profile=mi350x"],
            ["--profile=MI350X"],
        ):
            with self.subTest(option=option):
                result = self.run_wrapper(*option, "-q", program)
                self.assert_launched(result, "mi350x", f"-q {program}")

    def test_profile_option_program_without_device_code_for_it(self):
        program = self.make_program("gfx942")
        result = self.run_wrapper("--profile", "mi350x", program)
        self.assert_fails(
            result,
            "./app has no device code for gfx950 (built for gfx942); "
            "rebuild with --offload-arch=gfx950",
        )

    def test_environment_profile_program_without_device_code_for_it(self):
        program = self.make_program("gfx942")
        result = self.run_wrapper(program, ROCGDB_EMULATE_PROFILE="gfx950")
        self.assert_fails(result, "./app has no device code for gfx950")

    def test_profile_option_multi_target_program(self):
        program = self.make_program("gfx942", "gfx950")
        result = self.run_wrapper("--profile", "mi300x", program)
        self.assert_launched(result, "mi300x", program)

    def test_profile_option_host_only_program(self):
        program = self.make_program()
        result = self.run_wrapper("--profile", "mi350x", program)
        self.assert_launched(result, "mi350x", program)

    def test_profile_from_environment(self):
        result = self.run_wrapper(ROCGDB_EMULATE_PROFILE="gfx942")
        self.assert_launched(result, "mi300x", "")

    def test_profile_option_overrides_environment(self):
        result = self.run_wrapper(
            "--profile", "mi350x", ROCGDB_EMULATE_PROFILE="mi300x"
        )
        self.assert_launched(result, "mi350x", "")

    def test_profile_from_program(self):
        program = self.make_program("gfx942")
        result = self.run_wrapper(program)
        self.assert_launched(result, "mi300x", program)

    def test_profile_from_program_on_path(self):
        write_executable(self.work_dir / "path_bin" / "my_app", "gfx950\n")
        path = f"{self.work_dir / 'path_bin'}{os.pathsep}{os.environ['PATH']}"
        result = self.run_wrapper("my_app", PATH=path)
        self.assert_launched(result, "mi350x", "my_app")

    def test_program_after_options(self):
        program = self.make_program("gfx950")
        args = ["-q", "-ex", "break main", "-x", "cmds.gdb", "--args", program, "1"]
        result = self.run_wrapper(*args)
        self.assert_launched(result, "mi350x", " ".join(args))

    def test_offload_scan_leaves_no_files(self):
        program = self.make_program("gfx950")
        before = sorted(self.work_dir.iterdir())
        self.run_wrapper(program)
        self.assertEqual(sorted(self.work_dir.iterdir()), before)

    def test_profile_from_tree(self):
        self.set_tree_targets("gfx950")
        program = self.make_program("gfx942", "gfx950")
        result = self.run_wrapper(program)
        self.assert_launched(result, "mi350x", program)

    def test_no_profile_found(self):
        program = self.make_program("gfx942", "gfx950")
        result = self.run_wrapper(program)
        self.assert_fails(result, "could not tell which GPU to emulate")
        self.assertIn("Usage: rocgdb-emulate", result.stderr)

    def test_unknown_profile(self):
        result = self.run_wrapper("--profile", "mi100")
        self.assert_fails(result, "unknown profile 'mi100'")

    def test_missing_profile_value(self):
        result = self.run_wrapper("--profile")
        self.assert_fails(result, "--profile needs a value")

    def test_program_without_emulator_profile(self):
        program = self.make_program("gfx1100")
        result = self.run_wrapper(program)
        self.assert_fails(result, "built for gfx1100, which has no emulator profile")

    def test_profile_not_in_tree(self):
        result = self.run_wrapper("--profile", "mi450x")
        self.assert_fails(result, "mi450x emulates gfx1250, but")

    def test_missing_mirage(self):
        (self.tree / "bin" / "mirage").unlink()
        result = self.run_wrapper("--profile", "mi350x")
        self.assert_fails(result, "no mirage in")

    def test_debug_mode_does_not_run(self):
        result = self.run_wrapper("--profile", "mi350x", ROCGDB_WRAPPER_DEBUG="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("Executing:", result.stderr)
        self.assertIn("-- sh -c '[ -t 0 ] && stty", result.stderr)

    def test_help(self):
        result = self.run_wrapper("--help-emulate")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Usage: rocgdb-emulate", result.stdout)


if __name__ == "__main__":
    unittest.main()
