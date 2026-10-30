# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise Mesa's generated build environment with a real shared-library probe."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
MESA = REPO / "third-party/sysdeps/linux/amd-mesa"


@unittest.skipUnless(
    sys.platform.startswith("linux")
    and all(shutil.which(tool) for tool in ("cmake", "ninja", "gcc", "g++")),
    "Requires Linux CMake/Ninja and C/C++ compilers",
)
class MesaDasanEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mesa-dasan-env-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = self.root / "compiler/lib/clang/24/lib/x86_64-unknown-linux-gnu"
        self.runtime.mkdir(parents=True)
        source = self.root / "runtime.c"
        source.write_text("int runtime_marker(void) { return 7; }\n")
        subprocess.run(
            [
                "gcc",
                "-shared",
                "-fPIC",
                source,
                "-Wl,-soname,libclang_rt.dasan.so",
                "-o",
                self.runtime / "libclang_rt.dasan.so",
            ],
            check=True,
            capture_output=True,
        )
        self.compiler = self.root / "clang"
        self.compiler.write_text(
            "#!/usr/bin/env python3\n"
            "import os,sys\n"
            "if sys.argv[1:] == ['-print-file-name=libclang_rt.dasan.so']:\n"
            f"    print({str(self.runtime / 'libclang_rt.dasan.so')!r})\n"
            "else:\n"
            "    os.execvp('gcc', ['gcc', *sys.argv[1:]])\n"
        )
        self.compiler.chmod(0o755)

    def configure(self, sanitizer="DASAN", compiler=None):
        build = self.root / ("build-" + sanitizer)
        env = {**os.environ, "LD_LIBRARY_PATH": "/existing/compiler-libraries"}
        result = subprocess.run(
            [
                "cmake",
                "-G",
                "Ninja",
                "-S",
                MESA,
                "-B",
                build,
                f"-DTHEROCK_SOURCE_DIR={REPO}",
                f"-DCMAKE_C_COMPILER={compiler or self.compiler}",
                f"-DCMAKE_CXX_COMPILER={shutil.which('g++')}",
                f"-DTHEROCK_SANITIZER={sanitizer}",
                "-DCMAKE_C_FLAGS=-DKEEP_C_FLAGS=1",
                "-DCMAKE_CXX_FLAGS=-DKEEP_CXX_FLAGS=1",
                "-DTHEROCK_PRIVATE_INSTALL_RPATH_DIRS=lib/llvm/lib/clang/24/lib/x86_64-unknown-linux-gnu",
                "-DPATCHELF=/bin/true",
                "-DMESON_BUILD=/bin/true",
                f"-DSOURCE_DIR={self.root / 'source'}",
                f"-DCMAKE_INSTALL_PREFIX={self.root / 'install'}",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        return result, build

    def test_runtime_environment_covers_both_setups_compile_and_install(self):
        result, build = self.configure()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        command = subprocess.check_output(
            ["ninja", "-C", build, "-t", "commands", "meson_build"], text=True
        )
        expected = f"LD_LIBRARY_PATH={self.runtime}:/existing/compiler-libraries"
        # Both compiler-sanity setups and later generated build tools need it.
        self.assertEqual(command.count(expected), 4, command)
        self.assertIn("-fsanitize=daddress", command)
        self.assertIn("-shared-libsan", command)
        self.assertIn("KEEP_C_FLAGS=1", command)
        self.assertIn("KEEP_CXX_FLAGS=1", command)
        # Build-only lookup must not turn into an absolute installed RPATH.
        self.assertNotIn("-rpath," + str(self.runtime), command)
        self.assertIn("ORIGIN/../../llvm/lib/clang/24", command)

        # Reproduce the failed Meson check: link succeeds with the installation
        # RPATH, but execution cannot locate its runtime from meson-private/.
        private = build / "meson-private"
        private.mkdir()
        probe = private / "sanity.c"
        probe.write_text(
            "extern int runtime_marker(void);\n"
            "int main(void) { return runtime_marker() != 7; }\n"
        )
        binary = private / "sanity.exe"
        subprocess.run(
            [
                "gcc",
                probe,
                "-L" + str(self.runtime),
                "-lclang_rt.dasan",
                "-Wl,-rpath,$ORIGIN/../../llvm/lib/clang/24/lib/x86_64-unknown-linux-gnu",
                "-o",
                binary,
            ],
            check=True,
            capture_output=True,
        )
        before = subprocess.run(
            [binary], env={**os.environ, "LD_LIBRARY_PATH": ""}, capture_output=True
        )
        self.assertNotEqual(before.returncode, 0)
        self.assertIn(b"libclang_rt.dasan.so", before.stderr)
        after = subprocess.run(
            [binary],
            env={**os.environ, "LD_LIBRARY_PATH": expected.split("=", 1)[1]},
            capture_output=True,
        )
        self.assertEqual(after.returncode, 0, after.stderr)

    def test_non_dasan_environment_does_not_query_or_add_runtime(self):
        result, build = self.configure(sanitizer="OFF", compiler=shutil.which("gcc"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        command = subprocess.check_output(
            ["ninja", "-C", build, "-t", "commands", "meson_build"], text=True
        )
        self.assertNotIn("LD_LIBRARY_PATH=", command)
        self.assertNotIn("-shared-libsan", command)

    def test_missing_compiler_runtime_is_a_configuration_error(self):
        result, _ = self.configure(compiler=shutil.which("gcc"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "requires the selected compiler's libclang_rt.dasan.so", result.stderr
        )


if __name__ == "__main__":
    unittest.main()
