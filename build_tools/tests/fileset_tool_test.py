import hashlib
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
from _therock_utils.hash_util import calculate_hash

FILESET_TOOL = Path(__file__).parent.parent / "fileset_tool.py"

ARTIFACT_DESCRIPTOR_1 = r"""
[components.doc."example/stage"]
"""


def exec(args: list[str | Path], cwd: Path = FILESET_TOOL.parent) -> str:
    args = [str(arg) for arg in args]
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    return subprocess.check_output(
        args, cwd=str(cwd), stdin=subprocess.DEVNULL
    ).decode()


def write_text(p: Path, text: str):
    p.parent.mkdir(exist_ok=True, parents=True)
    p.write_text(text)


def is_windows():
    return platform.system() == "Windows"


def fset_executable(f):
    os.fchmod(f.fileno(), os.fstat(f.fileno()).st_mode | 0o111)


def is_executable(path: Path):
    return bool(os.stat(path).st_mode & 0o111)


class FilesetToolTest(unittest.TestCase):
    def setUp(self):
        override_temp = os.getenv("TEST_TMPDIR")
        if override_temp is not None:
            self.temp_context = None
            self.temp_dir = Path(override_temp)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.temp_context = tempfile.TemporaryDirectory()
            self.temp_dir = Path(self.temp_context.name)

    def tearDown(self):
        if self.temp_context:
            self.temp_context.cleanup()

    # Validates that the happy path flow of creating an artifact, archiving it,
    # expanding and flattening works. This does not exhaustively verify
    # all descriptor options.
    def testSimpleArtifact(self):
        input_dir = self.temp_dir / "input"
        artifact_dir = self.temp_dir / "artifact_dir"
        artifact_archive = self.temp_dir / "artifact.tar.xz"
        descriptor_file = self.temp_dir / "artifact.toml"
        hash_file = self.temp_dir / "artifact.tar.xz.sha256sum"
        flat1_dir = self.temp_dir / "flat1"
        flat2_dir = self.temp_dir / "flat2"
        write_text(descriptor_file, ARTIFACT_DESCRIPTOR_1)

        # One sample file and a symlink, and an executable.
        write_text(
            input_dir / "example" / "stage" / "share" / "doc" / "README.txt",
            "Hello World!",
        )
        Path(input_dir / "example" / "stage" / "share" / "doc" / "README").symlink_to(
            "README.txt"
        )
        if not is_windows():
            with open(
                input_dir / "example" / "stage" / "share" / "doc" / "executable", "wb"
            ) as f:
                f.write(b"Contents")
                fset_executable(f)
        exec(
            [
                sys.executable,
                FILESET_TOOL,
                "artifact",
                "--descriptor",
                descriptor_file,
                "--output-dir",
                artifact_dir,
                "--root-dir",
                input_dir,
                "--component",
                "doc",
            ]
        )

        # Validate artifact dir.
        manifest_lines = (
            (artifact_dir / "artifact_manifest.txt").read_text().strip().splitlines()
        )
        self.assertEqual(manifest_lines, ["example/stage"])
        self.assertEqual(
            (
                artifact_dir / "example" / "stage" / "share" / "doc" / "README.txt"
            ).read_text(),
            "Hello World!",
        )
        self.assertEqual(
            os.readlink(
                artifact_dir / "example" / "stage" / "share" / "doc" / "README"
            ),
            "README.txt",
        )
        if not is_windows():
            self.assertTrue(
                is_executable(
                    artifact_dir / "example" / "stage" / "share" / "doc" / "executable"
                )
            )

        # Archive it.
        exec(
            [
                sys.executable,
                FILESET_TOOL,
                "artifact-archive",
                artifact_dir,
                "-o",
                artifact_archive,
                "--hash-file",
                hash_file,
            ]
        )

        # Verify digest.
        expected_digest = calculate_hash(artifact_archive, "sha256").hexdigest()
        actual_digest = hash_file.read_text().strip()
        self.assertEqual(expected_digest, actual_digest)

        # Flatten the raw directory and verify.
        exec(
            [
                sys.executable,
                FILESET_TOOL,
                "artifact-flatten",
                artifact_dir,
                "-o",
                flat1_dir,
            ]
        )
        self.assertEqual(
            (flat1_dir / "share" / "doc" / "README.txt").read_text(),
            "Hello World!",
        )
        self.assertEqual(
            os.readlink(flat1_dir / "share" / "doc" / "README"),
            "README.txt",
        )
        if not is_windows():
            self.assertTrue(is_executable(flat1_dir / "share" / "doc" / "executable"))

        # Flatten the archive file and verify.
        flatten_output = exec(
            [
                sys.executable,
                FILESET_TOOL,
                "artifact-flatten",
                artifact_archive,
                "--verbose",
                "-o",
                flat2_dir,
            ]
        )
        self.assertListEqual(flatten_output.splitlines(), ["example/stage"])
        self.assertEqual(
            (flat2_dir / "share" / "doc" / "README.txt").read_text(),
            "Hello World!",
        )
        self.assertEqual(
            os.readlink(flat2_dir / "share" / "doc" / "README"),
            "README.txt",
        )
        if not is_windows():
            self.assertTrue(is_executable(flat2_dir / "share" / "doc" / "executable"))


if __name__ == "__main__":
    unittest.main()
