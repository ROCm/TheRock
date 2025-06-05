from pathlib import Path
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
import urllib.request
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from fetch_artifacts import (
    IndexPageParser,
    compute_enabled_artifacts,
    retrieve_s3_artifacts,
    ArtifactNotFoundExeption,
    FetchArtifactException,
)

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent.parent


def run_indexer_file(temp_dir):
    subprocess.run(
        [
            sys.executable,
            REPO_DIR / "third-party" / "indexer" / "indexer.py",
            "-f",
            "*.tar.xz*",
            temp_dir,
        ]
    )


def create_sample_tar_files(temp_dir):
    with open(temp_dir / "test.txt", "w") as file:
        file.write("Hello, World!")

    with tarfile.open(temp_dir / "empty_1.tar.xz", "w:xz") as tar:
        tar.add(temp_dir / "test.txt", arcname="test.txt")

    with tarfile.open(temp_dir / "empty_2.tar.xz", "w:xz") as tar:
        tar.add(temp_dir / "test.txt", arcname="test.txt")

    with tarfile.open(temp_dir / "empty_3.tar.xz", "w:xz") as tar:
        tar.add(temp_dir / "test.txt", arcname="test.txt")


class ArtifactsIndexPageTest(unittest.TestCase):
    def testComputeEnabledArtifacts(self):
        s3_artifacts = set()
        s3_artifacts.add("foo_dev_generic.tar.xz")
        s3_artifacts.add("foo_dev_targetABC.tar.xz")
        s3_artifacts.add("foo_dev_targetXYZ.tar.xz")
        s3_artifacts.add("bar_dev_generic.tar.xz")
        s3_artifacts.add("bar_test_generic.tar.xz")
        s3_artifacts.add("baz_dev_generic.tar.xz")

        enabled_artifacts = compute_enabled_artifacts(
            s3_artifacts=s3_artifacts,
            names="foo,bar",
            generic_only=False,
            target="targetABC",
            dev=True,
            lib=False,
            test=False,
        )

        self.assertIn("foo_dev_generic.tar.xz", enabled_artifacts)
        self.assertIn("foo_dev_targetABC.tar.xz", enabled_artifacts)
        self.assertNotIn("foo_dev_targetXYZ.tar.xz", enabled_artifacts)
        self.assertIn("bar_dev_generic.tar.xz", enabled_artifacts)
        self.assertNotIn("bar_test_generic.tar.xz", enabled_artifacts)
        self.assertNotIn("baz_dev_generic.tar.xz", enabled_artifacts)

    def testCreateIndexPage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            create_sample_tar_files(temp_dir)
            run_indexer_file(temp_dir)

            index_file_path = Path(temp_dir / "index.html")
            self.assertGreater(index_file_path.stat().st_size, 0)
            # Ensuring we have three tar.xz files
            parser = IndexPageParser()
            with open(temp_dir / "index.html", "r") as file:
                parser.feed(str(file.read()))
            self.assertEqual(len(parser.files), 3)

    @patch("urllib.request.urlopen")
    def testRetrieveS3Artifacts(self, mock_urlopen):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            create_sample_tar_files(temp_dir)
            run_indexer_file(temp_dir)

            with open(temp_dir / "index.html", "r") as file:
                mock_urlopen().__enter__().read.return_value = file.read()

            result = retrieve_s3_artifacts("123", "test")

            self.assertEqual(len(result), 3)
            self.assertTrue("empty_1.tar.xz" in result)
            self.assertTrue("empty_2.tar.xz" in result)
            self.assertTrue("empty_3.tar.xz" in result)

    @patch("urllib.request.urlopen")
    def testRetrieveS3ArtifactsNotFound(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.request.HTTPError(
            code=404, msg="ok", hdrs=None, fp=None, url=None
        )

        with self.assertRaises(ArtifactNotFoundExeption):
            retrieve_s3_artifacts("123", "test")

    @patch("urllib.request.urlopen")
    def testRetrieveS3ArtifactsFetchNotFound(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.request.HTTPError(
            code=400, msg="ok", hdrs=None, fp=None, url=None
        )

        with self.assertRaises(FetchArtifactException):
            retrieve_s3_artifacts("123", "test")


if __name__ == "__main__":
    unittest.main()
