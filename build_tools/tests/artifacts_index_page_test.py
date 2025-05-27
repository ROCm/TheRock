import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import unittest
from unittest.mock import patch
import urllib.request

sys.path.append('build_tools')
from fetch_artifacts import IndexPageParser, retrieve_s3_artifacts, ArtifactNotFoundExeption, FetchArtifactException

THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR / "test_dir"

def get_indexer_file():
    subprocess.run(["curl",
                    "--silent",
                    "--fail",
                    "--show-error",
                    "--location",
                    "https://raw.githubusercontent.com/joshbrunty/Indexer/6d8cbfd15d3853b482e6a49f2d875ded9188b721/indexer.py",
                    "--output",
                    TEST_DIR / "indexer.py"])
    pass

def run_indexer_file():
    subprocess.run(["python",
                    TEST_DIR / "indexer.py",
                    "-f",
                    "*.tar.xz*",
                    TEST_DIR])

def create_sample_tar_files():
    with open(TEST_DIR / "test.txt", "w") as file:
        file.write("Hello, World!")
        
    with tarfile.open(TEST_DIR / "empty_1.tar.xz", "w:xz") as tar:
        tar.add(TEST_DIR / "test.txt", arcname="test.txt")
    with tarfile.open(TEST_DIR / "empty_2.tar.xz", "w:xz") as tar:
        tar.add(TEST_DIR / "test.txt", arcname="test.txt")
    with tarfile.open(TEST_DIR / "empty_3.tar.xz", "w:xz") as tar:
        tar.add(TEST_DIR / "test.txt", arcname="test.txt")

class ArtifactsIndexPageTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        os.makedirs(TEST_DIR, exist_ok=True)
        create_sample_tar_files()
        get_indexer_file()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(TEST_DIR)
        pass

    def testCreateIndexPage(self):
        run_indexer_file()
        self.assertGreater(os.path.getsize(TEST_DIR / "index.html"), 0)
        # Ensuring we have three tar.xz files
        parser = IndexPageParser()
        with open(TEST_DIR / "index.html", "r") as file:
            parser.feed(str(file.read()))
        self.assertEqual(len(parser.files), 3)    
        
    @patch('urllib.request.urlopen')
    def testRetrieveS3Artifacts(self, mock_urlopen):
        with open(TEST_DIR / "index.html", "r") as file:
            mock_urlopen().__enter__().read.return_value = file.read()

        result = retrieve_s3_artifacts("123", "test")

        self.assertEqual(len(result), 3)
        self.assertTrue("empty_1.tar.xz" in result)
        self.assertTrue("empty_2.tar.xz" in result)
        self.assertTrue("empty_3.tar.xz" in result)

    @patch('urllib.request.urlopen')
    def testRetrieveS3ArtifactsNotFound(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.request.HTTPError(code=404, msg="ok", hdrs=None, fp=None, url=None)

        with self.assertRaises(ArtifactNotFoundExeption):
            retrieve_s3_artifacts("123", "test")
        
    @patch('urllib.request.urlopen')
    def testRetrieveS3ArtifactsFetchNotFound(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.request.HTTPError(code=400, msg="ok", hdrs=None, fp=None, url=None)

        with self.assertRaises(FetchArtifactException):
            retrieve_s3_artifacts("123", "test")

if __name__ == "__main__":
    unittest.main()
