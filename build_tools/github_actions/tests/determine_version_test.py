from pathlib import Path
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import determine_version


class DetermineVersionTest(unittest.TestCase):
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

    def test_dev_version(self):
        temp_file = self.temp_dir / "gh-env"
        os.environ["GITHUB_ENV"] = f"{temp_file}"
        determine_version.main(
            ["--rocm-version", "7.0.0.dev0+515115ea2cb85a0b71b5507ce56a627d14c7ae73"]
        )

        with open(temp_file, "r") as f:
            optional_build_prod_arguments = f.read()

        self.assertEqual(
            optional_build_prod_arguments,
            "optional_build_prod_arguments=--rocm-sdk-version ==7.0.0.dev0+515115ea2cb85a0b71b5507ce56a627d14c7ae73 --version-suffix +rocm7.0.0.dev0-515115ea2cb85a0b71b5507ce56a627d14c7ae73",
        )

    def test_nightly_version(self):
        temp_file = self.temp_dir / "gh-env"
        os.environ["GITHUB_ENV"] = f"{temp_file}"
        determine_version.main(["--rocm-version", "7.0.0rc20250707"])

        with open(temp_file, "r") as f:
            optional_build_prod_arguments = f.read()

        self.assertEqual(
            optional_build_prod_arguments,
            "optional_build_prod_arguments=--rocm-sdk-version ==7.0.0rc20250707 --version-suffix +rocm7.0.0rc20250707",
        )


if __name__ == "__main__":
    unittest.main()
