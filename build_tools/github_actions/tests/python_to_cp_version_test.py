from pathlib import Path
import os
import sys
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import python_to_cp_version


class PythonToCpVersionTest(unittest.TestCase):
    def test_312(self):
        python_version = "3.12"
        cp_version = python_to_cp_version.transform_python_version(
            python_version=python_version
        )

        self.assertEqual(
            cp_version,
            "cp312-cp312",
        )

    def test_313t(self):
        python_version = "3.13t"
        cp_version = python_to_cp_version.transform_python_version(
            python_version=python_version
        )

        self.assertEqual(
            cp_version,
            "cp313t-cp313t",
        )

    def test_invalid_version(self):
        python_version = "0"
        self.assertRaises(
            ValueError, python_to_cp_version.transform_python_version, python_version
        )


if __name__ == "__main__":
    unittest.main()
