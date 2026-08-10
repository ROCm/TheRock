"""
Unit tests for external-builds/jax/install_jax_wheels.py and
external-builds/jax/install_jax_test_requirements.py
"""

import argparse
import os
import sys
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
JAX_DIR = THIS_DIR.parents[2] / "external-builds" / "jax"

sys.path.insert(0, os.fspath(JAX_DIR))

import install_jax_test_requirements as requirements
import install_jax_wheels as wheels

INDEX_URL = "https://rocm.nightlies.amd.com/whl-multi-arch/"
FIND_LINKS_URL = "https://therock-artifacts.s3.amazonaws.com/12345-linux/index.html"


def wheel_args(**overrides) -> argparse.Namespace:
    values = dict(
        index_url=INDEX_URL,
        find_links="",
        plugin_package="jax_rocm10_plugin",
        pjrt_package="jax_rocm10_pjrt",
        plugin_version="0.11.0",
        pjrt_version="0.11.0",
        jax_version="0.11.0",
        jaxlib_version="",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class InstallJaxWheelsTest(unittest.TestCase):
    def test_plugin_and_pjrt_come_from_the_index(self):
        first = wheels.install_commands(wheel_args())[0]

        self.assertIn("--index-url", first)
        self.assertEqual(first[first.index("--index-url") + 1], INDEX_URL)
        self.assertIn("jax_rocm10_plugin==0.11.0", first)
        self.assertIn("jax_rocm10_pjrt==0.11.0", first)

    def test_without_a_built_jaxlib_both_come_from_pypi(self):
        commands = wheels.install_commands(wheel_args())

        self.assertEqual(len(commands), 2)
        self.assertNotIn("--index-url", commands[1])
        self.assertIn("jax==0.11.0", commands[1])
        self.assertIn("jaxlib==0.11.0", commands[1])

    def test_a_built_jaxlib_comes_from_the_index_and_jax_from_pypi(self):
        commands = wheels.install_commands(
            wheel_args(jaxlib_version="0.11.0.dev20260804")
        )

        self.assertEqual(len(commands), 3)
        self.assertIn("--index-url", commands[1])
        self.assertIn("jaxlib==0.11.0.dev20260804", commands[1])
        # jax itself is never built here, so it always comes from PyPI.
        self.assertNotIn("--index-url", commands[2])
        self.assertIn("jax==0.11.0", commands[2])

    def test_no_index_url_leaves_the_flag_out(self):
        commands = wheels.install_commands(wheel_args(index_url=""))

        for command in commands:
            self.assertNotIn("--index-url", command)

    def test_ci_wheels_come_from_a_find_links_page(self):
        first = wheels.install_commands(
            wheel_args(index_url="", find_links=FIND_LINKS_URL)
        )[0]

        self.assertIn("--find-links", first)
        self.assertEqual(first[first.index("--find-links") + 1], FIND_LINKS_URL)

    def test_a_find_links_page_wins_over_an_index(self):
        # An index that does not carry the run's wheels cannot install them, so
        # a caller that gave both meant the page.
        first = wheels.install_commands(wheel_args(find_links=FIND_LINKS_URL))[0]

        self.assertIn("--find-links", first)
        self.assertNotIn("--index-url", first)

    def test_missing_arguments_are_rejected(self):
        with self.assertRaises(SystemExit):
            wheels.main(["--plugin-package", "jax_rocm10_plugin"])


class InstallTestRequirementsTest(unittest.TestCase):
    def test_lock_file_name_per_python_version(self):
        self.assertEqual(
            requirements.lock_file(Path("jax"), "3.12").name,
            "requirements_lock_3_12.txt",
        )

    def test_commands_use_the_checkout_and_the_lock_file(self):
        commands = requirements.install_commands(Path("jax"), "3.12")

        joined = [" ".join(command) for command in commands]
        self.assertTrue(any(requirements.UV_REQUIREMENT in c for c in joined))
        self.assertTrue(
            any(
                os.path.join("jax", "build", "test-requirements.txt") in c
                for c in joined
            )
        )
        self.assertTrue(any("requirements_lock_3_12.txt" in c for c in joined))
        # The fresh-process retry reads the reports this writes.
        self.assertTrue(any(requirements.REPORT_REQUIREMENT in c for c in joined))

    def test_uv_is_installed_before_it_is_used(self):
        commands = requirements.install_commands(Path("jax"), "3.12")

        first = " ".join(commands[0])
        self.assertIn("pip", first)
        self.assertIn(requirements.UV_REQUIREMENT, first)
        self.assertIn("uv", commands[1])

    def test_python_version_is_required(self):
        with self.assertRaises(SystemExit):
            requirements.main(["--jax-dir", "jax"])


if __name__ == "__main__":
    unittest.main()
