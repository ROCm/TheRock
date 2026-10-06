# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_prod_wheels import _setup_common_build_env


class UseAsanEnvTest(unittest.TestCase):
    def test_use_asan_is_forwarded_when_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with mock.patch.dict(os.environ, {"USE_ASAN": "1"}):
                env = _setup_common_build_env(root, root, root, "gfx1100", None, False)
        self.assertEqual(env["USE_ASAN"], "1")

    def test_use_asan_is_absent_unless_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with mock.patch.dict(os.environ):
                os.environ.pop("USE_ASAN", None)
                build_env = _setup_common_build_env(
                    root, root, root, "gfx1100", None, False
                )
        self.assertNotIn("USE_ASAN", build_env)


if __name__ == "__main__":
    unittest.main()
