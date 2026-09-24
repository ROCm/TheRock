# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import unittest
from pathlib import Path


THEROCK_DIR = Path(__file__).resolve().parent.parent


class TsanComponentExecutionTest(unittest.TestCase):
    def test_component_execution_cannot_retry_tsan_into_green(self):
        workflow = (
            THEROCK_DIR / ".github" / "workflows" / "test_component.yml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "RETRY_THIS_STEP: ${{ inputs.build_variant == 'tsan' "
            "&& 'false' || 'true' }}",
            workflow,
        )
        self.assertIn(
            "RETRY_COUNT: ${{ inputs.build_variant == 'tsan' " "&& '0' || '1' }}",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
