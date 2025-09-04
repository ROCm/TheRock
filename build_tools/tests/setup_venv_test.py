from pathlib import Path
import sys
import unittest
import os
import re

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from setup_venv import (
    GFX_TARGET_REGEX,
)


class GfxRegexPatternTest(unittest.TestCase):
    def test_valid_match(self):
        html_snippet = '<a href="relpath/to/wherever/gfx103X-dgpu">gfx103X-dgpu</a><br><a href="/relpath/gfx120X-all">gfx120X-all</a>'
        matches = re.findall(GFX_TARGET_REGEX, html_snippet)
        self.assertEqual(["gfx103X-dgpu", "gfx120X-all"], matches)

    def test_match_without_suffix(self):
        html_snippet = "<a>gfx940</a><br><a>gfx1030</a>"
        matches = re.findall(GFX_TARGET_REGEX, html_snippet)
        self.assertEqual(["gfx940", "gfx1030"], matches)

    def test_invalid_match(self):
        html_snippet = "<a>gfx94000</a><br><a>gfx1030X-dgpu</a>"
        matches = re.findall(GFX_TARGET_REGEX, html_snippet)
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
