#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Unit tests for build_tools/resolve_docker_image.py.

Run from the repo root:
  python3 -m unittest build_tools.tests.resolve_docker_image_test -v
  python3 build_tools/tests/resolve_docker_image_test.py -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))
import resolve_docker_image
from resolve_docker_image import (
    ValidationError,
    get_image_ref,
    load_images,
    resolve,
    validate_entry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHA = "sha256:" + "a" * 64


def _entry(**kwargs) -> dict:
    """Build a minimal valid entry, overriding specific fields."""
    base = {"registry": "ghcr.io/rocm", "image": "myimage", "sha": _SHA}
    base.update(kwargs)
    return base


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# validate_entry
# ---------------------------------------------------------------------------


class ValidateEntryTest(unittest.TestCase):
    def _errors(self, key: str, entry) -> list[str]:
        return [e.message for e in validate_entry(key, entry)]

    def test_valid_sha_entry(self):
        self.assertEqual(self._errors("k", _entry()), [])

    def test_valid_tag_entry(self):
        entry = {"registry": "docker.io", "image": "python", "tag": "3.12-slim"}
        self.assertEqual(self._errors("k", entry), [])

    def test_not_a_dict(self):
        errs = self._errors("k", "just a string")
        self.assertIn("entry must be an object", errs[0])

    def test_missing_registry(self):
        entry = _entry()
        del entry["registry"]
        errs = self._errors("k", entry)
        self.assertTrue(any("registry" in e for e in errs))

    def test_empty_registry(self):
        errs = self._errors("k", _entry(registry=""))
        self.assertTrue(any("registry" in e for e in errs))

    def test_invalid_registry(self):
        errs = self._errors("k", _entry(registry="not a registry!!"))
        self.assertTrue(any("registry" in e for e in errs))

    def test_missing_image(self):
        entry = _entry()
        del entry["image"]
        errs = self._errors("k", entry)
        self.assertTrue(any("image" in e for e in errs))

    def test_image_with_uppercase(self):
        errs = self._errors("k", _entry(image="Bad_Image"))
        self.assertTrue(any("image" in e for e in errs))

    def test_invalid_sha_format(self):
        errs = self._errors("k", _entry(sha="sha256:ZZZZ"))
        self.assertTrue(any("sha256" in e for e in errs))

    def test_sha_wrong_length(self):
        errs = self._errors("k", _entry(sha="sha256:" + "a" * 63))
        self.assertTrue(any("sha256" in e for e in errs))

    def test_invalid_tag(self):
        errs = self._errors("k", _entry(sha=None, tag="tag with spaces"))
        self.assertTrue(any("tag" in e for e in errs))

    def test_neither_sha_nor_tag(self):
        errs = self._errors("k", _entry(sha=None))
        self.assertTrue(any("neither" in e or "one must be set" in e for e in errs))

    def test_sha_none_is_allowed_with_tag(self):
        entry = {
            "registry": "docker.io",
            "image": "ubuntu",
            "sha": None,
            "tag": "24.04",
        }
        self.assertEqual(self._errors("k", entry), [])

    def test_tag_none_is_allowed_with_sha(self):
        entry = _entry(tag=None)
        self.assertEqual(self._errors("k", entry), [])

    def test_sha_non_string(self):
        errs = self._errors("k", _entry(sha=12345))
        self.assertTrue(any("sha" in e for e in errs))

    def test_tag_non_string(self):
        errs = self._errors("k", _entry(sha=None, tag=99))
        self.assertTrue(any("tag" in e for e in errs))


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


class ResolveTest(unittest.TestCase):
    def test_sha_takes_priority_over_tag(self):
        entry = _entry(tag="latest")
        ref = resolve(entry, "k")
        self.assertIn(_SHA, ref)
        self.assertNotIn(":latest", ref)

    def test_sha_format(self):
        ref = resolve(_entry(), "k")
        self.assertEqual(ref, f"ghcr.io/rocm/myimage@{_SHA}")

    def test_tag_only(self):
        entry = {
            "registry": "docker.io",
            "image": "python",
            "sha": None,
            "tag": "3.12-slim",
        }
        self.assertEqual(resolve(entry, "k"), "docker.io/python:3.12-slim")

    def test_neither_raises(self):
        entry = _entry(sha=None)
        with self.assertRaises(ValueError):
            resolve(entry, "k")

    def test_invalid_entry_raises(self):
        with self.assertRaises(ValueError):
            resolve(_entry(registry="bad registry!!"), "k")

    def test_registry_trailing_slash_rejected_by_validator(self):
        # Trailing slashes are invalid per the registry regex; resolve() runs
        # validation first, so it raises rather than silently stripping.
        with self.assertRaises(ValueError):
            resolve(_entry(registry="ghcr.io/rocm/"), "k")


# ---------------------------------------------------------------------------
# get_image_ref (reads the real docker_images.json)
# ---------------------------------------------------------------------------


class GetImageRefTest(unittest.TestCase):
    def test_known_key_returns_string(self):
        ref = get_image_ref("therock_build_manylinux_x86_64")
        self.assertIsInstance(ref, str)
        self.assertTrue(
            ref.startswith("ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:")
        )

    def test_unknown_key_raises_key_error(self):
        with self.assertRaises(KeyError):
            get_image_ref("this_key_does_not_exist")

    def test_custom_json(self):
        data = {"img": {"registry": "r.io", "image": "foo", "tag": "v1"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp = Path(f.name)
        try:
            self.assertEqual(get_image_ref("img", path=tmp), "r.io/foo:v1")
        finally:
            tmp.unlink()


# ---------------------------------------------------------------------------
# load_images
# ---------------------------------------------------------------------------


class LoadImagesTest(unittest.TestCase):
    def test_comment_keys_stripped(self):
        data = {"_comment": "ignored", "real": _entry()}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp = Path(f.name)
        try:
            images = load_images(tmp)
            self.assertNotIn("_comment", images)
            self.assertIn("real", images)
        finally:
            tmp.unlink()


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------


def _run(argv: list[str]) -> tuple[int, str]:
    """Run main() capturing stdout; return (exit_code, stdout)."""
    import io

    buf = io.StringIO()
    with patch("sys.stdout", buf):
        code = resolve_docker_image.main(argv)
    return code, buf.getvalue()


class CLIGetImageTest(unittest.TestCase):
    def test_known_key_exits_zero(self):
        code, out = _run(["get-image", "python_slim"])
        self.assertEqual(code, 0)
        self.assertIn("python", out)

    def test_unknown_key_exits_nonzero(self):
        code, _ = _run(["get-image", "no_such_key"])
        self.assertNotEqual(code, 0)


class CLIListTest(unittest.TestCase):
    def test_list_shows_all_keys(self):
        code, out = _run(["list"])
        self.assertEqual(code, 0)
        self.assertIn("therock_build_manylinux_x86_64", out)
        self.assertIn("no_rocm_image_ubuntu24_04", out)


class CLIValidateTest(unittest.TestCase):
    def test_real_json_passes(self):
        code, _ = _run(["validate"])
        self.assertEqual(code, 0)

    def test_bad_json_fails(self):
        # Patch load_images so cmd_validate sees invalid data without touching
        # the real docker_images.json. _DOCKER_IMAGES_PATH can't be patched at
        # call time because default parameter values are bound at definition time.
        bad = {"bad": {"registry": "ok.io", "image": "img", "sha": "NOT_A_DIGEST"}}
        with patch.object(resolve_docker_image, "load_images", return_value=bad):
            code, _ = _run(["validate"])
        self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
