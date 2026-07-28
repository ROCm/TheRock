# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(
    0,
    os.fspath(Path(__file__).parent.parent),
)

import super_repo_signal

_SOURCE_SHA = "a" * 40
_THEROCK_SHA = "b" * 40


def _environment() -> dict[str, str]:
    return {
        "EXTERNAL_REPO_CONFIG_JSON": json.dumps(
            {
                "repository": "ROCm/rocm-libraries",
                "ref": "refs/pull/123/head",
                "checkout_path": "external-rocm-libraries",
                "source_package": "ROCM_LIBRARIES",
                "fetch_sources_args": ("--skip-submodules rocm-libraries"),
            }
        ),
        "SUPER_REPO_SOURCE_SHA": _SOURCE_SHA,
        "THEROCK_REPOSITORY": "ROCm/TheRock",
        "THEROCK_REQUESTED_REF": ("users/example/stage-reuse"),
        "THEROCK_RESOLVED_SHA": _THEROCK_SHA,
        "CALLING_WORKFLOW_REF": (
            "ROCm/rocm-libraries/"
            ".github/workflows/therock-ci.yml"
            "@refs/pull/123/merge"
        ),
        "REUSABLE_WORKFLOW_PATH": (".github/workflows/setup_multi_arch.yml"),
    }


class SuperRepoSignalTest(unittest.TestCase):
    def test_load_super_repo_signal(self):
        signal = super_repo_signal.load_super_repo_signal(_environment())

        self.assertEqual(
            signal.super_repo_repository,
            "ROCm/rocm-libraries",
        )
        self.assertEqual(
            signal.requested_source_ref,
            "refs/pull/123/head",
        )
        self.assertEqual(
            signal.super_repo_source_sha,
            _SOURCE_SHA,
        )
        self.assertEqual(
            signal.therock_repository,
            "ROCm/TheRock",
        )
        self.assertEqual(
            signal.requested_therock_ref,
            "users/example/stage-reuse",
        )
        self.assertEqual(
            signal.resolved_therock_sha,
            _THEROCK_SHA,
        )
        self.assertEqual(
            signal.overlay_path,
            "external-rocm-libraries",
        )
        self.assertEqual(
            signal.fetch_sources_args,
            "--skip-submodules rocm-libraries",
        )
        self.assertEqual(
            signal.reusable_workflow_path,
            ".github/workflows/setup_multi_arch.yml",
        )

    def test_render_super_repo_signal(self):
        signal = super_repo_signal.load_super_repo_signal(_environment())

        summary = super_repo_signal.render_super_repo_signal(signal)

        self.assertIn(
            "### Super-repo source/ref signal",
            summary,
        )
        self.assertIn(
            "<code>ROCm/rocm-libraries</code>",
            summary,
        )
        self.assertIn(
            f"<code>{_SOURCE_SHA}</code>",
            summary,
        )
        self.assertIn(
            f"<code>{_THEROCK_SHA}</code>",
            summary,
        )
        self.assertIn(
            "<code>external-rocm-libraries</code>",
            summary,
        )
        self.assertIn(
            "<code>--skip-submodules " "rocm-libraries</code>",
            summary,
        )
        self.assertIn(
            "Source fetch arguments",
            summary,
        )

    def test_render_escapes_markdown_table_values(self):
        env = _environment()
        config = json.loads(env["EXTERNAL_REPO_CONFIG_JSON"])
        config["ref"] = "branch|value\nsecond-line"
        env["EXTERNAL_REPO_CONFIG_JSON"] = json.dumps(config)

        signal = super_repo_signal.load_super_repo_signal(env)
        summary = super_repo_signal.render_super_repo_signal(signal)

        self.assertIn(
            "<code>branch&#124;value second-line</code>",
            summary,
        )

    def test_missing_optional_value_is_reported(self):
        env = _environment()
        env["THEROCK_REQUESTED_REF"] = ""

        signal = super_repo_signal.load_super_repo_signal(env)
        summary = super_repo_signal.render_super_repo_signal(signal)

        self.assertIn(
            "| Requested TheRock ref | " "_not provided_ |",
            summary,
        )

    def test_main_uses_injected_summary_writer(self):
        summaries: list[str] = []

        result = super_repo_signal.main(
            _environment(),
            append_summary=summaries.append,
        )

        self.assertEqual(result, 0)
        self.assertEqual(len(summaries), 1)
        self.assertIn(
            "Super-repo source/ref signal",
            summaries[0],
        )

    def test_invalid_external_repo_config_raises(self):
        cases = [
            (
                "{bad json",
                "invalid JSON",
            ),
            (
                '["not", "an", "object"]',
                "JSON object",
            ),
        ]

        for config_json, expected_error in cases:
            with self.subTest(config_json=config_json):
                env = _environment()
                env["EXTERNAL_REPO_CONFIG_JSON"] = config_json

                with self.assertRaisesRegex(
                    ValueError,
                    expected_error,
                ):
                    super_repo_signal.load_super_repo_signal(env)


if __name__ == "__main__":
    unittest.main()
