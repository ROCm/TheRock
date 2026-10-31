# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the actual workflow expression with a retained misrouted component.

The small expression adapter supports only the operators/functions this expression
uses. It evaluates the YAML expression, rather than a copied routing algorithm;
actionlint separately validates GitHub's syntax and allowed contexts.
"""

import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "build_tools/tests/fixtures/component_cpu_runner_routing.json"
TOKEN = re.compile(
    r"'(?:[^']|'')*'|!=|==|&&|\|\||[!(),]|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
)


def evaluate(expression: str, context: dict) -> str:
    expression = expression.strip().removeprefix("${{").removesuffix("}}")
    pieces = []
    end = 0
    for match in TOKEN.finditer(expression):
        if expression[end : match.start()].strip():
            raise ValueError("Unsupported expression token")
        token = match.group()
        if token.startswith("'"):
            pieces.append(repr(ast.literal_eval(token)))
        elif "." in token:
            pieces.append(f"lookup({token!r})")
        else:
            pieces.append(
                {"&&": "and", "||": "or", "!": "not", "true": "True"}.get(token, token)
            )
        end = match.end()
    if expression[end:].strip():
        raise ValueError("Unsupported expression suffix")

    def lookup(path: str):
        value = context
        for part in path.split("."):
            value = value.get(part, "") if isinstance(value, dict) else ""
        return value

    return eval(
        " ".join(pieces),
        {"__builtins__": {}},
        {
            "lookup": lookup,
            "contains": lambda value, part: str(part).lower() in str(value).lower(),
            "startsWith": lambda value, part: str(value)
            .lower()
            .startswith(str(part).lower()),
            "format": lambda value, *args: value.format(*args),
        },
    )


class ComponentCpuRunnerRoutingTest(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.workflow = yaml.load(
            (ROOT / ".github/workflows/test_artifacts.yml").read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
        self.expression = self.workflow["jobs"][self.fixture["workflow_job"]]["with"][
            "test_runs_on"
        ]
        self.context = copy.deepcopy(self.fixture["caller_context"])
        self.context["matrix"] = {
            "components": copy.deepcopy(self.fixture["component"])
        }

    def selected(self, context: dict | None = None) -> str:
        return evaluate(self.expression, context or self.context)

    def test_retained_inherited_dispatch_fails_before_and_routes_cpu_after(self):
        self.assertEqual(self.fixture["component"]["linux_cpu_runner"], True)
        self.assertEqual(
            evaluate(self.fixture["original_expression"], self.context),
            self.fixture["expected_before"],
        )
        self.assertEqual(self.selected(), self.fixture["expected_after"])

    def test_direct_test_artifacts_cpu_override_remains_intentional(self):
        self.context["github"][
            "workflow_ref"
        ] = "ROCm/TheRock/.github/workflows/test_artifacts.yml@refs/heads/main"
        self.assertEqual(self.selected(), self.fixture["expected_before"])
        self.context["inputs"]["test_runs_on"] = ""
        self.assertEqual(self.selected(), self.fixture["expected_after"])

    def test_branch_name_containing_workflow_path_is_not_direct_dispatch(self):
        self.context["github"][
            "workflow_ref"
        ] += "/.github/workflows/test_artifacts.yml@refs/heads/main"
        self.assertEqual(self.selected(), self.fixture["expected_after"])

    def test_cpu_non_dispatch_bot_and_empty_gpu_input_keep_component_runner(self):
        for event, actor, runner in [
            ("push", "sa-faizal", self.fixture["expected_before"]),
            ("pull_request", "sa-faizal", self.fixture["expected_before"]),
            (
                "workflow_dispatch",
                "github-actions[bot]",
                self.fixture["expected_before"],
            ),
            ("workflow_dispatch", "sa-faizal", ""),
        ]:
            with self.subTest(event=event, actor=actor, runner=runner):
                context = copy.deepcopy(self.context)
                context["github"].update(event_name=event, actor=actor)
                context["inputs"]["test_runs_on"] = runner
                self.assertEqual(self.selected(context), self.fixture["expected_after"])

    def test_gpu_manual_override_and_multi_gpu_order_unchanged(self):
        for cpu_marker in [False, None]:
            context = copy.deepcopy(self.context)
            component = context["matrix"]["components"]
            if cpu_marker is None:
                component.pop("linux_cpu_runner")
            else:
                component["linux_cpu_runner"] = cpu_marker
            component["multi_gpu_runner"] = "dedicated-multi-gpu"
            component["test_runner"] = "ordinary-single-gpu"
            with self.subTest(cpu_marker=cpu_marker):
                self.assertEqual(
                    self.selected(context), self.fixture["expected_before"]
                )
                context["inputs"]["test_runs_on"] = ""
                self.assertEqual(self.selected(context), "dedicated-multi-gpu")
                component["multi_gpu_runner"] = ""
                self.assertEqual(self.selected(context), "ordinary-single-gpu")

    def test_no_other_workflow_semantics_changed(self):
        self.workflow["jobs"][self.fixture["workflow_job"]]["with"].pop("test_runs_on")
        digest = hashlib.sha256(
            json.dumps(self.workflow, sort_keys=True).encode()
        ).hexdigest()
        self.assertEqual(
            digest, self.fixture["workflow_without_runner_expression_sha256"]
        )


if __name__ == "__main__":
    unittest.main()
