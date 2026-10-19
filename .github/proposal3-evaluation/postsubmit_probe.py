"""Exercise real pinned CI selectors without building ROCm or changing policy."""
from __future__ import annotations
import argparse
import copy
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--sha', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve() / 'build_tools/github_actions'))
import configure_multi_arch_ci as ci

os.environ.pop('CI_CONFIG_PATH', None)
def inputs(event: str, linux: list[str], windows: list[str]) -> ci.CIInputs:
    return ci.CIInputs(run_id='configuration-proof', event_name=event,
        commit_ref='develop', base_ref=None, build_variant='release',
        linux_amdgpu_families=[x.lower() for x in linux],
        windows_amdgpu_families=[x.lower() for x in windows],
        external_repo='{"repository":"ROCm/rocm-libraries"}')

# These inputs are the current pinned consumer workflow's literals, not guesses
# from a family default. Source snapshot is recorded in the evaluation manifest.
push = inputs('push', ['gfx94X', 'gfx950', 'gfx125X'], ['gfx110X'])
nightly = inputs('schedule', ['gfx94X', 'gfx950', 'gfx125X'], ['gfx1151'])
push_targets = ci.select_targets(push)
nightly_targets = ci.select_targets(nightly)
matrix = ci.get_all_families_for_trigger_types([])
normal = matrix['gfx94x']['linux']
removed = copy.deepcopy(normal)
removed['tests_on_trigger'] = [x for x in normal['tests_on_trigger'] if x != 'postsubmit']
context = ci.GitContext.empty()
baseline_gate = ci._should_run_tests_for_family(normal, push, context)
removed_gate = ci._should_run_tests_for_family(removed, push, context)
nightly_gate = ci._should_run_tests_for_family(removed, nightly, context)
assert baseline_gate[0], baseline_gate
assert not removed_gate[0], removed_gate
assert nightly_gate[0], nightly_gate
windows_gap = sorted(set(push_targets.windows_families) - set(nightly_targets.windows_families))
assert windows_gap == ['gfx110x'], windows_gap
result = {'source_sha': args.sha,
    'consumer_workflow_sha': '8535f49bfd5c7ab8fd23ca94169a48e2c5f6a819',
    'push_inputs': asdict(push), 'nightly_inputs': asdict(nightly),
    'push_targets': asdict(push_targets), 'nightly_targets': asdict(nightly_targets),
    'push_windows_families_absent_from_nightly': windows_gap,
    'retained_gfx94x_push_test_gate': {'original': baseline_gate,
        'after_removing_postsubmit_test_trigger': removed_gate,
        'nightly_after_removal': nightly_gate},
    'scope': 'Real selector and test policy execution; no ROCm build, GPU test, runner overlay, schedule/publish or cost qualification'}
args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
