#!/usr/bin/env python3
import json,pathlib,subprocess,sys
root=pathlib.Path(__file__).resolve().parent;entry=next(v for v in json.loads((root/'matrix.json').read_text())['include'] if v['id']==sys.argv[1])
cmd=[sys.executable,str(root/'consume_dasan_artifacts.py'),'--manifest',str(root/'producer-manifest.json'),'--stage',entry['stage'],'--output-dir','build','--evidence-dir','evidence','--cache-dir','downloads']
if entry.get('include_validation_bundle'):cmd+=['--include-validation-bundle']
for group in entry.get('dependency_groups',[]):cmd+=['--dependency-group',group]
subprocess.run(cmd,check=True)
if entry.get('supplemental_test_runner'):
    import shutil
    shutil.copy2(root/entry['supplemental_test_runner'],'build_tools/github_actions/test_executable_scripts/test_runner_fft_multi_gpu.py')
