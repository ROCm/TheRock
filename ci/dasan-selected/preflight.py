#!/usr/bin/env python3
import argparse,json,pathlib,subprocess,sys,os,yaml
p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parent;out=pathlib.Path('evidence');out.mkdir(exist_ok=True)
entries=[v for v in json.loads((root/'matrix.json').read_text())['include'] if v['stage']==a.stage]
cmd=[sys.executable,str(root/'consume_dasan_artifacts.py'),'--manifest',str(root/'producer-manifest.json'),'--stage',a.stage,'--output-dir','build','--evidence-dir','evidence','--cache-dir','downloads']
if a.stage=='compiler-runtime':cmd+=['--include-validation-bundle']
status=subprocess.run(cmd).returncode
consume_status_path=out/'consume-status.json'
consume_status=json.loads(consume_status_path.read_text()) if consume_status_path.is_file() else {}
setup_error=consume_status.get('error') if status else None
setup_reason=('producer-payload-incomplete' if setup_error and
              ('Selected groups have no manifests:' in setup_error or
               'Selected artifact groups contain no payload:' in setup_error)
              else 'artifact-setup-failed')
available=[];unavailable=[]
for entry in entries:
    missing=[pat for pat in entry.get('required_paths',[]) if not any(pathlib.Path('build').glob(pat))]
    for rel,categories in entry.get('required_yaml_categories',{}).items():
        file=pathlib.Path('build')/rel
        data=yaml.safe_load(file.read_text()) if file.is_file() else {}
        for category in categories:
            if not isinstance(data,dict) or category not in data:missing.append(rel+':category='+category)
    for label in entry.get('required_ctest_labels',[]):
        roots=[(pathlib.Path('build')/v).parent for v in entry['required_paths'] if v.endswith('CTestTestfile.cmake')]
        if not any(label in f.read_text(errors='replace') for root in roots for f in root.rglob('*.cmake')):missing.append('CTest label='+label)
    if status or missing or entry.get('kind')=='availability-only':
        row={'id':entry['id'],'component':entry['job_name'],'stage':a.stage,
             'reason':setup_reason if status else 'no-executable-suite' if entry.get('kind')=='availability-only' else 'missing-test-payload',
             'missing':[] if status else missing,'producer_run_id':entry.get('producer_run_id')}
        if status:
            row['setup_error']=setup_error or 'artifact consumer exited without an error record'
            row['unchecked_required_paths']=missing
        unavailable.append(row)
    else:available.append(entry)
report={'stage':a.stage,'artifact_setup_returncode':status,'artifact_setup_error':setup_error,
        'available':available,'unavailable':unavailable}
(out/('preflight-'+a.stage+'.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'stage':a.stage,'available_jobs':len(available),'unavailable_jobs':len(unavailable),'setup_returncode':status}))
summary=os.environ.get('GITHUB_STEP_SUMMARY')
if summary:
    with open(summary,'a') as f:
        f.write('### '+a.stage+' payload preflight\n\nAvailable jobs: '+str(len(available))+'; unavailable: '+str(len(unavailable))+'\n\n')
        for row in unavailable:
            detail=row.get('setup_error') or ', '.join(row['missing'])
            f.write('- '+row['id']+': '+row['reason']+(f' ({detail})' if detail else '')+'\n')
# Missing suites are reported to the final coverage gate; keep available stages runnable.
sys.exit(status)
