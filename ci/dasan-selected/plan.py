#!/usr/bin/env python3
import json,pathlib,os
root=pathlib.Path(__file__).resolve().parent;expected=json.loads((root/'producer-manifest.json').read_text())['producers'];available=[];unavailable=[];reports=[]
for producer in expected:
    stage=producer['stage'];found=list(pathlib.Path('preflight-results').rglob('preflight-'+stage+'.json'))
    if len(found)!=1:
        unavailable.append({'stage':stage,'reason':'missing-preflight-record'});continue
    report=json.loads(found[0].read_text());reports.append(report);available.extend(report['available']);unavailable.extend(report['unavailable'])
result={'include':available};pathlib.Path('dispatch-matrix.json').write_text(json.dumps(result,indent=2)+'\n');pathlib.Path('coverage-results.json').write_text(json.dumps({'available_jobs':len(available),'unavailable':unavailable,'producer_stages':len(reports)},indent=2)+'\n')
with open(os.environ['GITHUB_OUTPUT'],'a') as f:
    f.write('matrix='+json.dumps(result,separators=(',',':'))+'\n');f.write('count='+str(len(available))+'\n');f.write('missing='+str(len(unavailable))+'\n')
with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as f:
    f.write('## DASan gfx950 selected-artifact coverage\n\nRunnable jobs: **'+str(len(available))+'**. Unavailable jobs: **'+str(len(unavailable))+'**.\n\n')
    for v in unavailable:f.write('- '+v.get('id',v['stage'])+': '+v['reason']+'\n')
