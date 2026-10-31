"""Structure-only qualification of unchanged artifacts from two original producers."""
import datetime,hashlib,io,json,os,subprocess,sys,urllib.request,zipfile
from pathlib import Path
BASE='5b7cb88373362ab80dc0f19605915e534f4158aa';PRODUCER_SHA='17bf9b333993b6b12dd9ae867a9c16813cdfca0f'
OUT=Path('.perf-full');HERE=Path(__file__).resolve().parent
def save(name,value):OUT.mkdir(exist_ok=True);(OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
def fetch(path):
    class SafeRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,req,fp,code,msg,headers,newurl):
            value=super().redirect_request(req,fp,code,msg,headers,newurl)
            if value is not None:value.remove_header('Authorization')
            return value
    request=urllib.request.Request('https://api.github.com/repos/ROCm/TheRock/'+path,headers={'Authorization':'Bearer '+os.environ['GITHUB_TOKEN'],'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    with urllib.request.build_opener(SafeRedirect()).open(request,timeout=120) as response:raw=response.read(16*1024*1024+1)
    if len(raw)>16*1024*1024:raise RuntimeError('Unexpected metadata size')
    return raw
def api(path):return json.loads(fetch(path))
def pages(path,key):
    rows=[]
    for page in range(1,101):
        value=api(path+f'?per_page=100&page={page}')[key];rows+=value
        if len(value)<100:return rows
    raise RuntimeError('Pagination exceeded bound')
def main():
    if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('GITHUB_REPOSITORY')!='ROCm/TheRock':raise RuntimeError('Only TheRock Actions permitted')
    actual=subprocess.check_output(['git','-c',f'safe.directory={Path.cwd()}','rev-parse','HEAD'],text=True).strip()
    if actual!=BASE:raise RuntimeError('Frozen BASE and real Git checkout required')
    mode=os.environ['PERF_MODE'];profile=json.loads((HERE/'producer-profiles.json').read_text(encoding='utf-8'))[mode];run=os.environ['ARTIFACT_RUN_ID']
    if run!=str(profile['run_id']):raise RuntimeError('Producer ID/mode differs from allowed original full run')
    original=api('actions/runs/'+run)
    if original['head_sha']!=PRODUCER_SHA or original['head_branch']!='users/safaizal/full-ci-v2-01a19119' or original['event']!='workflow_dispatch' or original['display_title']!=f'Full HEAD {mode} validation (01a19119)':raise RuntimeError('Original producer identity mismatch')
    jobs=list({j['id']:j for j in pages(f'actions/runs/{run}/jobs','jobs')}.values());stages=[j for j in jobs if ' / Build Multi-Arch Stages / ' in j['name'] and ' / Stage - ' in j['name']]
    if not stages or any(j['conclusion']!='success' for j in stages):raise RuntimeError('Original product stages incomplete; no artifact rebuilding permitted')
    validators=[j for j in jobs if j['name'].endswith(' / Validate artifact structure')]
    if len(validators)!=1:raise RuntimeError('Expected one original structure validator')
    validator=validators[0];steps={s['name']:s['conclusion'] for s in validator['steps']}
    if validator['conclusion']!='failure' or steps.get('Verify frozen source and record full-run context')!='failure' or steps.get('Fetch artifact archives (no extract)')!='skipped' or steps.get('Validate artifact structure')!='skipped':raise RuntimeError('Restricted to pre-check instrumentation failure, not failed artifact assertions')
    if validator['labels']!=['aws-linux-scale-rocm-large']:raise RuntimeError('Original validator runner contract changed')
    artifacts=pages(f'actions/runs/{run}/artifacts','artifacts');chosen=[row for row in artifacts if row['name']==profile['setup_artifact'] and not row['expired']]
    if len(chosen)!=1:raise RuntimeError('Original setup artifact missing/ambiguous')
    archive=fetch(f"actions/artifacts/{chosen[0]['id']}/zip")
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        files=[row for row in zipped.infolist() if row.filename=='build-matrix.json']
        if len(files)!=1 or files[0].file_size>1024*1024:raise RuntimeError('Unexpected original matrix member')
        raw=zipped.read(files[0])
    if hashlib.sha256(raw).hexdigest()!=profile['setup_member_sha256']:raise RuntimeError('Original matrix bytes differ')
    config=json.loads(raw)['config']
    if config['dist_amdgpu_families']!='gfx950-dcgpu' or config['build_variant_label']!='asan-debug' or config['build_runs_on_small']!='aws-linux-scale-rocm-large' or config['skip_stages'] or config['prebuilt_stages']:raise RuntimeError('Original artifact/runner/configuration differs')
    sys.path.insert(0,str(Path('build_tools').resolve()));from _therock_utils.workflow_outputs import WorkflowOutputRoot
    route=WorkflowOutputRoot.from_workflow_run(run_id=run,platform='linux',github_repository='ROCm/TheRock',workflow_run=original,release_type='ci')
    if route.bucket!='therock-ci-artifacts' or route.prefix!=run+'-linux':raise RuntimeError('Original CI artifact read route differs')
    failures=[{'id':j['id'],'name':j['name'],'conclusion':j['conclusion'],'url':f"https://github.com/ROCm/TheRock/actions/runs/{run}/job/{j['id']}"} for j in jobs if j['conclusion'] in ('failure','timed_out','cancelled','action_required')]
    save('original-producer-qualification.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha':BASE,'mode':mode,'producer_run_id':run,'producer_instrumentation_sha':PRODUCER_SHA,
        'consumer_run_id':os.environ['GITHUB_RUN_ID'],'consumer_instrumentation_sha':os.environ['GITHUB_SHA'],'original_validator_job_id':validator['id'],'original_validator_failed_before_artifact_fetch_and_assertions':True,
        'producer_status':original['status'],'producer_conclusion':original['conclusion'],'original_failures':failures,'product_stage_job_ids':[j['id'] for j in stages],
        'setup_evidence':{'artifact_id':chosen[0]['id'],'artifact_name':chosen[0]['name'],'matrix_sha256':hashlib.sha256(raw).hexdigest()},'artifact_read_route':{'bucket':route.bucket,'prefix':route.prefix},
        'focused_numerics':'OFF: no GPU job exists','eligible_for_optimization_acceptance':False,
        'limits':'Only unexecuted artifact assertions are repaired. Original numerical/native outcomes stay unchanged; successful structure checks cannot accept an optimization or bypass real stock failures.'})
    save('original-job-status.json',{'jobs':[{k:j[k] for k in ('id','name','status','conclusion','labels','runner_name','steps')} for j in jobs]})
    with open(os.environ['GITHUB_STEP_SUMMARY'],'a',encoding='utf-8') as stream:stream.write(f"Structure-only repair reads producer {run}, mode {mode}. Stock failures remain recorded; no GPU/numerical rerun or optimization acceptance.\n")
if __name__=='__main__':main()
