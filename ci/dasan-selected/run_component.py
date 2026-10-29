#!/usr/bin/env python3
"""Execute pinned CI commands against the selected DASan artifact installation."""
import argparse, datetime, json, os, pathlib, re, signal, subprocess, sys, time

def main():
    p=argparse.ArgumentParser();p.add_argument('--component',required=True);p.add_argument('--matrix',required=True);p.add_argument('--sdk',default='build');p.add_argument('--evidence',default='evidence');a=p.parse_args()
    matrix=json.loads(pathlib.Path(a.matrix).read_text());entries=matrix.get('include',matrix) if isinstance(matrix,dict) else matrix
    matches=[v for v in entries if v.get('id',v.get('job_name'))==a.component]
    if len(matches)!=1: raise SystemExit('Expected exactly one matrix entry: '+a.component)
    c=matches[0];sdk=pathlib.Path(a.sdk).resolve();out=pathlib.Path(a.evidence).resolve();out.mkdir(exist_ok=True,parents=True)
    result={'component':c,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'preflight','sdk':str(sdk)}
    def save(): (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    missing=[pat for pat in c.get('required_paths',[]) if not any(sdk.glob(pat))]
    if missing:
        result.update(status='missing-test-payload',missing=missing);save();print('Missing test payload in selected producer artifacts:',missing,flush=True);return 2
    env=os.environ.copy();env.update({'ROCM_PATH':str(sdk),'HIP_PATH':str(sdk),'OUTPUT_ARTIFACTS_DIR':str(sdk),'THEROCK_BIN_DIR':str(sdk/'bin'),'AMDGPU_FAMILIES':'gfx950-dcgpu','AMDGPU_TARGETS':'gfx950','BUILD_VARIANT':'release','SANITIZER_MODE':'dasan','TEST_TYPE':'full','TEST_COMPONENT':c['job_name'],'SHARD_INDEX':str(c.get('shard',1)),'TOTAL_SHARDS':str(c.get('total_shards',1)),'GPU_ENABLE_PAL':'0','ROCM_KPACK_DEBUG':str(c.get('rocm_kpack_debug','1')),'PYTHONUNBUFFERED':'1'})
    env['PATH']=str(sdk/'bin')+':'+str(sdk/'lib/llvm/bin')+':'+env['PATH']
    runtime=subprocess.check_output([str(sdk/'lib/llvm/bin/clang'),'--print-file-name=libclang_rt.dasan.so'],text=True).strip()
    if not pathlib.Path(runtime).is_file(): raise RuntimeError('Selected compiler has no shared DASan runtime: '+runtime)
    if c.get('kind')=='compiler-dasan-lit': env.pop('LD_PRELOAD',None)
    else: env['LD_PRELOAD']=runtime
    env['LD_LIBRARY_PATH']=':'.join([str(pathlib.Path(runtime).parent),str(sdk/'lib'),str(sdk/'lib/llvm/lib'),str(sdk/'lib/rocm_sysdeps/lib')])
    if c.get('external_compile'):
        env.update(CC=str(sdk/'lib/llvm/bin/clang'),CXX=str(sdk/'lib/llvm/bin/clang++'),CFLAGS='-fsanitize=daddress -shared-libsan',CXXFLAGS='-fsanitize=daddress -shared-libsan',HIPFLAGS='-fsanitize=daddress -shared-libsan',LDFLAGS='-fsanitize=daddress -shared-libsan')
    cpu=env.get('KUBE_CPU_REQUEST','').split('.')[0]
    if cpu.isdigit(): env.update(OMP_NUM_THREADS=cpu,OPENBLAS_NUM_THREADS=cpu)
    result['runtime']=runtime;result['compile_flags']=env.get('CXXFLAGS');save()
    command=c['test_script'].replace(' --enable-cdash','')
    if c.get('supplemental_test_runner'):
        command=command.replace('build_tools/github_actions/test_executable_scripts/test_runner.py','build_tools/github_actions/test_executable_scripts/test_runner_fft_multi_gpu.py')
    result.update(status='running',command=command);save();start=time.monotonic()
    script='set -euo pipefail\n'+command+'\n'
    (out/'command.sh').write_text(script)
    with (out/'console.log').open('w') as log:
        proc=subprocess.Popen(['bash','-c',script],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try: code=proc.wait(timeout=int(c.get('timeout_minutes',180))*60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            result['status']='timeout';code=124
    text=(out/'console.log').read_text(errors='replace');print(text[-150000:],flush=True)
    result.update(returncode=code,seconds=round(time.monotonic()-start,2))
    if result['status']!='timeout':result['status']='passed' if code==0 else 'failed'
    if re.search(r'No tests were found|Empty test suite|\b0 tests passed\b|collected 0 items',text,re.I):result['status']='empty-test-suite';code=2
    if 'ERROR: DeviceAddressSanitizer:' in text:result['status']='failed';result['dasan_report']=True;code=1
    save()
    summary=os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary,'a') as f:f.write('### '+c['job_name']+'\n\nStatus: **'+result['status']+'**\n\nProducer stage: '+c['stage']+'; shard '+str(c.get('shard',1))+'/'+str(c.get('total_shards',1))+'\n')
    return code
if __name__=='__main__':sys.exit(main())
