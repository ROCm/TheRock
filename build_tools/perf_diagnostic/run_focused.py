#!/usr/bin/env python3
"""Focused SDK numerical checks. --plan is local/read-only; execution is CI-only."""
from __future__ import annotations
import argparse, datetime, hashlib, json, math, os, re, subprocess, sys, time
from pathlib import Path
import xml.etree.ElementTree as ET
BASE='5b7cb88373362ab80dc0f19605915e534f4158aa'
LIBS='f4e24d58b3f313f1a31068def1e35376e43c2f49'
PATCH='b74891bd117138d0e922407bf4cc24c1ee0482f08c685de0c1b36b662cb3aab9'
BOUNDARIES=(1,16,17,32,33,48,49,64,65)
FILTER='checkin_lapack/GETF2*/*:checkin_lapack/GETRF*/*:checkin_lapack/GETRI*/*'
EPSILON={'s':2**-23,'c':2**-23,'d':2**-52,'z':2**-52}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def run_logged(argv,log,timeout,env=None):
    """Preserve partial diagnostics even when a numerical subprocess times out."""
    try:
        result=subprocess.run(argv,text=True,capture_output=True,timeout=timeout,env=env)
    except subprocess.TimeoutExpired as error:
        decode=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else (x or '')
        log.write_text(decode(error.stdout)+'\nSTDERR:\n'+decode(error.stderr)+'\nTIMEOUT\n')
        raise
    log.write_text(result.stdout+'\nSTDERR:\n'+result.stderr)
    return result

def small_m(precision,pivot,batch,n):
    """One source-proven spker1 shape at inca1/n1..64; None is no pruning proof."""
    if precision in 'sd':
        if batch:return 1
        pairs=((20,1),(28,6),(36,8),(44,10),(52,16),(60,44)) if pivot else ((20,1),(28,6),(36,6),(44,10),(52,24))
        for hi,m in pairs:
            if n<=hi:return m
    elif batch:
        if pivot:
            if n<=36:return 1
            if n<=60:return 16
        elif n<=36:return 1
    elif pivot:
        if 1<n<=12:return 1
        if 12<n<=20:return 6
        if 20<n<=28:return 10
    else:
        for hi,m in ((12,32),(20,129),(28,289)):
            if n<=hi:return m
    return None

def cases():
    result={}
    def add(function,precision,m,n,coverage):
        key=(function,precision,m,n)
        if key in result:result[key]['coverage'].append(coverage);return
        argv=['-f',function,'-r',precision,'-n',str(n),'--lda',str(m),'-v','1','--iters','0']
        if function.startswith('getf2'):argv+=['-m',str(m)]
        if 'batched' in function:argv+=['--batch_count','3']
        result[key]={'function':function,'precision':precision,'m':m,'n':n,'coverage':[coverage],
            'argv':argv,'error_limit':min(m,n)*EPSILON[precision]}
    for precision in 'sdcz':
        for pivot in (True,False):
            for layout in ('','_batched','_strided_batched'):
                function='getf2'+('' if pivot else '_npvt')+layout
                for n in range(1,65):
                    m=small_m(precision,pivot,bool(layout),n)
                    add(function,precision,m or n,n,'source-selected-small-helper' if m else 'public API; changed-helper selection not established')
                for n in BOUNDARIES:add(function,precision,n,n,'square shard/fallback boundary')
    for prefix in ('getri','getri_npvt','getri_outofplace','getri_npvt_outofplace'):
        for layout in ('','_batched','_strided_batched'):
            for n in (range(1,65) if prefix=='getri' else BOUNDARIES):
                add(prefix+layout,'z',n,n,'getri_z public API/pointer-form coverage')
    return list(result.values())

def parse_error(stdout,limit):
    rows=re.findall(r'^=+\s*\nResults:\s*\n=+\s*\n([^\n]+)\n([^\n]+)',stdout,re.M)
    if len(rows)!=1:raise ValueError('Expected one Results table')
    names,values=(x.split() for x in rows[0])
    if len(names)!=len(values) or names.count('error')!=1:raise ValueError('Missing/ambiguous error column')
    error=float(values[names.index('error')])
    if not math.isfinite(error) or error<0 or error>limit:raise ValueError(f'Error {error} fails limit {limit}')
    return error

def check_gtest(path):
    tests=list(ET.parse(path).getroot().iter('testcase'))
    if not tests:raise ValueError('GTest ran zero cases')
    for family in ('GETF2','GETRF','GETRI'):
        subset=[t for t in tests if family in t.get('classname','')]
        for precision in ('float','double','float_complex','double_complex'):
            if not any(re.search(r'__'+precision+r'/',t.get('name','')) for t in subset):
                raise ValueError(f'GTest coverage missing: {family}/{precision}')
    for t in tests:
        if any(t.find(x) is not None for x in ('failure','error','skipped')):raise ValueError('Failed/error/skipped GTest')
        if t.get('status','run')!='run' or t.get('result','completed')!='completed':raise ValueError('Unexecuted GTest')
    return len(tests)

def guard(args):
    if sys.platform!='linux' or os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('GITHUB_REPOSITORY')!='ROCm/TheRock':
        raise RuntimeError('Execution restricted to Linux ROCm/TheRock Actions')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source_root,text=True).strip()
    if sha!=BASE:raise RuntimeError('Source differs from frozen HEAD')
    sdk=args.sdk_root.resolve();manifest=json.loads(args.sdk_manifest.read_text())
    expected={'source_sha':BASE,'rocm_libraries_gitlink':LIBS,'effective_stock_libraries_head':LIBS,
        'mode':args.mode,'joint_patch_sha256':PATCH if args.mode=='joint' else None,
        'producer_run_id':args.producer_run_id,'instrumentation_sha':args.expected_instrumentation_sha}
    for key,value in expected.items():
        if str(manifest.get(key))!=str(value):raise RuntimeError('SDK provenance mismatch: '+key)
    required={};verified=[]
    for row in manifest['files']:
        name=row['basename']
        if Path(name).name!=name or any(x in name for x in '*?[]'):
            raise RuntimeError('Invalid producer basename')
        matches=[p.resolve() for p in sdk.rglob(name) if p.is_file()
            and p.resolve().is_relative_to(sdk) and digest(p)==row['sha256']]
        matches=sorted(set(matches))
        if len(matches)!=1:raise RuntimeError('SDK checksum/path ambiguity: '+name)
        verified.append((row,matches[0]))
    for role in ('rocsolver_bench','rocsolver_test','librocsolver'):
        rows=[(r,p) for r,p in verified if r['role']==role]
        if len(rows)!=1:raise RuntimeError('Missing/ambiguous producer role: '+role)
        required[role]=rows[0][1]
    loaded=[]
    for role in ('rocsolver_bench','rocsolver_test'):
        text=subprocess.check_output(['ldd',str(required[role])],text=True)
        matches=re.findall(r'^\s*librocsolver\.so[^\s]*\s+=>\s+(\S+)',text,re.M)
        if len(matches)!=1 or Path(matches[0]).resolve()!=required['librocsolver']:raise RuntimeError('Actual loaded solver mismatch')
        loaded.append({'role':role,'ldd':text})
    if os.environ.get('HSA_XNACK')!='1':raise RuntimeError('Expected inherited gfx950:xnack+ ASan environment')
    info_candidates=sorted(set(p.resolve() for p in sdk.rglob('rocminfo') if p.is_file()))
    if len(info_candidates)!=1:raise RuntimeError('Missing/ambiguous SDK rocminfo')
    info=subprocess.check_output([str(info_candidates[0])],text=True,timeout=30)
    if not re.search(r'^\s*Name:\s*gfx950\s*$',info,re.M):raise RuntimeError('No gfx950 agent')
    loaded.append({'producer_files_verified':[{'role':r['role'],'path':str(p),'sha256':r['sha256']} for r,p in verified]})
    return required,manifest,loaded,info

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--sdk-root',type=Path);parser.add_argument('--source-root',type=Path,default=Path.cwd())
    parser.add_argument('--sdk-manifest',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--mode',choices=('baseline','joint'),default=os.environ.get('PERF_MODE'))
    parser.add_argument('--producer-run-id',default=os.environ.get('GITHUB_RUN_ID'))
    parser.add_argument('--expected-instrumentation-sha',default=os.environ.get('GITHUB_SHA'))
    parser.add_argument('--budget-s',type=int,default=3600)
    args=parser.parse_args();plan=cases()
    if args.plan:print(json.dumps({'case_count':len(plan),'gtest_filter':FILTER,'cases':plan},indent=2));return 0
    if any(getattr(args,k) is None for k in ('sdk_root','sdk_manifest','mode','producer_run_id','expected_instrumentation_sha','output')):
        parser.error('Execution requires SDK/provenance/mode/output identity arguments')
    args.output.mkdir(parents=True,exist_ok=False)
    summary={'started_utc':utc(),'source_sha':BASE,'mode':args.mode,'planned_bench_cases':len(plan),
        'passed':False,'completed_bench_cases':0,'cases':[],'limitations':[
        'Public API cases do not prove execution of every compiled internal helper/dimension.',
        'Stock ASan environment inherited; disabled ODR checks are not an ODR proof.',
        'Printed numerical error is gated; exit zero alone never counts as a numerical pass.']}
    start=time.monotonic()
    try:
        files,manifest,loaded,info=guard(args);summary['sdk_manifest']=manifest;summary['loaded_libraries']=loaded
        summary['environment']={k:os.environ.get(k) for k in ('ASAN_OPTIONS','HSA_XNACK','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','KUBE_CPU_REQUEST','LD_LIBRARY_PATH','LD_PRELOAD')}
        (args.output/'rocminfo.log').write_text(info)
        for index,case in enumerate(plan):
            remaining=args.budget_s-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError('Budget exhausted; numerical coverage incomplete')
            argv=[str(files['rocsolver_bench']),*case['argv']]
            log=args.output/f'bench-{index:04d}.log'
            proc=run_logged(argv,log,min(60,remaining))
            row={**case,'argv':argv,'exit_code':proc.returncode,'log':log.name,'utc':utc()};summary['cases'].append(row)
            if proc.returncode:raise RuntimeError(f'{log.name} exit{proc.returncode}')
            row['measured_error']=parse_error(proc.stdout,case['error_limit']);summary['completed_bench_cases']+=1
            if (index+1)%50==0:print(f'{utc()}: numerical cases {index+1}/{len(plan)} passed',flush=True)
        remaining=args.budget_s-(time.monotonic()-start)
        if remaining<=0:raise TimeoutError('No budget remains for required focused GTests')
        xml=args.output/'gtest.xml'
        argv=[str(files['rocsolver_test']),'--gtest_filter='+FILTER,'--gtest_output=xml:'+str(xml.resolve())]
        # This extra focused invocation covers all selected tests in one job;
        # inherited stock shard variables must not silently discard two thirds.
        test_env=os.environ.copy()
        for key in ('GTEST_TOTAL_SHARDS','GTEST_SHARD_INDEX','GTEST_SHARD_STATUS_FILE'):
            test_env.pop(key,None)
        proc=run_logged(argv,args.output/'gtest.log',remaining,env=test_env)
        summary['gtest_argv']=argv;summary['gtest_exit_code']=proc.returncode
        if proc.returncode:raise RuntimeError('Focused GTest failed')
        summary['gtest_completed_cases']=check_gtest(xml);summary['passed']=True
    except Exception as error:summary['failure']=repr(error)
    finally:
        summary['end_utc']=utc();summary['elapsed_s']=time.monotonic()-start
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('cases','sdk_manifest','loaded_libraries')},indent=2))
    return 0 if summary['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
