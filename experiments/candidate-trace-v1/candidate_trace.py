"""Candidate-only source/command qualification and six targeted trace replays."""
import argparse,hashlib,json,os,re,shlex,shutil,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;RECORDS=HERE/'candidate-records';OUT=Path('perf-records')
BASE='5b7cb88373362ab80dc0f19605915e534f4158aa';LIBS='f4e24d58b3f313f1a31068def1e35376e43c2f49'
PATCH='b74891bd117138d0e922407bf4cc24c1ee0482f08c685de0c1b36b662cb3aab9'
NAMES=('getri_z','getri_z_batched','getf2_c_3');UNSET=('ROCM_PATH','ROCM_DIR','HIP_PATH','HIP_DIR')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,data):OUT.mkdir(exist_ok=True);(OUT/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
def guard():
    if sys.platform!='linux' or os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('GITHUB_REPOSITORY')!='ROCm/TheRock':raise RuntimeError('Only Linux TheRock Actions permitted')
    if git('rev-parse','HEAD')!=BASE or git('-C','rocm-libraries','rev-parse','HEAD')!=LIBS:raise RuntimeError('Source pin mismatch')
    if digest(HERE/'rocsolver-specialized-shards.patch')!=PATCH:raise RuntimeError('Frozen joint patch changed')
def apply():
    guard();identity=read(HERE/'candidate-source-identity.json')
    if git('-C','rocm-libraries','diff','--name-only') or git('-C','rocm-libraries','ls-files','--others','--exclude-standard'):raise RuntimeError('Pre-candidate libraries source is not clean')
    for row in identity['files']:
        if row['base_sha256'] is not None and digest(Path('rocm-libraries')/row['path'])!=row['base_sha256']:raise RuntimeError('Frozen base source differs')
    subprocess.run(['git','-C','rocm-libraries','apply','--check',str(HERE/'rocsolver-specialized-shards.patch')],check=True)
    subprocess.run(['git','-C','rocm-libraries','apply',str(HERE/'rocsolver-specialized-shards.patch')],check=True)
def normalized(argv):
    result=[];index=0
    while index<len(argv):
        value=argv[index]
        if value in ('-o','-MF','-MT','-MQ'):index+=2;continue
        if value in ('-MD','-MMD','-MP'):index+=1;continue
        result.append(value);index+=1
    return result
def verify():
    guard();identity=read(HERE/'candidate-source-identity.json')
    tracked={r['path'] for r in identity['files'] if r['base_sha256'] is not None};added={r['path'] for r in identity['files'] if r['base_sha256'] is None}
    if set(git('-C','rocm-libraries','diff','--name-only').splitlines())!=tracked or set(git('-C','rocm-libraries','ls-files','--others','--exclude-standard').splitlines())!=added:raise RuntimeError('Candidate contains changes beyond the frozen joint patch')
    for row in identity['files']:
        if digest(Path('rocm-libraries')/row['path'])!=row['candidate_sha256']:raise RuntimeError('Candidate postimage differs: '+row['path'])
    subprocess.run(['git','-C','rocm-libraries','apply','--reverse','--check',str(HERE/'rocsolver-specialized-shards.patch')],check=True)
    nested=Path('build/math-libs/BLAS/rocSOLVER/build');cache=(nested/'CMakeCache.txt').read_text()
    for option in ('ROCSOLVER_SHARD_GETF2_SMALL','ROCSOLVER_SPLIT_GETRI_Z'):
        if option+':BOOL=ON' not in cache.splitlines():raise RuntimeError('Candidate option is not ON: '+option)
    database=read(nested/'compile_commands.json');rows=[]
    for name in NAMES:
        record=read(RECORDS/(name+'.json'));argv=record['argv'];source=argv[argv.index('-c')+1];matches=[r for r in database if r['file']==source]
        if len(matches)!=1:raise RuntimeError('Candidate compilation database source missing/nonunique: '+name)
        actual=matches[0].get('arguments') or shlex.split(matches[0]['command']);driver=next(i for i,v in enumerate(actual) if Path(v).name in ('clang++','clang','amdclang++'))
        actual=actual[driver:];row={'name':name,'actual_argv':actual,'frozen_candidate_argv':argv,'flags_equal_except_dependency_generation_output_destinations':normalized(actual)==normalized(argv),'source_sha256':digest(source),'cwd':matches[0]['directory']};rows.append(row)
    save('candidate-command-identity.json',{'commands':rows,'normalization':'Only -o/-MF/-MT/-MQ destinations and -MD/-MMD/-MP dependency generation omitted; all candidate codegen/ASan/gfx950/full-g flags compared.'})
    if not all(r['flags_equal_except_dependency_generation_output_destinations'] for r in rows):raise RuntimeError('Candidate flags do not match captured candidate commands')
    wrapper=Path(read(RECORDS/'getf2_c_3.json')['argv'][-1]);expected='#define ROCSOLVER_GETF2_SMALL_SHARD 3\n#include "'+str(Path.cwd()/'rocm-libraries/projects/rocsolver/library/src/specialized/roclapack_getf2_small_c.cpp')+'"\n'
    if wrapper.read_text()!=expected:raise RuntimeError('Generated c_3 wrapper differs')
    version=nested/'include/rocsolver/rocsolver-version.h';expected_version=HERE/'reference-candidate-rocsolver-version.h';shutil.copyfile(version,OUT/'candidate-generated-rocsolver-version.h')
    save('candidate-version-qualification.json',{'actual_sha256':digest(version),'reconstructed_pair1_candidate_sha256':digest(expected_version),'matches_reconstructed_reference':digest(version)==digest(expected_version),
        'reference_qualification':read(HERE/'version-reference-qualification.json'),'actual_macros':[l for l in version.read_text().splitlines() if l.startswith('#define ROCSOLVER_VERSION_')],
        'limits':'Reference header was reconstructed from pinned source/template and cohort CMake evidence, not captured verbatim in pair1; new actual bytes are retained.'})
    if digest(version)!=digest(expected_version):raise RuntimeError('Actual dirty-version header differs from reconstructed candidate reference')
    actual=read(OUT/'allocation-candidate-traces.json');reference=read(HERE/'pair1-allocation-baseline.json');tools={k:{'actual':v['sha256'],'pair1':reference['toolchain'].get(k,{}).get('sha256'),'equal':v['sha256']==reference['toolchain'].get(k,{}).get('sha256')} for k,v in actual['toolchain'].items()}
    save('candidate-toolchain-comparison.json',{'recorded_executables':tools,'recorded_clang_lld_executable_hashes_match_pair1':len(tools)==2 and all(v['equal'] for v in tools.values()),'limits':'Only captured clang/LLD executables, not every dynamically linked library; mismatch recorded without a false matching claim.'})
    (OUT/'candidate-source.diff').write_text(git('-C','rocm-libraries','diff'),encoding='utf-8');save('candidate-source-identity.json',identity)
    prime=[]
    for path in [Path(read(RECORDS/(name+'.json'))['argv'][-1]) for name in NAMES]+[version]:prime.append({'path':str(path),'sha256':digest(path),'bytes':path.stat().st_size})
    save('candidate-added-inputs-prime.json',{'files':prime,'limits':'Explicitly read added/generated inputs in addition to the frozen original input set; OS residency not proven.'})
def launch():
    guard();output=Path('perf-phase-probes');output.mkdir(exist_ok=False);jobs=[];start=time.time()
    for name in NAMES:
        expected='getf2-small' if name=='getf2_c_3' else 'getri-specialized'
        for traced in (False,True):
            label=name+('-trace' if traced else '-control');command=['cmake','-E','env',*[f'--unset={k}' for k in UNSET],'--',sys.executable,str(HERE/'trace_getf2.py'),'--record',str(RECORDS/(name+'.json')),'--source-root',str(Path.cwd()),'--expected-source',expected,'--output',str((output/label).resolve())]
            if traced:command+=['--trace','--device-link-trace']
            stream=(output/(label+'.log')).open('wb');jobs.append((label,command,subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT),stream))
    results=[]
    for label,command,process,stream in jobs:code=process.wait();stream.close();results.append({'label':label,'argv':command,'exit_code':code})
    (output/'cohort.json').write_text(json.dumps({'started_epoch_s':start,'ended_epoch_s':time.time(),'jobs':results,'limits':'N1 control/traced per candidate source, six compilers co-run; overhead/interference not separated. No whole solver build or GPU numerical validation.'},indent=2))
    return int(any(r['exit_code'] for r in results))
def after():
    guard();cohort=read('perf-phase-probes/cohort.json');rows=[]
    if len(cohort['jobs'])!=6 or any(r['exit_code'] for r in cohort['jobs']):raise RuntimeError('Six successful candidate replays required')
    for name in NAMES:
        for traced in (False,True):
            directory=Path('perf-phase-probes')/(name+('-trace' if traced else '-control'));m=read(directory/'replay-metadata.json')
            if m['exit_code'] or not m['objects']:raise RuntimeError('Missing successful candidate object metadata')
            row={'label':directory.name,'wall_s':m['wall_s'],'user_cpu_s':m['user_cpu_s'],'system_cpu_s':m['system_cpu_s'],'objects':m['objects'],'traces':[]}
            if traced:
                traces=[p for p in directory.glob('*.json') if p.name not in ('replay-metadata.json','capability.json')]
                if len(traces)<2 or not (directory/'device-lld.json').exists():raise RuntimeError('Candidate actual host/device traces missing')
                for path in traces:
                    payload=read(path);events=payload.get('traceEvents',[]) if isinstance(payload,dict) else payload;count=sum(e.get('ph')=='X' and not e.get('name','').startswith('Total ') for e in events)
                    if not count:raise RuntimeError('Candidate trace contains no ordinary duration events')
                    row['traces'].append({'file':str(path),'sha256':digest(path),'ordinary_X_count':count})
            rows.append(row)
    save('candidate-actual-trace-production.json',{'passed':True,'replays':rows,'limits':'Nested wall scopes are not CPU; six co-running commands differ in concurrency/context from the four original-source probes. Compare causality with this qualification; no new optimization or full-CI gain claim.'})
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('apply','verify','launch','after'));a=p.parse_args();result=globals()[a.phase]();return result or 0
if __name__=='__main__':raise SystemExit(main())
