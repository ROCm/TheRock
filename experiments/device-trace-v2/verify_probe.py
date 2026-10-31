"""Fail closed on missing original inputs/actual trace events; no experiment dispatch."""
import argparse, hashlib, json, os
from pathlib import Path
import shlex, subprocess, sys
BASE='5b7cb88373362ab80dc0f19605915e534f4158aa'
LIBS='f4e24d58b3f313f1a31068def1e35376e43c2f49'
HERE=Path(__file__).resolve().parent

def read(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,value): Path('perf-records',name).write_text(json.dumps(value,indent=2))
def git(*args): return subprocess.check_output(['git',*args],text=True).strip()
def guard():
    if sys.platform!='linux' or os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('GITHUB_REPOSITORY')!='ROCm/TheRock':raise RuntimeError('Only Linux TheRock Actions execution is permitted')
    if git('rev-parse','HEAD')!=BASE or git('-C','rocm-libraries','rev-parse','HEAD')!=LIBS:raise RuntimeError('Source pin mismatch')
    if git('-C','rocm-libraries','diff','--name-only') or git('-C','rocm-libraries','ls-files','--others','--exclude-standard'):raise RuntimeError('Solver source is not original')

def normalized(argv):
    result=[]; index=0
    while index<len(argv):
        value=argv[index]
        if value in ('-o','-MF','-MT','-MQ'):index+=2;continue
        if value in ('-MD','-MMD','-MP'):index+=1;continue
        result.append(value);index+=1
    return result

def before():
    guard()
    actual=read('perf-records/allocation-device-trace.json'); expected=read(HERE/'pair1-allocation-baseline.json')
    tools={name:{'actual':row.get('sha256'),'pair1':expected['toolchain'].get(name,{}).get('sha256'),
                 'equal':row.get('sha256')==expected['toolchain'].get(name,{}).get('sha256')}
           for name,row in actual['toolchain'].items()}
    tool_match=len(tools)==2 and set(tools)==set(expected['toolchain']) and all(row['equal'] for row in tools.values())
    result={'pair1_run_id':41199411426,'pair1_member':'perf-records/allocation-baseline.json','toolchain_hashes':tools,
        'recorded_clang_lld_executable_hashes_match_pair1':tool_match,'allocation_fields_equal':{k:actual[k]==expected[k] for k in ('host_logical_cpus','affinity','models')},
        'limits':'Compares only the clang and LLD executable hashes captured in pair1, not every dynamically linked toolchain library. Mismatch is recorded, never claimed matched. This remains N1 with four co-running compilers.'}
    save('pair1-toolchain-comparison.json',result)
    cache=Path('build/math-libs/BLAS/rocSOLVER/build/CMakeCache.txt').read_text()
    if any(line.endswith('=ON') and line.startswith(('ROCSOLVER_SHARD_GETF2_SMALL:','ROCSOLVER_SPLIT_GETRI_Z:')) for line in cache.splitlines()):raise RuntimeError('Candidate options unexpectedly enabled')
    database=read('build/math-libs/BLAS/rocSOLVER/build/compile_commands.json')
    rows=[]
    for name in ('getri_z','getf2_c'):
        frozen=read(HERE/'records'/f'{name}.json')['argv']; source=frozen[frozen.index('-c')+1]
        matches=[row for row in database if Path(row['file']).name==Path(source).name]
        if len(matches)!=1:raise RuntimeError('Cannot identify one actual command for '+name)
        row=matches[0]; argv=row.get('arguments') or shlex.split(row['command'])
        # Compilation databases may include the known launcher; compare from actual compiler onward.
        compiler=next((i for i,value in enumerate(argv) if Path(value).name in ('clang++','clang','amdclang++')),None)
        if compiler is None:raise RuntimeError('No actual compiler in configured command')
        argv=argv[compiler:]
        item={'name':name,'actual_argv':argv,'frozen_argv':frozen,'equal_except_output_dependency_destinations':normalized(argv)==normalized(frozen),'cwd':row['directory']}
        rows.append(item)
    save('configured-command-identity.json',{'commands':rows,'normalization':'Only -o/-MF/-MT/-MQ destinations and dependency-generation -MD/-MMD/-MP removed (CMake compdb omits these); all source/codegen/includes/architecture flags retained'})
    if not all(row['equal_except_output_dependency_destinations'] for row in rows):raise RuntimeError('Configured command flags differ from frozen replay inputs; inspect retained identity evidence')

def after():
    guard(); result=[]
    cohort=read('perf-phase-probes/cohort.json')
    if len(cohort['jobs'])!=4 or any(row['exit_code'] for row in cohort['jobs']):raise RuntimeError('Four successful replays required')
    for name in ('getri_z','getf2_c'):
        for traced in (False,True):
            directory=Path('perf-phase-probes')/(name+('-trace' if traced else '-control'))
            metadata=read(directory/'replay-metadata.json')
            if metadata['exit_code'] or not metadata['objects']:raise RuntimeError('Missing successful object metadata')
            row={'directory':str(directory),'wall_s':metadata['wall_s'],'user_cpu_s':metadata['user_cpu_s'],'system_cpu_s':metadata['system_cpu_s'],'objects':metadata['objects'],'traces':[]}
            if traced:
                traces=[p for p in directory.glob('*.json') if p.name not in ('replay-metadata.json','capability.json')]
                if not (directory/'device-lld.json').is_file() or len(traces)<2:raise RuntimeError('Require actual device LLD and host compiler traces')
                for path in traces:
                    payload=read(path); events=payload.get('traceEvents',[]) if isinstance(payload,dict) else payload
                    ordinary=[event for event in events if event.get('ph')=='X' and not event.get('name','').startswith('Total ')]
                    if not ordinary:raise RuntimeError('No actual duration events in '+str(path))
                    row['traces'].append({'path':str(path),'sha256':digest(path),'bytes':path.stat().st_size,'ordinary_duration_events':len(ordinary)})
            result.append(row)
    save('actual-trace-production.json',{'passed':True,'replays':result,'limits':'Wall durations are not CPU; four compilers co-run and N1 cannot separate overhead/interference or validate speedup.'})

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('before','after'));args=parser.parse_args()
    (before if args.phase=='before' else after)()
if __name__=='__main__':main()
