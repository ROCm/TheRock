"""Targeted profiler DEB repair and payload preservation; TheRock Actions only."""
import argparse,datetime,hashlib,json,os,re,shutil,subprocess,sys,time,urllib.request
from pathlib import Path

BASE='5b7cb88373362ab80dc0f19605915e534f4158aa'
PRODUCER='17bf9b333993b6b12dd9ae867a9c16813cdfca0f'
RUNS={'baseline':'41211496307','joint':'41212843218'}
JOBS={'baseline':124604240654,'joint':124604023601}
HERE=Path(__file__).resolve().parent;OUT=Path('.profiler-deb-evidence')
PACKAGE='amdrocm-profiler-base';PREFIX=Path('opt/rocm/core-asan-10.3')

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def save(name,value):
    OUT.mkdir(exist_ok=True);(OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')

def parse_depends(text):
    """Parse this contract's simple conjunctions; reject alternatives/extra syntax."""
    rows=[]
    for item in text.split(','):
        match=re.fullmatch(r'\s*([a-z0-9][a-z0-9+.-]*)(?:\s*\(\s*(<<|<=|=|>=|>>)\s*([^()\s]+)\s*\))?\s*',item)
        if match is None:raise RuntimeError('Unexpected dependency grammar: '+item)
        rows.append(list(match.groups()))
    return sorted(rows,key=lambda row:row[0])

def api(path):
    request=urllib.request.Request('https://api.github.com/repos/ROCm/TheRock/'+path,
        headers={'Authorization':'Bearer '+os.environ['GITHUB_TOKEN'],'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    with urllib.request.urlopen(request,timeout=120) as response:return json.load(response)

def guard():
    if sys.platform!='linux' or os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('GITHUB_REPOSITORY')!='ROCm/TheRock':raise RuntimeError('Only Linux TheRock Actions permitted')
    if subprocess.check_output(['git','-c',f'safe.directory={Path.cwd()}','rev-parse','HEAD'],text=True).strip()!=BASE:raise RuntimeError('Frozen BASE checkout required')
    if os.environ.get('PRODUCER_RUN_ID')!=RUNS[os.environ['MODE']]:raise RuntimeError('Original producer/mode mismatch')

def prepare():
    guard();mode=os.environ['MODE'];run=RUNS[mode];original=api('actions/runs/'+run)
    if original['head_sha']!=PRODUCER or original['head_branch']!='users/safaizal/full-ci-v2-01a19119' or original['event']!='workflow_dispatch' or original['display_title']!=f'Full HEAD {mode} validation (01a19119)':raise RuntimeError('Unexpected original producer identity')
    job=api('actions/jobs/'+str(JOBS[mode]));steps={s['name']:s['conclusion'] for s in job['steps']}
    if job['conclusion']!='failure' or steps.get('Build Packages')!='failure' or steps.get('Fetch Artifacts for all GPU families')!='success':raise RuntimeError('Original failure is not the retained package-build failure')
    config=Path('build_tools/packaging/linux/package.json');old=json.loads(config.read_text());patch=HERE/'profiler-disable-dwz.patch'
    expected=json.loads((HERE/'source-contract.json').read_text())
    if sha(config)!=expected['package_json_base_sha256'] or sha(patch)!=expected['patch_sha256']:raise RuntimeError('Frozen package source/patch mismatch')
    subprocess.run(['git','apply','--check',str(patch)],check=True);subprocess.run(['git','apply',str(patch)],check=True)
    new=json.loads(config.read_text());changed=[]
    for before,after in zip(old,new):
        if before!=after:
            name=before['Package'];copy=dict(after);flag=copy.pop('Disable_DWZ',None)
            if name!=PACKAGE or flag!='True' or before!=copy:raise RuntimeError('Patch changes more than profiler DWZ setting')
            changed.append(name);source_package_entries={'original':before,'patched':after}
    if len(old)!=len(new) or changed!=[PACKAGE]:raise RuntimeError('Unexpected JSON patch scope')
    save('source-and-producer.json',{'source_sha':BASE,'producer_run_id':run,'mode':mode,'producer_instrumentation_sha':PRODUCER,
        'consumer_run_id':os.environ['GITHUB_RUN_ID'],'consumer_instrumentation_sha':os.environ['GITHUB_SHA'],
        'original_packaging_job':job,'patch_sha256':sha(patch),'changed_package':PACKAGE,'original_package_entry':source_package_entries['original'],'patched_package_entry':source_package_entries['patched'],
        'scope':'Only supported Disable_DWZ=True; no compile, optimization, sanitizer or debug-level changes',
        'eligible_for_optimization_acceptance':False})
    sys.path.insert(0,str(Path('build_tools').resolve()))
    from _therock_utils.workflow_outputs import WorkflowOutputRoot
    from _therock_utils.artifact_backend import S3Backend
    from artifact_manager import find_available_artifacts,extract_artifact,ExtractRequest
    route=WorkflowOutputRoot.from_workflow_run(run_id=run,platform='linux',github_repository='ROCm/TheRock',workflow_run=original,release_type='ci')
    if route.bucket!='therock-ci-artifacts' or route.prefix!=run+'-linux':raise RuntimeError('Unexpected artifact read route')
    backend=S3Backend(route);needed={'rocprofiler-sdk','aqlprofile','profiler-hub'}
    available=set(backend.list_artifacts())
    selected=find_available_artifacts(needed,['generic'],available,{'test','dbg'})
    from _therock_utils.artifacts import ArtifactName
    if {ArtifactName.from_filename(name).name for name in selected}!=needed:raise RuntimeError('Required package artifact groups missing')
    if any(ArtifactName.from_filename(name).component not in {'lib','dev','run','doc'} for name in selected):raise RuntimeError('Unexpected component selection')
    # get_package_list() controls versioned Depends through real artifact presence.
    # Preserve the two direct package dependencies without fetching unrelated SDKs.
    markers=[]
    for group in ('core-runtime','sysdeps'):
        choices=find_available_artifacts({group},['generic'],available)
        if not choices:raise RuntimeError('Direct-dependency catalog artifact missing: '+group)
        marker=min(choices,key=lambda name:backend.s3_client.head_object(Bucket=route.bucket,Key=route.artifact(name).relative_path)['ContentLength'])
        markers.append(marker)
    selected+=markers
    cache=Path('output/downloads');artifacts=Path('output/artifacts');cache.mkdir(parents=True);artifacts.mkdir(parents=True)
    refs=[]
    for name in selected:
        target=cache/name;key=route.artifact(name).relative_path;head=backend.s3_client.head_object(Bucket=route.bucket,Key=key)
        backend.download_artifact(name,target);sidecar=cache/(name+'.sha256sum');backend.download_artifact(name+'.sha256sum',sidecar)
        matches=re.findall(r'\b[0-9a-fA-F]{64}\b',sidecar.read_text())
        if len(matches)!=1 or sha(target)!=matches[0].lower():raise RuntimeError('Original artifact checksum mismatch: '+name)
        after=backend.s3_client.head_object(Bucket=route.bucket,Key=key)
        if head['ETag']!=after['ETag'] or head['ContentLength']!=target.stat().st_size:raise RuntimeError('Original artifact changed during read')
        refs.append({'artifact':name,'role':'dependency_catalog_marker' if name in markers else 'package_payload','source_url':f'https://{route.bucket}.s3.amazonaws.com/{key}','sha256':sha(target),
            'checksum_sidecar_sha256':sha(sidecar),'bytes':target.stat().st_size,'etag':head['ETag'],'last_modified':head['LastModified'].isoformat()})
        result=extract_artifact(ExtractRequest(archive_path=target,output_dir=artifacts,delete_archive=False,flatten=False,bootstrap=False,cleaned_paths=None,cleaned_paths_lock=None))
        if result is None:raise RuntimeError('Original artifact extraction failed')
    save('input-artifacts.json',{'route':{'bucket':route.bucket,'prefix':route.prefix},'artifacts':refs,
        'selection':'Exact package payload Artifactory names/lib/dev/run/doc generic components; smallest genuine core-runtime/sysdeps archive preserves direct Depends catalog presence; no compiler or math artifacts fetched'})

def elf(path):
    from elftools.elf.elffile import ELFFile
    with path.open('rb') as stream:
        if stream.read(4)!=b'\x7fELF':return None
        stream.seek(0);file=ELFFile(stream);sections={}
        for section in file.iter_sections():
            if section.name=='.text' or section.name.startswith(('.debug_','.zdebug_','.hip','.llvm.offloading','.rocm')):
                sections.setdefault(section.name,[]).append({'sha256':hashlib.sha256(section.data()).hexdigest(),'bytes':len(section.data())})
        return sections

def before():
    guard()
    if sha('build_tools/packaging/linux/package.json')!=json.loads((HERE/'source-contract.json').read_text())['package_json_patched_sha256']:raise RuntimeError('Patched packaging configuration drift')
    sys.path.insert(0,str(Path('build_tools/packaging/linux').resolve()))
    from packaging_utils import filter_components_fromartifactory
    sources=filter_components_fromartifactory(PACKAGE,Path('output/artifacts').resolve(),'')
    if not sources or any(not Path(source).is_dir() for source in sources):raise RuntimeError('Required package source directories missing')
    rows={};sidecars={}
    for directory in sources:
        directory=Path(directory)
        for path in directory.rglob('*'):
            if path.is_symlink() or not path.is_file():continue
            sections=elf(path)
            key=path.relative_to(directory).as_posix()
            if sections is None:
                if path.suffix in {'.kpack','.a','.co','.hsaco'}:
                    value={'source_path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
                    if key in sidecars and sidecars[key]['sha256']!=value['sha256']:raise RuntimeError('Different binary sidecars overlap: '+key)
                    sidecars[key]=value
                continue
            if key in rows and rows[key]['sections']!=sections:raise RuntimeError('Different ELF payloads overlap: '+key)
            rows[key]={'source_path':str(path),'sections':sections,'file_sha256_before_stock_runpath_conversion':sha(path)}
    if not rows or not any('MatrixTranspose' in name and '.debug_str_offsets' in row['sections'] for name,row in rows.items()):raise RuntimeError('Expected failing profiler DWARF input not represented')
    save('elf-before.json',{'files':rows,'binary_sidecars':sidecars,'source_directories':[str(p) for p in sources],
        'limits':'Whole file hashes can change through existing stock RUNPATH to RPATH conversion; text/debug section contents are the invariant'})

def after():
    guard()
    if sha('build_tools/packaging/linux/package.json')!=json.loads((HERE/'source-contract.json').read_text())['package_json_patched_sha256']:raise RuntimeError('Patched packaging configuration drift')
    snapshot=json.loads((OUT/'elf-before.json').read_text());expected=snapshot['files'];packages=list(Path('output/packages').rglob('*.deb'));rows=[];payload=None
    for path in packages:
        fields=subprocess.check_output(['dpkg-deb','-f',str(path)],text=True)
        rows.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'complete_actual_binary_control':fields})
        name=subprocess.check_output(['dpkg-deb','-f',str(path),'Package'],text=True).strip()
        version=subprocess.check_output(['dpkg-deb','-f',str(path),'Version'],text=True).strip()
        if version!='10.3.0~dev20261031-'+os.environ['PRODUCER_RUN_ID']:raise RuntimeError('Package version/producer identity differs')
        if subprocess.check_output(['dpkg-deb','-f',str(path),'Architecture'],text=True).strip()!='amd64':raise RuntimeError('Package architecture differs')
        depends=subprocess.check_output(['dpkg-deb','-f',str(path),'Depends'],text=True).strip();parsed=parse_depends(depends)
        # Independently derived from frozen package.json and packaging_utils.py:
        # this package has Metapackage false and Gfxarch false. Neither binary
        # variant receives append_version_suffix() version relations.
        expected_dependencies={
            'amdrocm-profiler-base-asan10.3':sorted([[key,None,None] for key in ('libc6','amdrocm-runtime-asan10.3','amdrocm-sysdeps-asan10.3')]),
            'amdrocm-profiler-base-asan':[['amdrocm-profiler-base-asan10.3',None,None]],
        }
        if name not in expected_dependencies or parsed!=expected_dependencies[name]:raise RuntimeError('Exact original-source dependency contract differs: '+name)
        rows[-1]['depends']={'actual_text':depends,'parsed_names_operators_versions':parsed,'expected_names_operators_versions':expected_dependencies[name],'equal':True}
        if name=='amdrocm-profiler-base-asan10.3':
            if payload is not None:raise RuntimeError('Multiple payload packages')
            destination=OUT/'extracted';destination.mkdir();subprocess.run(['dpkg-deb','--extract',str(path),str(destination)],check=True);payload=destination/PREFIX
        elif name!='amdrocm-profiler-base-asan':raise RuntimeError('Unexpected package built: '+name)
    if payload is None or len(packages)!=2:raise RuntimeError('Versioned and meta profiler DEB packages required')
    actual={}
    for path in payload.rglob('*'):
        if path.is_file() and not path.is_symlink():
            sections=elf(path)
            if sections is not None:actual[path.relative_to(payload).as_posix()]=sections
    missing=sorted(set(expected)-set(actual));extra=sorted(set(actual)-set(expected))
    differences=[name for name,row in expected.items() if name in actual and row['sections']!=actual[name]]
    sidecar_errors=[name for name,row in snapshot['binary_sidecars'].items() if not (payload/name).is_file() or sha(payload/name)!=row['sha256']]
    save('payload-preservation.json',{'passed':not missing and not extra and not differences and not sidecar_errors,'elf_count':len(expected),
        'missing_elf':missing,'extra_elf':extra,'changed_text_or_debug_sections':differences,'actual_sections':actual,
        'changed_or_missing_binary_sidecars':sidecar_errors,'binary_sidecar_count':len(snapshot['binary_sidecars']),
        'invariant':'All regular payload ELF .text, debug, HIP/offload/kpack section contents and binary sidecar/archive hashes equal original input before stock RPATH conversion',
        'limits':'Selected ELF sections and binary sidecars only, not every payload byte, symlink/mode/doc property or ELF section; no GPU execution/full SDK ABI equivalence; DWZ bypass only for this package; no debug-removal/CI speedup claim'})
    save('built-packages.json',{'packages':rows,'product_s3_upload':False,'eligible_for_optimization_acceptance':False})
    if missing or extra or differences or sidecar_errors:raise RuntimeError('Profiler payload code/debug/sidecar bytes changed')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','before','after'));args=parser.parse_args();globals()[args.phase]()
if __name__=='__main__':main()
