"""Qualify original RPM repositories and record installed ELF/RPM-database digest comparisons."""
import argparse,datetime,gzip,hashlib,json,lzma,os,re,stat,subprocess,sys,time,urllib.parse,urllib.request,xml.etree.ElementTree as ET
from pathlib import Path
BASE='5b7cb88373362ab80dc0f19605915e534f4158aa';PRODUCER='17bf9b333993b6b12dd9ae867a9c16813cdfca0f'
HERE=Path(__file__).resolve().parent;OUT=Path('.perf-full/native-rpm')
REQUESTED='linux-gfx950-1gpu-ccs-ossci-rocm'

def save(name,value):OUT.mkdir(parents=True,exist_ok=True);(OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
def digest(raw):return hashlib.sha256(raw).hexdigest()
def api(path):
    request=urllib.request.Request('https://api.github.com/repos/ROCm/TheRock/'+path,headers={'Authorization':'Bearer '+os.environ['GITHUB_TOKEN'],'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    with urllib.request.urlopen(request,timeout=120) as response:return json.load(response)
def guard():
    if sys.platform!='linux' or os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('GITHUB_REPOSITORY')!='ROCm/TheRock':raise RuntimeError('Only Linux TheRock Actions permitted')
    if subprocess.check_output(['git','-c',f'safe.directory={Path.cwd()}','rev-parse','HEAD'],text=True).strip()!=BASE:raise RuntimeError('Frozen BASE required')
    profiles=json.loads((HERE/'profiles.json').read_text());mode=os.environ['PERF_MODE'];profile=profiles['producers'][mode]['profiles'][os.environ['OS_PROFILE']]
    if os.environ['PRODUCER_RUN_ID']!=str(profiles['producers'][mode]['run_id']) or os.environ['REPO_URL']!=profile['repo_url']:raise RuntimeError('Original producer/repository mismatch')
    for key,value in {'PKG_TYPE':'rpm','GFX_ARCH':'gfx950','INSTALL_PREFIX':'/opt/rocm/core','TEST_TYPE':'sanity','BUILD_VARIANT':'asan-debug','RELEASE_TYPE':'ci','GPG_KEY_URL':''}.items():
        if os.environ.get(key)!=value:raise RuntimeError('Frozen install environment differs: '+key)
    if os.environ.get('PERF_REQUESTED_RUNNER')!=REQUESTED:raise RuntimeError('Explicit user-authorized gfx950 runner required')
    return profiles,mode,profile
def read_url(url,cap=32*1024*1024):
    request=urllib.request.Request(url)
    with urllib.request.urlopen(request,timeout=120) as response:
        raw=response.read(cap+1);headers={k:response.headers.get(k) for k in ('ETag','Last-Modified','Content-Length')}
    if len(raw)>cap:raise RuntimeError('Repository metadata exceeds evidence bound')
    return raw,headers
def repo_snapshot():
    root=os.environ['REPO_URL'].rstrip('/')+'/';url=root+'repodata/repomd.xml';raw,headers=read_url(url)
    namespace={'r':'http://linux.duke.edu/metadata/repo'};document=ET.fromstring(raw);items=document.findall('r:data',namespace)
    primary=[item for item in items if item.attrib.get('type')=='primary']
    if len(primary)!=1:raise RuntimeError('One original primary metadata entry required')
    item=primary[0];href=item.find('r:location',namespace).attrib['href'];checksum=item.find('r:checksum',namespace)
    if urllib.parse.urlsplit(href).scheme or '..' in Path(href).parts or not href.startswith('repodata/'):raise RuntimeError('Unsafe repository metadata URL')
    payload,pheaders=read_url(urllib.parse.urljoin(root,href));algorithm=checksum.attrib['type']
    if algorithm not in {'sha256','sha512'} or hashlib.new(algorithm,payload).hexdigest()!=checksum.text:raise RuntimeError('Primary metadata checksum mismatch')
    if href.endswith('.gz'):xml=gzip.decompress(payload)
    elif href.endswith('.xz'):xml=lzma.decompress(payload)
    elif href.endswith('.xml'):xml=payload
    else:raise RuntimeError('Unsupported primary compression; preserve evidence and inspect')
    if len(xml)>128*1024*1024:raise RuntimeError('Expanded primary metadata exceeds bound')
    ns={'c':'http://linux.duke.edu/metadata/common'};packages=[]
    for package in ET.fromstring(xml).findall('c:package',ns):
        name=package.findtext('c:name',namespaces=ns)
        if not name.startswith('amdrocm'):continue
        location=package.find('c:location',ns).attrib['href']
        if urllib.parse.urlsplit(location).scheme or '..' in Path(location).parts:raise RuntimeError('Package location escapes original repository')
        version=package.find('c:version',ns).attrib;check=package.find('c:checksum',ns)
        packages.append({'name':name,'arch':package.findtext('c:arch',namespaces=ns),'version':version,
            'checksum_type':check.attrib['type'],'checksum':check.text,'location':location})
    if not packages:raise RuntimeError('Original RPM repository has no ROCm packages')
    return {'repomd_url':url,'repomd_sha256':digest(raw),'repomd_headers':headers,'primary_url':urllib.parse.urljoin(root,href),
        'primary_archive_sha256':digest(payload),'primary_headers':pheaders,'primary_checksum_verified':True,'packages':packages}
def prepare():
    profiles,mode,profile=guard();run=profiles['producers'][mode]['run_id'];original=api('actions/runs/'+str(run))
    if original['head_sha']!=PRODUCER or original['head_branch']!='users/safaizal/full-ci-v2-01a19119' or original['event']!='workflow_dispatch' or original['display_title']!=f'Full HEAD {mode} validation (01a19119)':raise RuntimeError('Unexpected original run identity')
    job=api('actions/jobs/'+str(profile['job_id']));steps={s['name']:s['conclusion'] for s in job['steps']}
    if job['conclusion']!='failure' or steps.get('Verify frozen source and record full-run context')!='failure' or steps.get('Package Installation Test')!='skipped' or steps.get('Package Uninstall Test')!='skipped':raise RuntimeError('Only previously unexecuted assertions after instrumentation failure may be continued')
    if job['labels']!=profile['original_runner_labels']:raise RuntimeError('Original allocation evidence differs')
    rpm=api('actions/jobs/'+str(profiles['producers'][mode]['rpm_build_job_id']))
    if rpm['conclusion']!='success':raise RuntimeError('Original RPM producer did not pass')
    repository=repo_snapshot();save('repository-before.json',repository)
    save('qualification.json',{'source_sha':BASE,'producer_run_id':run,'producer_instrumentation_sha':PRODUCER,
        'consumer_run_id':os.environ['GITHUB_RUN_ID'],'consumer_instrumentation_sha':os.environ['GITHUB_SHA'],'mode':mode,'os_profile':os.environ['OS_PROFILE'],
        'original_failed_job':job,'original_rpm_build_job':rpm,'source_raw_evidence':profile,
        'requested_runner':REQUESTED,'runner_name':os.environ.get('RUNNER_NAME'),
        'runner_difference':'Original native install used default gfx942 despite gfx950 artifact selection and failed before assertions; new explicit gfx950 route follows user instruction. Allocation is not matched.',
        'eligible_for_optimization_acceptance':False,'limits':'Repairs skipped stock install/uninstall checks, not original runtime failures or full CI; no speedup credit'})
def installed():
    guard();start=time.monotonic();cpu=time.process_time();raw=subprocess.check_output(['rpm','-qa','--qf','%{NAME}\t%{VERSION}\t%{RELEASE}\t%{ARCH}\n'],text=True)
    packages=[dict(zip(('name','version','release','arch'),line.split('\t'))) for line in raw.splitlines() if line.startswith('amdrocm')]
    before=json.loads((OUT/'repository-before.json').read_text());catalog={(p['name'],p['version']['ver'],p['version']['rel'],p['arch']):p for p in before['packages']}
    failures=[];file_rows=[];byte_count=0
    if not packages:raise RuntimeError('Stock install succeeded but no ROCm RPM is installed')
    for package in packages:
        key=tuple(package[k] for k in ('name','version','release','arch'))
        if key not in catalog:failures.append({'package':package,'reason':'Installed name/version/release/architecture tuple absent from original producer repository'});continue
        algorithm=subprocess.check_output(['rpm','-q','--qf','%{FILEDIGESTALGO}',package['name']],text=True).strip()
        if algorithm not in {'8','2','1'}:raise RuntimeError('Unsupported RPM file digest algorithm '+algorithm)
        hasher={'8':'sha256','2':'sha1','1':'md5'}[algorithm]
        dump=subprocess.check_output(['rpm','-q','--qf','[%{FILENAMES}\t%{FILEDIGESTS}\t%{FILEMODES:octal}\n]',package['name']],text=True)
        for line in dump.splitlines():
            name,expected,mode=line.split('\t');path=Path(name)
            if not name.startswith('/opt/rocm/core-asan-10.3/') or not stat.S_ISREG(int(mode,8)) or path.is_symlink():continue
            if not path.is_file():failures.append({'path':name,'reason':'Installed regular payload missing'});continue
            with path.open('rb') as stream:
                if stream.read(4)!=b'\x7fELF':continue
                stream.seek(0);value=hashlib.new(hasher)
                for block in iter(lambda:stream.read(4*1024*1024),b''):value.update(block)
            actual=value.hexdigest();size=path.stat().st_size;byte_count+=size
            file_rows.append({'package':package['name'],'path':name,'digest_type':hasher,'rpm_database_digest':expected,'actual_digest':actual,'bytes':size,'equal':actual==expected})
            if actual!=expected:failures.append({'path':name,'reason':'Installed ELF differs from RPM database payload digest'})
    save('installed-payload-integrity.json',{'packages':packages,'elf_files':file_rows,'failures':failures,'passed':bool(file_rows) and not failures,
        'wall_s':time.monotonic()-start,'observer_cpu_s':time.process_time()-cpu,'bytes_hashed':byte_count,
        'limits':'Two separate checks: installed name/version/release/architecture tuples match original repository metadata (Epoch is not compared); regular SDK ELF bytes match installed RPM-database file digests. ELF digests are not independently bound to the original repository .rpm checksum. No independent whole-.rpm transport hashing, full filesystem verification or GPU numerical coverage.'})
    if failures or not file_rows:raise RuntimeError('Installed RPM payload qualification failed')
def final():
    guard();before=json.loads((OUT/'repository-before.json').read_text());after=repo_snapshot()
    same=before['repomd_sha256']==after['repomd_sha256'] and before['primary_archive_sha256']==after['primary_archive_sha256']
    save('repository-after.json',after);save('repository-stability.json',{'original_metadata_unchanged':same,'eligible_for_optimization_acceptance':False})
    if not same:raise RuntimeError('Original RPM repository metadata changed during checks')
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','installed','final'));args=parser.parse_args();globals()[args.phase]()
if __name__=='__main__':main()
