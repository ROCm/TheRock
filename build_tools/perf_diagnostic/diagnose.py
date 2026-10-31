"""One original failing stock GTest, diagnostic only; retain original failures."""
import argparse,datetime,hashlib,importlib.util,io,json,os,re,subprocess,sys,zipfile
from pathlib import Path
from original_transport_reference import api,pages,fetch
BASE='5b7cb88373362ab80dc0f19605915e534f4158aa';LIBS='f4e24d58b3f313f1a31068def1e35376e43c2f49'
PRODUCER='17bf9b333993b6b12dd9ae867a9c16813cdfca0f';PATCH='b74891bd117138d0e922407bf4cc24c1ee0482f08c685de0c1b36b662cb3aab9'
RUNS={'baseline':'41211496307','joint':'41212843218'}
CTEST_SHA='104d41600b9d6284afdb9787fb2f0c304edf4337f5e0364555d8808164962c26'
CASE='checkin_lapack/GETF2_NPVT.__float/0';OUT=Path('.runtime-diagnostic');HERE=Path(__file__).resolve().parent
def save(name,value):OUT.mkdir(exist_ok=True);(OUT/name).write_text(json.dumps(value,indent=2))
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def guard():
 if sys.platform!='linux' or os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('GITHUB_REPOSITORY')!='ROCm/TheRock':raise RuntimeError('TheRock Linux Actions only')
 if subprocess.check_output(['git','-c',f'safe.directory={Path.cwd()}','rev-parse','HEAD'],text=True).strip()!=BASE:raise RuntimeError('Actual source HEAD must be BASE')
 mode=os.environ['PERF_MODE'];run=os.environ['ARTIFACT_RUN_ID']
 if RUNS.get(mode)!=run:raise RuntimeError('Original producer/mode mismatch')
 return mode,run
def qualify():
 mode,run=guard();original=api('actions/runs/'+run)
 if original['head_sha']!=PRODUCER or original['head_branch']!='users/safaizal/full-ci-v2-01a19119' or original['event']!='workflow_dispatch' or original['display_title']!=f'Full HEAD {mode} validation (01a19119)':raise RuntimeError('Original producer source mismatch')
 jobs=pages(f'actions/runs/{run}/jobs','jobs');stages=[j for j in jobs if ' / Build Multi-Arch Stages / ' in j['name'] and ' / Stage - ' in j['name']]
 if not stages or any(j['conclusion']!='success' for j in stages):raise RuntimeError('Original product stages incomplete')
 solver=[j for j in jobs if j['id']==(124604850010 if mode=='baseline' else 124604438298)]
 if len(solver)!=1 or solver[0]['conclusion']!='failure' or not any(s['name']=='Test' and s['conclusion']=='failure' for s in solver[0]['steps']):raise RuntimeError('Expected original real stock solver failure')
 artifacts=pages(f'actions/runs/{run}/artifacts','artifacts');name='full-solver-identity-'+mode+'-gfx950-dcgpu';chosen=[r for r in artifacts if r['name']==name and not r['expired']]
 if len(chosen)!=1:raise RuntimeError('Original producer identity missing/ambiguous')
 raw=fetch(f"actions/artifacts/{chosen[0]['id']}/zip")
 with zipfile.ZipFile(io.BytesIO(raw)) as archive:
  members=[r for r in archive.infolist() if r.filename=='solver-sdk-identity.json']
  if len(members)!=1 or members[0].file_size>1024*1024:raise RuntimeError('Invalid identity sidecar')
  manifest=archive.read(members[0])
 OUT.mkdir(exist_ok=True);(OUT/'solver-sdk-identity.json').write_bytes(manifest)
 identity=json.loads(manifest);expected={'source_sha':BASE,'rocm_libraries_gitlink':LIBS,'effective_stock_libraries_head':LIBS,'mode':mode,'producer_run_id':run,'instrumentation_sha':PRODUCER,'joint_patch_sha256':PATCH if mode=='joint' else None}
 if any(str(identity.get(k))!=str(v) for k,v in expected.items()):raise RuntimeError('Producer sidecar identity mismatch')
 save('producer-qualification.json',{'source_sha':BASE,'producer_run_id':run,'mode':mode,'producer_instrumentation_sha':PRODUCER,'consumer_run_id':os.environ['GITHUB_RUN_ID'],'consumer_instrumentation_sha':os.environ['GITHUB_SHA'],'original_solver_job':solver[0],'original_failures_unchanged':True,'eligible_for_optimization_acceptance':False,'sidecar_artifact_id':chosen[0]['id'],'sidecar_archive_sha256':hashlib.sha256(raw).hexdigest(),'sidecar_sha256':hashlib.sha256(manifest).hexdigest(),'artifact_route':'s3://therock-ci-artifacts/'+run+'-linux','purpose':'DIAGNOSTIC ONLY: effective child environment and actual loaded SDK byte gap; no numerical validation acceptance'})
def cmake_arg(value):
 if any(x in value for x in ('\n','\r',']]')):raise RuntimeError('Unsupported CMake argument')
 return '[['+value+']]'
def launch():
 mode,run=guard();sdk=Path('build').resolve();manifest=json.loads((OUT/'solver-sdk-identity.json').read_text());verified=[];roles={}
 if manifest['producer_run_id']!=run or manifest['mode']!=mode or manifest['instrumentation_sha']!=PRODUCER:raise RuntimeError('Identity changed since qualification')
 for row in manifest['files']:
  name=row['basename']
  if Path(name).name!=name or any(x in name for x in '*?[]'):raise RuntimeError('Invalid producer basename')
  matches=sorted(set(p.resolve() for p in sdk.rglob(name) if p.is_file() and p.resolve().is_relative_to(sdk) and digest(p)==row['sha256']))
  if len(matches)!=1:raise RuntimeError('Original producer bytes missing/ambiguous: '+name)
  verified.append({**row,'actual_path':str(matches[0])})
  if row['role'] in ('rocsolver_bench','rocsolver_test','librocsolver'):
   if row['role'] in roles:raise RuntimeError('Duplicate role')
   roles[row['role']]=matches[0]
 if set(roles)!=set(('rocsolver_bench','rocsolver_test','librocsolver')):raise RuntimeError('Required producer roles absent')
 save('verified-before-launch.json',{'files':verified,'original_producer_manifest_sha256':digest(OUT/'solver-sdk-identity.json'),'all_post_kpack_producer_files_verified':True})
 if os.environ.get('ASAN_OPTIONS')!='detect_odr_violation=0:quarantine_size_mb=600:verify_asan_link_order=0' or os.environ.get('HSA_XNACK')!='1':raise RuntimeError('Original stock ASan/xnack environment differs')
 if os.environ.get('THEROCK_CI_DEBUG')!='0' or os.environ.get('ROCM_KPACK_DEBUG')!='1' or os.environ.get('GPU_ENABLE_PAL')!='0':raise RuntimeError('Original stock Linux debug/backend environment differs')
 # Record all available SDK DSO bytes before launch separately from producer-side identity.
 dsos=[];seen=set()
 for path in sdk.rglob('*'):
  if '.so' in path.name and path.is_file() and path.resolve().is_relative_to(sdk) and path.resolve() not in seen:
   seen.add(path.resolve());dsos.append({'path':str(path.resolve()),'sha256':digest(path.resolve()),'bytes':path.stat().st_size})
 save('sdk-dso-before-launch.json',{'files':dsos,'scope':'Consumer SDK bytes before launch, not producer-side hashes for HIP DSOs absent from original sidecar'})
 spec=importlib.util.spec_from_file_location('stock_test_runner',Path('build_tools/github_actions/test_executable_scripts/test_runner.py'));runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
 if runner.TEST_COMPONENT!='rocsolver' or runner.TEST_TYPE!='quick' or str(runner.SHARD_INDEX)!='1' or str(runner.TOTAL_SHARDS)!='1':raise RuntimeError('Original component/shard/type differs')
 testdir=Path(runner.TEST_DIR).resolve();metadata=testdir/'CTestTestfile.cmake';original=metadata.read_bytes()
 if hashlib.sha256(original).hexdigest()!=CTEST_SHA:raise RuntimeError('Installed CTest metadata differs from both original captured archives')
 if (testdir/'generate_resource_spec').exists():raise RuntimeError('Unexpected resource helper: unsupported in this frozen contract')
 arches,excluded=runner.check_available_labels();cmd=runner.build_ctest_command('quick','gfx950',arches,excluded,None,'gfx950-dcgpu')
 shown=subprocess.check_output(cmd+['--show-only=json-v1'],cwd=runner.THEROCK_DIR,env=runner.environ_vars,text=True);(OUT/'original-ctest-show-only.json').write_text(shown)
 tests=json.loads(shown)['tests']
 if len(tests)!=1 or tests[0]['name']!='rocsolver-test_quick_suite':raise RuntimeError('Expected exactly original quick_suite')
 test=tests[0];props={p['name']:p['value'] for p in test.get('properties',[])}
 if set(props)-{'LABELS','TIMEOUT','WORKING_DIRECTORY'}:raise RuntimeError('Unsupported installed CTest property; do not omit environment/resource semantics')
 if float(props['TIMEOUT'])!=300 or Path(props.get('WORKING_DIRECTORY',str(testdir))).resolve()!=testdir:raise RuntimeError('Original CTest timeout/working directory differs')
 command=test['command'];exe=Path(command[0]).resolve()
 if exe!=roles['rocsolver_test'] or len(command)!=2 or not command[1].startswith('--gtest_filter='):raise RuntimeError('Unsupported wrapper or command; original semantics cannot be reconstructed')
 target=[str(exe),'--gtest_filter='+CASE];(OUT/'original-CTestTestfile.cmake').write_bytes(original)
 # Replace only quick add_test command: one filter plus observer prefix. Preserve all properties byte-for-byte and original installed working directory.
 lines=original.decode().splitlines(keepends=True);indices=[i for i,line in enumerate(lines) if line.startswith('add_test(rocsolver-test_quick_suite ')]
 if len(indices)!=1:raise RuntimeError('Ambiguous quick test definition')
 wrapped=[sys.executable,str(HERE/'observe.py'),'--output',str((OUT/'child-observation').resolve()),'--sdk-root',str(sdk),'--',*target]
 lines[indices[0]]='add_test(rocsolver-test_quick_suite '+' '.join(cmake_arg(s) for s in wrapped)+')\n';modified=''.join(lines).encode()
 save('diagnostic-command.json',{'original_ctest_argv':cmd,'original_test':test,'target_test_command':target,'instrumented_command':wrapped,'changed_product_argument':'gtest_filter only','original_ctest_sha256':CTEST_SHA,'modified_ctest_sha256':hashlib.sha256(modified).hexdigest(),'original_properties_preserved':True,'eligible_for_optimization_acceptance':False})
 (OUT/'instrumented-CTestTestfile.cmake').write_bytes(modified)
 code=None
 try:
  metadata.write_bytes(modified)
  with (OUT/'ctest.log').open('w') as log:
   proc=subprocess.Popen(cmd,cwd=runner.THEROCK_DIR,env=runner.environ_vars,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
   code=proc.wait()
 finally:metadata.write_bytes(original)
 restored=digest(metadata)==CTEST_SHA
 observation_path=OUT/'child-observation'/'observation.json';observation=json.loads(observation_path.read_text()) if observation_path.exists() else {}
 loaded=[r for r in observation.get('loaded_files',[]) if Path(r['path']).name.startswith('librocsolver.so')]
 actual_loaded_verified=len(loaded)==1 and loaded[0].get('sha256')==manifest['files'][next(i for i,r in enumerate(manifest['files']) if r['role']=='librocsolver')]['sha256'] and loaded[0].get('unchanged_since_observation') is True
 log=(OUT/'ctest.log').read_text();executed=bool(re.search(r'\[ RUN\s*\]\s*'+re.escape(CASE)+r'\s*$',log,re.M))
 save('diagnostic-summary.json',{'mode':mode,'producer_run_id':run,'ctest_exit_code':code,'actual_selected_case_started':executed,'actual_loaded_solver_matches_original_producer':actual_loaded_verified,'original_metadata_restored':restored,'observer_failure':observation.get('observer_failure'),'original_stock_outcomes_unchanged':True,'eligible_for_optimization_acceptance':False,'diagnostic_complete':executed and actual_loaded_verified and restored and not observation.get('observer_failure'),'note':'Test failure remains test failure. A passing reduced diagnostic cannot accept optimization or replace failed original stock tests.'})
 if not (executed and actual_loaded_verified and restored) or observation.get('observer_failure'):return 125
 return code
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('qualify','launch'));a=p.parse_args()
 if a.action=='qualify':qualify()
 else:raise SystemExit(launch())
