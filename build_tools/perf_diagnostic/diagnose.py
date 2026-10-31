"""Same-allocation ABBA filter diagnostic against one unchanged original joint SDK."""
import argparse,datetime,hashlib,importlib.util,io,json,os,re,subprocess,sys,time,zipfile
from pathlib import Path
from original_transport_reference import api,pages,fetch
BASE='5b7cb88373362ab80dc0f19605915e534f4158aa';LIBS='f4e24d58b3f313f1a31068def1e35376e43c2f49'
PRODUCER='17bf9b333993b6b12dd9ae867a9c16813cdfca0f';PATCH='b74891bd117138d0e922407bf4cc24c1ee0482f08c685de0c1b36b662cb3aab9'
RUNS={'joint':'41212843218'}
CTEST_SHA='104d41600b9d6284afdb9787fb2f0c304edf4337f5e0364555d8808164962c26'
CASE='checkin_lapack/GETF2_NPVT.__float/0';OUT=Path('.runtime-filter-control');HERE=Path(__file__).resolve().parent
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
 save('producer-qualification.json',{'source_sha':BASE,'producer_run_id':run,'mode':mode,'producer_instrumentation_sha':PRODUCER,'consumer_run_id':os.environ['GITHUB_RUN_ID'],'consumer_instrumentation_sha':os.environ['GITHUB_SHA'],'original_solver_job':solver[0],'original_failures_unchanged':True,'eligible_for_optimization_acceptance':False,'sidecar_artifact_id':chosen[0]['id'],'sidecar_archive_sha256':hashlib.sha256(raw).hexdigest(),'sidecar_sha256':hashlib.sha256(manifest).hexdigest(),'artifact_route':'s3://therock-ci-artifacts/'+run+'-linux','purpose':'DIAGNOSTIC ONLY same-allocation ABBA filter control, one original joint SDK; no focused or full-CI acceptance'})
def cmake_arg(value):
 if any(x in value for x in ('\n','\r',']]')):raise RuntimeError('Unsupported CMake argument')
 return '[['+value+']]'
def launch(filter_mode):
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
 target=[str(exe),*command[1:]] if filter_mode=='full' else [str(exe),'--gtest_filter='+CASE];(OUT/'original-CTestTestfile.cmake').write_bytes(original)
 # Add only an observer prefix. Preserve the complete original quick filter and all CTest properties/working directory.
 lines=original.decode().splitlines(keepends=True);indices=[i for i,line in enumerate(lines) if line.startswith('add_test(rocsolver-test_quick_suite ')]
 if len(indices)!=1:raise RuntimeError('Ambiguous quick test definition')
 wrapped=[sys.executable,str(HERE/'observe.py'),'--output',str((OUT/'child-observation').resolve()),'--sdk-root',str(sdk),'--',*target]
 lines[indices[0]]='add_test(rocsolver-test_quick_suite '+' '.join(cmake_arg(s) for s in wrapped)+')\n';modified=''.join(lines).encode()
 save('diagnostic-command.json',{'original_ctest_argv':cmd,'original_test':test,'target_test_command':target,'instrumented_command':wrapped,'changed_product_argument':None if filter_mode=='full' else 'gtest_filter only','filter_mode':filter_mode,'full_original_filter_unchanged':target[1:]==command[1:],'original_ctest_sha256':CTEST_SHA,'modified_ctest_sha256':hashlib.sha256(modified).hexdigest(),'original_properties_preserved':True,'eligible_for_optimization_acceptance':False})
 (OUT/'instrumented-CTestTestfile.cmake').write_bytes(modified)
 code=None
 try:
  metadata.write_bytes(modified)
  with (OUT/'ctest.log').open('w') as log:
   proc=subprocess.Popen([sys.executable,str(Path('build_tools/github_actions/test_executable_scripts/test_runner.py').resolve())],cwd=runner.THEROCK_DIR,env=os.environ.copy(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
   code=proc.wait()
 finally:metadata.write_bytes(original)
 restored=digest(metadata)==CTEST_SHA
 observation_path=OUT/'child-observation'/'observation.json';observation=json.loads(observation_path.read_text()) if observation_path.exists() else {}
 loaded=[r for r in observation.get('loaded_files',[]) if Path(r['path']).name.startswith('librocsolver.so')]
 actual_loaded_verified=len(loaded)==1 and loaded[0].get('sha256')==manifest['files'][next(i for i,r in enumerate(manifest['files']) if r['role']=='librocsolver')]['sha256'] and loaded[0].get('unchanged_since_observation') is True
 log=(OUT/'ctest.log').read_text();started=re.findall(r'\[ RUN\s*\]\s*(\S+)',log)
 running=[int(x) for x in re.findall(r'\[==========\] Running (\d+) tests?' ,log)]
 passed=[int(x) for x in re.findall(r'\[\s*PASSED\s*\]\s*(\d+) tests?' ,log)]
 expected=6528 if filter_mode=='full' else 1
 # GTest uses singular test for the one-case summary.
 full_count_verified=running==[expected] and passed==[expected] and len(started)==expected and len(set(started))==expected
 observer_ok=not observation.get('observer_failure') and observation.get('child_exit_code')==0
 gate=code==0 and full_count_verified and actual_loaded_verified and restored and observer_ok
 save('trial-result.json',{'mode':mode,'producer_run_id':run,'consumer_run_id':os.environ['GITHUB_RUN_ID'],'consumer_instrumentation_sha':os.environ['GITHUB_SHA'],'stock_test_runner_source':'unchanged BASE build_tools/github_actions/test_executable_scripts/test_runner.py','ctest_exit_code':code,'original_filter_unchanged':target[1:]==command[1:],'filter_mode':filter_mode,'expected_case_count':expected,'running_counts':running,'passed_counts':passed,'started_cases':len(started),'unique_started_cases':len(set(started)),'actual_loaded_solver_matches_original_producer':actual_loaded_verified,'original_metadata_restored':restored,'observer_failure':observation.get('observer_failure'),'diagnostic_trial_passed':gate,'original_stock_outcomes_unchanged':True,'eligible_for_full_CI_optimization_acceptance':False,'note':'Diagnostic trial only; original stock outcomes and skipped focused checks remain unchanged. All four ABBA children share one SDK/allocation; only the product filter differs.'})
 if not gate:return code if code else 125
 return 0

def series():
 import resource
 global OUT
 mode,run=guard();base=OUT;identity=(base/'solver-sdk-identity.json').read_bytes();qualification=(base/'producer-qualification.json').read_bytes()
 if mode!='joint' or run!='41212843218':raise RuntimeError('Only unchanged original joint producer permitted')
 plan=('single','full','full','single');rows=[];started=datetime.datetime.now(datetime.timezone.utc).isoformat()
 for number,filter_mode in enumerate(plan,1):
  OUT=base/f'trial-{number}-{filter_mode}';OUT.mkdir(exist_ok=False);(OUT/'solver-sdk-identity.json').write_bytes(identity);(OUT/'producer-qualification.json').write_bytes(qualification)
  begin=time.monotonic();self_begin=time.process_time();cpu_begin=resource.getrusage(resource.RUSAGE_CHILDREN);row={'index':number,'filter_mode':filter_mode,'expected_case_count':1 if filter_mode=='single' else 6528,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  print(f'ABBA diagnostic trial {number}/4 {filter_mode}: original joint SDK, unchanged stock ASan/XNACK',flush=True)
  try:
   row['launcher_exit_code']=launch(filter_mode)
   row['trial_result']=json.loads((OUT/'trial-result.json').read_text())
  except Exception as error:
   row['setup_or_instrumentation_error']=repr(error);row['launcher_exit_code']=125
  finally:
   cpu_end=resource.getrusage(resource.RUSAGE_CHILDREN);row.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_s=time.monotonic()-begin,controller_CPU_s=time.process_time()-self_begin,aggregate_descendant_user_CPU_s=cpu_end.ru_utime-cpu_begin.ru_utime,aggregate_descendant_system_CPU_s=cpu_end.ru_stime-cpu_begin.ru_stime,CPU_scope='Descendant aggregate includes BASE runner, CTest, observer and test; observer overhead is not isolated or calibrated')
   rows.append(row);OUT=base
   save('series-progress.json',{'started_utc':started,'planned_order':plan,'trials':rows,'eligible_for_optimization_acceptance':False})
  print(f'ABBA diagnostic trial {number}/4 {filter_mode}: exit {row["launcher_exit_code"]}',flush=True)
  # Real test failures continue, while unsupported source/provenance prevents
  # inventing a complete four-process experiment from unqualified inputs.
  if 'setup_or_instrumentation_error' in row:break
 completed=len(rows)==4 and not any('setup_or_instrumentation_error' in r for r in rows);all_passed=completed and all(r['launcher_exit_code']==0 for r in rows)
 environments=[]
 for number,filter_mode in enumerate(plan,1):
  path=base/f'trial-{number}-{filter_mode}'/'child-observation/observation.json'
  if path.exists():
   observation=json.loads(path.read_text());execs=observation.get('exec_observations',[]);environments.append({'trial':number,'actual_child_environment':execs[0].get('environment') if execs else None})
 observed=[r['actual_child_environment'] for r in environments];same_environment=len(observed)==4 and observed[0] is not None and all(e==observed[0] for e in observed)
 save('series-summary.json',{'source_sha':BASE,'mode':mode,'producer_run_id':run,'producer_instrumentation_sha':PRODUCER,'consumer_run_id':os.environ['GITHUB_RUN_ID'],'consumer_instrumentation_sha':os.environ['GITHUB_SHA'],'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'planned_order':plan,'completed_four_trials':completed,'all_four_trials_passed':all_passed,'same_recorded_actual_child_environment':same_environment,'actual_environments':environments,'trials':rows,'original_stock_outcomes_unchanged':True,'original_focused_checks_stay_unvalidated':True,'eligible_for_optimization_acceptance':False,'interpretation':'Diagnostic ABBA series in one allocation and installed SDK; only filter changes. Outcomes cannot establish candidate attribution or full-CI correctness. Sequential device/cache state and uncalibrated observation overhead remain limitations.'})
 return 0 if all_passed and same_environment else 1

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('qualify','series'));a=p.parse_args()
 if a.action=='qualify':qualify()
 else:raise SystemExit(series())

