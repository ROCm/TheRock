"""Affected-project gated retest: unchanged stock quick suite before frozen focused checks."""
import argparse,datetime,hashlib,importlib.util,io,json,os,re,subprocess,sys,time,threading,zipfile
from pathlib import Path
from original_transport_reference import api,pages,fetch
BASE='5b7cb88373362ab80dc0f19605915e534f4158aa';LIBS='f4e24d58b3f313f1a31068def1e35376e43c2f49'
PRODUCER='17bf9b333993b6b12dd9ae867a9c16813cdfca0f';PATCH='b74891bd117138d0e922407bf4cc24c1ee0482f08c685de0c1b36b662cb3aab9'
RUNS={'baseline':'41211496307','joint':'41212843218'}
CTEST_SHA='104d41600b9d6284afdb9787fb2f0c304edf4337f5e0364555d8808164962c26'
CASE='checkin_lapack/GETF2_NPVT.__float/0';OUT=Path('.project-correctness');HERE=Path(__file__).resolve().parent
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
 save('producer-qualification.json',{'source_sha':BASE,'producer_run_id':run,'mode':mode,'producer_instrumentation_sha':PRODUCER,'consumer_run_id':os.environ['GITHUB_RUN_ID'],'consumer_instrumentation_sha':os.environ['GITHUB_SHA'],'original_solver_job':solver[0],'original_failures_unchanged':True,'eligible_for_optimization_acceptance':False,'sidecar_artifact_id':chosen[0]['id'],'sidecar_archive_sha256':hashlib.sha256(raw).hexdigest(),'sidecar_sha256':hashlib.sha256(manifest).hexdigest(),'artifact_route':'s3://therock-ci-artifacts/'+run+'-linux','purpose':'Separate affected-project gated retest: original full quick suite then frozen focused checks only if successful; original full-CI failures remain unchanged; no full-CI optimization acceptance'})
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
 target=[str(exe),*command[1:]];(OUT/'original-CTestTestfile.cmake').write_bytes(original)
 # Add only an observer prefix. Preserve the complete original quick filter and all CTest properties/working directory.
 lines=original.decode().splitlines(keepends=True);indices=[i for i,line in enumerate(lines) if line.startswith('add_test(rocsolver-test_quick_suite ')]
 if len(indices)!=1:raise RuntimeError('Ambiguous quick test definition')
 wrapped=[sys.executable,str(HERE/'observe.py'),'--output',str((OUT/'child-observation').resolve()),'--sdk-root',str(sdk),'--',*target]
 lines[indices[0]]='add_test(rocsolver-test_quick_suite '+' '.join(cmake_arg(s) for s in wrapped)+')\n';modified=''.join(lines).encode()
 save('diagnostic-command.json',{'original_ctest_argv':cmd,'original_test':test,'target_test_command':target,'instrumented_command':wrapped,'changed_product_argument':None,'full_original_filter_unchanged':target[1:]==command[1:],'original_ctest_sha256':CTEST_SHA,'modified_ctest_sha256':hashlib.sha256(modified).hexdigest(),'original_properties_preserved':True,'eligible_for_optimization_acceptance':False})
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
 running=[int(x) for x in re.findall(r'\[==========\] Running (\d+) tests',log)]
 passed=[int(x) for x in re.findall(r'\[\s*PASSED\s*\]\s*(\d+) tests',log)]
 full_count_verified=running==[6528] and passed==[6528] and len(started)==6528 and len(set(started))==6528
 observer_ok=not observation.get('observer_failure') and observation.get('child_exit_code')==0
 gate=code==0 and full_count_verified and actual_loaded_verified and restored and observer_ok
 save('stock-project-gate.json',{'mode':mode,'producer_run_id':run,'consumer_run_id':os.environ['GITHUB_RUN_ID'],'consumer_instrumentation_sha':os.environ['GITHUB_SHA'],'stock_test_runner_source':'unchanged BASE build_tools/github_actions/test_executable_scripts/test_runner.py','ctest_exit_code':code,'original_filter_unchanged':target[1:]==command[1:],'expected_full_case_count':6528,'running_counts':running,'passed_counts':passed,'started_cases':len(started),'unique_started_cases':len(set(started)),'actual_loaded_solver_matches_original_producer':actual_loaded_verified,'original_metadata_restored':restored,'observer_failure':observation.get('observer_failure'),'stock_project_gate_passed':gate,'original_stock_outcomes_unchanged':True,'eligible_for_full_CI_optimization_acceptance':False,'note':'This is a separately instrumented per-project retest, never a replacement for original global stock outcomes. Focused checks must be skipped unless this whole gate passes.'})
 if not gate:return code if code else 125
 with open(os.environ['GITHUB_OUTPUT'],'a') as stream:stream.write('stock_gate_passed=true\n')
 return 0

def focused():
 mode,run=guard();gate=json.loads((OUT/'stock-project-gate.json').read_text())
 if gate.get('stock_project_gate_passed') is not True or gate['mode']!=mode or gate['producer_run_id']!=run or gate.get('consumer_run_id')!=os.environ['GITHUB_RUN_ID'] or gate.get('consumer_instrumentation_sha')!=os.environ['GITHUB_SHA']:raise RuntimeError('Full original stock project gate did not pass; focused checks forbidden')
 helper=HERE/'run_focused.py';expected='a10039fbbf6fc79b43315b49ee1d31ea2b165c3892c073796f535771b1b6c75e'
 if digest(helper)!=expected:raise RuntimeError('Focused helper differs from exact frozen full-run helper')
 if os.environ.get('LD_PRELOAD')!=os.environ.get('ASAN_RUNTIME_PATH') or not os.environ.get('LD_PRELOAD'):raise RuntimeError('Expected original extra-focused-only ASan preload scope')
 if os.environ.get('ASAN_OPTIONS')!='detect_odr_violation=0:quarantine_size_mb=600:verify_asan_link_order=0' or os.environ.get('HSA_XNACK')!='1':raise RuntimeError('Original focused ASan/xnack environment differs')
 from observe import KEYS
 sdk=Path('build').resolve();manifest=json.loads((OUT/'solver-sdk-identity.json').read_text());solver=next(r for r in manifest['files'] if r['role']=='librocsolver')
 command=[sys.executable,str(helper),'--sdk-root',str(sdk),'--source-root',str(Path.cwd()),'--sdk-manifest',str((OUT/'solver-sdk-identity.json').resolve()),'--output',str((OUT/'focused-numerics').resolve()),'--budget-s','7200','--mode',mode,'--producer-run-id',run,'--expected-instrumentation-sha',PRODUCER]
 observation={'purpose':'Separate focused-check child startup/DSO sampling; no full-CI acceptance','frozen_helper_sha256':expected,'command':command,'stock_filter_or_ASan_settings_changed':False,'focused_only_preload_matches_original_full_workflow':True,'focused_launch_environment':{k:os.environ.get(k) for k in KEYS},'roles':{},'eligible_for_full_CI_optimization_acceptance':False,'limitations':['Only the first observed bench and final GTest child are sampled, not all 1963 bench processes.','Mapping absence between samples is not a never-loaded proof.','Sampling and hashing overhead uncalibrated; no performance claim.'],'caps':{'single_map_bytes':1024*1024,'map_count':2,'total_map_bytes':2*1024*1024,'descendant_scan':32,'sampling_gap_records':40000}}
 gaps=[];last=None;start=time.monotonic();paths={};last_saved=0
 def descendants(pid):
  queue=[pid];seen=set()
  while queue and len(seen)<32:
   value=queue.pop()
   if value in seen:continue
   seen.add(value)
   try:queue.extend(int(x) for x in Path(f'/proc/{value}/task/{value}/children').read_text().split())
   except (FileNotFoundError,ProcessLookupError,PermissionError):pass
  return seen-{pid}
 with (OUT/'focused-wrapper.log').open('w') as log:
  proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  def copy_output():
   for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
  reader=threading.Thread(target=copy_output,daemon=True);reader.start()
  while proc.poll() is None:
   tick=time.monotonic()
   if last is not None and len(gaps)<40000:gaps.append(tick-last)
   last=tick
   for pid in descendants(proc.pid):
    try:
     process=Path('/proc')/str(pid);exe=Path(os.readlink(process/'exe')).resolve();role=exe.name
     if role not in ('rocsolver-bench','rocsolver-test') or role in observation['roles'] or not exe.is_relative_to(sdk):continue
     with (process/'maps').open('rb') as f:raw=f.read(1024*1024+1)
     if len(raw)>1024*1024:
      observation['sampling_failure']='Focused child map exceeds explicit cap'
      for owned in descendants(proc.pid):
       try:os.kill(owned,9)
       except ProcessLookupError:pass
      proc.terminate();break
     if b'librocsolver.so' not in raw:continue
     env=dict(x.split(b'=',1) for x in (process/'environ').read_bytes().split(b'\0') if b'=' in x)
     path=OUT/('focused-'+role+'-maps.txt');path.write_bytes(raw);mapped=[]
     for line in raw.decode(errors='replace').splitlines():
      fields=line.split(None,5)
      if len(fields)!=6 or not fields[5].startswith('/'):continue
      value=Path(fields[5])
      if value.is_file() and '.so' in value.name and (str(value) in paths or len(paths)<512):
       st=value.stat();paths[str(value)]={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'device':st.st_dev};mapped.append(str(value))
     observation['roles'][role]={'pid':pid,'exe':str(exe),'argv':(process/'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0'),'elapsed_s':tick-start,'environment':{k:env[k.encode()].decode(errors='replace') if k.encode() in env else None for k in KEYS},'maps':path.name,'maps_sha256':hashlib.sha256(raw).hexdigest(),'mapped_DSO_paths':sorted(set(mapped))}
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
   if tick-last_saved>=1:
    observation['running']=True;(OUT/'focused-observation.json').write_text(json.dumps(observation,indent=2));last_saved=tick
   time.sleep(.005 if 'rocsolver-bench' not in observation['roles'] else .1)
  code=proc.wait();reader.join(timeout=5)
 observation['loaded_files']=[]
 for pathname,original in paths.items():
  path=Path(pathname);st=path.stat();current={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'device':st.st_dev};observation['loaded_files'].append({'path':pathname,'sha256':digest(path),'unchanged_since_observation':current==original})
 mapped={r['path']:r for r in observation['loaded_files']};loaded_ok=set(observation['roles'])=={'rocsolver-bench','rocsolver-test'}
 for row in observation['roles'].values():
  candidates=[mapped[p] for p in row['mapped_DSO_paths'] if Path(p).name.startswith('librocsolver.so')]
  row['actual_solver_matches_original_producer']=len(candidates)==1 and candidates[0]['sha256']==solver['sha256'] and candidates[0]['unchanged_since_observation']
  loaded_ok=loaded_ok and row['actual_solver_matches_original_producer']
 result_path=OUT/'focused-numerics'/'summary.json';result=json.loads(result_path.read_text()) if result_path.exists() else {}
 passed=code==0 and not observation.get('sampling_failure') and result.get('passed') is True and result.get('completed_bench_cases')==1963 and result.get('gtest_completed_cases',0)>0 and loaded_ok
 observation.update(running=False,exit_code=code,elapsed_s=time.monotonic()-start,sampling={'count':len(gaps),'min_gap_s':min(gaps) if gaps else None,'max_gap_s':max(gaps) if gaps else None,'gaps_s':gaps,'gap_records_capped':len(gaps)==40000},actual_loaded_solver_verified=loaded_ok,focused_checks_passed=passed)
 (OUT/'focused-observation.json').write_text(json.dumps(observation,indent=2))
 return 0 if passed else (code if code else 125)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('qualify','launch','focused'));a=p.parse_args()
 if a.action=='qualify':qualify()
 elif a.action=='launch':raise SystemExit(launch())
 else:raise SystemExit(focused())
