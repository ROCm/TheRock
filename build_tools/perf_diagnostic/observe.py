"""Sample a direct SDK test child without ptrace, runtime flags, or environment changes.

Sampling can miss transient mappings. Missing observations fail closed; observed
absence never proves an object was never loaded. No perf/correctness acceptance.
"""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
MAX_MAPS=128;MAX_MAP_BYTES=16*1024*1024;MAX_SINGLE_MAP=8*1024*1024;MAX_PATHS=512;MAX_SAMPLES=4000
KEYS=('ASAN_OPTIONS','ASAN_RUNTIME_PATH','ASAN_SYMBOLIZER_PATH','LSAN_OPTIONS','UBSAN_OPTIONS','HSA_XNACK','HSA_OVERRIDE_GFX_VERSION','HSA_ENABLE_SDMA','HSA_ENABLE_DEBUG','HSA_FORCE_FINE_GRAIN_PCIE','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','LD_PRELOAD','LD_LIBRARY_PATH','LD_AUDIT','LD_BIND_NOW','LD_DEBUG','LD_DEBUG_OUTPUT','LD_ASSUME_KERNEL','LD_ORIGIN_PATH','LD_HWCAP_MASK','GLIBC_TUNABLES','ROCM_PATH','GTEST_SHARD_INDEX','GTEST_TOTAL_SHARDS','GTEST_FILTER','ROCM_KPACK_DEBUG','THEROCK_CI_DEBUG','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','KUBE_CPU_REQUEST','GPU_ENABLE_PAL')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def bounded_maps_read(path):
 # Each request is at most 8 MiB; one additional byte detects truncation.
 with path.open('rb') as stream:
  raw=stream.read(MAX_SINGLE_MAP);extra=stream.read(1)
 return raw,bool(extra)

def observe_child(command,sdk_root,output,*,proc_for_pid=None,exe_reader=None):
 output.mkdir(parents=True,exist_ok=False);sdk=sdk_root.resolve();expected=Path(command[0]).resolve();start=time.monotonic();paths={};seen=set();samples=[];child=None;written=0;last_saved=0;dropped=0;maps_disabled=False;read_cap_events=0;maps_skipped=0
 result={'started_utc':utc(),'command':command,'observer':'direct-child /proc sampling, 5ms first 2s then 100ms; no ptrace','environment_unchanged':True,'eligible_for_optimization_acceptance':False,'exec_observations':[],'map_events':[],'maps_coverage_incomplete':False,'maps_read_cap_events':0,'observer_terminated_child':False,
 'caps':{'raw_maps_count':MAX_MAPS,'raw_maps_bytes':MAX_MAP_BYTES,'single_maps_read_bytes':MAX_SINGLE_MAP,'max_read_request_bytes':MAX_SINGLE_MAP,'truncation_detection_extra_bytes':1,'loaded_paths':MAX_PATHS,'sample_timestamps':MAX_SAMPLES},
 'limitations':['Observer overhead uncalibrated; no performance claim.','Sampling can miss transient mappings; absence does not prove never loaded.','No ptrace is used because LeakSanitizer does not support ptrace.','Only allowlisted environment values retained; no inherited environment dump.']}
 try:
  child=subprocess.Popen(command);result['pid']=child.pid;proc=proc_for_pid(child.pid) if proc_for_pid else Path('/proc')/str(child.pid);captured=False
  while child.poll() is None:
   tick=time.monotonic()
   if len(samples)<MAX_SAMPLES:samples.append(tick-start)
   try:
    actual=(exe_reader(proc) if exe_reader else Path(os.readlink(proc/'exe'))).resolve()
    if actual==expected:
     if not captured:
      env=dict(x.split(b'=',1) for x in (proc/'environ').read_bytes().split(b'\0') if b'=' in x)
      cmd=(proc/'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
      if cmd!=command:raise RuntimeError('Actual child argv differs from CTest-resolved command')
      result['exec_observations'].append({'utc':utc(),'exe':str(actual),'argv':cmd,'environment':{k:env[k.encode()].decode(errors='replace') if k.encode() in env else None for k in KEYS}});captured=True
     if maps_disabled:
      maps_skipped+=1
     else:
      raw,oversized=bounded_maps_read(proc/'maps')
      if oversized:
       maps_disabled=True;read_cap_events+=1;dropped+=1
       result['maps_coverage_incomplete']=True;result['maps_read_cap_events']=read_cap_events
       result['maps_read_cap_reason']='snapshot exceeds 8 MiB; further maps reads/parsing disabled without terminating product'
      else:
       digest=hashlib.sha256(raw).hexdigest()
       if digest not in seen:
        if len(seen)<MAX_MAPS and written+len(raw)<=MAX_MAP_BYTES:
         seen.add(digest);name='maps-'+digest[:24]+'.txt';(output/name).write_bytes(raw);written+=len(raw);result['map_events'].append({'utc':utc(),'elapsed_s':tick-start,'path':name,'sha256':digest})
        else:dropped+=1
        for line in raw.decode(errors='replace').splitlines():
         fields=line.split(None,5)
         if len(fields)!=6 or not fields[5].startswith('/'):continue
         path=Path(fields[5])
         if str(path) in paths:continue
         if path.is_file() and ('.so' in path.name or path.resolve().is_relative_to(sdk)) and len(paths)<MAX_PATHS:
          st=path.stat();paths[str(path)]={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'device':st.st_dev}
   except (FileNotFoundError,ProcessLookupError,PermissionError):pass
   if tick-last_saved>=1:
    result['running']=True;result['sampling_partial']={'count':len(samples),'sample_elapsed_s':samples,'maps_bytes':written,'maps_dropped':dropped,'maps_reads_skipped_after_read_cap':maps_skipped};(output/'observation.json').write_text(json.dumps(result,indent=2));last_saved=tick
   time.sleep(.005 if tick-start<2 else .1)
  result['child_exit_code']=child.returncode;result['loaded_files']=[]
  for name,original in sorted(paths.items()):
   path=Path(name);row={'path':name,'within_sdk':path.resolve().is_relative_to(sdk),'observed_stat':original}
   try:
    st=path.stat();current={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'device':st.st_dev};row.update(sha256=sha(path),unchanged_since_observation=current==original)
   except OSError as e:row['unavailable']=type(e).__name__
   result['loaded_files'].append(row)
  if not captured:raise RuntimeError('Actual SDK child environment not sampled; diagnostic inconclusive')
  if not any(Path(r['path']).name.startswith('librocsolver.so') for r in result['loaded_files']):
   if result['maps_coverage_incomplete']:result['loaded_solver_mapping_unavailable_due_coverage']=True
   else:raise RuntimeError('Actual solver DSO mapping not sampled; diagnostic inconclusive')
 except BaseException as error:
  result['observer_failure']=repr(error)
  if child is not None and child.poll() is None:
   if result['maps_coverage_incomplete']:child.wait()
   else:result['observer_terminated_child']=True;child.kill();child.wait()
 finally:
  if child is not None:result['child_exit_code']=child.poll()
  gaps=[b-a for a,b in zip(samples,samples[1:])];result['running']=False;result['sampling']={'count':len(samples),'nominal_startup_interval_s':.005,'startup_window_s':2,'nominal_steady_interval_s':.1,'min_gap_s':min(gaps) if gaps else None,'max_gap_s':max(gaps) if gaps else None,'sample_elapsed_s':samples,'maps_bytes':written,'maps_dropped':dropped,'maps_reads_skipped_after_read_cap':maps_skipped,'sample_timestamps_capped':len(samples)==MAX_SAMPLES}
  result['finished_utc']=utc();result['elapsed_s']=time.monotonic()-start;(output/'observation.json').write_text(json.dumps(result,indent=2))
 return 125 if 'observer_failure' in result or result['maps_coverage_incomplete'] else child.returncode

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--sdk-root',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
 command=a.command[1:] if a.command[:1]==['--'] else a.command
 if sys.platform!='linux' or os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('GITHUB_REPOSITORY')!='ROCm/TheRock':raise RuntimeError('TheRock Linux Actions only')
 if not command or not Path(command[0]).is_absolute() or Path(command[0]).name!='rocsolver-test':raise RuntimeError('Only actual resolved direct rocsolver-test command supported')
 return observe_child(command,a.sdk_root,a.output)

if __name__=='__main__':raise SystemExit(main())
