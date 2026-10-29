#!/usr/bin/env python3
import ctypes,json,os,pathlib,subprocess,sys
sdk=pathlib.Path(os.environ['ROCM_PATH']);lib=ctypes.CDLL(str(sdk/'lib/libamdhip64.so'));lib.hipGetDeviceCount.argtypes=[ctypes.POINTER(ctypes.c_int)];lib.hipGetDeviceCount.restype=ctypes.c_int
count=ctypes.c_int();rc=lib.hipGetDeviceCount(ctypes.byref(count))
if rc or count.value<2:raise SystemExit('Full compiler DASan validation needs multiple visible GPUs; status='+str(rc)+' count='+str(count.value))
bundle=pathlib.Path('evidence/compiler-validation/dasan-lit-bundle')
subprocess.run([sys.executable,'test_tools/dasan_smoke.py','--rocm-path',str(sdk),'--llvm-lit',str(bundle/'llvm-lit.py'),'--lit-suite',str(bundle/'suite'),'--gpu-arch','gfx950','--gpu-device-count',str(count.value),'--openmp','--require-multi-device','--output-dir','evidence/lit-results'],check=True)
