# Normal CI input capture

`normal_ci_validation.py capture` uses only Python's standard library. It does
not require the process profiler or install packages in the wheel test venv.
The profiler and capture share the same visible cgroup-v2 hierarchy resolver.

SDK receipts retain nested component `dist` views and the installed component
view under `build/bin`, `build/llvm`, `build/lib/llvm`, `build/lib` and
`build/lib64`. Installed scans select named compiler/debugger/helper executables,
LLVM device bitcode and compiler/RTC/HSA shared libraries. They do not hash
unrelated BLAS, RCCL or other math-library payloads. The pinned rocGDB test source
declares its LLVM lookup roots; the retained make argv proves the specific
`build/bin/rocgdb` selection. The fixture records this distinction.

`complete` describes only that selection. Missing roots and tools are explicit;
an empty complete scan does not establish SDK or provider identity. Entry and
logical-byte limits, scan errors and reached limits are retained. Alias rows
retain their link targets while the capture fingerprint cache reuses resolved
file hashes. `capture_wall_s` and `new_unique_fingerprint_inputs` measure the SDK
scan separately. Logical bytes include aliases and are not physical I/O bytes.

Loader queries describe the capture environment. Test-local library-path changes,
later `dlopen` selection, runtime Python support and whole-archive completeness
remain outside that claim. No source, product, cache or test policy is changed.

Regression fixtures run the real guarded capture with `python -I -S`, proving
`psutil` is unavailable, and exercise both SDK layouts, missing inputs, limits
and exclusion of unrelated payloads without executing a compiler or ROCm binary.
