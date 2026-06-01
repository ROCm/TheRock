# 💥 Runtime Error Triage Checklist
## Specialized Guide for Runtime Failures (Segfaults, Kernel Errors, Crashes)

> **When to use:** Segmentation faults, GPU kernel failures, assertion errors, crashes during execution
>
> **Based on:** Analysis of runtime failures including flaky multi-node issues and kernel selection problems

---

## 🚦 Pre-Triage: Is This Really a Runtime Error?

### Quick Classification

**Runtime errors occur AFTER successful compilation:**
- ✅ Code compiled successfully
- ✅ Binary/library exists
- ❌ Crashes/errors during execution

**NOT runtime errors (different checklist):**
- Build failures → Use `checklist_build_failures.md`
- CMake configuration issues → Use `checklist_build_failures.md`
- Import errors (Python) → May be build or environment issue

---

## 🎯 Quick Triage Flowchart

**Start here to determine your path:**

```
1. Can you reproduce the error?
   ├─ YES, 100% of the time → Go to Section 3 (Versions)
   └─ NO, it's flaky/random → Go to Section 2 (Flaky Errors)

2. Do you have a last known good build/commit?
   ├─ YES → Go to Section 4 (Regression Analysis) FIRST
   └─ NO → Continue to Section 3

3. Is this a recent issue or long-standing?
   ├─ Recent (last 2 weeks) → Section 4 (Regression)
   └─ Long-standing → Section 3 (Versions) then Section 5 (Debug)

4. Ready to file issue?
   → Go to Section 7 (issue Requirements Summary)
```

---

## 📋 Section 1: Error Type Classification


- **Segmentation Fault (SIGSEGV)**
  ```
  Example: "Segmentation fault (core dumped)"
  Example: "Fatal Python error: Segmentation fault"
  ```

- **GPU Kernel Error**
  ```
  Example: "hipErrorLaunchFailure: Unspecified launch failure"
  Example: "HSA_STATUS_ERROR_MEMORY_FAULT"
  ```

- **Assertion Failure**
  ```
  Example: "Assertion `x != nullptr' failed"
  ```

- **Memory Error**
  ```
  Example: "Out of memory"
  Example: "hipErrorMemoryAllocation"
  ```

- **Numeric/Accuracy Error**
  ```
  Example: "NaN detected in output"
  ```

**Full stack trace:** (required)


**Key information:**
- Component: `_________________` (hipblasLt, Tensile, rocBLAS, etc.)
- Function: `_________________`
- Kernel name (if GPU): `_________________`

**Common stack trace patterns:**

| Pattern | Likely Cause | Route To |
|---------|--------------|----------|
| `hipblasLtMatmulAlgoGetHeuristic` | Kernel selection issue | Math team |
| `TensileHost::solutionSelection` | Tensile library issue | Math team |
| `rcclAllReduce` | Communication layer | RCCL team |
| `hipMalloc` / `hipMemcpy` | Memory allocation | ROCm Runtime |

---

## 📋 Section 2: Flaky/Intermittent Errors (If Applicable)

### 2.1 Reproducibility Assessment

**If error is NOT 100% reproducible:**

**Reproduction rate:** _____%

**Test at least 20 times to establish baseline:**
```bash
#!/bin/bash
TOTAL=20
FAILED=0
for i in $(seq 1 $TOTAL); do
  ./run_test.sh > run_$i.log 2>&1 || ((FAILED++))
done
echo "Failure rate: $((FAILED * 100 / TOTAL))%"
```

**Minimum to file issue:**
- Failure rate >10% (if <10%, too unreliable to debug)
- At least 3 failure logs captured
- At least 3 success logs captured (for comparison)

### 2.2 Scale-Dependent Flakiness

**If failure rate increases with scale, complete this:**

| Configuration | Runs | Failures | Rate |
|---------------|------|----------|------|
| 1 node, 1 GPU | 20 | ___ | ___% |
| 1 node, 8 GPUs | 20 | ___ | ___% |
| 4 nodes | 20 | ___ | ___% |
| 16 nodes | 10 | ___ | ___% |
| 32+ nodes | 10 | ___ | ___% |

**Pattern indicates:**
- Rate increases with nodes → Race condition/communication issue
- Rate increases with GPUs → GPU synchronization issue
- Rate constant → Timing-independent bug

**When does it fail:**
- During initialization
- During compilation (JIT/JAX)
- During kernel execution
- Random timing

---

## 📋 Section 3: Version & Environment (MANDATORY)

### 3.1 Multiple Installation Check (DO THIS FIRST)

**⚠️ Check for conflicting installations BEFORE checking versions**

**Search entire system for installations:**

```bash
# Search everywhere for hipBLASLt
find / -name "libhipblaslt.so*" 2>/dev/null
find / -name "hipblaslt-version.h" 2>/dev/null

# Search for Tensile
find / -name "libtensile.so*" 2>/dev/null
find / -name "Tensile" -type d 2>/dev/null | head -20

# Search for rocBLAS
find / -name "librocblas.so*" 2>/dev/null
```

**Check which libraries are actually being loaded:**

```bash
# For Python workloads:
ldd $(which python3) | grep -E "rocm|hip|hsa|hipblaslt|rocblas|tensile"

# For hipblaslt-bench:
ldd /opt/rocm/bin/hipblaslt-bench | grep -E "hipblaslt|rocblas|tensile"

# For other binaries (adjust path as needed):
ldd /path/to/your/binary | grep -E "hipblaslt|rocblas|tensile"
```

**Multiple installations found?** Yes No

**If multiple installations found:**
- **DO NOT PROCEED** - Report findings in issue
- List all locations found
- Document which libraries are being loaded (from ldd output)
- Let Math team determine resolution

**Required in issue if multiple found:**
```
Locations found:
- /opt/rocm-6.0.0/lib/libhipblaslt.so
- /opt/rocm-7.0.2/lib/libhipblaslt.so
- /home/user/local/lib/libhipblaslt.so

Libraries being loaded (from ldd):
[paste ldd output]
```

### 3.2 ROCm & Library Versions (Only After Confirming Single Installation)

**⚠️ Only run this section if Section 3.1 confirmed single clean installation**

**ROCm Version:**
```bash
cat /opt/rocm/.info/version
# Output: _______
```

**hipBLASLt Version from package:**
```bash
dpkg -l | grep hipblaslt     # Ubuntu/Debian
rpm -qa | grep hipblaslt     # RHEL/CentOS
# Output: _______
```

**hipBLASLt Version from header:**
```bash
cat /opt/rocm/include/hipblaslt/hipblaslt-version.h | grep HIPBLASLT_VERSION

# Output:
#define HIPBLASLT_VERSION_MAJOR _______
#define HIPBLASLT_VERSION_MINOR _______
#define HIPBLASLT_VERSION_PATCH _______
#define HIPBLASLT_VERSION_TWEAK _______
```

**If building from source:**
```bash
cd /workspace/rocm-libraries
git log -1 --format="Commit: %H%nDate: %ci"
# Output: _______
```

### 3.3 Related Libraries - Symbol Conflicts Check

**Check for duplicate symbols and ABI breaks:**

```bash
# Check rocBLAS for symbol conflicts
nm -D /opt/rocm/lib/librocblas.so | grep -E "hipblasLt|Tensile" > rocblas_symbols.txt

# Check hipSPARSELt for symbol conflicts
nm -D /opt/rocm/lib/libhipsparselt.so 2>/dev/null | grep -E "hipblasLt|Tensile" > hipsparselt_symbols.txt

# Check for ODR violations (One Definition Rule)
nm -D /opt/rocm/lib/lib*.so | grep " T " | sort | uniq -d > duplicate_symbols.txt

# Check ABI/SONAME compatibility
readelf -d /opt/rocm/lib/libhipblaslt.so | grep SONAME
readelf -d /opt/rocm/lib/librocblas.so | grep SONAME
readelf -d /opt/rocm/lib/libhipsparselt.so 2>/dev/null | grep SONAME
```

**Symbol conflicts or ODR violations found?** Yes No

**If yes, attach to issue:**
- rocblas_symbols.txt
- hipsparselt_symbols.txt
- duplicate_symbols.txt
- Output of all readelf commands

### 3.4 Hardware & System

- GPU Model: `_______` (MI300X, MI355X, MI325X)
- GFX IP: `_______` (gfx942, gfx950)
- Nodes: `___`, GPUs per node: `___`, Total: `___`
- OS: `_______` (Ubuntu 22.04, RHEL 9)
- Framework: `_______` version `_______` (if applicable)

### 3.5 Machine Access (MANDATORY)

**Provide ONE of:**

**Docker (preferred):**
```bash
docker pull <registry>/<image>:<tag>
docker run -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE <image> /bin/bash
```

**Bare Metal:**
- Machine: `_______`, Reservation: `_______`, Valid until: `_______`

---

## 📋 Section 4: Regression Analysis (DO THIS FIRST IF AVAILABLE)

**⚠️ If you know when it last worked, START HERE**

### 4.1 Bisection Information

**Last Known Good:**
- Build/Commit: `_______`
- Date: `_______`

**First Known Bad:**
- Build/Commit: `_______`
- Date: `_______`

**This is THE most valuable information for debugging**

### 4.2 Comparative Testing

**Run BOTH versions with debug flags and compare:**

```bash
# Last Good
git checkout <good-commit>
./install.sh -idc
export TENSILE_DB=0x8040
export HIPBLASLT_LOG_LEVEL=4
./run_test.sh > good_run.log 2>&1

# First Bad
git checkout <bad-commit>
./install.sh -idc
export TENSILE_DB=0x8040
export HIPBLASLT_LOG_LEVEL=4
./run_test.sh > bad_run.log 2>&1

# Compare kernel selections
diff <(grep -E "hipblasLt|Tensile" good_run.log) \
     <(grep -E "hipblasLt|Tensile" bad_run.log) > diff.txt
```

**Attach to issue:**
- good_run.log
- bad_run.log
- diff.txt

**Git Bisect (if building from source):**
```bash
git bisect start
git bisect bad <bad-commit>
git bisect good <good-commit>
# Test each checkout
git bisect log > bisect.log  # Attach this
```

### 4.3 Environment Comparison

| Aspect | Last Good | First Bad |
|--------|-----------|-----------|
| ROCm | _______ | _______ |
| hipBLASLt | _______ | _______ |
| Tensile | _______ | _______ |
| Framework | _______ | _______ |

---

## 📋 Section 5: Debug Instrumentation (MANDATORY)

### 5.1 Enable Debug Logging

**Set these environment variables BEFORE running test:**

```bash
# MANDATORY: Kernel selection logging
export TENSILE_DB=0x8040

# MANDATORY: hipBLASLt verbose logging
export HIPBLASLT_LOG_LEVEL=4
export HIPBLASLT_LOG_MASK=0xFFFFFFFF

# Helpful additional flags:
export AMD_LOG_LEVEL=4              # HIP runtime
export HIP_VISIBLE_DEVICES=0        # Single GPU test
```

**Run test and capture output:**
```bash
./run_test.sh > debug_run.log 2>&1
# OR
python3 train.py > debug_run.log 2>&1
```

**Required:** Attach debug_run.log to issue

### 5.2 hipblaslt-bench Reproduction (For GEMM Issues)

**Can you reproduce with hipblaslt-bench?**

```bash
hipblaslt-bench -f gemm_strided_batched \
  -m <M> -n <N> -k <K> \
  --batch_count <batch> \
  -r <precision> \
  -i <iterations>

# Example:
hipblaslt-bench -f gemm_strided_batched \
  -m 4096 -n 4096 -k 4096 \
  --batch_count 8 -r f16_r -i 100
```

**Result:** Reproduces Does not reproduce Not tested

**If reproduces:** Attach exact command to issue

### 5.3 Core Dump & Backtrace (For Segfaults)

```bash
# Enable core dumps
ulimit -c unlimited

# Run test (will create core file on crash)
./run_test.sh

# Analyze with gdb
gdb /path/to/binary core.xxx
(gdb) bt full
(gdb) info threads
```

**Attach to issue:**
- Core dump: `http://______`
- GDB backtrace (paste in issue)

### 5.4 Sanitizer Testing (Optional but Helpful)

```bash
# Rebuild with Address Sanitizer
./install.sh -c --address-sanitizer

# Run test
export ASAN_OPTIONS=detect_leaks=0:log_path=asan.log
./run_test.sh
```

**If ASAN detects issues:** Attach asan.log

---

## 📋 Section 6: Multi-Node Specific Debugging

**Only complete if failure is multi-node specific**

### 6.1 Scale Testing Results

| Config | Result |
|--------|--------|
| 1 node | Pass Fail |
| 2 nodes | Pass Fail |
| 4 nodes | Pass Fail |
| 16 nodes | Pass Fail |

**Pattern:** Fails only above ___ nodes

---

## 📋 Section 7: issue Requirements Summary

### 7.1 MANDATORY Information Checklist

**Before filing issue, ensure you have:**

**Core Information:**
- Error type and stack trace
- Reproducibility (deterministic or flaky with %)
- ROCm version: `cat /opt/rocm/.info/version`
- hipBLASLt version: `dpkg -l | grep hipblaslt`
- rocm-libraries commit (if source build): `git log -1 --format="%H"`

**Debug Logs:**
- Ran with `TENSILE_DB=0x8040` and `HIPBLASLT_LOG_LEVEL=4`
- Full execution log attached: `http://______`
- If regression: Both good and bad logs attached

**Environment:**
- Docker image: `docker pull <image>`
- OR Bare metal: Machine details

**Regression Info (if available):**
- Last good build/commit: `___`
- First bad build/commit: `___`
- Logs from both versions
- Kernel selection comparison

**For Segfaults:**
- Core dump available
- GDB backtrace attached

**For Flaky Errors:**
- Tested 20+ times
- Failure rate >10%
- Multiple failure logs
- Multiple success logs (for comparison)

**For GEMM Issues:**
- hipblaslt-bench command (if reproducible)
- Kernel selection logs

### 7.2 Complete issue Template

```markdown
**Title:** [Component][Error Type][GPU] Brief description

**ERROR TYPE:** [Segfault/GPU Kernel/Assertion/Memory/Other]

**REPRODUCIBILITY:**
[Deterministic 100%] OR [Flaky ___% - increases with scale]

**OBSERVED:**
[Stack trace]

**EXPECTED:**
[What should happen]

**IMPACT:**
[Who/what is blocked]

**VERSIONS:**
- ROCm: ___ (cat /opt/rocm/.info/version)
- hipBLASLt: ___ (dpkg -l | grep hipblaslt)
- rocm-libraries commit: ___ (if source build)
- Docker image: ___ (if using Docker)
- GPU: ___ (MI300X/MI355X/etc), GFX: ___
- Nodes/GPUs: ___ nodes, ___ GPUs/node
- Framework: ___ version ___

**MULTIPLE INSTALLATIONS CHECK:**
find /opt -name "rocm*" -type d → ___
# Multiple found: Yes/No
# If yes, which used: ___

**REGRESSION INFO:**
- Last known good: ___ (date: ___)
- First known bad: ___ (date: ___)
- Suspect change: ___
- Bisection: [Done/Not done]

**DEBUG LOGS (with TENSILE_DB=0x8040):**
- Good case (if regression): http://___
- Failing case: http://___
- Kernel comparison: http://___

**GEMM REPRODUCTION:**
hipblaslt-bench -f gemm_strided_batched \
  -m ___ -n ___ -k ___ \
  --batch_count ___ -r ___ -i ___
# Result: [Reproduces/Does not/Not tested]

**CORE DUMP (if segfault):**
- Location: http://___
- GDB backtrace:
  [Paste backtrace]

**Environment:**
Docker: `docker pull <image>`
OR Bare metal: Machine details

**FLAKY ERROR DATA (if applicable):**
| Config | Runs | Failures | Rate |
|--------|------|----------|------|
| 1 node | ___ | ___ | ___% |
| 4 nodes | ___ | ___ | ___% |
| 16 nodes | ___ | ___ | ___% |

**ADDITIONAL CONTEXT:**
[Workarounds attempted, hypotheses, etc.]
```
---

## 📋 Section 8: Triage Quick Start

### 8.1 Routing Decision Tree

```
START: Runtime error occurred

├─ Stack trace has hipblaslt/tensile/rocblas?
│  ├─ YES → Continue
│  └─ NO → Not Math library issue
│
├─ Can reproduce with hipblaslt-bench?
│  ├─ YES → Math team
│  └─ NO → Check next
│
├─ Error during kernel selection (check with TENSILE_DB=0x8040)?
│  ├─ YES → Math team
│  └─ NO → Check next
│
├─ Error in framework (JAX/PyTorch) code?
│  ├─ YES → Framework team (but attach hipBLASLt logs)
│  └─ NO → Math team (if mentions our libraries)
│
└─ Still unclear?
   → File with Math team, include ALL debug info
```

### 8.1 Routing Guide

```
═══════════════════════════════════════════════════════════════
              RUNTIME ERROR TRIAGE QUICK GUIDE
═══════════════════════════════════════════════════════════════

BEFORE FILING issue:
☑ Capture full stack trace
☑ Get ROCm version: cat /opt/rocm/.info/version
☑ Get commit: git log -1 --format="%H" (if source build)
☑ Check for multiple ROCm installations

REGRESSION? (if you know last good version):
☑ Run BOTH good and bad with TENSILE_DB=0x8040
☑ Compare kernel selections
☑ Attach logs from both

FLAKY ERROR? (if not 100% reproducible):
☑ Test 20+ times, document failure rate
☑ Test at different scales (nodes/GPUs)
☑ Capture 3+ failure logs, 3+ success logs

ALWAYS RUN WITH DEBUG FLAGS:
☑ export TENSILE_DB=0x8040
☑ export HIPBLASLT_LOG_LEVEL=4
☑ export HIPBLASLT_LOG_MASK=0xFFFFFFFF
☑ Attach full log

FOR SEGFAULTS:
☑ ulimit -c unlimited → capture core dump
☑ gdb <binary> core → bt full
☑ Optional: rebuild with --address-sanitizer

FOR GEMM ISSUES:
☑ Try hipblaslt-bench reproduction
☑ Include exact bench command if reproduces

MACHINE ACCESS:
☑ Docker: docker pull <image>
☑ OR Bare metal: Machine/Reservation details

═══════════════════════════════════════════════════════════════
```

---

## 📋 Appendix A: Debug Flag Reference (FAQ)

**⭐ This section is FAQ-worthy - consider posting to wiki/docs**

### Tensile Debug Flags

```bash
# MANDATORY for kernel issues:
export TENSILE_DB=0x8040        # Kernel selection logging

# Individual flags (can combine with |):
export TENSILE_DB=0x1           # Library selection
export TENSILE_DB=0x2           # Solution selection
export TENSILE_DB=0x4           # Kernel execution
export TENSILE_DB=0x8           # Memory operations
export TENSILE_DB=0x10          # Timing info
export TENSILE_DB=0x40          # Kernel selection (included in 0x8040)
export TENSILE_DB=0x8000        # Problem dimensions (included in 0x8040)

# Combine flags: 0x8040 = 0x8000 | 0x40
```

### hipBLASLt Debug Flags

```bash
# Logging level (0=off, 1=error, 2=warn, 3=info, 4=trace)
export HIPBLASLT_LOG_LEVEL=4

# Log mask - all subsystems:
export HIPBLASLT_LOG_MASK=0xFFFFFFFF

# Specific subsystems (can combine with |):
export HIPBLASLT_LOG_MASK=0x1   # API calls
export HIPBLASLT_LOG_MASK=0x2   # Kernel selection
export HIPBLASLT_LOG_MASK=0x4   # Performance data
```

### ROCm/HIP Debug Flags

```bash
# HIP runtime logging (0-4, 4=most verbose)
export AMD_LOG_LEVEL=4

# HSA debugging
export HSA_ENABLE_DEBUG=1

# Control GPU visibility
export HIP_VISIBLE_DEVICES=0,1,2,3  # Specific GPUs
export ROCR_VISIBLE_DEVICES=0       # Specific GPU for ROCr

# Synchronous execution (helps debug async errors)
export HIP_LAUNCH_BLOCKING=1
```

### install.sh Options Reference

```bash
# Common build commands:
./install.sh -idc             # Install + Dependencies + Clients
./install.sh -c --debug       # Clients + Debug build
./install.sh -c --address-sanitizer  # Clients + ASAN build
./install.sh -i               # Install only
./install.sh -d               # Dependencies
./install.sh -c               # Build clients (benchmarks/tests)
./install.sh --debug          # Debug build
./install.sh --address-sanitizer    # ASAN build
```

---

## 📋 Appendix B: Example Perfect issue

```markdown
**Title:** [hipBLASLt][Runtime][MI123] Random segfault when running JAX - hipblasLtMatmulAlgoGetHeuristic

**ERROR TYPE:** Segfault

**REPRODUCIBILITY:** Flaky, 4% single-node to 50% at 4 nodes

**OBSERVED:**
```
Fatal Python error: Segmentation fault
Thread 0x00007f8a4c7fa700:
  hipblasLtMatmulAlgoGetHeuristic
  TensileHost::solutionSelectionLibrary
Segmentation fault (core dumped)
```
Occurs randomly during JAX XLA compilation, before training starts.

**EXPECTED:** JAX compilation completes, training begins

**VERSIONS:**
- ROCm: A.B.C
- hipBLASLt: 1.2.3.be40066
- Docker: my:image
- GPU: MI123 (gfx456), 1-64 nodes, 8 GPUs/node
- Framework: JAX G.H.I

**MULTIPLE INSTALLATIONS:** No conflicts found

**REGRESSION:**
- Last good: JAX D.E.F
- First bad: JAX G.H.I
- Suspect: JAX XLA GEMM compilation changes

**DEBUG LOGS (TENSILE_DB=0x8040):**
- Success (1 node): http://example.com/good.log
- Failure (4 nodes): http://example.com/bad.log
- Comparison: http://example.com/diff.txt

**KERNEL DIFFERENCES:**
```diff
< Selected solution 42 for GEMM(4096,4096,4096)
> Selected solution 157 for GEMM(4096,4096,4096) → SEGFAULT
```

**GEMM BENCH:** Does NOT reproduce with hipblaslt-bench

**CORE DUMP:**
http://example.com/core.12345
```
#0 hipblasLtMatmulAlgoGetHeuristic ()
#1 TensileHost::solutionSelectionLibrary ()
#2 xla::gpu::GemmThunk::ExecuteOnStream ()
```

**MACHINE ACCESS:**
Docker: `docker pull my:image` (public)

**FLAKY DATA:**
| Config | Runs | Failures | Rate |
|--------|------|----------|------|
| 1 node | 50 | 2 | 4% |
| 4 nodes | 20 | 10 | 50% |

**CONTEXT:**
Appears to be race condition in kernel selection when JAX compilation
calls hipblasLtMatmulAlgoGetHeuristic from multiple threads.

Hypothesis: Solution selection database not thread-safe.

Suggested: Check TensileHost thread-safety, add mutex if needed.
```

---

## 📋 Appendix C: Sanitizer Build Reference

### Address Sanitizer (ASAN)

```bash
./install.sh -c --address-sanitizer

export ASAN_OPTIONS="detect_leaks=0:log_path=asan.log"
./run_test.sh
```

---

## 📋 Appendix D: Common Mistakes to Avoid

### ❌ DON'T:

1. File issue without running with `TENSILE_DB=0x8040`
   - ✅ DO: Always enable debug logging first

2. File flaky issue with <20 tests or <10% failure rate
   - ✅ DO: Establish solid reproduction rate

3. Skip regression analysis if you know last good version
   - ✅ DO: Compare good vs bad FIRST - most valuable debug info

4. Provide only partial logs or screenshots
   - ✅ DO: Attach complete logs from start to error

5. Forget to check for multiple ROCm installations
   - ✅ DO: Always run find /opt -name "rocm*"

6. File without environment information
   - ✅ DO: Provide Docker image or machine details

7. Mix multiple unrelated errors in one issue
   - ✅ DO: One issue per issue
