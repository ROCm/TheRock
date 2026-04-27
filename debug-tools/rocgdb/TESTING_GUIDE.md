# Testing Guide for Work-Item Commands

## Overview

This guide explains how to build and test the new work-item debugging commands in ROCgdb.

## What Has Been Implemented

### Core Commands ✅

1. **work-item** - Select work-item by HIP coordinates
2. **info work-items** - List all work-items with coordinates  
3. **Convenience variables** - 7 variables for work-item coordinates

### Implementation Details

- **Files modified:** `gdb/thread.c` (+576 lines)
- **Test files created:** 4 test files (900+ lines)
- **Documentation:** Complete user guide and examples

## Building ROCgdb

### Option 1: Build within TheRock (Recommended)

```bash
cd /home/sulakshm/working/Debugger/TheRock

# Configure TheRock (first time only)
cmake -B build -GNinja \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_ROCGDB=ON \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100

# Build rocgdb
ninja -C build rocgdb

# ROCgdb binary will be at:
# build/rocgdb/stage/opt/rocm/bin/rocgdb
```

### Option 2: Build rocgdb Standalone

```bash
cd /home/sulakshm/working/Debugger/TheRock/debug-tools/rocgdb/source

# Configure
mkdir build-standalone
cd build-standalone
../configure --prefix=/tmp/rocgdb-test \
  --enable-targets=x86_64-linux-gnu,amdgcn-amd-amdhsa \
  --with-python=python3

# Build (this will take 30-60 minutes)
make -j$(nproc)

# Install
make install

# ROCgdb binary will be at:
# /tmp/rocgdb-test/bin/gdb
```

### Option 3: Quick Syntax Check (No Full Build)

```bash
cd /home/sulakshm/working/Debugger/TheRock/debug-tools/rocgdb/source

# Just compile thread.c to check for syntax errors
g++ -c gdb/thread.c \
  -I. -Igdb -I../include \
  -DHAVE_CONFIG_H \
  -std=c++17 \
  -o /tmp/thread.o 2>&1 | head -50
```

## Running the Test Suite

### Prerequisites

1. ROCgdb must be built
2. HIP runtime must be available
3. AMD GPU must be present (or use emulation)

### Build Test Applications

```bash
cd /home/sulakshm/working/Debugger/TheRock/debug-tools/rocgdb/source/gdb/testsuite/gdb.rocm

# Compile test applications
hipcc -g work-item-test.cpp -o work-item-test

# Verify it runs
./work-item-test
```

### Run Tests with DejaGnu

```bash
cd /home/sulakshm/working/Debugger/TheRock/debug-tools/rocgdb/source/gdb/testsuite

# Create site.exp (first time only)
make site.exp

# Run work-item tests
runtest gdb.rocm/work-item.exp

# Run all work-item tests
runtest gdb.rocm/work-item*.exp

# Check results
cat gdb.sum | grep "^PASS\|^FAIL"
```

### Expected Test Results

**With Full Implementation:**
- work-item.exp: 16 tests (should mostly PASS)
- work-item-1d.exp: 2 tests (should PASS)
- work-item-3d.exp: 4 tests (should PASS)

**Known Limitations:**
- Breakpoint filtering tests will FAIL (not yet implemented)
- work-item apply tests will FAIL (not yet implemented)
- Other tests should PASS

## Manual Testing

### Test the Example Application

```bash
cd /home/sulakshm/working/Debugger/TheRock/debug-tools/rocgdb

# Compile example
hipcc -g work-item-guide-example.cpp -o work-item-guide-example

# Run under rocgdb
rocgdb ./work-item-guide-example
```

### Interactive Testing Session

```gdb
# In rocgdb:
(gdb) break matrix_add
Breakpoint 1 at 0x...: file work-item-guide-example.cpp, line 23.

(gdb) run
...
Breakpoint 1, matrix_add(...) at work-item-guide-example.cpp:23

# Test work-item selection
(gdb) work-item (2,3,0)[8,8,0]
[Switching to work-item (2,3,0)[8,8,0], thread N, lane M]

# Verify coordinates
(gdb) print my_block_x, my_block_y
$1 = 2
$2 = 3

(gdb) print my_thread_x, my_thread_y
$3 = 8
$4 = 8

# Test convenience variables
(gdb) print $_work_item_block_x
$5 = 2

(gdb) print $_work_item_thread_y
$6 = 8

(gdb) print $_work_item_global_id
$7 = 2248

# Test info work-items
(gdb) info work-items
Wave  Lane  State  Block      Thread     Global-ID  Target-ID
...
*42    25    A      (2,3,0)    [8,8,0]    2248       AMDGPU Lane...
...

# Test partial coordinates
(gdb) work-item [5,5,0]
[Switching to work-item (2,3,0)[5,5,0]...]

# Test query current
(gdb) work-item
[Current work-item is (2,3,0)[5,5,0], thread 42, lane 21]

# Test error handling
(gdb) work-item (99,99,0)[0,0,0]
Error: Block coordinate (99,99,0) is out of bounds for grid (8,8,1)

# Test with no active kernel
(gdb) delete breakpoints
(gdb) run
Program exited normally.
(gdb) work-item (0,0,0)[0,0,0]
Error: No thread selected
```

### Quick Validation Checklist

- [ ] `work-item (bx,by,bz)[tx,ty,tz]` switches to correct lane
- [ ] `work-item -bl bx,by,bz -wi tx,ty,tz` flag syntax works
- [ ] `work-item [tx,ty,tz]` uses current block
- [ ] `work-item` (no args) displays current work-item
- [ ] `info work-items` displays table with coordinates
- [ ] `$_work_item_block_x/y/z` return correct values
- [ ] `$_work_item_thread_x/y/z` return correct values
- [ ] `$_work_item_global_id` calculates correctly
- [ ] Out-of-bounds coordinates rejected with error
- [ ] Invalid syntax rejected with error
- [ ] Works with 1D grids (y=z=0)
- [ ] Works with 2D grids (z=0)
- [ ] Works with full 3D grids

## Troubleshooting

### Build Errors

**Problem:** Missing dependencies

```bash
# Install required packages (Ubuntu/Debian)
sudo apt-get install build-essential texinfo python3-dev \
  libexpat1-dev libncurses5-dev libgmp-dev libmpfr-dev \
  libipt-dev libbabeltrace-dev liblzma-dev
```

**Problem:** HIP not found

```bash
# Ensure ROCm is in PATH
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### Runtime Errors

**Problem:** "No thread selected" when kernel is running

**Solution:** Ensure you're stopped at a breakpoint inside the kernel

**Problem:** Work-item coordinates don't match expected

**Solution:** Check that you're using the correct block/thread dimensions from your kernel launch

**Problem:** info work-items shows no results

**Solution:** Ensure a kernel is actively stopped at a breakpoint

### Test Failures

**Problem:** Tests fail to build

```bash
# Ensure hipcc is in PATH
which hipcc

# Check GPU availability
rocminfo | grep "Marketing Name"
```

**Problem:** Breakpoint tests fail

**Expected:** Breakpoint filtering not fully implemented yet. Use manual conditions:
```gdb
break kernel if $_work_item_block_x == 1 && $_work_item_thread_x == 4
```

## Performance Notes

### Large Grids

For kernels with >10,000 work-items, `info work-items` limits output to 1000 items to prevent overwhelming the terminal.

To see specific work-items in large grids:
1. Use `work-item` to navigate directly to the work-item of interest
2. Use `info threads` to see waves, then navigate by wave/lane
3. Future enhancement: add filtering options to `info work-items`

### Build Time

- **Full rocgdb build:** 30-60 minutes (first time)
- **Incremental rebuild:** 2-5 minutes (after changes)
- **Syntax check only:** <1 second

## Next Steps

After validating the core implementation:

1. **Performance testing** - Test with grids of 100K+ work-items
2. **Breakpoint integration** - Implement native `break kernel work-item` syntax
3. **work-item apply** - Implement command application to specific work-items
4. **MI support** - Add Machine Interface for IDE integration
5. **Python API** - Expose to GDB Python for scripting
6. **Tab completion** - Add auto-completion for coordinates

## Reporting Issues

If you find bugs or have suggestions:

1. Check the test results: `cat gdb.sum`
2. Note your GPU model: `rocminfo | grep gfx`
3. Capture GDB output with `set debug threads on`
4. File issue with reproducible test case

## References

- User Guide: `WORK_ITEM_GUIDE.md`
- Implementation Status: `WORK_ITEM_IMPLEMENTATION_STATUS.md`
- Example Application: `work-item-guide-example.cpp`
- Test Suite: `source/gdb/testsuite/gdb.rocm/work-item*.exp`
