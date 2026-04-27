# Work-Item Commands Implementation Status

## Overview

Implementation of AIROCGDB-427: HIP threads/work-items as first-class citizens in ROCgdb.

**Branch:** `users/sulakshm/work-item-commands`  
**Base:** `amd-staging-rocgdb-16`  
**Approach:** Test-Driven Development (TDD)

---

## Phase 1: Test Infrastructure ✅ COMPLETE

### Test Applications Created

1. **work-item-test.cpp** - Comprehensive HIP test application
   - 2D grid: 2x2 blocks, 8x8 threads (256 work-items)
   - 1D grid: 4 blocks, 32 threads (128 work-items)
   - 3D grid: 2x2x2 blocks, 4x4x4 threads (512 work-items)
   - Breakpoint targets at specific coordinates
   - Selection test points

2. **work-item-guide-example.cpp** - Tutorial application
   - 128x128 matrix addition
   - Documented debugging session
   - Hands-on learning tool

### Test Suites Created

1. **work-item.exp** (600+ lines)
   - **Positive Tests:**
     - Basic work-item selection
     - Flag-based syntax (-bl, -wi)
     - Partial coordinates
     - info work-items display
     - Convenience variables
     - Breakpoints with work-item filters
     - Wildcard breakpoints
     - work-item apply commands
   
   - **Negative Tests:**
     - Invalid block coordinates (out of bounds)
     - Invalid thread coordinates (out of bounds)
     - Invalid syntax
     - Negative coordinates
     - No active dispatch error handling
   
   - **Edge Cases:**
     - Current work-item query
     - Wave boundary conditions
     - Filtered info work-items

2. **work-item-1d.exp** - 1D grid tests
   - 1D work-item selection
   - Convenience variables in 1D
   - Simplified 1D syntax

3. **work-item-3d.exp** - 3D grid tests
   - Full 3D coordinate selection
   - 3D convenience variables
   - 3D breakpoint filtering
   - 3D info work-items display

### Documentation Created

1. **WORK_ITEM_GUIDE.md** (12KB)
   - Complete command reference
   - Quick start guide
   - Usage examples
   - Best practices
   - Troubleshooting guide
   - Coordinate calculation reference

### Commits

✅ `b99e6f4f` - Add work-item test infrastructure and documentation  
✅ `f5bf65db` - Add work-item debugging user guide and example

---

## Phase 2: Core Implementation ✅ COMPLETE

### Commands Implemented

- ✅ `work-item (bx,by,bz)[tx,ty,tz]` - Selection command
  - Files: `gdb/thread.c`
  - Functions: `work_item_command()`, `parse_work_item_coords()`, `find_wave_lane_for_work_item()`
  - Lines: ~250 (implemented)
  - Features: Full/partial coords, flag syntax, validation, error handling

- ✅ `info work-items` - Listing command
  - Files: `gdb/thread.c`
  - Functions: `info_work_items_command()`
  - Lines: ~140 (implemented)
  - Features: Table output, state display, global ID calculation, truncation

- ✅ Convenience variables (7 total)
  - Files: `gdb/thread.c`
  - Variables: `$_work_item_block_{x,y,z}`, `$_work_item_thread_{x,y,z}`, `$_work_item_global_id`
  - Lines: ~180 (implemented)
  - Features: Lazy evaluation, automatic coordinate extraction

- ⏳ Breakpoint extensions (DEFERRED)
  - Reason: Complex integration, requires broader breakpoint refactoring
  - Workaround: Users can use convenience variables in conditions
  - Example: `break kernel.cpp:42 if $_work_item_block_x == 1 && $_work_item_thread_x == 4`

- ✅ Coordinate mapping
  - Files: `gdb/thread.c` (integrated into work_item_command)
  - Uses existing target API: `target_workgroup_grid_pos()`, `target_lane_workgroup_pos()`
  - Lines: ~80 (implemented in find_wave_lane_for_work_item)

### Implementation Summary

- **Total lines added:** 576 in `gdb/thread.c`
- **Commit:** 1ff4dab7352
- **Core functionality:** Complete and ready for testing
- **Advanced features:** Deferred to future enhancement

---

## Testing Status

### Current Status: Implementation Complete, Ready for Testing

Core commands are implemented. Tests should now pass for:
- ✅ work-item selection (full/partial/flag syntax)
- ✅ info work-items display
- ✅ Convenience variables
- ⏳ Breakpoint filtering (use manual conditions instead)
- ⏳ work-item apply (deferred)

### To Run Tests

```bash
cd /home/sulakshm/working/Debugger/TheRock

# Build rocgdb
ninja -C build rocgdb+build

# Run test suite
cd debug-tools/rocgdb/source/gdb/testsuite
make site.exp
runtest gdb.rocm/work-item.exp

# Run all work-item tests
runtest gdb.rocm/work-item*.exp
```

### Expected Test Results

**Current (Phase 1):**
- work-item.exp: FAIL (commands don't exist)
- work-item-1d.exp: FAIL (commands don't exist)
- work-item-3d.exp: FAIL (commands don't exist)

**After Phase 2 Implementation:**
- work-item.exp: PASS
- work-item-1d.exp: PASS
- work-item-3d.exp: PASS

---

## File Locations

### Test Files
```
debug-tools/rocgdb/source/gdb/testsuite/gdb.rocm/
├── work-item-test.cpp          # Test application
├── work-item.exp               # Main test suite
├── work-item-1d.exp            # 1D grid tests
└── work-item-3d.exp            # 3D grid tests
```

### Documentation
```
debug-tools/rocgdb/
├── WORK_ITEM_GUIDE.md                  # User guide
└── work-item-guide-example.cpp         # Example application
```

### Implementation Files (To Be Modified)
```
debug-tools/rocgdb/source/gdb/
├── thread.c                    # Main command implementation
├── amdgpu-tdep.c              # Coordinate mapping
├── breakpoint.c               # Breakpoint extensions
└── mi/mi-cmds.c               # Machine Interface (future)
```

---

## Next Steps

### Immediate (Phase 2)

1. **Implement work-item selection command**
   - Parse coordinate syntax
   - Map coordinates to wave/lane
   - Switch context

2. **Implement info work-items**
   - Iterate waves and lanes
   - Convert to HIP coordinates
   - Format output table

3. **Add convenience variables**
   - Create lazy evaluators
   - Register with GDB

4. **Extend breakpoint system**
   - Parse work-item qualifiers
   - Generate conditions
   - Test with wildcards

### Future Enhancements (Phase 3+)

- [ ] Machine Interface (MI) support
- [ ] Python API bindings
- [ ] Tab completion
- [ ] DAP (Debug Adapter Protocol) support
- [ ] Performance optimization for large grids
- [ ] work-item apply command variants

---

## Success Criteria

1. ✅ All test files created
2. ✅ Tests initially fail (TDD)
3. ✅ Documentation complete
4. ✅ Example application created
5. ⏳ Implementation complete (in progress)
6. ⏳ All tests pass (pending implementation)
7. ⏳ No regressions in existing tests
8. ⏳ Code review and merge

---

## References

- **JIRA:** AIROCGDB-427
- **Branch:** users/sulakshm/work-item-commands
- **Base Commit:** ec2e9fc7a3f (amd-staging-rocgdb-16)
- **Plan:** /home/sulakshm/.claude/plans/streamed-cuddling-tome.md

---

## Build and Test Commands

```bash
# Navigate to TheRock root
cd /home/sulakshm/working/Debugger/TheRock

# Build rocgdb component
ninja -C build rocgdb+build

# Run specific test
cd debug-tools/rocgdb/source/gdb/testsuite
runtest gdb.rocm/work-item.exp

# Run all work-item tests
runtest gdb.rocm/work-item*.exp

# Run full ROCm test suite (regression check)
make check RUNTESTFLAGS='gdb.rocm/*.exp'

# Test example application
cd /home/sulakshm/working/Debugger/TheRock/debug-tools/rocgdb
hipcc -g work-item-guide-example.cpp -o work-item-guide-example
./build/dist/rocm/bin/rocgdb ./work-item-guide-example
```

---

Last Updated: 2026-04-27
