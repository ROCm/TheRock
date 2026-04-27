# Work-Item Debugging Guide for ROCgdb

## Overview

The work-item debugging commands allow you to debug HIP kernels using familiar HIP coordinate terminology (blocks and threads) instead of low-level waves and lanes. This makes it much easier to navigate and debug GPU kernels when you're thinking in terms of your source code's coordinate space.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Commands Reference](#commands-reference)
3. [Convenience Variables](#convenience-variables)
4. [Examples](#examples)
5. [Tips and Best Practices](#tips-and-best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Selecting a Work-Item

Select a specific work-item using HIP coordinates:

```gdb
# Select work-item in block (1,0,0), thread [4,2,1]
(gdb) work-item (1,0,0)[4,2,1]
[Switching to work-item (1,0,0)[4,2,1], wave 3, lane 36]

# Alternative syntax using flags
(gdb) work-item -bl 1,0,0 -wi 4,2,1

# Use partial coordinates (inherits current block)
(gdb) work-item [4,2,1]
```

### Listing Work-Items

Display all active work-items with their HIP coordinates:

```gdb
(gdb) info work-items
Wave  Lane  State  Block      Thread     Global-ID
1     0     A      (0,0,0)    [0,0,0]    0
1     1     A      (0,0,0)    [1,0,0]    1
1     5     A      (0,0,0)    [5,0,0]    5
...
3     36    A      (1,0,0)    [4,2,1]    100
...
```

### Setting Breakpoints

Break on specific work-items:

```gdb
# Break on a specific work-item
(gdb) break my_kernel work-item (1,1,0)[4,4,0]
Breakpoint 1 at 0x7fff12345678: file kernel.cpp, line 42, work-item (1,1,0)[4,4,0]

# Break on thread [0,0,0] in any block (wildcard)
(gdb) break my_kernel work-item (*,*,*)[0,0,0]

# Break on all work-items in a specific block
(gdb) break my_kernel work-item (2,1,0)[*,*,*]
```

### Convenience Variables

Access current work-item coordinates:

```gdb
(gdb) print $_work_item_block_x
$1 = 1

(gdb) print $_work_item_thread_y
$2 = 2

(gdb) print $_work_item_global_id
$3 = 100
```

---

## Commands Reference

### `work-item` - Select Work-Item

**Syntax:**
```gdb
work-item (bx,by,bz)[tx,ty,tz]
work-item [tx,ty,tz]
work-item -bl bx,by,bz -wi tx,ty,tz
work-item
```

**Description:**
Switches to a specific work-item identified by its HIP block and thread coordinates.

**Parameters:**
- `(bx,by,bz)` - Block coordinates (X, Y, Z)
- `[tx,ty,tz]` - Thread coordinates within the block (X, Y, Z)
- `-bl` - Flag-based block specification
- `-wi` - Flag-based work-item (thread) specification
- No arguments - Display current work-item

**Examples:**
```gdb
# Full coordinates
(gdb) work-item (1,0,0)[4,2,1]

# Partial coordinates (use current block)
(gdb) work-item [7,3,0]

# Flag-based syntax
(gdb) work-item -bl 0,1,0 -wi 5,5,0

# Query current work-item
(gdb) work-item
[Current work-item is (1,0,0)[4,2,1], wave 3, lane 36]
```

---

### `info work-items` - List Work-Items

**Syntax:**
```gdb
info work-items [OPTION]...
```

**Description:**
Displays all work-items currently known to the debugger, showing their HIP coordinates along with the underlying wave/lane mapping.

**Options:**
- `-active` - Show only active work-items
- `-inactive` - Show only inactive work-items
- `-block bx,by,bz` - Filter by specific block
- `-dispatch N` - Filter by dispatch ID

**Examples:**
```gdb
# Show all work-items
(gdb) info work-items

# Show only active work-items
(gdb) info work-items -active

# Show work-items in a specific block
(gdb) info work-items -block 1,0,0
```

**Output Format:**
```
Wave  Lane  State  Block      Thread     Global-ID
1     0     A      (0,0,0)    [0,0,0]    0
1     1     A      (0,0,0)    [1,0,0]    1
*1    5     A      (0,0,0)    [5,0,0]    5     <- Current work-item
```

**Legend:**
- `*` - Current work-item
- State: `A` (Active), `I` (Inactive), `U` (Unused)

---

### `work-item apply` - Apply Command to Work-Items

**Syntax:**
```gdb
work-item apply (bx,by,bz)[tx,ty,tz] COMMAND
work-item apply all COMMAND
```

**Description:**
Executes a GDB command for specified work-items.

**Examples:**
```gdb
# Print variable for specific work-item
(gdb) work-item apply (1,0,0)[4,4,0] print local_var

# Print coordinates for all work-items in block
(gdb) work-item apply (0,0,0)[*,*,*] print $_work_item_thread_x

# Execute command on all work-items
(gdb) work-item apply all print result
```

---

### Breakpoint Extensions

**Syntax:**
```gdb
break LOCATION work-item (bx,by,bz)[tx,ty,tz]
```

**Description:**
Sets a breakpoint that only triggers for the specified work-item coordinates. Wildcards (`*`) are supported for matching multiple work-items.

**Examples:**
```gdb
# Break on specific work-item
(gdb) break kernel.cpp:42 work-item (1,1,0)[4,4,0]

# Break on thread [0,0,0] in any block
(gdb) break my_kernel work-item (*,*,*)[0,0,0]

# Break on all work-items in block (2,1,0)
(gdb) break my_kernel work-item (2,1,0)[*,*,*]

# Break on specific thread X across all blocks and Y/Z
(gdb) break my_kernel work-item (*,*,*)[5,*,*]
```

---

## Convenience Variables

ROCgdb provides convenience variables to access the current work-item's coordinates:

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `$_work_item_block_x` | Block X coordinate | 1 |
| `$_work_item_block_y` | Block Y coordinate | 0 |
| `$_work_item_block_z` | Block Z coordinate | 0 |
| `$_work_item_thread_x` | Thread X coordinate | 4 |
| `$_work_item_thread_y` | Thread Y coordinate | 2 |
| `$_work_item_thread_z` | Thread Z coordinate | 1 |
| `$_work_item_global_id` | Linear global ID | 100 |

**Usage Examples:**
```gdb
# Print all coordinates
(gdb) print $_work_item_block_x
$1 = 1
(gdb) print $_work_item_thread_y
$2 = 2

# Use in conditional expressions
(gdb) if $_work_item_block_x == 1
> print "In block 1"
> end

# Use in breakpoint conditions
(gdb) break kernel.cpp:42 if $_work_item_thread_x == 0
```

---

## Examples

### Example 1: Debug Specific Thread in Matrix Multiplication

```gdb
# Your kernel: matmul<<<grid(32,32), block(16,16)>>>()
# You want to debug work-item computing result[100][200]

(gdb) break matmul
(gdb) run

# result[100][200] is computed by:
# block(6,12) = (100/16, 200/16)
# thread[4,8] = (100%16, 200%16)

(gdb) work-item (6,12,0)[4,8,0]
[Switching to work-item (6,12,0)[4,8,0]]

(gdb) print row
$1 = 100
(gdb) print col
$2 = 200

(gdb) step
# Now debugging the specific work-item
```

### Example 2: Find Work-Item with Wrong Result

```gdb
# Set breakpoint at result verification
(gdb) break verify_result

(gdb) run
# Hit breakpoint on first work-item

# List all work-items
(gdb) info work-items

# Apply command to find which work-item has wrong result
(gdb) work-item apply all print result[gid]

# Found bad result at work-item (5,3,0)[7,2,0]
(gdb) work-item (5,3,0)[7,2,0]
(gdb) backtrace
(gdb) step
# Debug the problematic work-item
```

### Example 3: Debug Thread [0,0,0] Across All Blocks

```gdb
# Thread [0,0,0] often has special responsibilities
(gdb) break my_kernel work-item (*,*,*)[0,0,0]
Breakpoint 1 at 0x...: work-item (*,*,*)[0,0,0]

(gdb) run
# Hits on every block's thread [0,0,0]

(gdb) print $_work_item_block_x
$1 = 0
(gdb) continue
# Hits next block

(gdb) print $_work_item_block_x
$2 = 1
```

### Example 4: 1D Grid Debugging

```gdb
# For 1D grids, Y and Z are always 0
# vecadd<<<256, 512>>>()

(gdb) work-item (100,0,0)[250,0,0]
[Switching to work-item (100,0,0)[250,0,0]]

# Or use simplified coordinates
(gdb) work-item [250,0,0]

# Global ID = 100 * 512 + 250 = 51450
(gdb) print $_work_item_global_id
$1 = 51450
```

### Example 5: Check All Work-Items in a Wave

```gdb
# Wave 3 contains work-items 192-255 (for wave64)
(gdb) work-item (3,0,0)[0,0,0]    # First work-item in wave 3
(gdb) info lanes
# Shows all 64 lanes in current wave

# Check specific lane's work-item coordinates
(gdb) lane 36
(gdb) print $_work_item_block_x, $_work_item_thread_x
$1 = 3, 36
```

---

## Tips and Best Practices

### 1. Understanding Coordinate Systems

HIP uses a nested coordinate system:
- **Grid**: Contains multiple blocks
- **Block**: Contains multiple threads (work-items)
- **Thread** (Work-Item): The actual executing unit

```
Grid(bx,by,bz) -> Block -> Thread[tx,ty,tz]
```

### 2. 1D, 2D, and 3D Grids

- **1D Grid**: Use `(bx,0,0)[tx,0,0]`
- **2D Grid**: Use `(bx,by,0)[tx,ty,0]`
- **3D Grid**: Use `(bx,by,bz)[tx,ty,tz]`

Always specify all three dimensions, using 0 for unused dimensions.

### 3. Finding the Right Work-Item

Use `info work-items` to see all available work-items and their states. Active work-items are those currently executing.

### 4. Wildcards for Pattern Matching

Use `*` to match any value in breakpoint conditions:
- `(*,*,*)[0,0,0]` - All thread [0,0,0] across all blocks
- `(1,*,*)[*,*,*]` - All work-items in blocks with X=1
- `(*,0,*)[5,*,0]` - All thread X=5, Z=0 in blocks with Y=0

### 5. Combining with Existing Commands

Work-item commands integrate with existing GDB commands:
```gdb
# Navigate by thread/lane, then check work-item coords
(gdb) thread 10
(gdb) lane 5
(gdb) print $_work_item_block_x

# Or navigate by work-item, then use traditional commands
(gdb) work-item (1,0,0)[4,2,1]
(gdb) backtrace
(gdb) info locals
```

### 6. Performance with Large Grids

For kernels with millions of work-items, `info work-items` may be slow. Use filtering options:
```gdb
# Filter by block to reduce output
(gdb) info work-items -block 1,0,0

# Show only active work-items
(gdb) info work-items -active
```

---

## Troubleshooting

### Error: "No active dispatch"

**Problem**: You're trying to use work-item commands before a kernel has launched or after it completed.

**Solution**: Set a breakpoint inside a kernel and ensure the kernel is running:
```gdb
(gdb) break my_kernel
(gdb) run
(gdb) work-item (0,0,0)[0,0,0]
```

### Error: "Work-item out of bounds"

**Problem**: The coordinates you specified don't exist in the current grid.

**Solution**: Check your grid/block dimensions:
```gdb
(gdb) info work-items
# Look at the range of valid coordinates

# Or check grid dimensions in code
(gdb) print gridDim
(gdb) print blockDim
```

### Error: "Invalid syntax"

**Problem**: Coordinates are not properly formatted.

**Solution**: Ensure you use parentheses for blocks and brackets for threads:
```gdb
# Correct:
(gdb) work-item (1,0,0)[4,2,1]

# Incorrect:
(gdb) work-item 1,0,0,4,2,1
(gdb) work-item [1,0,0](4,2,1)
```

### Work-Item State is "Inactive"

**Problem**: The work-item exists but is not currently active (diverged control flow).

**Explanation**: Due to SIMD execution, some lanes may be inactive when branches diverge. Inactive work-items can still be inspected but aren't currently executing.

**Solution**: This is normal. You can still select and inspect inactive work-items:
```gdb
(gdb) work-item (1,0,0)[5,0,0]
(gdb) print variable_name
```

### Multiple Dispatches Active

**Problem**: Multiple kernels are running concurrently, and coordinates are ambiguous.

**Solution**: Specify the dispatch ID:
```gdb
(gdb) info dispatches
# Find the dispatch ID you want

(gdb) work-item -dispatch 1 (0,0,0)[0,0,0]
```

---

## Coordinate Calculation Reference

### Linear to Block/Thread Conversion

Given a linear global ID `gid`, convert to block/thread coordinates:

```python
# For 2D grid: grid(GX, GY), block(BX, BY)
block_x = (gid / (BX * BY)) % GX
block_y = (gid / (BX * BY)) / GX
thread_x = (gid % (BX * BY)) % BX
thread_y = (gid % (BX * BY)) / BX
```

### Block/Thread to Linear Conversion

Given block `(bx, by, bz)` and thread `[tx, ty, tz]`:

```python
# Block size
block_size = BX * BY * BZ

# Block linear ID
block_id = bz * (GY * GX) + by * GX + bx

# Thread within block
thread_in_block = tz * (BY * BX) + ty * BX + tx

# Global ID
global_id = block_id * block_size + thread_in_block
```

### Wave/Lane Calculation

```python
# For wave64 architectures
wave_id = global_id / 64
lane_id = global_id % 64

# For wave32 architectures
wave_id = global_id / 32
lane_id = global_id % 32
```

---

## Additional Resources

- **ROCgdb Documentation**: See the main ROCgdb user guide for general debugging information
- **HIP Programming Guide**: Understanding HIP's execution model helps with coordinate navigation
- **Example Programs**: See `work-item-guide-example.cpp` for a complete working example

---

## Feedback and Issues

If you encounter issues or have suggestions for improving the work-item debugging commands, please file an issue at:
https://github.com/ROCm/rocgdb/issues
