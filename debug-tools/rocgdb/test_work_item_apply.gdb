set confirm off
set breakpoint pending on
set non-stop off
set pagination off

file /home/sulakshm/working/Debugger/TheRock/debug-tools/rocgdb/source/gdb/testsuite/gdb.rocm/work-item-test

# Start program
break main
run

# Set breakpoint on kernel dispatch line
break work-item-test.cpp:197
continue

# Set breakpoint in kernel
break work_item_test_kernel
continue

# Test 1: Apply to specific work-item
echo \n=== Test 1: Apply to specific work-item ===\n
work-item apply (1,0,0)[4,2,0] print $_work_item_global_id

# Test 2: Apply to all threads in a block (should match 64 work-items)
echo \n=== Test 2: Apply to all threads in block (0,0,0) ===\n
work-item apply (0,0,0) print $_work_item_thread_x

# Test 3: Apply to thread [0,0,0] across all blocks (4 blocks = 4 work-items)
echo \n=== Test 3: Apply to thread [0,0,0] in all blocks ===\n
work-item apply [0,0,0] print $_work_item_block_x

quit
