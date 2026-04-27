# Simple GPU test that should work
set confirm off
set breakpoint pending on

file /home/sulakshm/working/Debugger/TheRock/debug-tools/rocgdb/source/gdb/testsuite/gdb.rocm/work-item-test

# Break on the kernel function itself
break work_item_test_2d_kernel

# Run
run

# Should be stopped in kernel now - test our commands
echo \n=== Testing work-item commands ===\n
info threads
info work-items
echo \n=== Test 1: Query current work-item ===\n
work-item
echo \n=== Test 2: Select specific work-item ===\n
work-item (0,0,0)[0,0,0]
echo \n=== Test 3: Check convenience variables ===\n
print $_work_item_block_x
print $_work_item_block_y
print $_work_item_thread_x
print $_work_item_thread_y
print $_work_item_global_id
echo \n=== Test 4: Select different work-item ===\n
work-item (1,0,0)[4,2,0]
print $_work_item_block_x
print $_work_item_thread_x
echo \n=== Test passed! ===\n
quit
