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

# Set wildcard breakpoint - all blocks, thread [0,0,0] only
echo \n=== Setting wildcard work-item breakpoint ===\n
break work_item_test_kernel work-item (*,*,*)[0,0,0]
info breakpoints 3

# Continue - should hit thread [0,0,0] in any block
echo \n=== Continuing - should hit thread [0,0,0] ===\n
continue

# Verify
echo \n=== Checking: should be thread [0,0,0] in some block ===\n
print $_work_item_thread_x
print $_work_item_thread_y
print $_work_item_thread_z
work-item

quit
