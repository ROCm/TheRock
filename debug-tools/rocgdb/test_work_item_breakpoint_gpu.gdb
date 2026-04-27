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

# Now we're just before kernel launch
# Set work-item breakpoint for specific work-item
echo \n=== Setting work-item breakpoint ===\n
break work_item_test_kernel work-item (1,0,0)[4,2,0]
info breakpoints

# Continue - should hit only the specified work-item
echo \n=== Continuing to work-item breakpoint ===\n
continue

# Check that we hit the right work-item
echo \n=== Verifying we're at correct work-item ===\n
print $_work_item_block_x
print $_work_item_block_y
print $_work_item_thread_x
print $_work_item_thread_y
work-item

quit
