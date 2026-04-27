set confirm off
set breakpoint pending on

# Load the test application
file /home/sulakshm/working/Debugger/TheRock/debug-tools/rocgdb/source/gdb/testsuite/gdb.rocm/work-item-test

# Break on main first
break main
run

# Now break on the kernel - use line number to be sure
break work-item-test.cpp:115

# Continue to kernel
continue

# If we get here, we're in the kernel!
echo \n=== SUCCESS: Stopped in kernel! ===\n
info threads
info work-items

# Test work-item commands
work-item (0,0,0)[0,0,0]
print $_work_item_block_x
print $_work_item_thread_x

work-item (1,1,0)[4,4,0]
print $_work_item_block_x
print $_work_item_block_y
print $_work_item_thread_x
print $_work_item_thread_y

echo \n=== All tests completed! ===\n
quit
