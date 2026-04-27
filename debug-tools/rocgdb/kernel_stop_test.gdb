set confirm off
set breakpoint pending on
set non-stop off

file /home/sulakshm/working/Debugger/TheRock/debug-tools/rocgdb/source/gdb/testsuite/gdb.rocm/work-item-test

# Start program
break main
run

# Set breakpoint on kernel dispatch line
break work-item-test.cpp:197
continue

# Now we're just before kernel launch
# Set breakpoint in the kernel code using the mangled name
break work_item_test_kernel

# Continue - should hit kernel
set pagination off
continue

# We should be in kernel now
echo \n=== Checking if we're in kernel ===\n
info threads
backtrace

# Try our commands
info work-items
work-item
quit
