set confirm off
set breakpoint pending on

file /home/sulakshm/working/Debugger/TheRock/debug-tools/rocgdb/source/gdb/testsuite/gdb.rocm/work-item-test

# Test 1: work-item breakpoint with full coordinates
echo \n=== Test 1: Breakpoint with full coordinates ===\n
break work_item_test_kernel work-item (1,0,0)[4,2,0]
info breakpoints

# Test 2: work-item breakpoint with wildcards
echo \n=== Test 2: Breakpoint with wildcards ===\n
break work_item_test_kernel work-item (*,*,0)[0,0,0]
info breakpoints

# Test 3: work-item breakpoint with thread-only
echo \n=== Test 3: Breakpoint with thread-only ===\n
break work_item_test_kernel work-item [7,7,0]
info breakpoints

quit
