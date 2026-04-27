# Quick test script for work-item commands
set confirm off
break matrix_add
run
# Test work-item command (should work if stopped at breakpoint)
work-item
# Test info work-items (should display table)
info work-items
# Test convenience variables
print $_work_item_block_x
print $_work_item_thread_x
print $_work_item_global_id
quit
