# Manual test script for work-item commands
set confirm off
file ./work-item-guide-example
break matrix_add
run
# Should be stopped in kernel now
info threads
info work-items
work-item (0,0,0)[0,0,0]
print $_work_item_block_x
print $_work_item_thread_x
work-item (2,3,0)[8,8,0]
print $_work_item_block_x
print $_work_item_block_y
print $_work_item_thread_x
print $_work_item_thread_y
print my_block_x
print my_block_y
print my_thread_x
print my_thread_y
info work-items
quit
