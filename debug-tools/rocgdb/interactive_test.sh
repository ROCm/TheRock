#!/bin/bash
# Interactive GPU test for work-item commands

cd /home/sulakshm/working/Debugger/TheRock/debug-tools/rocgdb

SYSDEPS_LIBS=$(find /home/sulakshm/working/Debugger/TheRock/build -path "*/dist/lib/rocm_sysdeps/lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${SYSDEPS_LIBS}:/home/sulakshm/working/Debugger/TheRock/build/debug-tools/amd-dbgapi/dist/lib:/home/sulakshm/working/Debugger/TheRock/build/dist/rocm/lib"
ROCGDB=/home/sulakshm/working/Debugger/TheRock/build/debug-tools/rocgdb/build/rocgdb-build-py3.11/gdb/gdb

echo "Starting ROCgdb with work-item-guide-example..."
echo "At the (gdb) prompt, try:"
echo "  set breakpoint pending on"
echo "  break matrix_add"
echo "  run"
echo "  # When stopped:"
echo "  info work-items"
echo "  work-item (2,3,0)[8,8,0]"
echo "  print \$_work_item_block_x"
echo ""

$ROCGDB ./work-item-guide-example
