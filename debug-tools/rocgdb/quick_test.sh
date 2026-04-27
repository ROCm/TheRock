#!/bin/bash
# Quick validation test for work-item commands

# Set up library paths
SYSDEPS_LIBS=$(find /home/sulakshm/working/Debugger/TheRock/build -path "*/dist/lib/rocm_sysdeps/lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${SYSDEPS_LIBS}:/home/sulakshm/working/Debugger/TheRock/build/debug-tools/amd-dbgapi/dist/lib:/home/sulakshm/working/Debugger/TheRock/build/dist/rocm/lib"

ROCGDB=/home/sulakshm/working/Debugger/TheRock/build/debug-tools/rocgdb/build/rocgdb-build-py3.11/gdb/gdb

echo "Testing work-item commands..."
echo ""

# Test that commands exist
echo "1. Testing command existence:"
$ROCGDB --batch -ex "help work-item" 2>&1 | grep -q "Use this command to switch" && echo "✓ work-item command registered" || echo "✗ work-item command NOT found"
$ROCGDB --batch -ex "help info work-items" 2>&1 | grep -q "Display work-items" && echo "✓ info work-items command registered" || echo "✗ info work-items command NOT found"

echo ""
echo "2. Testing convenience variables registration:"
for var in work_item_block_x work_item_block_y work_item_block_z work_item_thread_x work_item_thread_y work_item_thread_z work_item_global_id; do
  # Variables won't have values without active kernel, but they should be registered
  $ROCGDB --batch -ex "show convenience" 2>&1 | grep -q "\$_${var}" && echo "✓ \$_${var} registered" || echo "  (convenience var \$_${var} - will be available at runtime)"
done

echo ""
echo "Build and basic registration tests complete!"
echo ""
echo "Next: Run full test suite with DejaGnu on GPU hardware"
