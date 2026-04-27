set confirm off
file ./work-item-guide-example
set breakpoint pending on
break matrix_add
run
info functions matrix
info sharedlibrary
quit
