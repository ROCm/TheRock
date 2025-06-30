#!/bin/bash
# Does basic sanity checks of the setup_ccache.py mechanism. This is meant to
# be run interactively when updating setup_ccache.py without the need of
# verifying a full build.
set -e

td="$(cd $(dirname $0) && pwd)"
repo_root="$(cd $td/../../.. && pwd)"
setup_ccache_script="$repo_root/build_tools/setup_ccache.py"

eval "$($setup_ccache_script --dir $td/.ccache --init)"
echo "CCACHE_CONFIGPATH=$CCACHE_CONFIGPATH"

CXX="${CXX:-c++}"
ccache --version
ccache --zero-stats
rm -f $td/hello.o $td/hello.o.*
ccache debug=true $CXX -c $td/hello.cc -o $td/hello.o
ccache --show-stats
