#!/bin/bash
# Applies patches that we are maintaining on the ROCM sources.
# These are developed against HEAD.
# Must be run from the repo checkout dir.
set -euxo pipefail

repo_dir="$(pwd)"
this_dir="$(cd $(dirname $0) && pwd)"
root_dir="$(cd $this_dir/.. && pwd)"

function stash_changes() {
  local repo_name="$1"
  cd $repo_dir/$repo_name
  git add -A
  git stash
}

function apply_patch() {
  local repo_name="$1"
  local patch_file="$2"
  cd $repo_dir/$repo_name
  echo "Applying $patch_file to $repo_name"
  patch -p1 < $root_dir/patches/$patch_file
}

echo "Running from $PWD"

# Example:
# stash_changes clr
# apply_patch clr clr-disable-hipconfig-check.patch
# apply_patch clr clr-respect-no-versioned-soname.patch
