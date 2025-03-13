#!/bin/bash
# Saves patches that are maintained locally to make TheRock work with a given
# branch.
# Usage: save_patches.sh base_tag project_name patch_subdir
#
# Example:
#   save_patches.sh m/amd-mainline hipBLASLt amd-mainline
#
# This will save patch files for all commits to hipBLASLt since the commit
# "m/amd-mainline". Patch files will be saved in patches/amd-mainline/hipBLASLT
#
# The base_tag must be checked out on the current branch. All commits above it
# will be dumped as patches.
set -euo pipefail

this_dir="$(cd $(dirname $0) && pwd)"
root_dir="$(cd $this_dir/.. && pwd)"

# Command line args.
set +eu
version_tag="$1"
shift
project_name="$1"
shift
patch_subdir="$1"
shift
set -eu

if [ -z "$version_tag" ] || [ -z "$project_name" ] || [ -z "$patch_subdir" ]; then
  echo "Syntax: save_patches.sh <version_tag> <project_name> <patch_subdir>"
  exit 1
fi

# Directories.
patch_dir="$root_dir/patches/$patch_subdir/$project_name"
mkdir -p $patch_dir
source_dir="$(git config --file $root_dir/.gitmodules --get submodule.${project_name}.path)"
if ! [ -d "$source_dir" ]; then
  echo "Source directory not found: $source_dir"
  exit 1
fi

# Remove existing patch files.
for existing in $patch_dir/*.patch; do
  if [ -f "$existing" ]; then
    rm "$existing"
  fi
done

# And format.
(cd $source_dir && git format-patch -o $patch_dir "$version_tag")
