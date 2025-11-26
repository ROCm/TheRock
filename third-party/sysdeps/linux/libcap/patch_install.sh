#!/bin/bash
set -e

PREFIX="${1:?Expected install prefix argument}"
PATCHELF="${PATCHELF:-patchelf}"
THEROCK_SOURCE_DIR="${THEROCK_SOURCE_DIR:?THEROCK_SOURCE_DIR not defined}"
Python3_EXECUTABLE="${Python3_EXECUTABLE:?Python3_EXECUTABLE not defined}"

declare -A _unique_targets
while IFS= read -r -d '' candidate; do
  real_path=$(readlink -f "$candidate" 2>/dev/null || printf '%s' "$candidate")
  if [ -f "$real_path" ]; then
    _unique_targets["$real_path"]=1
  fi
done < <(find "${PREFIX}/lib" -maxdepth 1 \( -name "libcap.so*" -o -name "libpsx.so*" \) -print0)

patch_targets=("${!_unique_targets[@]}")
if [ ${#patch_targets[@]} -eq 0 ]; then
  echo "libcap patch_install: no shared libraries found under ${PREFIX}/lib" >&2
  exit 1
fi

"$Python3_EXECUTABLE" "$THEROCK_SOURCE_DIR/build_tools/patch_linux_so.py" \
  --patchelf "${PATCHELF}" --add-prefix rocm_sysdeps_ \
  "${patch_targets[@]}"

for target in "${patch_targets[@]}"; do
  if [ -f "$target" ]; then
    base_name=$(basename "$target")
    "${PATCHELF}" --set-soname "$base_name" "$target"
  fi
done

# pc files are not output with a relative prefix. Sed it to relative.
if [ -d "$PREFIX/lib/pkgconfig" ]; then
  for pcfile in "$PREFIX/lib/pkgconfig"/*.pc; do
    if [ -f "$pcfile" ]; then
      sed -i -E 's|^prefix=.+|prefix=${pcfiledir}/../..|' "$pcfile"
      sed -i -E 's|^exec_prefix=.+|exec_prefix=${pcfiledir}/../..|' "$pcfile"
      sed -i -E 's|^libdir=.+|libdir=${prefix}/lib|' "$pcfile"
      sed -i -E 's|^includedir=.+|includedir=${prefix}/include|' "$pcfile"
    fi
  done
fi
