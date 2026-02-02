#!/usr/bin/bash
set -e

echo "Patching ncurses sources to rename main library..."
SOURCE_DIR="${1:?Source directory must be given}"

# Determine if we're patching source files or generated Makefiles
if [ -f "$SOURCE_DIR/configure" ]; then
  CONFIGURE="$SOURCE_DIR/configure"
  echo "Patching source directory (before configure)..."

  # Change LIB_NAME from 'ncurses' to 'rocm_sysdeps_ncurses'
  sed -i 's/^LIB_NAME=ncurses$/LIB_NAME=rocm_sysdeps_ncurses/' "$CONFIGURE"
  # Patch where cf_libname is set from cf_dir for ncurses module
  sed -i 's/^\([[:space:]]*\)cf_libname=$cf_dir$/\1cf_libname=$cf_dir\n\1[ "$cf_dir" = "ncurses" ] \&\& cf_libname="rocm_sysdeps_ncurses"/' "$CONFIGURE"

  # Patch gen-pkgconfig.in to add a case for ncursesw in the name substitution
  # This ensures gen-pkgconfig generates rocm_sysdeps_ncursesw.pc (not ncursesw.pc)
  # which can then be properly renamed by patch_install.py
  if [ -f "$SOURCE_DIR/misc/gen-pkgconfig.in" ]; then
    echo "Patching gen-pkgconfig.in to handle ncursesw library name..."
    # Insert the ncursesw case right after panel*) case but before ncurses++*) case
    # The pattern 'ncursesw*)' is specific and won't match 'ncurses++w'
    sed -i '/panel\*)[[:space:]]*name=.*PANEL_LIBRARY.*;;/a\
	ncursesw*)	name="$MAIN_LIBRARY"	;;' "$SOURCE_DIR/misc/gen-pkgconfig.in"
  fi

else
  # This is the build directory - patch generated Makefiles after configure
  echo "Patching build directory (after configure)..."
  # Patch generated Makefile
  find "$SOURCE_DIR" -name "Makefile" -type f | while read -r makefile; do
    if grep -qE "libncursesw|-lncursesw" "$makefile" 2>/dev/null; then
      echo "  Patching $makefile..."
      # Patch all libncursesw variants (including _g debug library)
      sed -i 's/libncursesw\.so/librocm_sysdeps_ncursesw.so/g' "$makefile"
      sed -i 's/libncursesw\.a/librocm_sysdeps_ncursesw.a/g' "$makefile"
      sed -i 's/libncursesw_g\.a/librocm_sysdeps_ncursesw_g.a/g' "$makefile"
      # Fix linker flags
      sed -i 's/-lncursesw/-lrocm_sysdeps_ncursesw/g' "$makefile"
    fi
  done
fi

echo "NCurses patching completed."
