# Copyright 2019-2025 Yury Gribov
#
# The MIT License (MIT)
# 
# Use of this source code is governed by MIT license that can be
# found in the LICENSE.txt file.

if test -n "${TRAVIS:-}"; then
  set -x
fi

if test -z "${ARCH:-}"; then
  ARCH=$(uname -m)
fi

CFLAGS="-Wall -Wextra -Wconversion -Wsign-conversion -Wuninitialized -Werror ${CFLAGS:-}"

if test $ARCH = $(uname -m); then
  # Native
  TARGET=$ARCH
  PREFIX=
  INTERP=
else
  # Simulate
  # (see .github/workflows/ci.yml for list of needed packages)
  case "$ARCH" in
  arm*hf | armhf-*)
    TARGET=armhf
    PREFIX=arm-linux-gnueabihf-
    INTERP="qemu-arm -L /usr/arm-linux-gnueabihf -E LD_LIBRARY_PATH=.${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    CFLAGS="$CFLAGS -mthumb-interwork -mfpu=neon"
    ;;
  arm | arm-* | armel-*)
    TARGET=arm
    PREFIX=arm-linux-gnueabi-
    INTERP="qemu-arm -L /usr/arm-linux-gnueabi -E LD_LIBRARY_PATH=.${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
  aarch64*)
    TARGET=aarch64
    PREFIX=aarch64-linux-gnu-
    INTERP="qemu-aarch64 -L /usr/aarch64-linux-gnu -E LD_LIBRARY_PATH=.${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
  i[0-9]86*)
    TARGET=i686
    PREFIX=
    INTERP=
    CFLAGS="$CFLAGS -m32"
    ;;
  mips | mips-*)
    TARGET=mips
    PREFIX=mips-linux-gnu-
    INTERP="qemu-mips -L /usr/mips-linux-gnu -E LD_LIBRARY_PATH=.${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
  mipsel | mipsel-*)
    TARGET=mipsel
    PREFIX=mipsel-linux-gnu-
    INTERP="qemu-mipsel -L /usr/mipsel-linux-gnu -E LD_LIBRARY_PATH=.${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
  mips64el | mips64el-*)
    TARGET=mips64el
    PREFIX=mips64el-linux-gnuabi64-
    INTERP="qemu-mips64el -L /usr/mips64el-linux-gnuabi64 -E LD_LIBRARY_PATH=.${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
  ppc64 | powerpc64 | powerpc64-*)
    TARGET=powerpc64
    PREFIX=powerpc64-linux-gnu-
    INTERP="qemu-ppc64 -L /usr/powerpc64-linux-gnu -E LD_LIBRARY_PATH=.${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
  ppc64le | powerpc64le | powerpc64le-*)
    TARGET=powerpc64le
    PREFIX=powerpc64le-linux-gnu-
    INTERP="qemu-ppc64le -L /usr/powerpc64le-linux-gnu -E LD_LIBRARY_PATH=.${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
  rv64 | riscv4 | riscv64-*)
    TARGET=riscv64
    PREFIX=riscv64-linux-gnu-
    INTERP="qemu-riscv64 -L /usr/riscv64-linux-gnu -E LD_LIBRARY_PATH=.${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
  *)
    echo >&2 "Unsupported target: $ARCH"
    exit 1
    ;;
  esac
fi

CFLAGS="-Wall -Wextra -Werror ${CFLAGS:-}"

if uname | grep -q BSD; then
  CC=$PREFIX${CC:-cc}
  CXX=$PREFIX${CXX:-c++}
else
  CC=$PREFIX${CC:-gcc}
  CXX=$PREFIX${CXX:-g++}
fi

if uname | grep -q BSD; then
  LIBS='-pthread'
elif echo "$CC" | grep -q musl-gcc; then
  # Just libc.so in musl
  LIBS=
else
  LIBS='-ldl -pthread'
fi

# Do not bother with non-native targets
if test -z "$INTERP" \
    && ! echo "$CC" | grep -q musl-gcc \
    && uname | grep -q Linux \
    && echo 'int main() {}' | $CC $CFLAGS -fsanitize=thread -x c - -o /dev/null 2> /dev/null; then
  TSAN_AVAILABLE=1
else
  TSAN_AVAILABLE=
fi
