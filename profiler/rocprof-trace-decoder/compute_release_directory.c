#include <features.h>
#include <stdio.h>

int main(int argc, char **argv) {
#ifdef __GLIBC__
  int glibc_major = __GLIBC__;
  int glibc_minor = __GLIBC_MINOR__;

  // We have binaries for a sparse set of glibc versions. We generally want
  // the binary targeting the newest glibc that we support. Populate an
  // available version.
  int selected_glibc_major = 0;
  int selected_glibc_minor = 0;

  if (glibc_major == 2) {
    selected_glibc_major = 2;
    if (glibc_minor >= 28) {
      selected_glibc_minor = 28;
    } else {
      fprintf(stderr, "unsupported glibc minor version %d.%d\n", glibc_major,
              glibc_minor);
      return 1;
    }
  } else {
    fprintf(stderr, "unsupported glibc version %d.%d\n", glibc_major,
            glibc_minor);
    return 1;
  }

  // Now select the available architecture.
  const char *selected_arch = 0;
#ifdef __x86_64__
  selected_arch = "x86_64";
#endif

  if (!selected_arch) {
    fprintf(stderr, "unsupported machine architecture\n");
    return 1;
  }

  printf("linux_glibc_%d_%d_%s\n", selected_glibc_major, selected_glibc_minor,
         selected_arch);
  return 0;
#endif

  fprintf(stderr, "unsupported standard c library\n");
  return 1;
}
