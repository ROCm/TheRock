#include <dlfcn.h>
#include <stdio.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Syntax error: Expected library path\n");
    return 1;
  }
  char *lib_path = argv[1];
  int rc;
  void *h = dlopen(lib_path, RTLD_NOW);
  if (!h) {
    fprintf(stderr, "ERROR: Could not dlopen \"%s\"\n", lib_path);
    return 1;
  }
  int(*hipRuntimeGetVersion)(int*) = (int(*)(int*))dlsym(h, "hipRuntimeGetVersion");
  if (!hipRuntimeGetVersion) {
    fprintf(stderr, "ERROR: Could not resolve symbol hipRuntimeGetVersion\n");
    dlclose(h);
    return 1;
  }
  int version = -1;
  rc = hipRuntimeGetVersion(&version);
  if (rc != 0) {
    fprintf(stderr, "ERROR: hipRuntimeGetVersion returned %d\n", rc);
    return 2;
  }
  printf("HIP VERSION: %x\n", version);

  rc = dlclose(h);
  if (rc != 0) {
    fprintf(stderr, "ERROR: dlclose(): %d\n", rc);
    return 3;
  }
  return 0;
}
