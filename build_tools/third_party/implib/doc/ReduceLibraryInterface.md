Sometimes you may want to reduce public interface of existing closed-source shared library (e.g. if it's a third-party lib which erroneously exports too many unrelated symbols).

To achieve this you can generate a wrapper with limited number of symbols and override the callback which loads the library to use `dlmopen` instead of `dlopen` (and thus does not pollute the global namespace):

```
$ cat mysymbols.txt
foo
bar
$ cat mycallback.c
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
#endif

// Dlopen callback that loads library to dedicated namespace
void *mycallback(const char *lib_name) {
  void *h = dlmopen(LM_ID_NEWLM, lib_name, RTLD_LAZY | RTLD_DEEPBIND);
  if (h)
    return h;
  fprintf(stderr, "dlmopen failed: %s\n", dlerror());
  exit(1);
}

$ implib-gen.py --dlopen-callback=mycallback --symbol-list=mysymbols.txt libxyz.so
$ ... # Link your app with libxyz.tramp.S, libxyz.init.c and mycallback.c
```

Similar approach can be used if you want to provide a common interface for several libraries with partially intersecting interfaces (see [this example](tests/multilib/run.sh) for more details).

