Sometimes you may need to rename API of existing shared library to avoid name clashes.

To achieve this you can generate a wrapper with _renamed_ symbols which call to old, non-renamed symbols in original library loaded via `dlmopen` instead of `dlopen` (to avoid polluting global namespace):

```
$ cat mycallback.c
... Same as before ...
$ implib-gen.py --dlopen-callback=mycallback --symbol_prefix=MYPREFIX_ libxyz.so
$ ... # Link your app with libxyz.tramp.S, libxyz.init.c and mycallback.c
```
