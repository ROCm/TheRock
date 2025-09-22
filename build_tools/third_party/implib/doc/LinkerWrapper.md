Generation of wrappers may be automated via linker wrapper `scripts/ld`.
Adding it to `PATH` (in front of normal `ld`) would by default result
in all dynamic libs (besides system ones) to be replaced with wrappers.
Explicit list of libraries can be specified by exporting
`IMPLIBSO_LD_OPTIONS` environment variable:
```
export IMPLIBSO_LD_OPTIONS='--wrap-libs attr,acl'
```
For more details run with
```
export IMPLIBSO_LD_OPTIONS=--help
```

Atm linker wrapper is only meant for testing.
